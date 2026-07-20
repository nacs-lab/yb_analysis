"""Box/image TEMPORAL synchronization monitor.

Detects the ONE error this is meant to find: the box overlay and the image it is
drawn over coming from DIFFERENT published frames -- a temporal mismatch, not a
detection or grid-drift question.

Why it can happen (see dashboard.py):
  * `DashboardRenderer.update()` publishes ONE atomic snapshot per frame: the
    per-site overlay (`logicals`/`grid_locations`) and the frame's PNG are
    encoded together and stamped with a monotonic `_write_seq`.
  * BUT the browser (and any client) fetches the two pieces with SEPARATE HTTP
    requests: the overlay via `GET /api/snapshot`, the pixels via
    `GET /api/live/image{1,2}`. Each independently reads the *current* shared
    buffer. If `update()` publishes a new frame BETWEEN those two requests, the
    client composites frame K's boxes over frame K+1's image (or vice-versa) ->
    the green/red boxes belong to a different shot than the atoms shown.
  * The window widens with array size: the snapshot pickle (~MB of per-site
    arrays) and the PNG are both larger, so the two fetches take longer and are
    more likely to straddle a publish. Hence "worse at large arrays".

The test is an IDENTITY check, not a pixel check:
  1. GET /api/snapshot            -> S_boxes = _write_seq (the overlay's frame)
  2. GET /api/live/image{1,2}     -> S_img  = X-Frame-Seq header (the PNG's frame)
  3. If S_img != S_boxes, the overlay and the image are from different published
     frames == a temporal box/image mismatch this poll.

That is the whole diagnosis. No thresholds, no re-detection, no grid math -- it
compares which shot each half came from.

Two identification methods, best-available:
  * EXACT -- the `X-Frame-Seq` response header on /api/live/imageN (added to
    dashboard.py alongside this script). Needs a backend restart to deploy.
  * DEGRADED (no restart needed) -- the per-frame (vlo, vhi, shape) fingerprint:
    the snapshot keeps `_img{,2}_vlo/_vhi/_shape` for its frame, and
    /api/live/imageN?json=1 returns the same for the served frame. If the two
    fingerprints DIFFER, the halves came from different frames == a real
    temporal gap. One-sided: differing PROVES a gap, but equal does NOT prove
    sync (two frames can share a percentile clip). The monitor auto-uses EXACT
    when the header is present, else DEGRADED, and labels which in every line.

Run (yb_analysis env):
  python -m yb_analysis.scripts.box_sync_monitor                 # loopback, follow
  python -m yb_analysis.scripts.box_sync_monitor --once
  python -m yb_analysis.scripts.box_sync_monitor --url http://100.86.15.43:8050
  python -m yb_analysis.scripts.box_sync_monitor --image 2 --scans 2 --jsonl sync.jsonl

`--scans N` auto-stops after N scans complete (a scan_id that produced frames
then went Stopped / was replaced by a new scan_id). Fetch ORDER matters: the
monitor alternates snapshot-first / image-first each poll so a mismatch from
either interleaving is observable (the browser issues both and does not
guarantee order either).

Exit codes: 0 = no temporal mismatch observed, 3 = dashboard unreachable,
5 = temporal mismatch detected.
"""

import argparse
import sys
import time
import json
import urllib.request
import urllib.error


def _fetch(url, timeout=10.0, want_headers=False):
    """GET url. Returns (body_bytes, headers_dict) or (None, None) on failure."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            body = r.read()
            hdrs = {k.lower(): v for k, v in r.headers.items()}
            return body, hdrs
    except (urllib.error.URLError, OSError):
        return None, None


def _get_json(url, timeout=10.0):
    body, _ = _fetch(url, timeout=timeout)
    if body is None:
        return None
    try:
        return json.loads(body.decode('utf-8'))
    except Exception:
        return None


def _snapshot_ident(base, which, timeout):
    """Fetch /api/snapshot and return (ident, snap) where `ident` identifies the
    OVERLAY's published frame:
      * ('seq', _write_seq)             -- always available
    plus a secondary per-frame FINGERPRINT of the frame's image
      * ('fp', (vlo, vhi, tuple(shape))) -- the percentile clip + shape the
        renderer computed for THIS frame's image, kept in the snapshot even
        though the heavy data URI is stripped.
    Returns ({'seq':..,'fp':..}, snap) or (None, None)."""
    snap = _get_json(base + '/api/snapshot', timeout=timeout)
    if snap is None:
        return None, None
    if which == 2 and int(snap.get('num_images') or 1) >= 2:
        vlo, vhi, shp = snap.get('_img2_vlo'), snap.get('_img2_vhi'), snap.get('_img2_shape')
    else:
        vlo, vhi, shp = snap.get('_img_vlo'), snap.get('_img_vhi'), snap.get('_img_shape')
    ident = {'seq': snap.get('_write_seq'),
             'fp': (vlo, vhi, tuple(shp) if shp is not None else None)}
    return ident, snap


def _image_ident(base, which, timeout):
    """Identify the frame the DISPLAYED PNG came from, two ways:
      * seq: the X-Frame-Seq response header (exact; requires the deployed stamp)
      * fp:  (vlo, vhi, shape) via /api/live/imageN?json=1 (the same per-frame
             fingerprint the snapshot carries -- works WITHOUT the header stamp)
    Returns ({'seq':int|None,'fp':tuple|None}, reason)."""
    path = '/api/live/image2' if which == 2 else '/api/live/image1'
    # Header seq (fast, exact) -- HEAD-like GET; we don't need the bytes here.
    seq = None
    body, hdrs = _fetch(base + path, timeout=timeout)
    if body is None:
        return None, 'no-frame'
    if hdrs and hdrs.get('x-frame-seq', '').strip():
        try:
            seq = int(hdrs['x-frame-seq'].strip())
        except ValueError:
            seq = None
    # Fingerprint via the json variant (vlo/vhi/shape only; small vs the URI).
    meta = _get_json(base + path + '?json=1', timeout=timeout)
    fp = None
    if meta is not None:
        shp = meta.get('shape')
        fp = (meta.get('vlo'), meta.get('vhi'),
              tuple(shp) if shp is not None else None)
    return {'seq': seq, 'fp': fp}, 'ok'


def _browser_ident(base, which, timeout):
    """Replicate what the BROWSER actually composites, to test the real path
    (the bare-endpoint check in poll_once tests a phantom race the browser never
    runs -- the browser never fetches /api/live/imageN with no ?t=).

    The live array panel is ONE figure from /api/live/figures?group=snapshot:
      * the green/red boxes are BAKED into that figure JSON at frame K
      * the image is referenced as layout.images[0].source = /api/live/imageN?t=K
    So the figure's own frame K IS the boxes' frame, and the ?t=K in the image URL
    is the frame the browser then fetches. We:
      1. GET the snapshot-group figure, read the array panel's image source (?t=K)
         -> K is the BOXES' frame (box_seq).
      2. GET that EXACT pinned URL, read its X-Frame-Seq -> the frame the server
         actually served for those boxes (img_seq).
      3. box/image match iff img_seq == K. With the per-seq ring deployed this
         should hold even while the frame advances; without it, the server ignores
         ?t= and serves current -> the straddle reappears.
    Returns ({'box_seq':K,'img_seq':S}, reason)."""
    name = 'array2' if which == 2 else 'array'
    fig = _get_json(base + '/api/live/figures?which=' + name, timeout=timeout)
    if not fig:
        return None, 'no-figure'
    # ?which= returns the single figure dict {data, layout}. Dig out the image src.
    try:
        src = fig['layout']['images'][0]['source']
    except (KeyError, IndexError, TypeError):
        return None, 'no-baked-image'      # waiting/placeholder panel this poll
    if not isinstance(src, str) or '/api/live/image' not in src:
        # baked base64 (use_img_url off) -> boxes+image already atomic, nothing to straddle
        return {'box_seq': None, 'img_seq': None, 'atomic': True}, 'baked'
    # box_seq = the ?t=K the figure baked (the boxes' frame).
    box_seq = None
    if '?t=' in src:
        try:
            box_seq = int(src.split('?t=', 1)[1].split('&', 1)[0])
        except ValueError:
            box_seq = None
    # Fetch the EXACT pinned URL the browser would (relative -> absolute).
    img_url = src if src.startswith('http') else base + src
    body, hdrs = _fetch(img_url, timeout=timeout)
    if body is None:
        return None, 'no-frame'
    img_seq = None
    if hdrs and hdrs.get('x-frame-seq', '').strip():
        try:
            img_seq = int(hdrs['x-frame-seq'].strip())
        except ValueError:
            img_seq = None
    return {'box_seq': box_seq, 'img_seq': img_seq}, 'ok'


def poll_once_browser(base, which, timeout):
    """The REAL browser-path check (see _browser_ident). One figure fetch + the
    pinned image fetch; mismatch iff the served image frame != the ?t= the figure
    baked its boxes at. This is what validates the per-seq ring fix."""
    bid, reason = _browser_ident(base, which, timeout)
    snap = _get_json(base + '/api/snapshot', timeout=timeout)
    if bid is None:
        if snap is None:
            return {'status': 'unreachable'}
        return {'status': reason}      # no-figure / no-baked-image / no-frame
    if bid.get('atomic'):
        # Image baked into the figure JSON -> inherently coherent.
        return {'status': 'ok', 'image': which, 'order': 'browser',
                'method': 'baked', 'mismatch': False,
                's_box': None, 's_img': None,
                'scan_id': (snap or {}).get('scan_id')}
    box_seq, img_seq = bid['box_seq'], bid['img_seq']
    rec = {'status': 'ok', 'image': which, 'order': 'browser', 'method': 'seq',
           's_box': box_seq, 's_img': img_seq,
           'scan_id': (snap or {}).get('scan_id'),
           'num_sites': (snap or {}).get('num_sites'),
           'num_images': (snap or {}).get('num_images')}
    if box_seq is None or img_seq is None:
        rec['status'] = 'no-ident'
        rec['method'] = None
        rec['mismatch'] = None
        return rec
    rec['delta'] = int(img_seq - box_seq)
    rec['mismatch'] = (img_seq != box_seq)
    return rec


def poll_once(base, which, order, timeout):
    """One temporal-identity check. `order` in {'snap-img', 'img-snap'} controls
    which half is fetched first (a mismatch can appear from either interleaving).

    Compares the overlay's frame identity against the image's. Prefers the exact
    X-Frame-Seq stamp; falls back to the (vlo, vhi, shape) fingerprint when the
    stamp is not deployed. The fingerprint is ONE-SIDED: a difference PROVES the
    two came from different frames (real mismatch); equal fingerprints do NOT
    prove sync (two frames can share a clip). So `method` is reported and a
    fingerprint 'match' is treated as 'consistent', not 'proven-synced'.
    Returns a result dict."""
    if order == 'img-snap':
        img_id, reason = _image_ident(base, which, timeout)
        box_id, snap = _snapshot_ident(base, which, timeout)
    else:
        box_id, snap = _snapshot_ident(base, which, timeout)
        img_id, reason = _image_ident(base, which, timeout)

    if snap is None and img_id is None:
        return {'status': 'unreachable'}
    if box_id is None:
        return {'status': 'no-snapshot'}
    if img_id is None:
        return {'status': reason}

    base_rec = {
        'status': 'ok', 'image': which, 'order': order,
        's_box': box_id['seq'], 's_img': img_id['seq'],
        'fp_box': list(box_id['fp']) if box_id['fp'] else None,
        'fp_img': list(img_id['fp']) if img_id['fp'] else None,
        'scan_id': (snap or {}).get('scan_id'),
        'num_sites': (snap or {}).get('num_sites'),
        'num_images': (snap or {}).get('num_images'),
    }

    # Exact path: both seqs known -> compare directly.
    if box_id['seq'] is not None and img_id['seq'] is not None:
        base_rec['method'] = 'seq'
        base_rec['delta'] = int(img_id['seq'] - box_id['seq'])
        base_rec['mismatch'] = (img_id['seq'] != box_id['seq'])
        return base_rec

    # Degraded path: fingerprint. Requires BOTH sides to have a usable fp.
    if box_id['fp'] and img_id['fp'] and None not in box_id['fp'] and None not in img_id['fp']:
        base_rec['method'] = 'fingerprint'
        base_rec['mismatch'] = (box_id['fp'] != img_id['fp'])
        return base_rec

    # Neither method usable this poll.
    base_rec['status'] = 'no-ident'
    base_rec['method'] = None
    base_rec['mismatch'] = None
    return base_rec


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--url', default='http://127.0.0.1:8050',
                    help='dashboard base URL (default loopback)')
    ap.add_argument('--image', type=int, default=2, choices=(1, 2),
                    help='which display frame to audit (default 2 = final)')
    ap.add_argument('--interval', type=float, default=0.5,
                    help='seconds between polls (default 0.5 -- poll FAST to '
                         'catch the narrow publish-straddle window)')
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--timeout', type=float, default=10.0)
    ap.add_argument('--jsonl', default=None,
                    help='append one JSON record per poll to this file')
    ap.add_argument('--scans', type=int, default=0,
                    help='auto-stop after this many scans COMPLETE (a scan_id '
                         'that produced frames then went Stopped, or was '
                         'replaced by a new scan_id). 0 = run until Ctrl-C.')
    ap.add_argument('--browser', action='store_true',
                    help='test the REAL browser path: fetch the snapshot-group '
                         'figure, read the array panel image URL (/api/live/'
                         'imageN?t=K, K = the boxes\' baked frame), fetch THAT '
                         'pinned URL, compare its served frame to K. This is what '
                         'validates the per-seq ring fix -- the DEFAULT (bare-'
                         'endpoint) check tests a race the browser never runs '
                         '(it never fetches the image without ?t=).')
    args = ap.parse_args(argv)

    base = args.url.rstrip('/')
    jf = open(args.jsonl, 'a') if args.jsonl else None

    def emit(rec):
        if jf is not None:
            rec = dict(rec)
            rec['t'] = time.time()
            jf.write(json.dumps(rec) + '\n')
            jf.flush()

    polled = mismatches = 0
    seen_seqs = set()
    methods = set()
    # --- two-scan tracking ---
    completed = 0            # scans that finished while we watched
    active_sid = None        # scan_id currently producing frames
    active_had_frames = False
    per_scan = {}            # sid -> {polls, mism, seqs:set, method}

    def _summarize_scan(sid):
        s = per_scan.get(sid)
        if not s:
            return
        adv = len(s['seqs'])
        print(f'  scan {sid}: {s["mism"]}/{s["polls"]} mismatch polls, '
              f'{adv} distinct frames, method={s.get("method") or "?"}')

    order_toggle = 'snap-img'
    try:
        while True:
            if args.browser:
                res = poll_once_browser(base, args.image, args.timeout)
            else:
                res = poll_once(base, args.image, order_toggle, args.timeout)
            order_toggle = 'img-snap' if order_toggle == 'snap-img' else 'snap-img'
            st = res['status']
            emit(res)

            if st == 'unreachable':
                print('dashboard unreachable at', base, file=sys.stderr)
                if args.once:
                    return 3
                time.sleep(args.interval)
                continue
            if st in ('no-snapshot', 'no-frame', 'no-ident', 'no-figure',
                      'no-baked-image', 'baked'):
                # Transient: waiting for a live frame (idle backend, camera init,
                # or a poll that caught a mid-write). Not fatal, not conclusive.
                if args.once:
                    return 0
                time.sleep(args.interval)
                continue

            # --- scan-completion bookkeeping (drives --scans auto-stop) ---
            sid = res.get('scan_id')
            seq = res.get('s_box')
            frame_advanced = seq is not None and seq not in seen_seqs
            if sid is not None:
                if active_sid is None:
                    active_sid = sid
                elif sid != active_sid:
                    # scan_id rolled over -> the previous scan completed.
                    if active_had_frames:
                        completed += 1
                        print(f'--- scan {active_sid} complete '
                              f'({completed}/{args.scans or "inf"}) ---')
                        _summarize_scan(active_sid)
                    active_sid = sid
                    active_had_frames = False
                ps = per_scan.setdefault(
                    sid, {'polls': 0, 'mism': 0, 'seqs': set(), 'method': None})
                ps['polls'] += 1
                if seq is not None:
                    ps['seqs'].add(seq)
                if res.get('method'):
                    ps['method'] = res['method']

            polled += 1
            if seq is not None:
                seen_seqs.add(seq)
            if res.get('method'):
                methods.add(res['method'])
            if frame_advanced:
                active_had_frames = True

            mism = res.get('mismatch')
            method = res.get('method')
            if mism:
                mismatches += 1
                if sid is not None:
                    per_scan[sid]['mism'] += 1
                if method == 'seq':
                    print(f'[MISMATCH] {method} order={res["order"]:8s} '
                          f'boxes@{res["s_box"]} image@{res["s_img"]} '
                          f'(delta={res.get("delta"):+d})  <- overlay & frame '
                          f'from DIFFERENT published frames')
                else:
                    print(f'[MISMATCH] {method} order={res["order"]:8s} '
                          f'boxes fp={res["fp_box"]} image fp={res["fp_img"]}  '
                          f'<- overlay & frame fingerprints DIFFER (proves '
                          f'different frames)')
            else:
                tag = 'consistent' if method == 'fingerprint' else 'ok'
                print(f'[{tag}] {method} order={res["order"]:8s} '
                      f'seq={res.get("s_box")}')

            if args.once:
                return 5 if mism else 0

            # auto-stop after N completed scans
            if args.scans and completed >= args.scans:
                break

            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        if jf is not None:
            jf.close()

    # Fold in the still-active scan if it produced frames (e.g. we stopped
    # mid-scan or the last of N never rolled over).
    if active_sid is not None and active_had_frames and args.scans \
            and completed < args.scans:
        completed += 1

    method_note = ('+'.join(sorted(methods)) or 'none')
    if method_note == 'fingerprint':
        method_note += (' (degraded: X-Frame-Seq not deployed; a mismatch '
                        'still PROVES a temporal gap, but "consistent" polls '
                        'do not prove sync)')
    advancing = len(seen_seqs) > 1

    print(f'\n=== per-scan summary ({completed} scan(s) observed) ===')
    for sid in per_scan:
        _summarize_scan(sid)

    if polled and mismatches:
        print(f'\nVERDICT: temporal box/image mismatch DETECTED -- '
              f'{mismatches}/{polled} polls, {len(seen_seqs)} distinct frames, '
              f'method={method_note}.', file=sys.stderr)
        return 5
    if not advancing:
        print(f'\nVERDICT: inconclusive -- frame never advanced '
              f'({len(seen_seqs)} distinct frame(s) over {polled} polls). '
              f'Backend idle/stopped; re-arm during a running scan.')
        return 0
    print(f'\nVERDICT: no temporal mismatch -- 0/{polled} polls, '
          f'{len(seen_seqs)} distinct frames, method={method_note}.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
