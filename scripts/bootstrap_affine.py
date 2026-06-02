"""Bootstrap (and optionally commit) the global SLM->camera affine from a
loading scan and its WGS pattern.

Flow:
  1. Get the pattern's extracted knm positions (order="col") — via the
     deployed ``/slm/initialize_loading_pattern`` endpoint if available,
     else by calling ``server._derive_grid_from_phase`` through ``/eval``
     (same code the endpoint wraps). Persist the pattern record.
  2. Load the scan's averaged loading image + crop ROI.
  3. Fit the affine via ``affine_transform.bootstrap_from_scan`` (90°-family
     seed + coarse-to-fine ICP; generous centroid radius for defocus).
  4. Verify: compare ``detect_atom`` mean intensity on the affine-mapped grid
     vs. the (possibly stale) day-folder gridLocations.txt.
  5. With ``--apply`` and an accepted fit, commit it to the affine store.

Usage::

    python -m yb_analysis.scripts.bootstrap_affine \
        --scan-id 20260601_151712 --pattern 33x33_uniform \
        --phase phase/33x33_uniform.pt --zernike            # no zernike
    # add --apply to write the registry record + commit the affine
"""

import argparse
import json
import os
import sys

import numpy as np
import requests
from requests.auth import HTTPBasicAuth

from yb_analysis import config


def _auth():
    pw = '174171'
    p = getattr(config, 'SLM_PASSWORD_PATH', None)
    if p:
        try:
            with open(p) as f:
                pw = f.read().strip()
        except OSError:
            pass
    return HTTPBasicAuth('admin', pw)


def _knm_via_eval(phase_path, zernike, order, url):
    """Derive (knm, phases, lattice) on the SLM PC through /eval — the same
    server._derive_grid_from_phase the endpoint wraps. Base = phase minus
    any baked zernike (defocus-independent extraction)."""
    base = (url or config.SLM_URL).rstrip('/')
    z = list(zernike or [])
    code = f'''
import json, numpy as np, torch
ph = torch.load({phase_path!r}, weights_only=False, map_location="cpu")
arr = ph.detach().cpu().numpy() if hasattr(ph,"detach") else np.asarray(ph)
arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
z = {z!r}
if any(c != 0.0 for c in z):
    from slmnet.experimental import tools as T
    zt = T.build_zernike_phase_gpu(list(z), shape=arr.shape, device="cpu")
    arr = (arr - zt.detach().cpu().numpy().astype(np.float32)).astype(np.float32)
import hashlib; sha = hashlib.sha256(arr.tobytes()).hexdigest()
coords, info = server._derive_grid_from_phase(arr, order={order!r}, name="bootstrap")
coords = np.asarray(coords)
out = {{"sha": sha, "n": int(coords.shape[0]), "knm": coords.tolist(),
        "phases": np.asarray(info.get("phases")).ravel().tolist() if info.get("phases") is not None else None,
        "n_rows": int(info.get("n_rows")) if info.get("n_rows") is not None else None,
        "n_cols": int(info.get("n_cols")) if info.get("n_cols") is not None else None,
        "pitch_x": float(info.get("pitch_x")) if info.get("pitch_x") is not None else None,
        "pitch_y": float(info.get("pitch_y")) if info.get("pitch_y") is not None else None}}
print("RESULT " + json.dumps(out))
'''
    r = requests.post(base + '/eval',
                      json={'code': code, 'session': 'bootstrap-affine'},
                      auth=_auth(), timeout=(3.0, 180.0),
                      verify=config.SLM_VERIFY_TLS)
    r.raise_for_status()
    d = r.json()
    if not d.get('ok'):
        raise RuntimeError(f"/eval failed: {d.get('error') or d.get('stderr')}")
    for line in (d.get('stdout') or '').splitlines():
        if line.startswith('RESULT '):
            return json.loads(line[len('RESULT '):])
    raise RuntimeError('no RESULT from /eval')


def _scan_dir(scan_id):
    sid = scan_id.replace('_', '')
    day = sid[:8]
    return os.path.join(config.DATA_DIR, day, f'data_{day}_{sid[8:]}')


def _avg_image_and_roi(scan_id):
    import h5py
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    sdir = _scan_dir(scan_id)
    sid = scan_id.replace('_', '')
    h5p = os.path.join(sdir, f'data_{sid[:8]}_{sid[8:]}.h5')
    with h5py.File(h5p, 'r', libver='latest', swmr=True) as f:
        n = f['imgs'].shape[0]
        acc = np.zeros(f['imgs'].shape[1:], np.float64)
        for i in range(n):
            acc += f['imgs'][i].astype(np.float64)
        avg = acc / n
    # ROI from scan config, else expConfig default.
    roi = None
    try:
        from yb_analysis.io.mat_reader import load_scan_config_from_mat
        cfg = load_scan_config_from_mat(
            os.path.join(sdir, f'data_{sid[:8]}_{sid[8:]}.mat'))
        r = cfg.get('roi')
        if r is not None and len(np.ravel(r)) == 4:
            roi = [float(v) for v in np.ravel(r)]
    except Exception as ex:
        print(f'(scan ROI unavailable: {ex})')
    if roi is None:
        try:
            roi = list(config.read_orca_config()['roi'])
        except Exception:
            roi = [1000.0, 100.0, 2100.0, 2100.0]
        print(f'using fallback ROI {roi}')
    return avg, roi


def _stale_grid(scan_id):
    sid = scan_id.replace('_', '')
    glt = os.path.join(config.DATA_DIR, sid[:8], 'gridLocations.txt')
    if os.path.isfile(glt):
        try:
            return np.loadtxt(glt, skiprows=1)[:, :2]
        except Exception:
            return None
    return None


def _contrast(avg, grid_yx, w=4):
    """Windowed (integrated, background-subtracted) signal at grid points vs.
    random background points. Windowed because the spots are defocused/faint
    — single-pixel sampling misses them, but the integrated signal over the
    spot is preserved. grid_yx in cropped pixels."""
    H, W = avg.shape
    g = np.asarray(grid_yx, dtype=np.float64)
    bg_level = float(np.median(avg))
    box = (2 * w + 1) ** 2

    def wsum(pts):
        vals = []
        for y, x in pts:
            yi, xi = int(round(y)), int(round(x))
            if w <= yi < H - w and w <= xi < W - w:
                vals.append(avg[yi - w:yi + w + 1, xi - w:xi + w + 1].sum()
                            - bg_level * box)
        return np.asarray(vals) if vals else np.array([0.0])

    gi = wsum(g)
    rng = np.random.default_rng(0)
    rg = np.column_stack([rng.uniform(g[:, 0].min(), g[:, 0].max(), len(g)),
                          rng.uniform(g[:, 1].min(), g[:, 1].max(), len(g))])
    bg = wsum(rg)
    return {'grid_signal': float(gi.mean()), 'bg_signal': float(bg.mean()),
            'ratio': float(gi.mean() / max(abs(bg.mean()), 1e-6)),
            'n_used': int(len(gi))}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scan-id', required=True, help='e.g. 20260601_151712')
    p.add_argument('--pattern', required=True, help='e.g. 33x33_uniform')
    p.add_argument('--phase', required=True, help='server-side base phase .pt')
    p.add_argument('--zernike', nargs='*', type=float, default=[],
                   help='baked ANSI coeffs (empty = none)')
    p.add_argument('--order', default='col')
    p.add_argument('--spot-sigma', type=float, default=5.0)
    p.add_argument('--apply', action='store_true',
                   help='write registry record + commit the affine')
    p.add_argument('--url', default=None)
    args = p.parse_args(argv)

    import yb_analysis.analysis.affine_transform as aff
    import yb_analysis.analysis.pattern_registry as reg

    print(f'[1] deriving {args.pattern} positions via /eval ...')
    res = _knm_via_eval(args.phase, args.zernike, args.order, args.url)
    knm = np.asarray(res['knm'], dtype=np.float64)
    print(f'    n_sites={res["n"]} sha={res["sha"][:12]} '
          f'rows={res.get("n_rows")} cols={res.get("n_cols")}')

    record = {
        'name': args.pattern, 'base_phase_path': args.phase,
        'legacy_zerniked': bool(any(c != 0 for c in args.zernike)),
        'baked_zernike': list(args.zernike) or None,
        'base_sha256': res['sha'], 'default_loading_zernike': None,
        'order': args.order, 'fft_shape': [4096, 4096], 'threshold': 0.30,
        'min_dist': None, 'n_sites': res['n'], 'knm': res['knm'],
        'phases': res.get('phases'),
        'lattice': {'n_rows': res.get('n_rows'), 'n_cols': res.get('n_cols'),
                    'pitch_x': res.get('pitch_x'), 'pitch_y': res.get('pitch_y')},
        'source_endpoint': '/eval:_derive_grid_from_phase',
        'created_iso': reg._now_iso(), 'updated_iso': reg._now_iso(),
    }

    print(f'[2] loading scan {args.scan_id} avg image + ROI ...')
    avg, roi = _avg_image_and_roi(args.scan_id)
    print(f'    avg {avg.shape} ROI={roi}')

    print('[3] fitting affine (bootstrap) ...')
    cand = aff.bootstrap_from_scan(avg, record, roi, spot_sigma=args.spot_sigma,
                                   scan_id=args.scan_id.replace('_', ''))
    print(f'    accept={cand["accept"]} reason={cand["reason"]} '
          f'coverage={cand["coverage"]:.3f} n_pairs={cand["n_pairs"]}/{cand["n_sites"]} '
          f'rms_px={cand["rms_px"]:.3f}')
    if cand.get('A') is not None:
        print(f'    rotation_deg={cand.get("rotation_deg"):.2f} '
              f'scale_x={cand.get("scale_x"):.4f} scale_y={cand.get("scale_y"):.4f}')

    if cand.get('A') is not None:
        print('[4] verification — detect contrast on affine grid vs stale grid')
        grid_aff = aff.apply_affine_cropped(aff._knm_to_xy(knm), cand['A'], roi)
        c_aff = _contrast(avg, grid_aff)
        print(f'    affine grid: signal/bg ratio={c_aff["ratio"]:.1f} '
              f'(grid_signal={c_aff["grid_signal"]:.1f} bg={c_aff["bg_signal"]:.1f}, '
              f'n={c_aff["n_used"]})')
        stale = _stale_grid(args.scan_id)
        if stale is not None:
            c_st = _contrast(avg, stale)
            print(f'    stale  grid: signal/bg ratio={c_st["ratio"]:.1f} '
                  f'(grid_signal={c_st["grid_signal"]:.1f}, n={c_st["n_used"]})')

    if args.apply:
        if not cand.get('accept'):
            print('\n[!] fit not accepted; NOT committing. '
                  '(use a better scan or adjust gates.)')
            return 1
        reg.write_pattern(record)
        entry = aff.commit_update(cand)
        print(f'\n[5] committed affine to {aff._affine_path()}')
        print(f'    registry record at {reg._record_path(args.pattern)}')
        print(json.dumps({k: entry[k] for k in (
            'rotation_deg', 'scale_x', 'scale_y', 'rms_px', 'coverage',
            'last_scan_id', 'updated_iso')}, indent=2))
    else:
        print('\n(dry-run; pass --apply to write the registry record + commit)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
