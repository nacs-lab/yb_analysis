"""Probe the running dashboard's /api/* endpoints and report what's there.

Usage (in any python env with `requests`):

    python -m yb_analysis.scripts.diag_dashboard
    python -m yb_analysis.scripts.diag_dashboard --scan-id 20260529025015
    python -m yb_analysis.scripts.diag_dashboard --host http://localhost:8050

Prints one section per endpoint with a *minimal* summary -- enough to
tell whether the server is returning real data, empty data, or an
error. No business logic, no fixes -- just visibility.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request


def _get(url, timeout=5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            body = r.read()
            ct = r.headers.get('Content-Type', '')
            return r.status, ct, body
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get('Content-Type', ''), e.read()
    except Exception as e:
        return None, '', str(e).encode()


def _jload(body):
    try:
        return json.loads(body.decode('utf-8'))
    except Exception:
        return None


def section(title):
    print()
    print('=' * 72)
    print(title)
    print('=' * 72)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='http://localhost:8050')
    ap.add_argument('--scan-id', default=None,
                    help='14-digit scan_id to probe analysis endpoints')
    args = ap.parse_args()
    host = args.host.rstrip('/')

    # ---- /api/status -------------------------------------------------
    section('/api/status')
    status, ct, body = _get(f'{host}/api/status')
    print(f'HTTP {status}  ({ct})  bytes={len(body)}')
    if status == 200:
        d = _jload(body) or {}
        keys = sorted(d.keys())
        print(f'  keys ({len(keys)}): {keys[:12]}{"..." if len(keys) > 12 else ""}')

    # ---- /api/snapshot -----------------------------------------------
    section('/api/snapshot')
    status, ct, body = _get(f'{host}/api/snapshot')
    print(f'HTTP {status}  ({ct})  bytes={len(body)}')
    if status == 200:
        d = _jload(body) or {}
        print(f'  scan_id           = {d.get("scan_id")}')
        print(f'  scan_name         = {d.get("scan_name")}')
        print(f'  num_sites         = {d.get("num_sites")}')
        print(f'  n_accum_shots     = {d.get("n_accum_shots")}')
        print(f'  thresholds (len)  = {len(d.get("thresholds") or [])}')
        print(f'  logicals (len)    = {len(d.get("logicals") or [])}')
        print(f'  cur_intensities   = {len(d.get("cur_intensities") or [])}')
        print(f'  loading_history   = {len(d.get("loading_history") or [])}')
        print(f'  loading_rates     = {len(d.get("loading_rates") or [])}')
        print(f'  grid_locations    = {len(d.get("grid_locations") or [])}')
        print(f'  _img_shape        = {d.get("_img_shape")}')
        print(f'  _img2_shape       = {d.get("_img2_shape")}')

    # ---- /api/live/figures -- THE MAIN PLOT ENDPOINT -----------------
    section('/api/live/figures   (Live tab plots)')
    status, ct, body = _get(f'{host}/api/live/figures')
    print(f'HTTP {status}  ({ct})  bytes={len(body)}')
    if status == 200:
        d = _jload(body) or {}
        figs = d.get('figures') or {}
        for name, fig in figs.items():
            if fig is None:
                print(f'  {name:10s}  -> NULL  (server returned None / error)')
                continue
            data = fig.get('data') or []
            layout = fig.get('layout') or {}
            anns = (layout.get('annotations') or [])
            ann_texts = [a.get('text', '')[:32] for a in anns]
            # First trace summary
            first_trace_summary = ''
            if data:
                t0 = data[0]
                ttype = t0.get('type', 'scatter')
                xlen = len(t0.get('x') or [])
                ylen = len(t0.get('y') or [])
                first_trace_summary = f'{ttype} x={xlen} y={ylen}'
            else:
                first_trace_summary = '(no traces -> _waiting() placeholder)'
            print(f'  {name:10s}  traces={len(data):2d}  '
                  f'first={first_trace_summary:30s}  ann={ann_texts[:1]}')

    # ---- /api/runs/<scan_id>/analysis --------------------------------
    if args.scan_id:
        section(f'/api/runs/{args.scan_id}/analysis   (Analysis tab plots)')
        status, ct, body = _get(f'{host}/api/runs/{args.scan_id}/analysis',
                                timeout=30)
        print(f'HTTP {status}  ({ct})  bytes={len(body)}')
        if status == 200:
            d = _jload(body) or {}
            print(f'  scan_id        = {d.get("scan_id")}')
            print(f'  n_params       = {d.get("n_params")}')
            print(f'  n_shots        = {d.get("n_shots")}')
            print(f'  unpack_error   = {d.get("unpack_error")}')
            sweep = d.get('sweep') or {}
            print(f'  sweep.cols     = {sweep.get("cols")}')
            print(f'  sweep.dims     = {sweep.get("dims")}')
            vals = sweep.get('values') or []
            print(f'  sweep.values   = {len(vals)} cols, first col len = '
                  f'{len(vals[0]) if vals else 0}')
            summary = d.get('summary') or {}
            sm = summary.get('survival_mean') or []
            lr = summary.get('loading_rate') or []
            print(f'  summary.survival_mean   len={len(sm)}   '
                  f'first 5 = {sm[:5]}')
            print(f'  summary.loading_rate    len={len(lr)}   '
                  f'first 5 = {lr[:5]}')
            print(f'  -> survival chart will render? '
                  f'{"YES" if len(sm) > 0 else "NO (empty)"}')
            print(f'  -> loading  chart will render? '
                  f'{"YES" if len(lr) > 0 else "NO (empty)"}')
            print(f'  code.present   = {(d.get("code") or {}).get("present")}')
            print(f'  grid.present   = {(d.get("grid") or {}).get("present")}')
        elif status:
            print(body.decode('utf-8', errors='replace')[:500])

    # ---- /api/runs/list ----------------------------------------------
    section('/api/runs/list   (last 5)')
    status, ct, body = _get(f'{host}/api/runs/list?max=5')
    print(f'HTTP {status}  ({ct})  bytes={len(body)}')
    if status == 200:
        d = _jload(body) or {}
        for r in (d.get('runs') or [])[:5]:
            print(f'  {r.get("scan_id")}  has_diag={r.get("has_diag")}  '
                  f'has_code={r.get("has_code")}  has_grid={r.get("has_grid")}  '
                  f'name={r.get("name")}')

    print()


if __name__ == '__main__':
    main()
