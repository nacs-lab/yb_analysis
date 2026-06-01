"""Lab-side backfill: fetch SLM-server `/runs/<run_id>/analysis` for the
24 matched pre-Phase-1 rearrangement runs and write it as
`<scan_dir>/slm_analysis.json`.

Companion to `slm_backfill/slm_pc_backfill.py` (which runs on the SLM PC
and renames code-snapshot dirs). This script needs no SLM-side changes
because `/runs/<run_id>/analysis` already accepts the raw ISO run_id —
the bit-order-correct lattice, paths-per-shot, and survival-vs-distance
the dashboard wants are already computed server-side for these runs.

Reads:
- `slm_backfill/slm_runid_mapping.json` (or wherever `--mapping` points).
- `/runs/<run_id>/analysis` on the SLM server for each pairing.

Writes (per pairing):
- `<scan_dir>/slm_analysis.json` — verbatim response + `synced_at_iso`
  + `_backfill_mapping` block (slm_run_id, delta_seconds, confidence)
  so a future reader knows where it came from.
- `<scan_dir>/slm_runid.txt` — single-line file with the raw SLM ISO
  run_id, so subsequent fetches (e.g. of `/slm/runs/<run_id>/code`)
  can use it without re-running the matcher.

Idempotent: skips scan dirs that already have a fresh
`slm_analysis.json` (mtime newer than `--max-age-h`, default 24 h).
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional


logger = logging.getLogger('backfill_slm_analysis')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mapping', type=Path,
                   default=Path('slm_backfill/slm_runid_mapping.json'),
                   help='path to slm_runid_mapping.json')
    p.add_argument('--slm-url', default='http://100.114.207.118:8551',
                   help='SLM server base URL (default: Tailscale)')
    p.add_argument('--data-root', type=Path,
                   default=Path(r'D:\OneDrive - Harvard University\Documents - Yb\Data'),
                   help='lab data root containing <YYYYMMDD>/data_<scan_id>/')
    p.add_argument('--max-age-h', type=float, default=24.0,
                   help='skip scan dirs whose slm_analysis.json is fresher than this (hours)')
    p.add_argument('--apply', action='store_true',
                   help='actually write files (default: dry-run, just print)')
    p.add_argument('--scan-id', action='append', default=None,
                   help='restrict to one or more scan_ids (repeatable)')
    p.add_argument('--timeout', type=float, default=30.0,
                   help='per-request HTTP timeout (s)')
    p.add_argument('-v', '--verbose', action='store_true')
    return p.parse_args()


def fetch_analysis(slm_url: str, run_id: str, timeout: float) -> Optional[dict]:
    """GET /runs/<run_id>/analysis. Returns parsed JSON or None on
    failure (logged). Raw ISO run_id is URL-quoted because `:` etc.
    need escaping in the path component."""
    path = f'/runs/{urllib.parse.quote(run_id, safe="")}/analysis'
    try:
        with urllib.request.urlopen(slm_url + path, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        logger.warning('  HTTP %s for %s: %s', e.code, run_id, e.read()[:200])
    except (urllib.error.URLError, TimeoutError) as e:
        logger.warning('  network error for %s: %s', run_id, e)
    except json.JSONDecodeError as e:
        logger.warning('  bad JSON for %s: %s', run_id, e)
    return None


def resolve_scan_dir(data_root: Path, scan_id: str) -> Optional[Path]:
    """scan_id (YYYYMMDDHHMMSS) -> data_root/YYYYMMDD/data_YYYYMMDD_HHMMSS."""
    if len(scan_id) != 14 or not scan_id.isdigit():
        return None
    day, hms = scan_id[:8], scan_id[8:]
    candidate = data_root / day / f'data_{day}_{hms}'
    return candidate if candidate.is_dir() else None


def is_fresh(path: Path, max_age_h: float) -> bool:
    if not path.is_file():
        return False
    age_h = (time.time() - path.stat().st_mtime) / 3600.0
    return age_h <= max_age_h


def write_atomic(path: Path, content: str, *, encoding='utf-8'):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(content, encoding=encoding)
    tmp.replace(path)


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(message)s')

    mapping = json.loads(args.mapping.read_text(encoding='utf-8'))
    pairings = mapping.get('pairings', [])
    if not pairings:
        print('ERROR: no pairings in mapping file', file=sys.stderr)
        return 1

    if args.scan_id:
        wanted = set(args.scan_id)
        pairings = [p for p in pairings if p['matlab_scan_id'] in wanted]
        if not pairings:
            print(f'No pairings match --scan-id={args.scan_id}', file=sys.stderr)
            return 1

    print(f'Mapping:          {args.mapping}')
    print(f'SLM URL:          {args.slm_url}')
    print(f'Data root:        {args.data_root}')
    print(f'Max age:          {args.max_age_h} h (skip fresh sidecars)')
    print(f'Mode:             {"APPLY" if args.apply else "DRY-RUN"}')
    print(f'Pairings:         {len(pairings)}')
    print()

    n_ok = n_skip_fresh = n_skip_no_dir = n_err = 0
    for i, p in enumerate(pairings, 1):
        slm_run_id = p['slm_run_id']
        scan_id    = p['matlab_scan_id']
        delta      = p['matched_delta_seconds']
        conf       = p.get('shot_count_confidence', '?')
        prefix     = f'[{i:2}/{len(pairings)}] scan_id={scan_id}  delta={delta:+.2f}s  conf={conf:10}'

        scan_dir = resolve_scan_dir(args.data_root, scan_id)
        if scan_dir is None:
            print(f'{prefix}  SKIP (no scan dir at expected path)')
            n_skip_no_dir += 1
            continue

        analysis_path = scan_dir / 'slm_analysis.json'
        runid_path    = scan_dir / 'slm_runid.txt'

        if is_fresh(analysis_path, args.max_age_h):
            print(f'{prefix}  SKIP (fresh sidecar: {analysis_path.name})')
            n_skip_fresh += 1
            continue

        if not args.apply:
            print(f'{prefix}  WOULD FETCH /runs/{slm_run_id!r}/analysis -> {analysis_path}')
            n_ok += 1
            continue

        print(f'{prefix}  fetching...', end=' ', flush=True)
        result = fetch_analysis(args.slm_url, slm_run_id, args.timeout)
        if result is None:
            print('ERR')
            n_err += 1
            continue
        # Stamp the response with backfill provenance so a future reader
        # can tell this came from the lab-side script, not the auto-sync.
        result['synced_at_iso'] = datetime.datetime.now().isoformat(timespec='seconds')
        result['_backfill_mapping'] = {
            'slm_run_id':            slm_run_id,
            'matlab_scan_id':        scan_id,
            'matched_delta_seconds': delta,
            'shot_count_confidence': conf,
            'shot_count_slm':        p.get('shot_count_slm'),
            'shot_count_lab':        p.get('shot_count_lab'),
            'source':                args.slm_url,
            'script':                'backfill_slm_analysis.py',
        }
        try:
            write_atomic(analysis_path, json.dumps(result, indent=2,
                                                    default=str))
            write_atomic(runid_path, slm_run_id + '\n')
            size_kb = analysis_path.stat().st_size / 1024
            print(f'OK ({size_kb:.0f} KB)')
            n_ok += 1
        except OSError as e:
            print(f'WRITE ERR: {e}')
            n_err += 1

    print()
    print(f'Summary:  ok/would={n_ok}  skip_fresh={n_skip_fresh}  '
          f'skip_no_dir={n_skip_no_dir}  err={n_err}')
    if not args.apply:
        print()
        print('Dry-run only. Re-run with --apply to fetch.')
    return 0 if n_err == 0 else 2


if __name__ == '__main__':
    sys.exit(main())
