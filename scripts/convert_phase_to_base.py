"""One-time conversion of zerniked WGS phase files to their BASE form.

Part of the loading-pattern affine migration (see PYTHON_FRONTEND_PLAN /
the ~/.claude plan).  Our canonical artifact is now the **base
(unzerniked) phase**: extraction runs on the base (defocus-independent),
and the loading defocus is re-applied only when writing to the SLM.

Existing files (e.g. ``phase/3270_z4eq4.pt``) have a Zernike baked in.
This script computes ``base = stored - build_zernike(coeffs)`` and saves
it to ``phase/base/<name>.pt`` on the SLM PC.  For a pattern with no
baked Zernike (e.g. ``33x33_uniform``) the base equals the stored file
and we just copy it.

The ``.pt`` files and ``build_zernike_phase_gpu`` live on the SLM PC, so
the work is done there via the SLM server's ``/eval`` endpoint (cwd =
``slm/``).  This is a one-time admin tool, not part of the live path.

The new ``/slm/initialize_loading_pattern`` endpoint also accepts
``legacy_zerniked=true`` + ``baked_zernike=[...]`` so you can transition
WITHOUT converting files first; this script just makes the base canonical.

Usage (dry-run prints the plan; --apply performs it)::

    python -m yb_analysis.scripts.convert_phase_to_base \
        --pattern 3270_z4eq4 --zernike 0 0 0 0 -4 --apply
    python -m yb_analysis.scripts.convert_phase_to_base \
        --pattern 33x33_uniform --zernike --apply     # no zernike -> copy
"""

import argparse
import json
import sys

import requests
from requests.auth import HTTPBasicAuth

from yb_analysis import config


def _auth():
    pw = '174171'
    pw_path = getattr(config, 'SLM_PASSWORD_PATH', None)
    if pw_path:
        try:
            with open(pw_path) as f:
                pw = f.read().strip()
        except OSError:
            pass
    return HTTPBasicAuth('admin', pw)


def _eval(code, *, url=None, timeout=(3.0, 120.0)):
    """Run a snippet on the SLM server; return the parsed /eval JSON."""
    base = (url or config.SLM_URL).rstrip('/')
    r = requests.post(base + '/eval',
                      json={'code': code, 'session': 'convert-phase-to-base'},
                      auth=_auth(), timeout=timeout, verify=config.SLM_VERIFY_TLS)
    r.raise_for_status()
    return r.json()


def convert(pattern, zernike, *, src=None, dst=None, apply=False,
            force=False, url=None):
    """Convert one pattern's stored phase to its base form on the SLM PC.

    Parameters
    ----------
    pattern : str
        Pattern name, e.g. ``"3270_z4eq4"``.
    zernike : list[float]
        ANSI coeffs baked into the stored phase ([] / None = none -> copy).
    src, dst : str | None
        Server-side paths; default ``phase/<pattern>.pt`` ->
        ``phase/base/<pattern>.pt``.
    apply : bool
        If False (default), only report the plan. If True, write the base.
    force : bool
        Overwrite an existing base file.
    """
    src = src or f'phase/{pattern}.pt'
    dst = dst or f'phase/base/{pattern}.pt'
    coeffs = list(zernike or [])
    nz = [(i, c) for i, c in enumerate(coeffs) if c]
    # The eval code is fully self-contained and prints a JSON line.
    code = f'''
import os, json, hashlib
import numpy as np, torch
src, dst = {src!r}, {dst!r}
coeffs = {coeffs!r}
apply_, force_ = {bool(apply)!r}, {bool(force)!r}
out = {{"src": src, "dst": dst, "coeffs": coeffs}}
if not os.path.exists(src):
    out["error"] = "src not found"
else:
    obj = torch.load(src, weights_only=False, map_location="cpu")
    arr = obj.detach().cpu().numpy() if hasattr(obj, "detach") else np.asarray(obj)
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    out["src_shape"] = list(arr.shape)
    out["src_sha"] = hashlib.sha256(arr.tobytes()).hexdigest()[:16]
    if any(c != 0.0 for c in coeffs):
        from slmnet.experimental import tools as T
        z = T.build_zernike_phase_gpu(list(coeffs), shape=arr.shape, device="cpu")
        z = z.detach().cpu().numpy().astype(np.float32)
        base = (arr - z).astype(np.float32)
        out["subtracted"] = True
    else:
        base = arr
        out["subtracted"] = False
    base = np.ascontiguousarray(base, dtype=np.float32)
    out["base_sha"] = hashlib.sha256(base.tobytes()).hexdigest()[:16]
    out["dst_exists"] = os.path.exists(dst)
    if apply_ and (force_ or not os.path.exists(dst)):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        torch.save(torch.from_numpy(base), dst)
        out["wrote"] = True
    else:
        out["wrote"] = False
print("RESULT " + json.dumps(out))
'''
    res = _eval(code, url=url)
    if not res.get('ok'):
        raise RuntimeError(f"/eval failed: {res.get('error') or res.get('stderr')}")
    out = {}
    for line in (res.get('stdout') or '').splitlines():
        if line.startswith('RESULT '):
            out = json.loads(line[len('RESULT '):])
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pattern', required=True, help='pattern name, e.g. 3270_z4eq4')
    p.add_argument('--zernike', nargs='*', type=float, default=[],
                   help='ANSI coeffs baked in (empty = none -> copy)')
    p.add_argument('--src', default=None, help='server src .pt (default phase/<pattern>.pt)')
    p.add_argument('--dst', default=None, help='server dst .pt (default phase/base/<pattern>.pt)')
    p.add_argument('--apply', action='store_true', help='write the base file (default dry-run)')
    p.add_argument('--force', action='store_true', help='overwrite an existing base file')
    p.add_argument('--url', default=None, help='SLM server URL (default config.SLM_URL)')
    args = p.parse_args(argv)

    out = convert(args.pattern, args.zernike, src=args.src, dst=args.dst,
                  apply=args.apply, force=args.force, url=args.url)
    print(json.dumps(out, indent=2))
    if out.get('error'):
        return 1
    if not args.apply:
        print('\n(dry-run; pass --apply to write the base file)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
