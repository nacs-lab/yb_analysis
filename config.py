"""Experiment configuration constants — mirrors expConfig.m defaults."""

import os
import tempfile

# Default data storage path (override via environment or constructor arg)
PATH_PREFIX = os.environ.get('YB_PATH_PREFIX', r'D:\OneDrive - Harvard University\Documents - Yb')
DATA_DIR = os.path.join(PATH_PREFIX, 'Data')

# ZMQ server URL for AnalysisClient — must match consts.MatlabURL in
# the MATLAB expConfig.m.
MATLAB_URL = 'tcp://127.0.0.1:1408'

# Fixed ports
DASHBOARD_PORT = 8050
TEST_SERVER_PORT = 1500

# ---- Dashboard live-image rendering -------------------------------------
# The live "Tweezer Array" panels render at ~670 px, so a full-sensor camera
# frame (e.g. 4096x2304) is needlessly huge: the old encoder shipped a ~12 MB
# base64 PNG *per image, per update*, and with two-array + middle frame that's
# ~38 MB the browser must base64- + PNG-decode + raster every tick — the
# freeze seen under heavy imaging. Downsample the displayed frame to at most
# this many pixels on the long edge before encoding; overlays stay aligned
# because the raster is stretched back to the original extent. None disables.
# Kept generous (1400) so the ON state still has zoom headroom; toggle the
# dashboard's Downsample switch OFF for full native resolution when zooming in.
DASH_IMAGE_MAX_DIM = 1400
# PNG compression level (0-9) for the displayed frame. 0 = none (old behavior:
# fast but large); 1 is nearly free and shrinks real (sparse) frames a lot.
DASH_IMAGE_PNG_COMPRESSION = 1

# Background MATLAB runner (SequenceRunner.m). MATLAB_EXE defaults to
# "matlab" on PATH; MATLAB_ROOT is the matlab_new/ directory used as the
# startup path so addpath(genpath(pwd)) finds the full tree.
def _default_matlab_exe():
    """Pick a sensible MATLAB path per OS. Override with $YB_MATLAB_EXE."""
    override = os.environ.get('YB_MATLAB_EXE')
    if override:
        return override
    if os.name == 'nt':
        # Production lab PC runs R2023a; try common install roots.
        for p in (r'C:\Program Files\MATLAB\R2023a\bin\matlab.exe',
                  r'C:\Program Files\MATLAB\R2025b\bin\matlab.exe'):
            if os.path.exists(p):
                return p
        return r'C:\Program Files\MATLAB\R2023a\bin\matlab.exe'
    # macOS
    for p in ('/Applications/MATLAB_R2023a.app/bin/matlab',
              '/Applications/MATLAB_R2025b.app/bin/matlab'):
        if os.path.exists(p):
            return p
    return 'matlab'


MATLAB_EXE = _default_matlab_exe()
MATLAB_ROOT = os.environ.get(
    'YB_MATLAB_ROOT',
    os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'matlab_new')))
# Stable per-user location (NOT the OS temp dir, which Storage Sense / cleanup
# wipes — that silently reset the scan queue/history on restart). Mirrors
# ExptServer._state_dir(); the engine's ExptServer.QUEUE_PATH is the file of
# record, this constant is kept consistent for any future consumer.
RUNNER_QUEUE_PATH = os.path.join(
    os.environ.get('YB_NACSCTL_DIR')
    or os.path.join(os.environ.get('LOCALAPPDATA') or os.path.expanduser('~'),
                    'nacsctl'),
    'runner_queue.json')

# ---- Sequence backend selection ---------------------------------------------
# The monitor is a backend-agnostic ZMQ client: it talks to whichever backend
# hosts the ExptServer at MATLAB_URL. Two backends exist:
#   'matlab'  -> matlab_new/YbExptCtrl/SequenceRunner.m (legacy production)
#   'pyctrl'  -> the pyctrl Python front-end run loop (now the default)
# Switching is done live from the monitor GUI (a "Restart All"-style handoff
# that relaunches run_monitor with the new --backend). The two never run at
# once (one DCAM camera handle, one ZMQ port), so the handoff in run_monitor
# tears the old backend down before the new one binds.
DEFAULT_BACKEND = os.environ.get('YB_BACKEND', 'pyctrl')
VALID_BACKENDS = ('matlab', 'pyctrl')

# pyctrl backend launch. PYCTRL_PYTHON is the interpreter that has a libnacs
# build (NOT the yb_analysis env — that one lacks the engine). PYCTRL_MODULE is
# the run-loop entry point spawned as `python -m <module> <url>`; it hosts the
# same ExptServer the MATLAB runner does. PYCTRL_CWD is the pyctrl package root
# placed on sys.path. These are Phase-5 deliverables; until the module exists,
# switching to pyctrl simply brings up a backend-down GUI you can switch back
# from (run_monitor degrades gracefully when the backend fails to boot).
def _default_pyctrl_python():
    override = os.environ.get('YB_PYCTRL_PYTHON')
    if override:
        return override
    # Prefer the pyctrl engine venv: it has BOTH the libnacs engine AND pylablib
    # (the Orca camera dependency). Default to the Python 3.12 venv
    # (.venv-engine-py312); fall back to the legacy Python 3.8 venv (.venv-engine),
    # then to a bare Python38 (which has the engine but NOT pylablib, so a backend
    # spawned with it boots camera-less and the monitor's Camera card shows
    # "camera unavailable (pylablib not opened)").
    pyctrl_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), '..', 'pyctrl'))
    if os.name == 'nt':
        for venv_name in ('.venv-engine-py312', '.venv-engine'):
            venv = os.path.join(pyctrl_root, venv_name, 'Scripts', 'python.exe')
            if os.path.exists(venv):
                return venv
        for p in (r'C:\Users\Ybtweezer-PC2\AppData\Local\Programs\Python\Python38\python.exe',
                  r'C:\Python38\python.exe'):
            if os.path.exists(p):
                return p
        return r'C:\Python38\python.exe'
    for venv_name in ('.venv-engine-py312', '.venv-engine'):
        venv = os.path.join(pyctrl_root, venv_name, 'bin', 'python')
        if os.path.exists(venv):
            return venv
    return 'python3'


PYCTRL_PYTHON = _default_pyctrl_python()
PYCTRL_MODULE = os.environ.get('YB_PYCTRL_MODULE', 'launcher.run_loop.runner')
PYCTRL_CWD = os.environ.get(
    'YB_PYCTRL_CWD',
    os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'pyctrl')))

# ---- Orca camera config (read from / written to expConfig.m) ----

_EXPCONFIG_PATH = os.path.join(MATLAB_ROOT, 'expConfig.m')


def read_orca_config():
    """Parse consts.Orca.ROI and consts.Orca.ExposureTime from expConfig.m."""
    import re
    roi = [0, 0, 4096, 2304]
    exposure = 0.1
    try:
        with open(_EXPCONFIG_PATH, 'r') as f:
            text = f.read()
        m = re.search(r'consts\.Orca\.ROI\s*=\s*\[([^\]]+)\]', text)
        if m:
            roi = [int(float(x)) for x in m.group(1).split()]
        m = re.search(r'consts\.Orca\.ExposureTime\s*=\s*([0-9.eE+-]+)', text)
        if m:
            exposure = float(m.group(1))
    except Exception:
        pass
    return {'roi': roi, 'exposure_time': exposure}


def write_orca_roi(roi):
    """Update consts.Orca.ROI in expConfig.m in-place."""
    import re
    try:
        with open(_EXPCONFIG_PATH, 'r') as f:
            text = f.read()
        roi_str = '%d %d %d %d' % tuple(roi)
        text = re.sub(
            r'(consts\.Orca\.ROI\s*=\s*\[)[^\]]+(\])',
            r'\g<1>' + roi_str + r'\2',
            text)
        with open(_EXPCONFIG_PATH, 'w') as f:
            f.write(text)
    except Exception as e:
        raise RuntimeError(f'Failed to update expConfig.m: {e}')


def write_orca_exposure(exposure_time):
    """Update consts.Orca.ExposureTime (seconds) in expConfig.m in-place."""
    import re
    try:
        # Read in binary to preserve the file's original line endings.
        with open(_EXPCONFIG_PATH, 'rb') as f:
            raw = f.read()
        text = raw.decode('utf-8')
        # Detect the dominant newline style so any insertion matches.
        newline = '\r\n' if '\r\n' in text else '\n'
        exp_str = ('%g' % float(exposure_time))
        new_text, n = re.subn(
            r'(consts\.Orca\.ExposureTime\s*=\s*)[0-9.eE+-]+',
            r'\g<1>' + exp_str,
            text)
        if n == 0:
            # Field missing — append after the ROI line using its newline.
            new_text, m = re.subn(
                r'(consts\.Orca\.ROI\s*=\s*\[[^\]]+\];[^\r\n]*(?:\r?\n))',
                r'\1consts.Orca.ExposureTime = ' + exp_str + ';' + newline,
                text, count=1)
            if m == 0:
                # Neither ExposureTime nor ROI found — append at EOF.
                if not text.endswith(('\n', '\r\n')):
                    text += newline
                new_text = text + 'consts.Orca.ExposureTime = ' + exp_str + ';' + newline
        with open(_EXPCONFIG_PATH, 'wb') as f:
            f.write(new_text.encode('utf-8'))
    except Exception as e:
        raise RuntimeError(f'Failed to update expConfig.m: {e}')


# DataManager update intervals (in number of sequences)
UPDATE_GRID_INTERVAL = 50
UPDATE_GRID_BATCH_SIZE = 50
UPDATE_THRES_INTERVAL = 200
UPDATE_THRES_BATCH_SIZE = 200
UPDATE_LOADING_INTERVAL = 50  # update loading rates every 50 shots (like grid)
UPDATE_HIST_BATCH_SIZE = 2000  # accumulate this many shots for histogram
UPDATE_HIST_INTERVAL = 200     # recompute histograms every N shots

# ---- Live per-N-shots EWMA self-calibration (loading-pattern scans only) ----
# Every N loading shots we nudge BOTH the global SLM->camera affine (TRANSLATION
# only) and the per-pattern detection thresholds toward what the recent shots
# show, as an exponentially-weighted moving average (no stored queue; just blend
# with a weight). The EWMA weight w gives an effective memory of ~ N/w shots, so
# w = N / memory. With N = 10 and a ~100-shot memory, w ~= 0.1. Every update is
# persisted (affine_transform.json / per-pattern threshold.mat) AND appended to a
# shot-stamped jsonl audit log so the drift can be analysed offline.
AFFINE_LIVE_INTERVAL = 10        # seqs between affine translation updates
AFFINE_LIVE_BATCH = 30           # recent img1 frames averaged for the shift estimate
AFFINE_LIVE_SEARCH_RANGE = 4     # px cross-correlation half-width (drift/N shots is sub-px)
AFFINE_LIVE_EMA = 0.1            # EWMA weight per update (~N/EMA = 100-shot memory)
THRES_LIVE_INTERVAL = 10         # seqs between cheap per-site threshold EWMA updates
THRES_LIVE_WINDOW = 100          # recent shots used for each cheap threshold estimate
THRES_LIVE_EMA = 0.1             # EWMA weight for the cheap threshold update
THRES_LIVE_MIN_PER_SIDE = 3      # min recent shots on each side of the threshold to trust a site

# ---- Structural threshold-fit guard (degenerate-fit rejection + drift clamp) ----
# A full double-Gaussian refit is REJECTED (thresholds NOT replaced, NOT saved to
# either store) when the fitted empty/atom peaks are not credibly separated — the
# failure mode where a low-loading / near-unimodal histogram collapses both
# Gaussians onto one peak and the cheap tracker then explodes the per-site spread.
# Separation must exceed BOTH an absolute floor AND a multiple of the empty width,
# for at least THRES_FIT_MIN_SEP_FRAC of the converged sites; the pooled loaded
# fraction (at the new thresholds) must also be plausible.
THRES_FIT_MIN_SEP_ABS = 1.5      # ADU: absolute min (mu_atom - mu_empty) for an accepted site
THRES_FIT_MIN_SEP_SIGMA = 2.0    # also require separation >= this * sigma_empty
THRES_FIT_MIN_SEP_FRAC = 0.5     # >= this fraction of converged sites must clear the separation
THRES_FIT_MIN_LOADING = 0.05     # min pooled loaded-fraction (at the new thresholds) to trust a fit
# Cheap inter-fit tracker: each per-site threshold is clamped to stay at least
# this fraction of (mu_atom - mu_empty) inside EACH peak, so a threshold can never
# drift onto or past a peak (the spread-explosion mechanism). Requires a valid
# placement-ratio anchor from an ACCEPTED full fit; with no anchor the cheap update
# holds the loaded thresholds rather than flying blind on r=0.5.
THRES_LIVE_VALLEY_MARGIN = 0.15  # keep threshold within [mu_e + m*sep, mu_a - m*sep]
# On-load sanity: a stored per-pattern threshold vector whose per-site std exceeds
# this (ADU) is flagged degraded (the symptom of a corrupted store) and surfaced as
# a loud dashboard + log warning; it is re-anchored by the first ACCEPTED full fit
# rather than swapped for the (pattern-ambiguous) day-folder thresholds.
THRES_LOADED_MAX_SPREAD = 1.5    # ADU
# Cross-run accumulation: carry frame-0 (loading) intensities for the SAME pattern
# across DataManager (scan) boundaries so the 200-shot full fit can fire over
# several short runs, as long as they are within this many seconds of each other.
# In-memory only (module scope); lost on process restart.
THRES_ACCUM_CROSS_RUN_WINDOW_S = 3600   # ~1 hour
THRES_ACCUM_CROSS_RUN_MAX = 2000        # cap carried shots per pattern (== UPDATE_HIST_BATCH_SIZE)

# ---- Blank / dropped-frame rejection ----
# A camera/acquisition glitch occasionally yields a whole-frame ~0 image: every
# masked-site intensity reads essentially 0 because the camera (or ZMQ) returned no
# real frame (e.g. the 2026-06-19 47x47_feedbackwarm4 runs whose imgs were a
# degenerate 4x4 of background and whose intensities were exactly 0). Such a shot
# carries NO atom information, yet a double-Gaussian threshold refit reads the
# 0-vs-pedestal toggle as "empty vs atom", lands per-site cuts in the empty gap
# (corrupting detection), and stretches the per-site histograms so wide the real
# empty/atom doublet collapses into a single bar. A shot whose MAX masked intensity
# is below this floor is treated as blank and EXCLUDED from the threshold-fit
# accumulators (live + cross-run) and the per-site histograms. Set well below any
# real masked sum (which always sits on the camera pedestal) but above pure zero, so
# a legitimately-empty atom shot (still on the pedestal) is never rejected.
BLANK_FRAME_FLOOR = 1.0          # ADU: shot is blank/dropped if max(intensities) < this
# Backstop in the full-fit guard: reject a refit when more than this fraction of the
# accumulated shots are blank (whole-frame ~0), in case a blank slips past the
# ingestion filter (e.g. a cross-run buffer carried in before the filter existed).
THRES_FIT_MAX_BLANK_FRAC = 0.02
# Cap the shots used per full Gaussian refit AND per histogram rebuild. The per-site
# double-Gaussian fit (one bounded least-squares per site) is the dominant live cost;
# beyond a few hundred shots more data barely improves a per-site histogram while the
# histogram/percentile passes keep scaling with shot count (the cross-run buffer can
# reach 2000). Use the most recent THRES_FIT_MAX_SHOTS only.
THRES_FIT_MAX_SHOTS = 400
# Robust per-site histogram range (DISPLAY + the bins the fit reads): clip each site's
# intensities to this central percentile band before choosing the 50-bin range, so a
# single hot pixel / outlier shot can't stretch the bins so wide the empty/atom
# doublet collapses into one bar. (lo, hi) in percent.
HIST_DISPLAY_CLIP_PCT = (0.5, 99.5)

# --- img2 spot-shape GMM detector (yb_analysis/detection/spot_shape_model.py) ---
# img2 (the post-protocol frame) is detected by a spot-SHAPE GMM classifier
# rather than an intensity threshold: when nearly every site is loaded the img2
# intensity histogram is unimodal and has no clean cut, but the spot shape still
# discriminates loaded vs empty. Set to '' (empty) to disable and fall back to
# intensity thresholding for img2. Artifacts live in spot_shape_ml/model/
# (override the dir with $YB_SPOT_SHAPE_ML_DIR). Variants: 'A' = full 81-D
# patch shape, curated training (decisive / near-binary posteriors); 'B' =
# PCA-5, whole dataset; 'C' = PCA-5, curated (graded posteriors). The per-site
# posterior P(loaded) is stored as the "% certainty" alongside the logicals.
IMG2_SHAPE_MODEL_VARIANT = os.environ.get('YB_IMG2_SHAPE_MODEL', 'A')

# Number of completed scans shown in the queue history panel
QUEUE_HISTORY_DISPLAY = 30

# ---- SLM server (separate machine, reached over Tailscale) ----
#
# The SLM PC runs SLMnet's FastAPI server. By default we connect to its plaintext
# HTTP loopback companion port (8551) over Tailscale. No basic-auth inside the
# tailnet; outside it the SLM PC is unreachable.
#
# Defaults reflect the current lab Tailscale assignments. Override with env
# vars if those change or for a different deployment.
SLM_URL = os.environ.get('YB_SLM_URL', 'http://100.114.207.118:8551')
SLM_VERIFY_TLS = os.environ.get('YB_SLM_VERIFY_TLS', '0') == '1'

# The scope_control dashboard is hosted OUT OF PROCESS (on whichever PC can reach
# the bench scopes -- currently the rearrangement / SLM-server box over Tailscale).
# The dashboard's "Scope" tab just iframes its embed page ("{SCOPE_URL}/dashboard").
# No port is opened here. Override the host with YB_SCOPE_URL when it moves.
SCOPE_URL = os.environ.get('YB_SCOPE_URL', 'http://100.114.207.118:8600')

# The yb_monitor service (oven temp + wavemeter + laser PID locks) runs OUT OF
# PROCESS on the temp-monitor PC (tailnet-only). The "Monitor" tab iframes its
# embed page ("{MONITOR_URL}/dashboard?embed=1") DIRECTLY -- never server-side
# proxied: the monitor's control gate trusts the viewer's tailnet IP, which only a
# direct browser iframe preserves. Override the host with YB_MONITOR_URL.
MONITOR_URL = os.environ.get('YB_MONITOR_URL', 'http://100.118.221.34:8060')
SLM_PASSWORD_PATH = os.environ.get('YB_SLM_PASSWORD_PATH', None)
LAB_PC_TAILSCALE_IP = os.environ.get('YB_LAB_TAILSCALE_IP', '100.86.15.43')

# Per-endpoint poll cadence (ms) — different panels tolerate different
# staleness. Mapped to a background thread per endpoint in slm_proxy.py.
SLM_POLL_INTERVALS_MS = {
    'health':         2000,
    'devices':       10000,
    'lock_status':    2000,
    'camera_png':     2000,
    'phase_png':      2000,
    'rearrange_diag': 5000,
}

# HTTP timeouts for SLM proxy / sync calls. Tuple is (connect, read).
# Connect is tight: Tailscale handshake is fast or it's not happening at all.
# Read is more generous: PNG / JSON responses may be 50–500ms.
SLM_HTTP_TIMEOUT_S = (2.0, 5.0)

# ---- Molecube (FPGA1 DDS/TTL/clock) control daemon ----
#
# FPGA1's DDS / TTL / clock are fronted by the molecube2 C++ ZMQ daemon (the same
# daemon the libnacs engine submits sequences to, and that the Next.js page at
# https://yb.nigrp.org/s/zynq/1/dds drives). The dashboard's /api/molecube/* routes
# talk to it through yb_analysis.control.molecube_client -- ALWAYS through the daemon,
# never the devices directly.
#
# !!! These endpoints are MASTER-GATED (closed) by default for safety -- see the
#     MOLECUBE GATE block in plotting/dashboard.py. This URL is only used once the
#     gate is opened. Point YB_MOLECUBE_URL at the local mock for development.
MOLECUBE_URL = os.environ.get('YB_MOLECUBE_URL', 'tcp://192.168.0.174:7777')
MOLECUBE_TIMEOUT_MS = int(os.environ.get('YB_MOLECUBE_TIMEOUT_MS', '2000'))


def _read_molecube_max_ttl_chn():
    """Authoritative TTL channel count for the dashboard's Molecube panel.

    The molecube daemon's own ``get_max_ttl`` is unreliable here (an older daemon
    build replies with the 1-byte error status, read as ``1`` -> only ch 0-1), so
    we take the count from the SAME engine ``config.yml`` the libnacs zynq backend
    uses (``FPGA1.config.max_ttl_chn``, currently 55 -> channels 0..55, 2 banks).
    Env override: ``YB_MOLECUBE_MAX_TTL_CHN``."""
    override = os.environ.get('YB_MOLECUBE_MAX_TTL_CHN')
    if override:
        try:
            return int(override)
        except ValueError:
            pass
    import re
    here = os.path.dirname(__file__)
    for p in (os.path.join(here, '..', 'pyctrl', 'config.yml'),
              os.path.join(MATLAB_ROOT, 'config.yml')):
        try:
            with open(os.path.normpath(p)) as f:
                m = re.search(r'max_ttl_chn:\s*(\d+)', f.read())
            if m:
                return int(m.group(1))
        except Exception:
            pass
    return 55


MOLECUBE_MAX_TTL_CHN = _read_molecube_max_ttl_chn()
# Public web UI for the molecube daemon (the Next.js page above). The Hardware
# tab's Molecube sub-view links its "open source" (↗) action here, mirroring how
# the iframe sub-views link to their standalone dashboards. Override per host.
MOLECUBE_WEB_URL = os.environ.get('YB_MOLECUBE_WEB_URL', 'https://yb.nigrp.org/s/zynq/1/dds')
