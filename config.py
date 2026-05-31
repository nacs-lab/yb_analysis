"""Experiment configuration constants — mirrors expConfig.m defaults."""

import os
import tempfile

# Default data storage path (override via environment or constructor arg)
PATH_PREFIX = os.environ.get('YB_PATH_PREFIX', r'D:\OneDrive - Harvard University\Documents - Yb')
DATA_DIR = os.path.join(PATH_PREFIX, 'Data')

# ZMQ server URL for AnalysisClient — must match consts.MatlabURL in
# matlab_new/expConfig.m (line 9).
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
RUNNER_QUEUE_PATH = os.path.join(
    tempfile.gettempdir(), 'nacsctl', 'runner_queue.json')

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
