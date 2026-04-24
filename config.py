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
