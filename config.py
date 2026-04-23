"""Experiment configuration constants — mirrors expConfig.m defaults."""

import os

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
    override = os.environ.get('YB_MATLAB_EXE')
    if override:
        return override
    if os.name == 'nt':
        return r'C:\Program Files\MATLAB\R2025b\bin\matlab.exe'
    # macOS / Linux
    mac_default = '/Applications/MATLAB_R2025b.app/bin/matlab'
    if os.path.exists(mac_default):
        return mac_default
    return 'matlab'


MATLAB_EXE = _default_matlab_exe()
MATLAB_ROOT = os.environ.get(
    'YB_MATLAB_ROOT',
    os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'matlab_new')))
RUNNER_QUEUE_PATH = os.path.join(
    os.environ.get('TEMP', '/tmp'), 'nacsctl', 'runner_queue.json')

# DataManager update intervals (in number of sequences)
UPDATE_GRID_INTERVAL = 50
UPDATE_GRID_BATCH_SIZE = 50
UPDATE_THRES_INTERVAL = 200
UPDATE_THRES_BATCH_SIZE = 200
UPDATE_LOADING_INTERVAL = 50  # update loading rates every 50 shots (like grid)
UPDATE_HIST_BATCH_SIZE = 2000  # accumulate this many shots for histogram
UPDATE_HIST_INTERVAL = 200     # recompute histograms every N shots
