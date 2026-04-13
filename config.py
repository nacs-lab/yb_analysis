"""Experiment configuration constants — mirrors expConfig.m defaults."""

import os

# Default data storage path (override via environment or constructor arg)
PATH_PREFIX = os.environ.get('YB_PATH_PREFIX', r'D:\OneDrive - Harvard University\Documents - Yb')
DATA_DIR = os.path.join(PATH_PREFIX, 'Data')

# ZMQ server URL for AnalysisClient
MATLAB_URL = 'tcp://127.0.0.1:1400'

# Fixed ports
DASHBOARD_PORT = 8050
TEST_SERVER_PORT = 1500

# DataManager update intervals (in number of sequences)
UPDATE_GRID_INTERVAL = 50
UPDATE_GRID_BATCH_SIZE = 50
UPDATE_THRES_INTERVAL = 200
UPDATE_THRES_BATCH_SIZE = 200
UPDATE_LOADING_INTERVAL = 50  # update loading rates every 50 shots (like grid)
UPDATE_HIST_BATCH_SIZE = 2000  # accumulate this many shots for histogram
UPDATE_HIST_INTERVAL = 200     # recompute histograms every N shots
