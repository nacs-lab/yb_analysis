#!/usr/bin/env python3
"""Entry point: Yb experiment monitor with tkinter control panel + Plotly Dash dashboard.

- Control panel: tkinter on main thread
- Dashboard: Plotly Dash in separate process (http://localhost:8050)
- Data processing: background thread in control panel

Usage:
    python -m yb_analysis.scripts.run_monitor [--url URL]
"""

import argparse
import atexit
import logging
import os
import signal
import subprocess
import sys

from yb_analysis.config import MATLAB_URL, DASHBOARD_PORT


def _kill_port(port):
    """Kill any process listening on the given TCP port (Windows)."""
    try:
        out = subprocess.check_output(
            ['netstat', '-ano'], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if f':{port} ' in line and 'LISTENING' in line:
                pid = int(line.strip().split()[-1])
                if pid == os.getpid():
                    continue
                logging.info('Killing stale process on port %d (pid=%d)', port, pid)
                subprocess.call(['taskkill', '/PID', str(pid), '/F'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        logging.debug('Port cleanup failed: %s', e)


def main():
    parser = argparse.ArgumentParser(description='Yb Experiment Monitor')
    parser.add_argument('--url', default=MATLAB_URL,
                        help=f'ZMQ server URL (default: {MATLAB_URL})')
    parser.add_argument('--refresh', type=int, default=2,
                        help='Refresh rate in seconds (default: 2)')
    parser.add_argument('--port', type=int, default=DASHBOARD_PORT,
                        help=f'Dashboard web server port (default: {DASHBOARD_PORT})')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable debug logging')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    # Kill stale processes on our ports
    _kill_port(args.port)

    from yb_analysis.acquisition.zmq_client import ZmqClient
    from yb_analysis.plotting.dashboard import DashboardRenderer
    from yb_analysis.gui.control_panel import ControlPanel
    from yb_analysis.io.preload import load_background_data

    logging.info('Connecting to %s', args.url)
    client = ZmqClient(args.url, refresh_rate=args.refresh)
    dashboard = DashboardRenderer(port=args.port)

    # Register cleanup for clean exit
    def _cleanup():
        logging.info('Shutting down...')
        dashboard.close()

    atexit.register(_cleanup)

    # Pre-load background data from most recent scan on disk
    bg_data = load_background_data()
    if bg_data is not None:
        logging.info('Loaded background: %d sites, thresholds + grid', bg_data['num_sites'])
        dashboard.update(bg_data)
    else:
        dashboard.start()

    logging.info('Dashboard at http://localhost:%d', args.port)
    app = ControlPanel(client, dashboard)
    app.mainloop()


if __name__ == '__main__':
    main()
