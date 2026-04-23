#!/usr/bin/env python3
"""Entry point: Yb experiment monitor with tkinter control panel + Plotly Dash dashboard.

- Control panel: tkinter on main thread
- Dashboard: Plotly Dash in separate process (http://localhost:8050)
- Data processing: background thread in control panel
- Sequence runner: background MATLAB process (SequenceRunner.m), spawned
  on startup and shut down on exit. Pass --no-runner to skip (e.g. when a
  MATLAB runner is already running in another window).

Usage:
    python -m yb_analysis.scripts.run_monitor [--url URL] [--no-runner] [--mock]
"""

import argparse
import atexit
import logging

from yb_analysis.acquisition.port_utils import kill_port
from yb_analysis.config import (
    MATLAB_URL, DASHBOARD_PORT, MATLAB_EXE, MATLAB_ROOT,
)


def main():
    parser = argparse.ArgumentParser(description='Yb Experiment Monitor')
    parser.add_argument('--url', default=MATLAB_URL,
                        help=f'ZMQ server URL (default: {MATLAB_URL})')
    parser.add_argument('--refresh', type=int, default=2,
                        help='Refresh rate in seconds (default: 2)')
    parser.add_argument('--port', type=int, default=DASHBOARD_PORT,
                        help=f'Dashboard web server port (default: {DASHBOARD_PORT})')
    parser.add_argument('--no-runner', action='store_true',
                        help='Do not spawn the background MATLAB SequenceRunner')
    parser.add_argument('--mock', action='store_true',
                        help='Launch the runner with NACS_MOCK=1 (stub libnacs)')
    parser.add_argument('--matlab-exe', default=MATLAB_EXE,
                        help=f'MATLAB binary (default: {MATLAB_EXE})')
    parser.add_argument('--matlab-root', default=MATLAB_ROOT,
                        help=f'matlab_new directory (default: {MATLAB_ROOT})')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable debug logging')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    # Clear any stale listener on the dashboard port
    kill_port(args.port)

    # Start the background MATLAB runner (owns the ZMQ server at args.url)
    runner = None
    if not args.no_runner:
        from yb_analysis.acquisition.runner_launcher import RunnerLauncher
        runner = RunnerLauncher(
            matlab_exe=args.matlab_exe,
            matlab_root=args.matlab_root,
            url=args.url,
            mock=args.mock,
        )
        logging.info('Starting MATLAB SequenceRunner (mock=%s)...', args.mock)
        runner.start()

    from yb_analysis.acquisition.zmq_client import ZmqClient
    from yb_analysis.plotting.dashboard import DashboardRenderer
    from yb_analysis.gui.control_panel import ControlPanel
    from yb_analysis.io.preload import load_background_data

    logging.info('Connecting to %s', args.url)
    client = ZmqClient(args.url, refresh_rate=args.refresh)
    dashboard = DashboardRenderer(port=args.port)

    def _cleanup():
        logging.info('Shutting down...')
        try:
            dashboard.close()
        except Exception:
            pass
        if runner is not None:
            try:
                runner.stop()
            except Exception as e:
                logging.warning('Runner stop failed: %s', e)

    atexit.register(_cleanup)

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
