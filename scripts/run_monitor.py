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
import signal

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
    parser.add_argument('--reuse-runner', action='store_true',
                        help='Reuse an already-running SequenceRunner if present, '
                             'and DO NOT shut it down on exit. Mitigates the '
                             'DCAM/AslDma zombie issue (one MATLAB stays alive '
                             'across GUI sessions).')
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
            reuse=args.reuse_runner,
        )
        logging.info('Starting MATLAB SequenceRunner (mock=%s)...', args.mock)
        runner.start()

    from yb_analysis.acquisition.zmq_client import ZmqClient
    from yb_analysis.plotting.dashboard import DashboardRenderer
    from yb_analysis.gui.control_panel import ControlPanel
    from yb_analysis.io.preload import load_background_data, bootstrap_today_from
    from yb_analysis.config import read_orca_config

    logging.info('Connecting to %s', args.url)
    client = ZmqClient(args.url, refresh_rate=args.refresh)

    orca_cfg = read_orca_config()
    dashboard = DashboardRenderer(port=args.port)

    def _cleanup():
        logging.info('Shutting down...')
        try:
            client.camera_close()
        except Exception:
            pass
        try:
            dashboard.close()
        except Exception:
            pass
        if runner is not None:
            try:
                runner.stop()
            except Exception as e:
                logging.warning('Runner stop failed: %s', e)
        # Final safety net: kill whatever still holds the ZMQ port.
        # Covers cases where runner.stop() failed or the JVM is orphaned.
        from yb_analysis.acquisition.port_utils import kill_port
        try:
            kill_port(int(args.url.rsplit(':', 1)[-1].split('/')[0]))
        except Exception:
            pass

    atexit.register(_cleanup)

    bg_data, init_dir = load_background_data()
    if bg_data is not None:
        logging.info('Loaded background: %d sites, thresholds + grid', bg_data['num_sites'])
        init_status = f'{bg_data["num_sites"]} sites loaded'
        if init_dir:
            today_name, copied = bootstrap_today_from(init_dir)
            if today_name and copied:
                logging.info('Bootstrapped today (%s) with %d files from %s',
                             today_name, len(copied), init_dir)
                init_status = f'{init_status} → copied to {today_name}'
        dashboard.update(bg_data)
    else:
        init_dir = None
        init_status = 'No data found'
        dashboard.start()

    logging.info('Dashboard at http://localhost:%d', args.port)
    app = ControlPanel(client, dashboard, init_dir=init_dir, init_status=init_status)
    app._camera_pane.set_roi(orca_cfg['roi'])
    app._camera_pane.set_exposure(orca_cfg['exposure_time'])
    # Kick off camera init in the background so the GUI shows "Connecting..."
    # from its very first render instead of blocking before the window opens.
    logging.info('Starting camera init in background (ROI=%s, Exposure=%g s)...',
                 orca_cfg['roi'], orca_cfg['exposure_time'])
    app._camera_pane._on_connect()

    # Ctrl-C in the terminal: route through the UI's normal close path so
    # the dashboard + runner shut down cleanly.
    def _on_sigint(signum, frame):
        logging.info('Signal %s received — closing UI', signum)
        try:
            app.after(0, app._on_close)
        except Exception:
            app.quit()
    signal.signal(signal.SIGINT, _on_sigint)
    # Also catch Ctrl-Break on Windows (SIGBREAK). Without this, sending
    # Ctrl-Break (or CTRL_BREAK_EVENT to a CREATE_NEW_PROCESS_GROUP child)
    # terminates Python without running atexit, leaving the MATLAB runner
    # orphaned. Used by the zombie-repro test harness; harmless otherwise.
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, _on_sigint)

    app.mainloop()


if __name__ == '__main__':
    main()
