"""Reproduce the user's flow: start runner -> submit LACScan -> wait -> stop -> count zombies.

Run with:  python -m yb_analysis.scripts._zombie_repro [N]
"""
import logging
import os
import subprocess
import sys
import time

from yb_analysis.acquisition.runner_launcher import RunnerLauncher
from yb_analysis.acquisition.zmq_client import ZmqClient
from yb_analysis.config import MATLAB_EXE, MATLAB_ROOT, MATLAB_URL


def snapshot():
    ps = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -match 'MATLAB|dcamtray' } | "
        "ForEach-Object { '{0}|{1}|{2}|{3}' -f $_.ProcessId, $_.Name, $_.ParentProcessId, $_.CreationDate }"
    )
    try:
        out = subprocess.check_output(
            ['powershell', '-NoProfile', '-NonInteractive', '-Command', ps],
            text=True, timeout=10)
    except Exception as e:
        return [('ERROR', str(e), '', '')]
    rows = []
    for line in out.splitlines():
        line = line.strip()
        if line:
            parts = line.split('|')
            rows.append(tuple(parts) if len(parts) == 4 else (line, '', '', ''))
    return rows


def show(title, rows):
    print(f"--- {title} ---")
    if not rows:
        print("  (none)")
    for row in rows:
        pid, name, ppid, ctime = row
        print(f"  PID={pid:<6} name={name:<15} parentPID={ppid}  created={ctime}")


def submit_lac_scan():
    """Launch a separate matlab.exe to run LACScan and exit. Submission is
    async (ZMQ submit_job returns immediately), so this returns ~5s after
    MATLAB boots."""
    cmd = [
        MATLAB_EXE, '-nodesktop', '-nosplash', '-batch',
        f"addpath(genpath('{MATLAB_ROOT}')); LACScan; pause(2);"
    ]
    print(f"  launching submitter MATLAB...")
    rc = subprocess.call(cmd, timeout=120)
    print(f"  submitter MATLAB exited rc={rc}")


def cycle(i, runner_url):
    print(f"\n========== CYCLE {i} ==========")
    runner = RunnerLauncher(
        matlab_exe=MATLAB_EXE,
        matlab_root=MATLAB_ROOT,
        url=runner_url,
    )
    print(">>> runner.start()")
    runner.start(boot_timeout=90.0)

    print(">>> camera_init")
    client = ZmqClient(runner_url, refresh_rate=2)
    client.camera_init([0, 0, 4096, 2304], exposure_time=0.001)

    print(">>> submit LACScan via separate MATLAB")
    submit_lac_scan()

    wait_s = int(os.environ.get('YB_REPRO_WAIT', '20'))
    print(f">>> sleep {wait_s}s for scan to (try to) run")
    time.sleep(wait_s)

    show(f"BEFORE STOP (cycle {i})", snapshot())

    print(">>> client.camera_close")
    try:
        client.camera_close()
    except Exception as e:
        print(f"  camera_close FAILED: {e}")
    time.sleep(1)

    print(">>> runner.stop(grace=20)")
    t0 = time.monotonic()
    runner.stop(grace=20.0)
    print(f"  stopped in {time.monotonic() - t0:.1f}s")
    time.sleep(2)

    rows = snapshot()
    show(f"AFTER STOP (cycle {i})", rows)
    matlab_left = [r for r in rows if 'MATLAB' in r[1] or 'matlab' in r[1].lower()]
    return len(matlab_left), rows


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S')
    print(f"MATLAB_EXE  = {MATLAB_EXE}")
    print(f"MATLAB_ROOT = {MATLAB_ROOT}")
    print(f"URL         = {MATLAB_URL}")
    print(f"N (cycles)  = {n}")
    print()
    show("BEFORE", snapshot())

    results = []
    for i in range(1, n + 1):
        zombies, rows = cycle(i, MATLAB_URL)
        results.append((i, zombies, rows))

    print("\n" + "=" * 50)
    print("FINAL ZOMBIE TALLY")
    print("=" * 50)
    for i, z, _ in results:
        print(f"  cycle {i}: {z} MATLAB process(es) left behind")

    final = snapshot()
    print(f"  total at end: {sum(1 for r in final if 'matlab' in r[1].lower())}")
    show("FINAL", final)


if __name__ == '__main__':
    main()
