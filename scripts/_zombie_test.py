"""Standalone test: spawn RunnerLauncher, init camera, stop, look for zombies.

Run with:  python -m yb_analysis.scripts._zombie_test
"""
import logging
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
        "ForEach-Object { '{0}|{1}|{2}' -f $_.ProcessId, $_.Name, $_.ParentProcessId }"
    )
    try:
        out = subprocess.check_output(
            ['powershell', '-NoProfile', '-NonInteractive', '-Command', ps],
            text=True, timeout=10)
    except Exception as e:
        return [('ERROR', str(e), '')]
    rows = []
    for line in out.splitlines():
        line = line.strip()
        if line:
            parts = line.split('|')
            rows.append(tuple(parts) if len(parts) == 3 else (line, '', ''))
    return rows


def show(title, rows):
    print(f"--- {title} ---")
    if not rows:
        print("  (none)")
    for pid, name, ppid in rows:
        print(f"  PID={pid:<6} name={name:<15} parentPID={ppid}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S')
    print(f"MATLAB_EXE   = {MATLAB_EXE}")
    print(f"MATLAB_ROOT  = {MATLAB_ROOT}")
    print(f"URL          = {MATLAB_URL}")
    print()

    show("BEFORE", snapshot())

    runner = RunnerLauncher(
        matlab_exe=MATLAB_EXE,
        matlab_root=MATLAB_ROOT,
        url=MATLAB_URL,
    )
    print("\n>>> runner.start()")
    t0 = time.monotonic()
    runner.start(boot_timeout=90.0)
    print(f"   booted in {time.monotonic() - t0:.1f}s")

    show("AFTER START", snapshot())

    print("\n>>> camera_init (triggers OrcaInit -> dcamtray spawn)")
    client = ZmqClient(MATLAB_URL, refresh_rate=2)
    try:
        client.camera_init([0, 0, 4096, 2304], exposure_time=0.001)
        print("   camera_init OK")
    except Exception as e:
        print(f"   camera_init FAILED: {e}")

    time.sleep(2)
    show("AFTER camera_init", snapshot())

    print("\n>>> client.camera_close")
    try:
        client.camera_close()
        print("   camera_close OK")
    except Exception as e:
        print(f"   camera_close FAILED: {e}")
    time.sleep(2)
    show("AFTER camera_close", snapshot())

    print("\n>>> runner.stop(grace=15)")
    t0 = time.monotonic()
    runner.stop(grace=15.0)
    print(f"   stopped in {time.monotonic() - t0:.1f}s")
    time.sleep(2)
    show("AFTER STOP", snapshot())

    print("\n=== ZOMBIE CHECK ===")
    final = snapshot()
    matlab_zombies = [r for r in final if 'matlab' in r[1].lower() or 'MATLAB' in r[1]]
    dcam_zombies = [r for r in final if 'dcam' in r[1].lower()]
    print(f"  MATLAB zombies:   {len(matlab_zombies)}")
    print(f"  dcamtray zombies: {len(dcam_zombies)}")
    if matlab_zombies:
        print("  MATLAB zombies are bad — accumulation regression!")
        for r in matlab_zombies:
            print(f"    {r}")
        return 1
    print("  CLEAN")
    return 0


if __name__ == '__main__':
    sys.exit(main())
