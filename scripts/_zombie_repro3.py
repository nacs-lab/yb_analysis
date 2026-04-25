"""Reproduce the user's flow exactly: launch monitor, wait, then send WM_CLOSE
to the tkinter window — the SAME Windows message the X-button click generates.

Run with:  python -m yb_analysis.scripts._zombie_repro3 [N] [wait_seconds]
"""
import ctypes
from ctypes import wintypes
import os
import subprocess
import sys
import time

user32 = ctypes.WinDLL('user32', use_last_error=True)
WM_CLOSE = 0x0010

EnumWindows = user32.EnumWindows
EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
GetWindowThreadProcessId = user32.GetWindowThreadProcessId
GetWindowTextW = user32.GetWindowTextW
GetClassNameW = user32.GetClassNameW
PostMessageW = user32.PostMessageW
IsWindowVisible = user32.IsWindowVisible


def find_top_windows_for_pid(target_pid):
    """Return list of (hwnd, title, classname) for top-level windows owned by pid."""
    found = []

    def callback(hwnd, lparam):
        if not IsWindowVisible(hwnd):
            return True
        pid = wintypes.DWORD()
        GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value != target_pid:
            return True
        buf = ctypes.create_unicode_buffer(512)
        GetWindowTextW(hwnd, buf, 512)
        title = buf.value
        cls_buf = ctypes.create_unicode_buffer(256)
        GetClassNameW(hwnd, cls_buf, 256)
        cls = cls_buf.value
        found.append((hwnd, title, cls))
        return True

    EnumWindows(EnumWindowsProc(callback), 0)
    return found


def snapshot():
    ps = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -match 'MATLAB|matlab|dcamtray' } | "
        "ForEach-Object { '{0}|{1}|{2}|{3}|{4}|{5}' -f $_.ProcessId, $_.Name, "
        "$_.ParentProcessId, $_.ThreadCount, $_.HandleCount, $_.CreationDate }"
    )
    try:
        out = subprocess.check_output(
            ['powershell', '-NoProfile', '-NonInteractive', '-Command', ps],
            text=True, timeout=10)
    except Exception as e:
        return [('ERROR', str(e), '', '', '', '')]
    rows = []
    for line in out.splitlines():
        line = line.strip()
        if line:
            parts = line.split('|')
            if len(parts) == 6:
                rows.append(tuple(parts))
    return rows


def show(title, rows):
    print(f"--- {title} ---")
    if not rows:
        print("  (none)")
        return
    for pid, name, ppid, tc, hc, ctime in rows:
        print(f"  PID={pid:<6} {name:<14} parent={ppid:<6} threads={tc} handles={hc} created={ctime}")


def cycle(i, wait_s):
    print(f"\n========== CYCLE {i} ==========")
    show(f"BEFORE start (cycle {i})", snapshot())

    print(">>> launch run_monitor")
    extra_args = os.environ.get('YB_REPRO_EXTRA_ARGS', '').split()
    cmd = [
        r'C:\Users\Ybtweezer-PC2\anaconda3\envs\yb_analysis\python.exe',
        '-m', 'yb_analysis.scripts.run_monitor', '--verbose',
    ] + extra_args
    log_path = os.path.join(os.environ['TEMP'],
                            f'_zombie_repro3_cycle{i}.log')
    log_f = open(log_path, 'w', encoding='utf-8', errors='replace')
    print(f"  log -> {log_path}")
    proc = subprocess.Popen(
        cmd,
        cwd=r'c:\msys64\home\Ybtweezer-PC2\projects\experiment-control',
        stdout=log_f,
        stderr=subprocess.STDOUT,
    )
    print(f"  monitor PID = {proc.pid}")

    print(f">>> wait {wait_s}s for camera init + GUI to appear")
    time.sleep(wait_s)

    show(f"AFTER camera init (cycle {i})", snapshot())

    # Find the tkinter window owned by this Python process
    print(">>> finding tkinter window for monitor process...")
    deadline = time.monotonic() + 10
    wins = []
    while time.monotonic() < deadline:
        wins = find_top_windows_for_pid(proc.pid)
        if wins:
            break
        time.sleep(0.5)
    if not wins:
        print(f"  no visible windows for PID {proc.pid}!")
    for hwnd, title, cls in wins:
        print(f"  hwnd={hwnd:#x} class={cls!r} title={title!r}")

    # Send WM_CLOSE to the main tk window — exactly what clicking the X does.
    print(">>> send WM_CLOSE to tkinter window(s)")
    for hwnd, title, cls in wins:
        if 'Tk' in cls or 'tk' in cls.lower():
            print(f"  PostMessage(WM_CLOSE) -> hwnd={hwnd:#x} ({title!r})")
            PostMessageW(hwnd, WM_CLOSE, 0, 0)

    print(">>> wait up to 90s for monitor to exit")
    try:
        rc = proc.wait(timeout=90)
        print(f"  monitor exited rc={rc}")
    except subprocess.TimeoutExpired:
        print("  monitor did NOT exit, terminating")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    time.sleep(3)
    rows = snapshot()
    show(f"AFTER shutdown (cycle {i})", rows)
    return rows


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    wait_s = int(sys.argv[2]) if len(sys.argv) > 2 else 75
    print(f"N (cycles)  = {n}")
    print(f"wait per cycle = {wait_s}s")
    show("BASELINE", snapshot())

    results = []
    for i in range(1, n + 1):
        rows = cycle(i, wait_s)
        results.append((i, rows))

    print("\n" + "=" * 50)
    print("FINAL ZOMBIE TALLY")
    print("=" * 50)
    for i, rows in results:
        m = sum(1 for r in rows if 'matlab' in r[1].lower())
        d = sum(1 for r in rows if r[1].lower() == 'dcamtray.exe')
        print(f"  cycle {i}: {m} MATLAB process(es), {d} dcamtray remaining")

    show("FINAL", snapshot())


if __name__ == '__main__':
    main()
