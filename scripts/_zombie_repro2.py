"""Reproduce: run the monitor, wait for camera init, quit. Look for zombies.

Mirrors the user's exact flow. Uses an inline harness that runs the monitor's
main() but monkey-patches ControlPanel to schedule its OWN _on_close after a
delay — same code path the user's GUI X-button click triggers, no signals
involved.

Run with:  python -m yb_analysis.scripts._zombie_repro2 [N] [wait_seconds]
"""
import os
import subprocess
import sys
import time
import textwrap


HARNESS_SCRIPT = textwrap.dedent('''
    import sys
    sys.argv = ['run_monitor', '--verbose']

    # Monkey-patch ControlPanel to auto-quit after a delay.
    # SAME code path the GUI X-button click triggers.
    import yb_analysis.gui.control_panel as cp_module
    _orig_init = cp_module.ControlPanel.__init__

    def patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        delay_ms = {wait_ms}
        print("[harness] scheduling _on_close in " + str(delay_ms) + " ms",
              flush=True)
        self.after(delay_ms, self._on_close)

    cp_module.ControlPanel.__init__ = patched_init

    from yb_analysis.scripts.run_monitor import main
    main()
''')


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

    # Write the harness script to a temp file
    tmp = os.path.join(os.environ['TEMP'], f'_zombie_harness_{os.getpid()}.py')
    with open(tmp, 'w') as f:
        f.write(HARNESS_SCRIPT.format(wait_ms=wait_s * 1000))

    print(f">>> launch run_monitor (auto-quit in {wait_s}s)")
    cmd = [
        r'C:\Users\Ybtweezer-PC2\anaconda3\envs\yb_analysis\python.exe',
        tmp,
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=r'c:\msys64\home\Ybtweezer-PC2\projects\experiment-control',
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"  monitor PID = {proc.pid}")

    # Wait for monitor to exit on its own (auto-close fires after wait_s)
    print(f">>> wait up to {wait_s + 90}s for monitor to exit cleanly")
    try:
        rc = proc.wait(timeout=wait_s + 90)
        print(f"  monitor exited rc={rc}")
    except subprocess.TimeoutExpired:
        print("  monitor did NOT exit, terminating")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    try:
        os.remove(tmp)
    except Exception:
        pass

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
