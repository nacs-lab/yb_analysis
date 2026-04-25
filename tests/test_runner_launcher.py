"""Tests for RunnerLauncher — uses a small Python stand-in to exercise
the spawn → ping → force-kill lifecycle without requiring a real MATLAB."""
import itertools
import os
import subprocess
import sys
import tempfile
import textwrap
import time

import pytest


_port_counter = itertools.count(14600)


def _next_url():
    return f"tcp://127.0.0.1:{next(_port_counter)}"


@pytest.fixture
def fake_runner_script(tmp_path):
    """A Python script that mimics SequenceRunner: binds ExptServer and
    loops forever. RunnerLauncher.stop() force-kills it."""
    script = tmp_path / "fake_runner.py"
    expserver_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'matlab_new', 'YbExpServer'))
    script.write_text(textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {expserver_dir!r})
        from ExptServer import ExptServer
        url = sys.argv[1]
        srv = ExptServer(url)
        try:
            while True:
                time.sleep(0.1)
        finally:
            srv.stop_worker()
    """))
    return str(script)


def test_spawn_ping_shutdown(fake_runner_script, tmp_path, monkeypatch):
    """RunnerLauncher should:
      1. kill-stale-port (no-op since port is free),
      2. spawn the process,
      3. wait for ping to succeed,
      4. on stop(): force-kill the process.
    """
    from yb_analysis.acquisition.runner_launcher import RunnerLauncher

    url = _next_url()
    rl = RunnerLauncher(
        matlab_exe=sys.executable,
        matlab_root=str(tmp_path),
        url=url,
    )

    # Replace the .start() spawn command with our python stand-in. Easiest:
    # monkeypatch subprocess.Popen called from RunnerLauncher.
    orig_popen = subprocess.Popen
    captured = {}

    def spy_popen(cmd, *args, **kwargs):
        # ignore the matlab command RunnerLauncher built; run our stand-in
        captured['cmd'] = cmd
        return orig_popen([sys.executable, fake_runner_script, url], *args, **kwargs)

    monkeypatch.setattr('yb_analysis.acquisition.runner_launcher.subprocess.Popen', spy_popen)

    rl.start(boot_timeout=10.0)
    assert rl.is_alive()
    # sanity: the originally-constructed command used the -nodesktop flags
    assert '-nodesktop' in captured['cmd']
    assert '-r' in captured['cmd']

    rl.stop(grace=5.0)
    assert not rl.is_alive()


def test_boot_timeout_if_nothing_responds(tmp_path, monkeypatch):
    from yb_analysis.acquisition.runner_launcher import RunnerLauncher

    url = _next_url()

    # A Popen that runs `sleep` — never binds a port. start() should time
    # out, tear down the process, and raise TimeoutError.
    orig_popen = subprocess.Popen

    def spy_popen(cmd, *args, **kwargs):
        return orig_popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                          *args, **kwargs)

    monkeypatch.setattr('yb_analysis.acquisition.runner_launcher.subprocess.Popen', spy_popen)

    rl = RunnerLauncher(
        matlab_exe=sys.executable,
        matlab_root=str(tmp_path),
        url=url,
    )
    with pytest.raises(TimeoutError):
        rl.start(boot_timeout=2.0)
    # RunnerLauncher should have cleaned the process up
    assert not rl.is_alive()
