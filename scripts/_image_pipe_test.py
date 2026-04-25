"""End-to-end verifier for the image-passing pipe across runner sessions.

Uses raw ZMQ (no ZmqClient/AnalysisUser) so the test doesn't introduce
an additional poller that races with MATLAB's ExptServer worker.

Each cycle:
  1. Spawn runner via RunnerLauncher (force-kills any prior runner)
  2. Send camera_init via raw ZMQ
  3. Block until camera reports connected=True (or fail). Skipping this
     and submitting a scan during MATLAB's ~25s OrcaInit reproduced an
     intermittent "0 frames" failure on this rig.
  4. Submit LACScan via a separate `matlab.exe -batch`
  5. Poll get_seq_num via raw ZMQ until at least one sequence completes
     or wait_s elapses
  6. Send camera_close via raw ZMQ
  7. runner.stop

Per-cycle MATLAB stdout is captured to /tmp/matlab_cycle<N>.log so a
failing cycle leaves a forensic trail.

Run with:  python -m yb_analysis.scripts._image_pipe_test [N] [wait_s]
"""
import json
import logging
import os
import subprocess
import sys
import time

import zmq

from yb_analysis.acquisition.runner_launcher import RunnerLauncher
from yb_analysis.config import MATLAB_EXE, MATLAB_ROOT, MATLAB_URL


def raw_zmq_call(url, frames, reply='string', timeout_ms=10000):
    """Send a multi-frame REQ to ExptServer, return the reply.

    Mirrors zmq_client._q_call but uses a fresh socket each call so we
    can't get tangled with anyone else's REQ-state."""
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(url)
    try:
        for i, f in enumerate(frames):
            flag = zmq.SNDMORE if i < len(frames) - 1 else 0
            if isinstance(f, str):
                sock.send_string(f, flag)
            else:
                sock.send(f, flag)
        if sock.poll(timeout_ms) == 0:
            return None
        if reply == 'string':
            return sock.recv_string()
        elif reply == 'int':
            return int.from_bytes(sock.recv(), 'little')
        elif reply == 'json':
            return json.loads(sock.recv())
        else:
            return sock.recv()
    finally:
        sock.close(linger=0)


def camera_init_raw(url, roi, exposure_time):
    payload = json.dumps({'roi': list(roi), 'exposure_time': float(exposure_time)})
    return raw_zmq_call(url, ['camera_init', payload], reply='string',
                        timeout_ms=10000)


def camera_close_raw(url):
    return raw_zmq_call(url, ['camera_close'], reply='string', timeout_ms=5000)


def query_seq_num(url, timeout_ms=5000):
    """nseq is monotonic — never reset. Use that, not nseq_imgs."""
    return raw_zmq_call(url, ['get_seq_num'], reply='int',
                        timeout_ms=timeout_ms)


def submit_lac_scan():
    cmd = [
        MATLAB_EXE, '-nodesktop', '-nosplash', '-batch',
        f"addpath(genpath('{MATLAB_ROOT}')); LACScan; pause(2);"
    ]
    print('  launching submitter MATLAB...')
    rc = subprocess.call(cmd, timeout=180)
    print(f'  submitter MATLAB exited rc={rc}')


def camera_status_raw(url, timeout_ms=2000):
    return raw_zmq_call(url, ['camera_status'], reply='json',
                        timeout_ms=timeout_ms)


def wait_for_camera_connected(url, timeout_s=45.0):
    """Poll camera_status after camera_init until connected=True or
    error. Returns (ok, last_status_dict)."""
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        try:
            st = camera_status_raw(url, timeout_ms=500)
        except Exception:
            st = None
        if isinstance(st, dict):
            last = st
            if st.get('connected'):
                return True, st
            if st.get('error'):
                return False, st
        time.sleep(0.5)
    return False, last or {}


def cycle(i, wait_s, url):
    print(f'\n========== CYCLE {i} ==========')
    # Per-cycle MATLAB stdout — RunnerLauncher reads YB_MATLAB_LOGFILE
    # from its OWN env and injects -logfile into the child cmdline.
    log_path = os.path.abspath(f'/tmp/matlab_cycle{i}.log')
    try:
        os.makedirs('/tmp', exist_ok=True)
    except Exception:
        pass
    os.environ['YB_MATLAB_LOGFILE'] = log_path
    runner = RunnerLauncher(
        matlab_exe=MATLAB_EXE,
        matlab_root=MATLAB_ROOT,
        url=url,
    )
    print(f'  matlab logfile: {log_path}')
    print('>>> runner.start()')
    runner.start(boot_timeout=90.0)

    print('>>> camera_init (raw ZMQ)')
    reply = camera_init_raw(url, [0, 0, 4096, 2304], 0.001)
    print(f'  reply: {reply!r}')

    print('>>> waiting for camera to report connected=True')
    cam_ok, cam_status = wait_for_camera_connected(url, timeout_s=45.0)
    print(f'  cam_ok={cam_ok}  status={cam_status}')

    n0 = query_seq_num(url)
    print(f'  seq_num at start: {n0}')

    print('>>> submit LACScan')
    submit_lac_scan()

    print(f'>>> poll get_seq_num for up to {wait_s}s')
    t0 = time.monotonic()
    last_count = n0 if n0 is not None else 0
    first_image_at = None
    while time.monotonic() - t0 < wait_s:
        count = query_seq_num(url, timeout_ms=3000)
        if count is not None and count > last_count:
            if first_image_at is None:
                first_image_at = time.monotonic() - t0
                print(f'  first seq finished at +{first_image_at:.1f}s '
                      f'(seq_num {last_count} -> {count})')
            last_count = count
        time.sleep(0.5)
    received = last_count - (n0 if n0 is not None else 0)
    images_flowed = received > 0
    print(f'  final seq_num = {last_count}')
    print(f'  total sequences this cycle: {received} '
          f'({"PASS" if images_flowed else "FAIL — pipe broken!"})')

    print('>>> camera_close (raw ZMQ) + runner.stop')
    try:
        camera_close_raw(url)
    except Exception as e:
        print(f'  camera_close failed: {e}')
    t_stop = time.monotonic()
    runner.stop(grace=20.0)
    print(f'  runner.stop took {time.monotonic() - t_stop:.2f}s')

    return images_flowed, received


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    wait_s = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S')
    print(f'MATLAB_EXE  = {MATLAB_EXE}')
    print(f'MATLAB_ROOT = {MATLAB_ROOT}')
    print(f'URL         = {MATLAB_URL}')
    print(f'N (cycles)  = {n}')
    print(f'wait per cycle = {wait_s}s')

    results = []
    for i in range(1, n + 1):
        ok, received = cycle(i, wait_s, MATLAB_URL)
        results.append((i, ok, received))

    print('\n' + '=' * 60)
    print('IMAGE PIPELINE VERDICT')
    print('=' * 60)
    any_broken = False
    for i, ok, received in results:
        marker = 'PASS' if ok else 'BROKEN'
        if not ok:
            any_broken = True
        print(f'  cycle {i}: {marker} ({received} sequences)')
    print('=' * 60)
    if any_broken:
        print('OVERALL: PIPE BROKEN in at least one cycle.')
        sys.exit(1)
    print('OVERALL: ALL CYCLES PASSED — pipe is stable across sessions.')


if __name__ == '__main__':
    main()
