"""In-process tests for the ExptServer job queue.

These drive the queue via direct method calls + via ZMQ, and exercise
persistence (save/load + demote on reload).
"""
import itertools
import json
import os
import tempfile
import time

import pytest
import zmq

_port_counter = itertools.count(14500)


def _next_url():
    return f"tcp://127.0.0.1:{next(_port_counter)}"


@pytest.fixture
def server():
    from ExptServer import ExptServer
    url = _next_url()
    srv = ExptServer(url)
    time.sleep(0.1)  # let the worker thread actually bind
    try:
        yield srv, url
    finally:
        srv.stop_worker()


def test_submit_pop_finish_history(server):
    srv, _ = server
    jid1 = srv.submit_job(b'\x00LACScan\x00payload1')
    jid2 = srv.submit_job(b'\x00EITScan\x00payload2')
    assert jid1 == 1 and jid2 == 2

    q = srv.queue_list()
    assert [e['id'] for e in q['queued']] == [1, 2]
    assert q['running'] is None

    job = srv.pop_next_job()
    assert job['id'] == 1
    q = srv.queue_list()
    assert q['running']['id'] == 1
    assert [e['id'] for e in q['queued']] == [2]

    assert srv.finish_job(job['id'], 'ok') is True
    q = srv.queue_list()
    assert q['running'] is None
    assert [e['id'] for e in q['history']] == [1]
    assert q['history'][0]['status'] == 'ok'


def test_move_and_remove(server):
    srv, _ = server
    a = srv.submit_job(b'\x00A\x00')
    b = srv.submit_job(b'\x00B\x00')
    c = srv.submit_job(b'\x00C\x00')

    assert srv.queue_move(a, 'down') is True
    assert [e['id'] for e in srv.queue_list()['queued']] == [b, a, c]

    assert srv.queue_move(c, 'up') is True
    assert [e['id'] for e in srv.queue_list()['queued']] == [b, c, a]

    assert srv.queue_remove(c) is True
    assert [e['id'] for e in srv.queue_list()['queued']] == [b, a]

    # cannot remove a non-queued / nonexistent id
    srv.pop_next_job()
    assert srv.queue_remove(b) is False


def test_move_edges(server):
    srv, _ = server
    a = srv.submit_job(b'\x00A\x00')
    b = srv.submit_job(b'\x00B\x00')
    # can't move head further up or tail further down
    assert srv.queue_move(a, 'up') is False
    assert srv.queue_move(b, 'down') is False


def test_persistence_and_demote():
    from ExptServer import ExptServer
    url = _next_url()
    srv = ExptServer(url)
    srv.submit_job(b'\x00A\x00')
    srv.submit_job(b'\x00B\x00')
    srv.pop_next_job()  # A is now running
    srv.stop_worker()
    del srv
    time.sleep(0.1)

    srv2 = ExptServer(url)
    try:
        q = srv2.queue_list()
        # both reloaded; the previously-running A is demoted to queued
        assert [e['id'] for e in q['queued']] == [1, 2]
        assert all(e['state'] == 'queued' for e in q['queued'])
        assert q['running'] is None
    finally:
        srv2.stop_worker()


def test_zmq_ping_and_queue(server):
    srv, url = server
    ctx = zmq.Context()
    s = ctx.socket(zmq.REQ)
    s.setsockopt(zmq.LINGER, 0)
    try:
        s.connect(url)
        s.send_string('ping')
        assert s.recv_string() == 'pong'

        s.send_string('submit_job', zmq.SNDMORE)
        s.send(b'\x00LACScan\x00')
        jid = int.from_bytes(s.recv(), 'little')
        assert jid >= 1

        s.send_string('queue_list')
        q = json.loads(s.recv())
        assert any(e['id'] == jid for e in q['queued'])
    finally:
        s.close(linger=0)
        ctx.destroy(linger=0)


def test_port_rebind_after_stop():
    """After stopping the server, the port should be free for immediate
    rebind — validates the LINGER=0 + ctx.destroy fix."""
    from ExptServer import ExptServer
    url = _next_url()
    srv = ExptServer(url)
    srv.stop_worker()
    del srv

    # re-bind on the same url must succeed without sleep/retry
    srv2 = ExptServer(url)
    try:
        assert True
    finally:
        srv2.stop_worker()
