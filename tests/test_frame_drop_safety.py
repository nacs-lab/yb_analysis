"""Frame-drop safety in DataManager (two-array survival pairing).

A camera/compute stall can deliver a sequence with fewer than ``pSeq`` frames
(e.g. the loading image but not the final image). The img1/img2 split downstream
is purely by frame parity (``idx % pSeq`` in process_data, ``[0::pSeq]`` /
``[pSeq-1::pSeq]`` in save_data), so a single short sequence would phase-flip
that split for EVERY later sequence and silently scramble survival pairing —
the failure behind scan 20260608_111039 (logicals_img1=244, logicals_img2=186).

These tests pin the two safety layers:
  1. store_new_data drops an incomplete sequence whole (its seq_id too) so the
     processing buffers stay 1:1 and a multiple of pSeq.
  2. save_data's demux stays pSeq-aware and trims any orphan tail, so the saved
     logicals_img1 / logicals_img2 / seq_ids always agree in length.
"""

import time
import threading

import h5py
import numpy as np
import pytest

from yb_analysis.acquisition.data_manager import DataManager
from yb_analysis.io.hdf5_store import create_scan_file


def _drop_safety_dm(pSeq=2):
    """Bare DM with just the attributes store_new_data touches."""
    dm = DataManager.__new__(DataManager)
    dm.num_images_per_seq = pSeq
    dm._dropped_seqs = 0
    dm._dropped_seq_ids = []
    dm.frame_size = (4, 4)
    dm._frame_size_fixed = True          # skip the autodetect/HDF5-retry block
    dm._file_created = True
    dm.num_sites = 3
    dm.img_buffer = None                 # skip the ring-buffer push
    dm._imgs_to_process = []
    dm._seq_ids_to_process = []
    dm._img_cnt_grid = dm._img_cnt_refit = dm._img_cnt_loading = 0
    dm._img_cnt_affine = dm._img_cnt_thres_live = 0
    dm._diag_pull_cnt = 0                 # incremented in store_new_data (live-target pull cadence)
    return dm


def _seq_block(n_frames, val=1):
    return np.full((4, 4, n_frames), val, dtype=np.int16)


def test_store_new_data_drops_incomplete_sequence():
    dm = _drop_safety_dm(pSeq=2)
    # seq 2 delivers only 1 frame (final image lost) -> must be dropped whole.
    info = {
        'imgs': [_seq_block(2), _seq_block(1), _seq_block(2)],
        'seq_ids': np.array([1, 2, 3], dtype=np.int64),
    }
    dm.store_new_data(info)

    # Buffers stay 1:1 and a multiple of pSeq: 2 complete seqs -> 4 frames.
    assert len(dm._imgs_to_process) == 4
    assert dm._seq_ids_to_process == [1, 3]
    assert len(dm._imgs_to_process) == dm.num_images_per_seq * len(dm._seq_ids_to_process)
    # The dropped sequence is counted and identified.
    assert dm._dropped_seqs == 1
    assert dm._dropped_seq_ids == [2]
    # Cadence counters advance only for COMPLETE sequences actually stored.
    assert dm._img_cnt_grid == 2
    assert dm._img_cnt_refit == 2


def test_store_new_data_all_complete_keeps_everything():
    dm = _drop_safety_dm(pSeq=2)
    info = {
        'imgs': [_seq_block(2), _seq_block(2), _seq_block(2)],
        'seq_ids': np.array([5, 6, 7], dtype=np.int64),
    }
    dm.store_new_data(info)
    assert dm._dropped_seqs == 0
    assert dm._seq_ids_to_process == [5, 6, 7]
    assert len(dm._imgs_to_process) == 6


def test_store_new_data_pseq3_two_round():
    """pSeq=3 (two-round rearrangement): full=3 frames, short block dropped."""
    dm = _drop_safety_dm(pSeq=3)
    info = {
        'imgs': [_seq_block(3), _seq_block(2), _seq_block(3)],
        'seq_ids': np.array([1, 2, 3], dtype=np.int64),
    }
    dm.store_new_data(info)
    assert dm._dropped_seqs == 1 and dm._dropped_seq_ids == [2]
    assert dm._seq_ids_to_process == [1, 3]
    assert len(dm._imgs_to_process) == 6  # 2 complete * pSeq 3


def _save_dm(tmp_path, pSeq=2, num_sites=3, num_sites_img2=3):
    dm = DataManager.__new__(DataManager)
    dm.num_images_per_seq = pSeq
    dm.num_sites = num_sites
    dm.num_sites_img2 = num_sites_img2
    dm.is_two_array = True
    dm._save_two_array = True
    dm.frame_size = (4, 4)
    dm.config = {}
    dm.fname = str(tmp_path / 'data_test.h5')
    dm._file_created = True
    dm._save_lock = threading.Lock()
    create_scan_file(dm.fname, {}, dm.frame_size, num_sites,
                     two_array=True, num_sites_img2=num_sites_img2)
    return dm


def _wait_for_rows(path, key, n, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with h5py.File(path, 'r') as f:
                if key in f and f[key].shape[0] >= n:
                    return
        except (OSError, KeyError):
            pass
        time.sleep(0.05)


def test_save_data_demux_trims_orphan_and_keeps_img1_img2_equal(tmp_path, caplog):
    """An orphan frame in the buffer must be trimmed, not phase-flip the demux."""
    dm = _save_dm(tmp_path, pSeq=2)
    # 5 frames = 2 complete sequences + 1 orphan (the defensive case).
    n = 5
    dm._imgs_to_save = [np.full((4, 4), i, dtype=np.int16) for i in range(n)]
    dm._logicals_to_save = [np.array([i % 2, 0, 1], dtype=bool) for i in range(n)]
    dm._intensities_to_save = [np.full(3, float(i)) for i in range(n)]
    dm._seq_ids_to_save = [10, 20]  # 2 complete sequences

    with caplog.at_level('WARNING'):
        dm.save_data()
    _wait_for_rows(dm.fname, 'logicals_img1', 2)

    with h5py.File(dm.fname, 'r') as f:
        l1 = f['logicals_img1'].shape[0]
        l2 = f['logicals_img2'].shape[0]
        s = f['seq_ids'].shape[0]
        imgs = f['imgs'].shape[0]
    # The invariant the analysis side relies on: equal, in-phase, no orphan.
    assert l1 == l2 == s == 2
    assert imgs == 4  # orphan frame trimmed from imgs too
    assert any('orphan frame' in r.message for r in caplog.records)


def test_save_data_demux_even_buffer_no_warning(tmp_path, caplog):
    dm = _save_dm(tmp_path, pSeq=2)
    n = 4  # exactly 2 complete sequences, no orphan
    dm._imgs_to_save = [np.full((4, 4), i, dtype=np.int16) for i in range(n)]
    dm._logicals_to_save = [np.array([1, 0, 1], dtype=bool) for _ in range(n)]
    dm._intensities_to_save = [np.full(3, float(i)) for i in range(n)]
    dm._seq_ids_to_save = [10, 20]

    with caplog.at_level('WARNING'):
        dm.save_data()
    _wait_for_rows(dm.fname, 'logicals_img1', 2)

    with h5py.File(dm.fname, 'r') as f:
        assert f['logicals_img1'].shape[0] == 2
        assert f['logicals_img2'].shape[0] == 2
        assert f['imgs'].shape[0] == 4
    assert not any('orphan frame' in r.message for r in caplog.records)
