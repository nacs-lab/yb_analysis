"""live_hist_data rebin throttling (config.REBIN_HIST_INTERVAL).

The per-site histogram rebuild scans the whole intensity accumulator, so it is
throttled to every REBIN_HIST_INTERVAL new shots instead of every shot. These
tests pin:
  (a) live_hist_data refreshes on the first shot, then only at the interval;
  (b) at every refresh point its content equals an every-shot rebuild
      (self._compute_hist_data over the current accumulator);
  (c) a consumer that needs current bins (the full Gaussian refit forces a
      _rebin_histograms) always sees fresh content, even mid-interval;
  (d) after an accumulator reset the cadence restarts and panels stay populated.
"""

import numpy as np

from yb_analysis.acquisition.data_manager import DataManager
from yb_analysis.detection.buffers import RingBuffer
from yb_analysis.config import REBIN_HIST_INTERVAL


def _mask(box=9, sigma=2.0):
    m = np.zeros((box, box))
    c = box // 2
    for i in range(box):
        for j in range(box):
            m[i, j] = np.exp(-((i - c) ** 2 + (j - c) ** 2) / (2 * sigma ** 2))
    return m


def _hist_dm(M=5, box=9, H=80, W=90):
    """Bare single-array (pSeq=1) DM with just the attributes process_data's
    accumulate + rebin path touches."""
    dm = DataManager.__new__(DataManager)
    dm.is_init = False
    dm.num_images_per_seq = 1
    dm.is_two_array = False
    dm.num_sites = M
    ys = np.linspace(10, H - 10, M)
    xs = np.linspace(10, W - 10, M)
    dm.grid_locations = np.column_stack([ys, xs]).astype(np.float64)
    # thresholds is a read-only property (live_thresholds or loaded_thresholds);
    # its value is irrelevant to the histogram content this test pins.
    dm.live_thresholds = None
    dm.loaded_thresholds = np.full(M, 1e9)
    # No Gaussian fits -> the per-shot common-mode normalization is a no-op
    # (it needs a per-site mu_empty / atom reference), which is what this
    # histogram-cadence test wants: raw intensities, unchanged logicals.
    dm.live_gauss_fits = None
    dm.loaded_gauss_fits = None
    dm._cm = None
    dm._cm_ref_cache = {}
    dm._cm_logged = set()
    dm.mask_mat = _mask(box)
    dm._accum_pattern_name = None
    dm._logicals_to_save = []
    dm._intensities_to_save = []
    dm._imgs_to_save = []
    dm._intensity_accum = []
    dm._intensity_accum_img2 = []
    dm._hist_last_rebin_n = 0
    dm.live_hist_data = None
    dm._blank_shot_count = 0
    dm._scan_logicals = []
    dm._seq_total = 0
    dm._last_batch_seq_ids = []
    dm._seq_ids_to_process = []
    dm._seq_ids_to_save = []
    dm._imgs_to_process = []
    dm.log_buffer = RingBuffer(4000, (M,), 'float64')
    dm._frame_size = (H, W)
    return dm, (H, W)


def _feed_one(dm, frame_shape, rng, sid):
    img = rng.integers(180, 6000, size=frame_shape, dtype=np.uint16)
    dm._imgs_to_process = [img]
    dm._seq_ids_to_process = [sid]
    dm.process_data()


def _fresh_hist(dm):
    """What an every-shot rebuild would produce right now."""
    return dm._compute_hist_data(np.array(dm._intensity_accum), dm.num_sites)


def _hist_equal(a, b):
    if a is None or b is None:
        return a is b
    if len(a) != len(b):
        return False
    for sa, sb in zip(a, b):
        if not (np.array_equal(sa['counts'], sb['counts'])
                and np.array_equal(sa['bin_centers'], sb['bin_centers'])):
            return False
    return True


def test_rebin_cadence_and_content():
    dm, fshape = _hist_dm()
    rng = np.random.default_rng(1)

    # First shot: must refresh (panels never empty) and match a fresh rebuild.
    _feed_one(dm, fshape, rng, 1)
    assert dm.live_hist_data is not None
    assert _hist_equal(dm.live_hist_data, _fresh_hist(dm))
    assert dm._hist_last_rebin_n == 1

    # Shots 2 .. REBIN_HIST_INTERVAL: no refresh (delta since last rebin < interval).
    for k in range(2, REBIN_HIST_INTERVAL + 1):
        prev = dm.live_hist_data
        _feed_one(dm, fshape, rng, k)
        assert dm.live_hist_data is prev, 'refreshed early at shot %d' % k
        assert dm._hist_last_rebin_n == 1
    # It really is stale now (accum grew but hist did not).
    assert not _hist_equal(dm.live_hist_data, _fresh_hist(dm))

    # One interval past the last rebin (delta == REBIN_HIST_INTERVAL) triggers a
    # refresh, matching a fresh rebuild over the whole accumulator.
    _feed_one(dm, fshape, rng, REBIN_HIST_INTERVAL + 1)
    assert len(dm._intensity_accum) == REBIN_HIST_INTERVAL + 1
    assert dm._hist_last_rebin_n == REBIN_HIST_INTERVAL + 1
    assert _hist_equal(dm.live_hist_data, _fresh_hist(dm))


def test_forced_rebin_gives_fresh_content_midinterval():
    """The full-fit consumer calls _rebin_histograms directly; it must refresh to
    the current accumulator regardless of the throttle counter."""
    dm, fshape = _hist_dm()
    rng = np.random.default_rng(2)
    for k in range(1, REBIN_HIST_INTERVAL + 3):     # a couple shots past a refresh
        _feed_one(dm, fshape, rng, k)
    # Advance a few more without hitting the interval so hist is stale.
    base = dm._hist_last_rebin_n
    for k in range(REBIN_HIST_INTERVAL + 3, base + REBIN_HIST_INTERVAL):
        _feed_one(dm, fshape, rng, k + 100)
        if dm._hist_last_rebin_n != base:
            break
    stale = dm.live_hist_data
    # Force (what update_data does right before _fit_gaussians).
    dm._rebin_histograms()
    assert _hist_equal(dm.live_hist_data, _fresh_hist(dm))
    assert dm._hist_last_rebin_n == len(dm._intensity_accum)
    # And it actually changed something vs the stale snapshot (unless we happened
    # to force at the exact refresh point).
    if len(dm._intensity_accum) != base:
        assert not _hist_equal(stale, dm.live_hist_data)


def test_cadence_restarts_after_accum_reset():
    """Simulate the 2000-shot rotation clearing the accumulator + live_hist_data:
    the next shot must rebin so a panel is never empty."""
    dm, fshape = _hist_dm()
    rng = np.random.default_rng(3)
    for k in range(1, REBIN_HIST_INTERVAL + 1):
        _feed_one(dm, fshape, rng, k)
    assert dm.live_hist_data is not None
    # Rotation (mirrors process_data's UPDATE_HIST_BATCH_SIZE branch).
    dm._intensity_accum.clear()
    dm.live_hist_data = None
    _feed_one(dm, fshape, rng, 9999)
    assert dm.live_hist_data is not None
    assert dm._hist_last_rebin_n == 1
    assert _hist_equal(dm.live_hist_data, _fresh_hist(dm))
