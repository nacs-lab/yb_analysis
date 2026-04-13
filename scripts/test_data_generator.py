#!/usr/bin/env python3
"""Generate synthetic experiment data that mimics the Yb tweezer camera output.

Produces realistic images with:
- Poisson-distributed background (~200 counts, std ~4)
- Gaussian spots at tweezer sites when atoms are present
- ~50% loading probability per site
- Bimodal intensity distribution (empty vs loaded) matching real thresholds

Can run standalone as a fake ExptServer, feeding images to the Python monitor
via ZMQ, or generate a batch of test data to disk.

Usage:
    # Run as fake server (feeds monitor via ZMQ):
    python -m yb_analysis.scripts.test_data_generator --mode server

    # Generate test data to disk:
    python -m yb_analysis.scripts.test_data_generator --mode disk --n-seq 100

    # Quick sanity check (show one image):
    python -m yb_analysis.scripts.test_data_generator --mode show
"""

import argparse
import os
import time
import logging
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Realistic parameters (from 20260403 data)
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = dict(
    frame_size=(2300, 3000),       # (H, W) — matches real camera
    num_sites=22,
    box_size=9,
    mask_sigma=2.0,
    bg_mean=200.0,                 # background counts
    bg_std=4.0,                    # background noise
    atom_signal_mean=10.0,         # extra counts at site center when atom present
    atom_signal_std=2.0,           # variation in atom signal
    atom_psf_sigma=2.0,            # point-spread function sigma (pixels)
    loading_prob=0.5,              # probability atom is present per site
    num_images_per_seq=2,          # images per sequence (loading + survival)
    survival_prob=0.85,            # if loaded, probability of surviving to image 2
)


def make_grid(num_sites, frame_size, rng=None, margin=None):
    """Generate a square tweezer grid within the frame.

    Sites are arranged in a regular NxM grid centered in the frame,
    matching real tweezer array layout.

    Returns (num_sites, 2) array of [y, x] positions.
    """
    H, W = frame_size
    if margin is None:
        margin = min(200, H // 5, W // 5)
    # Find grid dimensions closest to square
    n_cols = int(np.ceil(np.sqrt(num_sites * (W - 2*margin) / (H - 2*margin))))
    n_rows = int(np.ceil(num_sites / n_cols))
    # Spacing
    spacing_x = (W - 2*margin) / max(n_cols - 1, 1)
    spacing_y = (H - 2*margin) / max(n_rows - 1, 1)
    ys, xs = [], []
    for r in range(n_rows):
        for c in range(n_cols):
            if len(ys) >= num_sites:
                break
            ys.append(margin + r * spacing_y)
            xs.append(margin + c * spacing_x)
    return np.column_stack([ys, xs]).astype(np.float64)


def make_gaussian_mask(box_size, sigma):
    """Create the Gaussian weighting mask used for detection."""
    center = box_size // 2
    yy, xx = np.mgrid[0:box_size, 0:box_size]
    mask = np.exp(-((xx - center)**2 + (yy - center)**2) / (2 * sigma**2))
    mask /= mask.sum()
    return mask


def generate_image(grid, atom_present, params, rng=None):
    """Generate a single synthetic camera image.

    Parameters
    ----------
    grid : ndarray (M, 2)
        Site positions [y, x].
    atom_present : ndarray (M,), bool
        Which sites have atoms.
    params : dict
        Image generation parameters.
    rng : np.random.Generator

    Returns
    -------
    img : ndarray (H, W), int16
    """
    rng = rng or np.random.default_rng()
    H, W = params['frame_size']
    psf_sigma = params['atom_psf_sigma']
    half = params['box_size'] // 2 + 3  # draw slightly wider than box

    # Background: Poisson-like (use normal approx for speed on large images)
    img = rng.normal(params['bg_mean'], params['bg_std'], size=(H, W))

    # Add Gaussian spots at loaded sites
    for i in range(len(grid)):
        if not atom_present[i]:
            continue
        y0, x0 = int(round(grid[i, 0])), int(round(grid[i, 1]))
        signal = max(0, rng.normal(params['atom_signal_mean'], params['atom_signal_std']))

        # Stamp a small Gaussian patch
        y_lo = max(0, y0 - half)
        y_hi = min(H, y0 + half + 1)
        x_lo = max(0, x0 - half)
        x_hi = min(W, x0 + half + 1)

        yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
        patch = signal * np.exp(-((xx - x0)**2 + (yy - y0)**2) / (2 * psf_sigma**2))
        img[y_lo:y_hi, x_lo:x_hi] += patch

    return np.clip(img, 0, 32767).astype(np.int16)


def generate_sequence(grid, params, rng=None):
    """Generate a full sequence (loading image + survival image).

    Returns
    -------
    images : ndarray (H, W, num_images_per_seq), int16
    atom_loaded : ndarray (M,), bool
    atom_survived : ndarray (M,), bool
    """
    rng = rng or np.random.default_rng()
    n_imgs = params['num_images_per_seq']
    M = len(grid)

    # Loading: each site has loading_prob chance
    atom_loaded = rng.random(M) < params['loading_prob']
    img1 = generate_image(grid, atom_loaded, params, rng)

    images = [img1]

    if n_imgs >= 2:
        # Survival: loaded atoms survive with survival_prob
        atom_survived = atom_loaded & (rng.random(M) < params['survival_prob'])
        img2 = generate_image(grid, atom_survived, params, rng)
        images.append(img2)
    else:
        atom_survived = atom_loaded

    # Stack as (H, W, n_imgs)
    stack = np.stack(images, axis=-1)
    return stack, atom_loaded, atom_survived


def compute_expected_threshold(params):
    """Estimate what the detection threshold should be.

    The threshold sits between the empty-site and loaded-site intensity
    distributions, weighted by the Gaussian mask.
    """
    mask = make_gaussian_mask(params['box_size'], params['mask_sigma'])
    # Empty site: just background, weighted sum ≈ bg_mean * sum(mask) = bg_mean
    # (since mask sums to 1 after normalization... but DetectAtom uses unnormalized sum)
    bg_intensity = params['bg_mean'] * mask.sum()
    atom_intensity = bg_intensity + params['atom_signal_mean'] * (
        2 * np.pi * params['atom_psf_sigma']**2 * mask.sum()
        / (params['box_size']**2)
    )
    # Simple midpoint
    threshold = (bg_intensity + atom_intensity) / 2
    return threshold


# ---------------------------------------------------------------------------
# Fake ZMQ Server
# ---------------------------------------------------------------------------

def run_fake_server(url, params, n_seq=None, rate=2.0, overrides=None):
    """Run a fake ExptServer that sends synthetic images via ZMQ.

    Compatible with AnalysisClient.get_imgs() wire format.
    Sends `rate` sequences per second.

    Parameters
    ----------
    url : str
        ZMQ bind URL (e.g. tcp://127.0.0.1:1313).
    params : dict
    n_seq : int or None
        Total sequences to send. None = run forever.
    rate : float
        Sequences per second.
    """
    import zmq

    rng = np.random.default_rng(12345)
    scan_id = int(time.strftime('%Y%m%d%H%M%S'))

    # Build a scan config — copies from latest real scan if available
    grid = make_grid(params['num_sites'], params['frame_size'], rng)
    fname = _write_scan_config(scan_id, grid, params)

    # If we copied a real scan, read its grid, frame size, and intensity calibration
    try:
        from yb_analysis.io.mat_reader import load_scan_config_from_mat
        cfg = load_scan_config_from_mat(fname)
        real_grid_x = np.asarray(cfg.get('initGridLocationsX', []), dtype=np.float64).ravel()
        real_grid_y = np.asarray(cfg.get('initGridLocationsY', []), dtype=np.float64).ravel()
        if len(real_grid_x) > 0:
            grid = np.column_stack([real_grid_y, real_grid_x])
            params = dict(params)
            params['num_sites'] = len(grid)
            fs = np.asarray(cfg.get('frameSize', [0, 0]), dtype=np.float64).ravel()
            if fs[0] > 0:
                params['frame_size'] = (int(fs[1]), int(fs[0]))  # (H, W)
                params['box_size'] = int(np.asarray(cfg.get('boxSize', 9)).flat[0])
                params['mask_sigma'] = float(np.asarray(cfg.get('maskSigma', 2.0)).flat[0])

            # Calibrate fake image pixel-level params so that detect_atom
            # produces masked intensities matching the real Gaussian fits.
            #
            # Key relationships (mask is normalized, sum=1):
            #   bg_mean     → mu_empty   (1:1, since mask sum = 1)
            #   bg_std      → sig_empty  (linear: sig_empty ≈ bg_std * 0.147)
            #   signal_mean → mu_atom    (delta = signal_mean * 0.523 for psf_sigma=2)
            #   signal_std  → sig_atom   (combined with bg noise)
            try:
                from yb_analysis.io.preload import _parse_gauss_fits_struct
                from scipy.io import loadmat
                from datetime import datetime, timedelta
                data_base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(fname))), '')
                thresh_file = None
                for delta_d in [0, 1, 2]:
                    day = (datetime.now() - timedelta(days=delta_d)).strftime('%Y%m%d')
                    candidate = os.path.join(data_base, day, 'threshold.mat')
                    if os.path.isfile(candidate):
                        thresh_file = candidate
                        break
                if thresh_file is not None:
                    td = loadmat(thresh_file, squeeze_me=True)
                    gf = _parse_gauss_fits_struct(td.get('gaussFitsStruct'))
                    if gf:
                        valid = [g['params'] for g in gf if g.get('params') is not None]
                        if valid:
                            avg_mu_e = np.mean([p[0] for p in valid])
                            avg_sig_e = np.mean([p[1] for p in valid])
                            avg_mu_a = np.mean([p[3] for p in valid])
                            avg_sig_a = np.mean([p[4] for p in valid])

                            # Compute mask properties for calibration
                            mask = make_gaussian_mask(params['box_size'], params['mask_sigma'])
                            # Empirical ratios from testing:
                            # sig_empty = bg_std * ||mask||_2 where ||mask||_2 = sqrt(sum(mask^2))
                            mask_l2 = np.sqrt(np.sum(mask**2))
                            # delta_intensity = signal_mean * sum(psf * mask)
                            psf_sigma = params['atom_psf_sigma']
                            center = params['box_size'] // 2
                            yy, xx = np.mgrid[0:params['box_size'], 0:params['box_size']]
                            psf = np.exp(-((xx-center)**2 + (yy-center)**2) / (2*psf_sigma**2))
                            coupling = np.sum(psf * mask)  # how much a unit-signal spot contributes

                            params['bg_mean'] = avg_mu_e
                            params['bg_std'] = avg_sig_e / mask_l2
                            params['atom_signal_mean'] = (avg_mu_a - avg_mu_e) / coupling
                            # sig_atom^2 = sig_empty^2 + (signal_std * coupling)^2
                            sig_atom_from_signal = np.sqrt(max(0, avg_sig_a**2 - avg_sig_e**2))
                            params['atom_signal_std'] = sig_atom_from_signal / coupling

                            logger.info('Calibrated: bg=%.1f±%.1f, signal=%.2f±%.2f '
                                        '(target mu_e=%.1f±%.2f, mu_a=%.1f±%.2f)',
                                        params['bg_mean'], params['bg_std'],
                                        params['atom_signal_mean'], params['atom_signal_std'],
                                        avg_mu_e, avg_sig_e, avg_mu_a, avg_sig_a)
            except Exception as e:
                logger.debug('Could not calibrate from day folder: %s', e)

            logger.info('Using real config: %d sites, frame=%s', params['num_sites'], params['frame_size'])
    except Exception as e:
        logger.warning('Could not read real config, using synthetic: %s', e)

    # CLI overrides take priority over real scan config
    if overrides:
        params = dict(params)
        if 'frame_size' in overrides:
            params['frame_size'] = overrides['frame_size']
        if 'num_sites' in overrides:
            params['num_sites'] = overrides['num_sites']
        grid = make_grid(params['num_sites'], params['frame_size'], rng, margin=100)
        # Rewrite scan config with correct frameSize/grid (synthetic, not copied)
        fname = _write_scan_config(scan_id, grid, params, synthetic=True)
        logger.info('CLI override: %d sites, frame=%s', params['num_sites'], params['frame_size'])

    # Write day folder calibration files so DataManager finds matching loaded fits
    _write_day_folder_calibration(scan_id, grid, params)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.ROUTER)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(url)
    logger.info('Fake server listening on %s (scan_id=%d)', url, scan_id)

    seq_num = 0
    status = 'running'

    def _send_string(addr, msg_str, flag=0):
        """Match ExptServer.safe_send_string: [addr, empty, string]."""
        sock.send(addr, zmq.SNDMORE)
        sock.send(b'', zmq.SNDMORE)
        sock.send_string(msg_str, flag)

    def _send_bytes(addr, data, flag=0):
        """Match ExptServer.safe_send: [addr, empty, bytes]."""
        sock.send(addr, zmq.SNDMORE)
        sock.send(b'', zmq.SNDMORE)
        sock.send(data, flag)

    try:
        while n_seq is None or seq_num < n_seq:
            if sock.poll(int(1000 / max(rate, 0.1))):
                # ROUTER framing: [identity, empty_delimiter, message]
                addr = sock.recv()
                _delim = sock.recv()  # empty delimiter frame
                msg = sock.recv_string()
                logger.debug('Received: %s', msg)

                if msg == 'get_status':
                    _send_string(addr, f'Sequence is {status}')
                elif msg == 'get_imgs':
                    batch_size = max(1, int(rate))
                    data = _encode_batch(grid, params, scan_id,
                                         seq_num, batch_size, rng)
                    _send_bytes(addr, data)
                    seq_num += batch_size
                    logger.info('Sent %d sequences (total: %d)', batch_size, seq_num)
                elif msg == 'get_seq_num':
                    _send_bytes(addr, seq_num.to_bytes(8, 'little'))
                elif msg == 'get_num_imgs':
                    _send_bytes(addr, seq_num.to_bytes(8, 'little'))
                elif msg == 'abort_seq':
                    status = 'stopped'
                    _send_string(addr, 'Sequence Aborted')
                    logger.info('Abort received')
                elif msg == 'pause_seq':
                    status = 'paused'
                    _send_string(addr, 'Sequence Paused')
                elif msg == 'start_seq':
                    status = 'running'
                    _send_string(addr, 'Sequence Started')
                elif msg == 'get_config':
                    sock.send(addr, zmq.SNDMORE)
                    sock.send(b'', zmq.SNDMORE)
                    sock.send_string(time.strftime('%Y%m%d'), zmq.SNDMORE)
                    sock.send_string(time.strftime('%H%M%S'))
                else:
                    _send_string(addr, '')
    except KeyboardInterrupt:
        logger.info('Shutting down fake server')
    finally:
        sock.close()
        ctx.term()


def _encode_batch(grid, params, scan_id, start_seq, batch_size, rng):
    """Encode a batch of sequences in the AnalysisClient.get_imgs() wire format.

    Format: [num_seqs, <per-sequence blocks separated by 0>]
    Each sequence: scan_id, seq_id, s1, s2, s3, <pixel data>, 0

    Uses numpy tobytes() instead of tolist() for ~7x speedup on large images.
    """
    H, W = params['frame_size']
    n_imgs = params['num_images_per_seq']
    chunks = [np.array([float(batch_size)], dtype=np.float64)]

    for i in range(batch_size):
        seq_id = start_seq + i + 1
        stack, _, _ = generate_sequence(grid, params, rng)

        header = np.array([scan_id, seq_id, H, W, n_imgs], dtype=np.float64)
        pixels = stack.astype(np.float64).ravel(order='F')
        trailer = np.array([0.0], dtype=np.float64)
        chunks.extend([header, pixels, trailer])

    return np.concatenate(chunks).tobytes()


def _find_latest_real_scan():
    """Find the most recent real scan .mat file."""
    import glob
    from yb_analysis.config import PATH_PREFIX
    data_dir = os.path.join(PATH_PREFIX, 'Data')
    for dd in sorted(glob.glob(os.path.join(data_dir, '2*')), reverse=True):
        scans = sorted(glob.glob(os.path.join(dd, 'data_*', 'data_*.mat')))
        if scans:
            return scans[-1]
    return None


def _write_scan_config(scan_id, grid, params, synthetic=False):
    """Write scan config by copying from the latest real scan .mat file.

    Falls back to synthetic config if no real scan exists or synthetic=True.
    """
    import shutil
    from yb_analysis.io.scan_directory import scan_id_to_stamps, make_scan_dir, make_scan_fname

    date_stamp, time_stamp = scan_id_to_stamps(scan_id)
    dname, _, _ = make_scan_dir(date_stamp, time_stamp)
    fname, _, _ = make_scan_fname(date_stamp, time_stamp, dname)

    # Try to copy from latest real scan (skip if synthetic forced)
    real_scan = None if synthetic else _find_latest_real_scan()
    if real_scan is not None:
        shutil.copy2(real_scan, fname)
        logger.info('Copied scan config from %s', real_scan)
        return fname

    # Fallback: write synthetic config
    mask = make_gaussian_mask(params['box_size'], params['mask_sigma'])
    threshold = compute_expected_threshold(params)
    thresholds = np.full(params['num_sites'], threshold)
    infidelities = np.full(params['num_sites'], 0.01)

    import h5py
    with h5py.File(fname, 'w') as f:
        g = f.create_group('Scan')
        g.create_dataset('frameSize', data=np.array([[params['frame_size'][1]],
                                                       [params['frame_size'][0]]], dtype=np.float64))
        g.create_dataset('NumImages', data=np.array([[params['num_images_per_seq']]], dtype=np.float64))
        g.create_dataset('boxSize', data=np.array([[params['box_size']]], dtype=np.float64))
        g.create_dataset('maskSigma', data=np.array([[params['mask_sigma']]], dtype=np.float64))
        g.create_dataset('isInit', data=np.array([[0]], dtype=np.float64))
        g.create_dataset('initGridLocationsX', data=grid[:, 1].reshape(1, -1))
        g.create_dataset('initGridLocationsY', data=grid[:, 0].reshape(1, -1))
        g.create_dataset('initThresholds', data=thresholds.reshape(-1, 1))
        g.create_dataset('initInfidelities', data=infidelities.reshape(-1, 1))
        g.create_dataset('NumPerGroup', data=np.array([[100]], dtype=np.float64))
        g.create_dataset('Params', data=np.ones((100, 1), dtype=np.float64))
        g.create_dataset('Rearrangement', data=np.array([[0]], dtype=np.float64))
        g.create_dataset('Repetition', data=np.array([[10]], dtype=np.float64))
        g.create_dataset('PlotScale', data=np.array([[1]], dtype=np.float64))
        g.create_dataset('roi', data=np.array([[0], [0],
                                                [params['frame_size'][1]],
                                                [params['frame_size'][0]]], dtype=np.float64))

    logger.info('Wrote synthetic scan config to %s', fname)
    return fname


# ---------------------------------------------------------------------------
# Disk mode: generate batch to HDF5
# ---------------------------------------------------------------------------

def generate_to_disk(params, n_seq=100, output_dir=None):
    """Generate synthetic data and save to HDF5.

    Creates a complete scan directory with config + images + logicals.
    """
    from yb_analysis.io.hdf5_store import create_scan_file, append_block

    rng = np.random.default_rng(42)
    grid = make_grid(params['num_sites'], params['frame_size'], rng)
    scan_id = int(time.strftime('%Y%m%d%H%M%S'))

    # Write .mat config
    mat_fname = _write_scan_config(scan_id, grid, params)
    h5_fname = os.path.splitext(mat_fname)[0] + '.h5'

    H, W = params['frame_size']
    create_scan_file(h5_fname, {'scan_id': scan_id}, (H, W), params['num_sites'])

    mask = make_gaussian_mask(params['box_size'], params['mask_sigma'])

    for seq in range(n_seq):
        stack, loaded, survived = generate_sequence(grid, params, rng)
        # stack is (H, W, n_imgs)
        n_imgs = stack.shape[2]

        imgs_block = np.transpose(stack, (2, 0, 1))  # (n_imgs, H, W)
        logicals_list = []
        intensities_list = []

        from yb_analysis.detection.detect_atom import detect_atom
        threshold = compute_expected_threshold(params)
        thresholds = np.full(params['num_sites'], threshold)

        for k in range(n_imgs):
            status, intens = detect_atom(
                imgs_block[k].astype(np.float64), grid, thresholds, mask
            )
            logicals_list.append(status)
            intensities_list.append(intens)

        logicals_block = np.array(logicals_list, dtype=bool)
        intensities_block = np.array(intensities_list, dtype=np.float64)
        seq_ids = np.array([seq + 1], dtype=np.int64)

        append_block(h5_fname, imgs_block.astype(np.int16),
                     logicals_block, intensities_block, seq_ids)

        if (seq + 1) % 10 == 0:
            logger.info('Generated %d/%d sequences', seq + 1, n_seq)

    logger.info('Saved %d sequences to %s', n_seq, h5_fname)
    return h5_fname


# ---------------------------------------------------------------------------
# Show mode: display one image
# ---------------------------------------------------------------------------

def show_sample(params):
    """Generate and display a single sample image with site annotations."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    grid = make_grid(params['num_sites'], params['frame_size'], rng)
    atom_present = rng.random(params['num_sites']) < params['loading_prob']

    img = generate_image(grid, atom_present, params, rng)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Full image
    axes[0].imshow(img, cmap='gray', vmin=190, vmax=220)
    axes[0].set_title(f'Full image ({img.shape[1]}x{img.shape[0]})')
    for i, (y, x) in enumerate(grid):
        color = 'lime' if atom_present[i] else 'red'
        axes[0].plot(x, y, 'o', color=color, markersize=4, markeredgewidth=0.5)

    # Zoomed view of first loaded site
    loaded_idx = np.where(atom_present)[0]
    if len(loaded_idx) > 0:
        site = loaded_idx[0]
        y0, x0 = int(grid[site, 0]), int(grid[site, 1])
        r = 30
        crop = img[max(0, y0-r):y0+r, max(0, x0-r):x0+r]
        axes[1].imshow(crop, cmap='hot')
        axes[1].set_title(f'Zoomed: site {site} (atom present)')
    else:
        axes[1].text(0.5, 0.5, 'No atoms loaded', transform=axes[1].transAxes,
                     ha='center', va='center')

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Replay mode: send real images from a .mat file
# ---------------------------------------------------------------------------

def run_replay_server(url, mat_path, rate=2.0):
    """Replay real images from a .mat data file via ZMQ.

    Reads imgs, seq_ids, and Scan config from the .mat file.
    Copies day folder calibration files to today's folder (never writes to source).
    """
    import zmq
    import shutil
    import h5py

    scan_id = int(time.strftime('%Y%m%d%H%M%S'))

    # Read the source .mat file
    with h5py.File(mat_path, 'r') as f:
        nimgs_per_seq = int(f['Scan/NumImages'][0, 0])
        total_frames = f['imgs'].shape[2]
        # HDF5 stores as (W, H, frames) in MATLAB convention
        dim0, dim1 = f['imgs'].shape[0], f['imgs'].shape[1]
        seq_ids = f['seq_ids'][:].ravel().astype(int)
        n_seqs = len(seq_ids)
        logger.info('Replay: %s — %d seqs, dims=(%d,%d), %d imgs/seq',
                     os.path.basename(mat_path), n_seqs, dim0, dim1, nimgs_per_seq)

    # Copy scan config .mat to today's scan dir (never write to source date)
    from yb_analysis.io.scan_directory import scan_id_to_stamps, make_scan_dir, make_scan_fname
    from yb_analysis.config import PATH_PREFIX
    date_stamp, time_stamp = scan_id_to_stamps(scan_id)
    dname, _, _ = make_scan_dir(date_stamp, time_stamp)
    fname, _, _ = make_scan_fname(date_stamp, time_stamp, dname)
    # Symlink instead of copying (source .mat can be 24GB+)
    mat_abs = os.path.abspath(mat_path)
    try:
        os.symlink(mat_abs, fname)
    except OSError:
        # Symlink may fail on Windows without admin; fall back to hardlink or just reference
        try:
            os.link(mat_abs, fname)
        except OSError:
            shutil.copy2(mat_abs, fname)
    logger.info('Linked scan config to %s', fname)

    # Copy day folder calibration files from source date to today
    src_day = os.path.dirname(os.path.dirname(mat_path))
    dst_day = os.path.join(PATH_PREFIX, 'Data', date_stamp)
    for cal_file in ['gridLocations.txt', 'threshold.mat', 'histData.mat']:
        src = os.path.join(src_day, cal_file)
        dst = os.path.join(dst_day, cal_file)
        if os.path.isfile(src) and not os.path.isfile(dst):
            shutil.copy2(src, dst)
            logger.info('Copied %s to today', cal_file)

    # ZMQ server
    ctx = zmq.Context()
    sock = ctx.socket(zmq.ROUTER)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(url)
    logger.info('Replay server on %s (scan_id=%d, %d seqs at %.1f/s)',
                url, scan_id, n_seqs, rate)

    seq_cursor = 0
    status = 'running'

    def _send_string(addr, msg):
        sock.send(addr, zmq.SNDMORE)
        sock.send(b'', zmq.SNDMORE)
        sock.send_string(msg)

    try:
        while seq_cursor < n_seqs:
            if not sock.poll(int(1000 / max(rate, 0.1))):
                continue
            addr = sock.recv()
            _ = sock.recv()
            msg = sock.recv_string()

            if msg == 'get_status':
                _send_string(addr, f'Sequence is {status}')
            elif msg == 'get_config':
                sock.send(addr, zmq.SNDMORE)
                sock.send(b'', zmq.SNDMORE)
                sock.send_string(date_stamp, zmq.SNDMORE)
                sock.send_string(time_stamp)
            elif msg == 'get_imgs':
                batch = max(1, int(rate))
                end = min(seq_cursor + batch, n_seqs)
                # Read images from .mat file on demand
                with h5py.File(mat_path, 'r') as f:
                    chunks = [np.array([float(end - seq_cursor)], dtype=np.float64)]
                    for s in range(seq_cursor, end):
                        frame_start = s * nimgs_per_seq
                        frame_end = frame_start + nimgs_per_seq
                        # HDF5 is already in MATLAB memory layout.
                        # Send with header (H, W) and pixels in F-order (column-major)
                        # to match what the real ExptServer sends.
                        raw = f['imgs'][:, :, frame_start:frame_end]  # (dim0, dim1, pSeq)
                        H_img, W_img = raw.shape[1], raw.shape[0]  # HDF5 is (W, H, f)
                        header = np.array([scan_id, seq_ids[s], H_img, W_img, nimgs_per_seq],
                                          dtype=np.float64)
                        pixels = raw.astype(np.float64).ravel(order='F')
                        chunks.extend([header, pixels, np.array([0.0], dtype=np.float64)])
                    data = np.concatenate(chunks).tobytes()

                sock.send(addr, zmq.SNDMORE)
                sock.send(b'', zmq.SNDMORE)
                sock.send(data)
                seq_cursor = end
                logger.info('Replayed %d sequences (total: %d/%d)',
                            end - seq_cursor + batch, seq_cursor, n_seqs)
            elif msg == 'get_seq_num':
                sock.send(addr, zmq.SNDMORE)
                sock.send(b'', zmq.SNDMORE)
                sock.send(seq_cursor.to_bytes(8, 'little'))
            elif msg == 'abort_seq':
                status = 'stopped'
                _send_string(addr, 'Sequence Aborted')
            else:
                _send_string(addr, '')
    except KeyboardInterrupt:
        logger.info('Replay finished')
    finally:
        sock.close()
        ctx.term()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Yb test data generator')
    parser.add_argument('--mode', choices=['server', 'disk', 'show', 'replay'],
                        default='show', help='Operation mode')
    parser.add_argument('--replay', type=str, metavar='PATH',
                        help='Path to .mat file to replay (shortcut for --mode replay)')
    from yb_analysis.config import TEST_SERVER_PORT
    parser.add_argument('--url', default=f'tcp://127.0.0.1:{TEST_SERVER_PORT}',
                        help=f'ZMQ bind URL for server mode (default: {TEST_SERVER_PORT})')
    parser.add_argument('--n-seq', type=int, default=None,
                        help='Number of sequences (default: unlimited)')
    parser.add_argument('--rate', type=float, default=2.0,
                        help='Sequences per second (server mode)')
    parser.add_argument('--small', action='store_true',
                        help='Use small 200x200 images with 6 sites (faster)')
    parser.add_argument('--frame-size', type=int, nargs=2, metavar=('H', 'W'),
                        help='Image size in pixels, e.g. --frame-size 2300 4000')
    parser.add_argument('--num-sites', type=int,
                        help='Number of tweezer sites, e.g. --num-sites 1024')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    params = dict(DEFAULT_PARAMS)
    if args.small:
        params['frame_size'] = (200, 200)
        params['num_sites'] = 6
    if args.frame_size:
        params['frame_size'] = tuple(args.frame_size)
    if args.num_sites:
        params['num_sites'] = args.num_sites

    # --replay shortcut
    if args.replay:
        args.mode = 'replay'

    if args.mode == 'replay':
        replay_path = args.replay
        if not replay_path:
            parser.error('--replay requires a .mat file path')
        import re
        m = re.search(r':(\d+)$', args.url)
        if m:
            _kill_port(int(m.group(1)))
        run_replay_server(args.url, replay_path, rate=args.rate)
    elif args.mode == 'show':
        show_sample(params)
    elif args.mode == 'disk':
        generate_to_disk(params, n_seq=args.n_seq)
    elif args.mode == 'server':
        # Kill stale process on the ZMQ port
        import re
        m = re.search(r':(\d+)$', args.url)
        if m:
            port = int(m.group(1))
            _kill_port(port)
        overrides = {}
        if args.frame_size:
            overrides['frame_size'] = tuple(args.frame_size)
        if args.num_sites:
            overrides['num_sites'] = args.num_sites
        try:
            run_fake_server(args.url, params, n_seq=args.n_seq, rate=args.rate,
                            overrides=overrides or None)
        finally:
            _cleanup_test_data()


def _write_day_folder_calibration(scan_id, grid, params):
    """Write gridLocations.txt + threshold.mat to the day folder so DataManager
    finds matching loaded fits/thresholds from shot 1."""
    from yb_analysis.io.scan_directory import scan_id_to_stamps
    from yb_analysis.config import PATH_PREFIX
    from scipy.io import savemat

    date_stamp, _ = scan_id_to_stamps(scan_id)
    day_dir = os.path.join(PATH_PREFIX, 'Data', date_stamp)
    os.makedirs(day_dir, exist_ok=True)
    M = len(grid)

    # Grid
    np.savetxt(os.path.join(day_dir, 'gridLocations.txt'),
               grid, header='Y\tX', delimiter='\t', comments='')

    # Synthetic thresholds and gauss fits based on calibrated params
    mask = make_gaussian_mask(params['box_size'], params['mask_sigma'])
    mask_l2 = np.sqrt(np.sum(mask**2))
    psf_sigma = params['atom_psf_sigma']
    center = params['box_size'] // 2
    yy, xx = np.mgrid[0:params['box_size'], 0:params['box_size']]
    psf = np.exp(-((xx-center)**2 + (yy-center)**2) / (2*psf_sigma**2))
    coupling = np.sum(psf * mask)

    mu_e = params['bg_mean']
    sig_e = params['bg_std'] * mask_l2
    mu_a = mu_e + params['atom_signal_mean'] * coupling
    sig_a = np.sqrt(sig_e**2 + (params['atom_signal_std'] * coupling)**2)
    threshold = (mu_e + mu_a) / 2

    thresholds = np.full(M, threshold)
    infidelities = np.full(M, 0.02)
    gs = np.empty(M, dtype=[('params', 'O')])
    for s in range(M):
        gs[s]['params'] = np.array([mu_e, sig_e, 0.5, mu_a, sig_a, 0.5])

    savemat(os.path.join(day_dir, 'threshold.mat'), {
        'thresholds': thresholds,
        'infidelities': infidelities,
        'gaussFitsStruct': gs,
    })

    # histData.mat — synthetic 50-bin histograms from the Gaussian fit params
    from scipy.stats import norm
    hs = np.empty(M, dtype=[('counts', 'O'), ('bin_centers', 'O')])
    edges = np.linspace(mu_e - 5*sig_e, mu_a + 5*sig_a, 51)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for s in range(M):
        density = 0.5 * norm.pdf(centers, mu_e, sig_e) + 0.5 * norm.pdf(centers, mu_a, sig_a)
        hs[s]['counts'] = density
        hs[s]['bin_centers'] = centers
    savemat(os.path.join(day_dir, 'histData.mat'), {'histData': hs})

    logger.info('Wrote day folder calibration: %d sites to %s '
                '(gridLocations.txt, threshold.mat, histData.mat)', M, day_dir)


def _kill_port(port):
    """Kill any process listening on the given TCP port (Windows)."""
    import subprocess
    try:
        out = subprocess.check_output(['netstat', '-ano'], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if f':{port} ' in line and 'LISTENING' in line:
                pid = int(line.strip().split()[-1])
                if pid == os.getpid():
                    continue
                logger.info('Killing stale process on port %d (pid=%d)', port, pid)
                subprocess.call(['taskkill', '/PID', str(pid), '/F'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        logger.debug('Port cleanup failed: %s', e)


def _cleanup_test_data():
    """Delete all test-generated scan directories from today's data folder."""
    import glob
    import shutil
    from datetime import datetime

    today = datetime.now().strftime('%Y%m%d')
    from yb_analysis.config import PATH_PREFIX
    data_dir = os.path.join(PATH_PREFIX, 'Data', today)

    if not os.path.isdir(data_dir):
        return

    removed = 0
    for scan_dir in glob.glob(os.path.join(data_dir, 'data_*')):
        # Only remove directories that contain .h5 files (our output)
        h5_files = glob.glob(os.path.join(scan_dir, '*.h5'))
        if h5_files and os.path.isdir(scan_dir):
            try:
                shutil.rmtree(scan_dir)
                removed += 1
            except Exception:
                pass  # OneDrive or other process may lock files

    if removed:
        logger.info('Cleaned up %d test scan directories from %s', removed, data_dir)

    # Do NOT delete gridLocations.txt, threshold.mat, histData.mat
    # from the day folder — they may contain real calibration data.


if __name__ == '__main__':
    main()
