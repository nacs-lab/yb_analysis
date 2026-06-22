"""HDF5 storage for scan data — chunked, appendable, with atomic save.

Replaces MATLAB matfile() incremental write pattern.
"""

import os
import time

# The live data dir lives under a OneDrive-synced folder. OneDrive (and AV, or a
# concurrent reader) can hold a transient lock on the .h5 file; HDF5's own file
# locking then fails the open with a Windows ERROR_LOCK_VIOLATION
# (GetLastError()=33), which silently dropped a whole block of saved data
# (see problem-memory bug-hdf5-append-lock-onedrive-silent-loss). Disabling
# HDF5's lock is safe here — the writer is already serialized by
# DataManager._save_lock — and removes that failure mode at the source. Set
# before h5py imports the HDF5 C library (HDF5 also re-reads it per open).
os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

import logging

logger = logging.getLogger(__name__)


def _open_h5_append(path, retries=12, base_delay=0.1, max_delay=1.0):
    """Open ``path`` in append mode, retrying transient OS-level file locks.

    Even with HDF5's own locking disabled, an external process (OneDrive sync,
    antivirus, a concurrent reader) can briefly hold the file, so a single open
    can still fail with OSError (Windows ERROR_LOCK_VIOLATION /
    ERROR_SHARING_VIOLATION). Retry with exponential backoff (~8 s worst case)
    so a transient lock no longer silently loses a block of data. Re-raises the
    last error if every attempt fails (the caller records it as save_health).
    """
    if h5py is None:
        raise ImportError("h5py is required for HDF5 storage")
    delay = base_delay
    last = None
    for attempt in range(retries):
        try:
            return h5py.File(path, 'a')
        except OSError as e:
            last = e
            if attempt < retries - 1:
                logger.warning('HDF5 open (append) locked, retry %d/%d in %.2fs: %s',
                               attempt + 1, retries, delay, e)
                time.sleep(delay)
                delay = min(max_delay, delay * 1.7)
    raise last


def create_scan_file(path, scan_config, frame_size, num_sites,
                     two_array=False, num_sites_img2=0,
                     img2_logicals_source=None):
    """Create a new HDF5 scan file with resizable datasets.

    Parameters
    ----------
    path : str
        Output file path (e.g. data_20260403_152030.h5).
    scan_config : dict
        Scan configuration to store as attributes.
    frame_size : tuple of (int, int)
        Image dimensions (H, W).
    num_sites : int
        Number of tweezer sites in image-1's grid.
    two_array : bool
        If True, also create per-image logicals/intensities datasets
        (``logicals_img1`` / ``logicals_img2`` / ``intensities_img1`` /
        ``intensities_img2``, each shaped ``(NSeqs, Mi)``) and set the
        ``two_array=True`` file attribute. The interleaved ``logicals`` /
        ``intensities`` datasets are not created in this mode.
    num_sites_img2 : int
        Number of tweezer sites in image-2's grid (required when
        ``two_array=True``).
    img2_logicals_source : str or None
        Provenance for how ``logicals_img2`` was produced. When set (e.g.
        ``'gmm_shape_model_C'``), ``logicals_img2`` came from a spot-shape
        MODEL rather than an intensity threshold; a ``certainties_img2``
        dataset (per-site posterior P(loaded), same shape as
        ``logicals_img2``) is created alongside, and the tag is stored as the
        ``logicals_img2_source`` file + dataset attribute. None -> threshold
        detection (no certainties dataset), the default.
    """
    if h5py is None:
        raise ImportError("h5py is required for HDF5 storage")

    H, W = frame_size
    tmp = path + '.tmp'

    with h5py.File(tmp, 'w') as f:
        # Resizable image dataset: (N, H, W), int16, chunked
        f.create_dataset(
            'imgs', shape=(0, H, W), maxshape=(None, H, W),
            dtype='int16', chunks=(1, H, W), compression='gzip',
            compression_opts=1,
        )
        if two_array:
            # Per-image datasets: one row per captured sequence, per image.
            f.attrs['two_array'] = True
            f.create_dataset(
                'logicals_img1', shape=(0, num_sites),
                maxshape=(None, num_sites), dtype='bool',
                chunks=(64, num_sites))
            f.create_dataset(
                'logicals_img2', shape=(0, num_sites_img2),
                maxshape=(None, num_sites_img2), dtype='bool',
                chunks=(64, max(num_sites_img2, 1)))
            f.create_dataset(
                'intensities_img1', shape=(0, num_sites),
                maxshape=(None, num_sites), dtype='float64',
                chunks=(64, num_sites))
            f.create_dataset(
                'intensities_img2', shape=(0, num_sites_img2),
                maxshape=(None, num_sites_img2), dtype='float64',
                chunks=(64, max(num_sites_img2, 1)))
            # img2 logicals from a spot-shape MODEL -> record provenance and a
            # per-site posterior "% certainty" dataset alongside the logicals.
            if img2_logicals_source:
                src = str(img2_logicals_source)
                f.attrs['logicals_img2_source'] = src
                f['logicals_img2'].attrs['source'] = src
                cert = f.create_dataset(
                    'certainties_img2', shape=(0, num_sites_img2),
                    maxshape=(None, num_sites_img2), dtype='float32',
                    chunks=(64, max(num_sites_img2, 1)))
                cert.attrs['source'] = src
                cert.attrs['meaning'] = 'per-site P(loaded) posterior for logicals_img2'
        else:
            # Legacy single-array layout: (nFrames, num_sites) interleaved.
            f.create_dataset(
                'logicals', shape=(0, num_sites), maxshape=(None, num_sites),
                dtype='bool', chunks=(64, num_sites),
            )
            f.create_dataset(
                'intensities', shape=(0, num_sites), maxshape=(None, num_sites),
                dtype='float64', chunks=(64, num_sites),
            )
        # Sequence IDs
        f.create_dataset(
            'seq_ids', shape=(0,), maxshape=(None,),
            dtype='int64', chunks=(64,),
        )

        # Store simple scan config fields as attributes (skip complex ones)
        cfg = f.create_group('scan_config')
        for key, val in scan_config.items():
            try:
                if isinstance(val, (int, float, str, bool)):
                    cfg.attrs[key] = val
                elif isinstance(val, np.ndarray) and val.size <= 100:
                    cfg.attrs[key] = val
                # Skip dicts, nested structs, large arrays
            except Exception:
                pass

    os.replace(tmp, path)


def append_block(path, imgs_block, logicals_block, intensities_block,
                 seq_ids_block, logicals_img2_block=None,
                 intensities_img2_block=None, proba_img2_block=None):
    """Append a block of data to an existing HDF5 file.

    Parameters
    ----------
    path : str
    imgs_block : ndarray, shape (N, H, W), int16
    logicals_block : ndarray, bool
        Single-array mode: shape (N, M) interleaved.
        Two-array mode: shape (NSeqs, M1), image-1 logicals.
    intensities_block : ndarray, float64
        Single-array mode: shape (N, M) interleaved.
        Two-array mode: shape (NSeqs, M1), image-1 intensities.
    seq_ids_block : ndarray, shape (K,), int64
        One seq_id per sequence (not per image).
    logicals_img2_block : ndarray or None
        If non-None, two-array mode: shape (NSeqs, M2), image-2 logicals.
    intensities_img2_block : ndarray or None
        If non-None, two-array mode: shape (NSeqs, M2), image-2 intensities.
    proba_img2_block : ndarray or None
        If non-None, two-array mode: shape (NSeqs, M2), the spot-shape model's
        per-site posterior P(loaded) for ``logicals_img2`` (the "% certainty"),
        appended to the ``certainties_img2`` dataset.
    """
    if h5py is None:
        raise ImportError("h5py is required for HDF5 storage")

    two_array = logicals_img2_block is not None

    with _open_h5_append(path) as f:
        # Always append the imgs block as-is (interleaved frames).
        if 'imgs' not in f:
            shape = (0,) + imgs_block.shape[1:]
            maxshape = (None,) + imgs_block.shape[1:]
            chunks = (1,) + imgs_block.shape[1:]
            f.create_dataset('imgs', shape=shape, maxshape=maxshape,
                             dtype=imgs_block.dtype, chunks=chunks)
        ds = f['imgs']
        cur = ds.shape[0]
        n_new = imgs_block.shape[0]
        ds.resize(cur + n_new, axis=0)
        ds[cur:cur + n_new] = imgs_block

        if two_array:
            pairs = [
                ('logicals_img1', logicals_block),
                ('logicals_img2', logicals_img2_block),
                ('intensities_img1', intensities_block),
                ('intensities_img2', intensities_img2_block),
            ]
            if proba_img2_block is not None:
                pairs.append(('certainties_img2',
                              np.asarray(proba_img2_block, dtype='float32')))
        else:
            pairs = [
                ('logicals', logicals_block),
                ('intensities', intensities_block),
            ]

        for ds_name, block in pairs:
            if ds_name not in f:
                shape = (0,) + block.shape[1:]
                maxshape = (None,) + block.shape[1:]
                chunks = (64,) + tuple(max(s, 1) for s in block.shape[1:])
                f.create_dataset(ds_name, shape=shape, maxshape=maxshape,
                                 dtype=block.dtype, chunks=chunks)
            ds = f[ds_name]
            cur = ds.shape[0]
            n_new = block.shape[0]
            ds.resize(cur + n_new, axis=0)
            ds[cur:cur + n_new] = block

        if 'seq_ids' not in f:
            f.create_dataset('seq_ids', shape=(0,), maxshape=(None,),
                             dtype='int64', chunks=(64,))
        ds = f['seq_ids']
        cur = ds.shape[0]
        n_new = seq_ids_block.shape[0]
        ds.resize(cur + n_new, axis=0)
        ds[cur:cur + n_new] = seq_ids_block


def read_scan_file(path):
    """Read an HDF5 scan file.

    Returns
    -------
    dict with keys: 'imgs', 'logicals', 'intensities', 'seq_ids', 'scan_config'
    """
    if h5py is None:
        raise ImportError("h5py is required for HDF5 storage")

    data = {}
    with h5py.File(path, 'r') as f:
        data['imgs'] = f['imgs'][:]
        data['logicals'] = f['logicals'][:]
        if 'intensities' in f:
            data['intensities'] = f['intensities'][:]
        data['seq_ids'] = f['seq_ids'][:]
        if 'scan_config' in f:
            data['scan_config'] = dict(f['scan_config'].attrs)
        else:
            data['scan_config'] = {}
    return data
