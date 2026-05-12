"""HDF5 storage for scan data — chunked, appendable, with atomic save.

Replaces MATLAB matfile() incremental write pattern.
"""

import os
import numpy as np

try:
    import h5py
except ImportError:
    h5py = None


def create_scan_file(path, scan_config, frame_size, num_sites,
                     two_array=False, num_sites_img2=0):
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
                 intensities_img2_block=None):
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
    """
    if h5py is None:
        raise ImportError("h5py is required for HDF5 storage")

    two_array = logicals_img2_block is not None

    with h5py.File(path, 'a') as f:
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
