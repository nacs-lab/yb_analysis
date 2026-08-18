"""HDF5 storage for scan data — chunked, appendable, with atomic save.

Replaces MATLAB matfile() incremental write pattern.

Two on-disk layouts, both written by ``create_scan_file``:

COMBINED (legacy, and what ``image_path=None`` still writes bit-identically) --
one ``data_<stamp>.h5`` carrying everything::

    /imgs                 uint16 (nFrames,H,W)  chunks (1,H,W), gzip-1
    /logicals[_img1/_img2/_mid], /intensities[...], /certainties_img2
    /seq_ids              int64  (nSeqs,)
    /scan_config          group of scalar attrs

SPLIT (``image_path=<...>``) -- the bulk frames move to a sibling
``image_<stamp>.h5`` so the analysis path only ever opens the small file::

    data_<stamp>.h5   everything above EXCEPT /imgs (absent, not a stub), plus
                      attrs schema_version=1, images_external=True,
                      images_file='image_<stamp>.h5', num_images_per_seq,
                      frame_size
    image_<stamp>.h5  /imgs           (identical dtype/chunks/compression)
                      /seq_ids        int64 (nSeqs,)  same values+order as data
                      /frame_seq_ids  int64 (nFrames,) = repeat(seq_ids, pSeq)
                      attrs schema_version=1, layout='images',
                      num_images_per_seq, frame_size, data_file, scan_id,
                      committed_frames

``committed_frames`` is a pessimistic watermark written LAST inside the append
handle (h5py publishes a resize before the bytes land), so a live reader counts
shots as ``min(data:/seq_ids rows, committed_frames // pSeq)``. The image file is
created BEFORE the data file, and each block appends images BEFORE data, so a
crash can only leave images with no data rows behind them -- never data rows
promising images that are absent. ``append_images_block`` heals such orphan image
rows on the next block (see its docstring).
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


def _create_image_file(image_path, data_path, frame_size, num_images_per_seq):
    """Create the SPLIT layout's bulk ``image_<stamp>.h5`` (tmp + os.replace).

    Holds ``/imgs`` (identical dtype/chunking/compression to the combined
    layout's -- the tuning is deliberately unchanged by the split), plus the
    per-sequence ``/seq_ids`` and the per-frame ``/frame_seq_ids`` that make a
    frame row self-describing, and the ``committed_frames`` watermark.
    """
    H, W = frame_size
    tmp = image_path + '.tmp'
    with h5py.File(tmp, 'w') as f:
        # Same tuning as the combined layout's /imgs -- see create_scan_file.
        f.create_dataset(
            'imgs', shape=(0, H, W), maxshape=(None, H, W),
            dtype='uint16', chunks=(1, H, W), compression='gzip',
            compression_opts=1,
        )
        f.create_dataset(
            'seq_ids', shape=(0,), maxshape=(None,),
            dtype='int64', chunks=(64,),
        )
        f.create_dataset(
            'frame_seq_ids', shape=(0,), maxshape=(None,),
            dtype='int64', chunks=(256,),
        )
        f.attrs['schema_version'] = 1
        f.attrs['layout'] = 'images'
        if num_images_per_seq is not None:
            f.attrs['num_images_per_seq'] = int(num_images_per_seq)
        f.attrs['frame_size'] = (int(H), int(W))
        f.attrs['data_file'] = os.path.basename(data_path)
        scan_id = _scan_id_of(data_path)
        if scan_id:
            f.attrs['scan_id'] = scan_id
        # Pessimistic watermark: h5py publishes a resize before the data lands,
        # so a live reader must trust this, not /imgs.shape[0].
        f.attrs['committed_frames'] = 0
    os.replace(tmp, image_path)


def _scan_id_of(data_path):
    """``.../data_20260818_101500.h5`` -> ``'20260818_101500'`` ('' if absent)."""
    base = os.path.splitext(os.path.basename(str(data_path)))[0]
    _, sep, stamp = base.partition('_')
    return stamp if sep else ''


def create_scan_file(path, scan_config, frame_size, num_sites,
                     two_array=False, num_sites_img2=0,
                     img2_logicals_source=None, save_mid=False,
                     num_sites_mid=None, image_path=None,
                     num_images_per_seq=None):
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
    save_mid : bool
        If True (NumImages >= 3, two-round rearrangement), also create
        ``logicals_mid`` / ``intensities_mid`` datasets (shape ``(NSeqs,
        num_sites_mid)``) for the MIDDLE (verify) frame -- the post-rearrangement
        occupancy that is detected + displayed live but was previously never
        persisted (only frame-0 -> img1 and the FINAL frame -> img2 were saved).
    num_sites_mid : int or None
        Number of tweezer sites in the MIDDLE frame's grid. None -> ``num_sites``
        (the middle frame detects on the img1 grid, the legacy behaviour). A
        two-round rearrangement whose middle pattern is its own array (e.g. a
        2198-site kagome between a 3013 tri load and a 2078 kagome target) MUST
        pass its own count, else the middle bits are stored at the wrong width.
    image_path : str or None
        None (default) -> the COMBINED layout: one file with ``/imgs`` inside,
        byte-identical to what this function always wrote. When set -> the SPLIT
        layout: ``/imgs`` moves to ``image_path`` (created FIRST, as its own
        atomic tmp + os.replace) and ``path`` gets no ``/imgs`` dataset at all
        (absent, not a stub -- a stub would turn a loud miss into a silent
        "0 frames") plus the ``images_external`` / ``images_file`` attrs.
        Image-file-first ordering means a crash between the two creates leaves an
        invisible orphan image file, never a data file promising absent images.
    num_images_per_seq : int or None
        Frames per sequence (NumImages). Stored as the ``num_images_per_seq``
        attr on both files of a split pair -- it is what lets a reader turn
        frame rows into shots without dividing across the two files. None ->
        the attr is simply omitted (the files are still created).
    """
    if h5py is None:
        raise ImportError("h5py is required for HDF5 storage")

    H, W = frame_size
    tmp = path + '.tmp'

    # SPLIT layout: the bulk file goes down FIRST and completely, so the data
    # file never exists while its images do not.
    if image_path is not None:
        _create_image_file(image_path, path, frame_size, num_images_per_seq)

    with h5py.File(tmp, 'w') as f:
        if image_path is None:
            # Resizable image dataset: (N, H, W), uint16, chunked. uint16 (not
            # int16) so full 16-bit camera counts >= 32768 store without wrapping to
            # negatives; readers cast to float and are dtype-agnostic (old int16
            # files still load unchanged).
            f.create_dataset(
                'imgs', shape=(0, H, W), maxshape=(None, H, W),
                dtype='uint16', chunks=(1, H, W), compression='gzip',
                compression_opts=1,
            )
        else:
            f.attrs['schema_version'] = 1
            f.attrs['images_external'] = True
            f.attrs['images_file'] = os.path.basename(image_path)
            if num_images_per_seq is not None:
                f.attrs['num_images_per_seq'] = int(num_images_per_seq)
            f.attrs['frame_size'] = (int(H), int(W))
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
            # Middle (verify) frame: post-rearrangement occupancy, detected on
            # the MIDDLE pattern's own grid (num_sites_mid; defaults to the img1
            # grid when the scan declared no distinct middle pattern). Only for
            # NumImages >= 3.
            if save_mid:
                n_mid = int(num_sites if num_sites_mid is None else num_sites_mid)
                f.attrs['save_mid'] = True
                lm = f.create_dataset(
                    'logicals_mid', shape=(0, n_mid),
                    maxshape=(None, n_mid), dtype='bool',
                    chunks=(64, max(n_mid, 1)))
                lm.attrs['meaning'] = ('middle (verify) frame occupancy: '
                                       'post-rearrangement, pre-science')
                f.create_dataset(
                    'intensities_mid', shape=(0, n_mid),
                    maxshape=(None, n_mid), dtype='float64',
                    chunks=(64, max(n_mid, 1)))
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


def _append_rows(f, name, block, chunks):
    """Append ``block`` to ``f[name]``, lazily creating the dataset."""
    if name not in f:
        shape = (0,) + block.shape[1:]
        maxshape = (None,) + block.shape[1:]
        f.create_dataset(name, shape=shape, maxshape=maxshape,
                         dtype=block.dtype, chunks=chunks)
    ds = f[name]
    cur = ds.shape[0]
    n_new = block.shape[0]
    ds.resize(cur + n_new, axis=0)
    ds[cur:cur + n_new] = block
    return ds.shape[0]


def append_images_block(image_path, imgs_block, seq_ids_block, num_images,
                        expected_seq_rows=None):
    """Append one block of frames to the SPLIT layout's ``image_<stamp>.h5``.

    Writes ``/imgs`` (the frames), ``/seq_ids`` (one id per sequence, same values
    and order as the data file's) and ``/frame_seq_ids``
    (``repeat(seq_ids_block, num_images)``), then sets the ``committed_frames``
    attr LAST -- all inside ONE ``_open_h5_append`` handle, so the OneDrive
    lock-retry (bug-hdf5-append-lock-onedrive-silent-loss) covers this file too
    and the watermark can never claim frames whose bytes have not landed.

    ORPHAN SELF-HEAL (the reviewed amendment). Callers write images BEFORE the
    data rows, so a failed data append -- or a crash between the two -- leaves
    image rows with no shots behind them. Appending the NEXT block on top of
    those orphans would shift every later shot's positional join
    ``(shot-1)*pSeq + frame``, and the dashboard / avg-image would silently show
    the WRONG shot. So when ``expected_seq_rows`` (the data file's committed
    sequence-row count) is given and ``/imgs`` holds more than
    ``num_images * expected_seq_rows`` rows, the three datasets are resized DOWN
    to alignment first, with a WARNING naming the trimmed count. Rule of thumb:
    prefer a hole in the images over a shift in the images.

    Parameters
    ----------
    image_path : str
    imgs_block : ndarray, shape (N, H, W), uint16
    seq_ids_block : ndarray, shape (K,), int64
        One seq_id per sequence. ``N`` must equal ``num_images * K``; a mismatch
        is logged (the caller trimmed a partial sequence) rather than silently
        written with a wrong per-frame id mapping.
    num_images : int
        Frames per sequence (pSeq).
    expected_seq_rows : int or None
        Sequence rows already durable in the DATA file. None -> no alignment
        check (nothing to compare against).
    """
    if h5py is None:
        raise ImportError("h5py is required for HDF5 storage")

    imgs_block = np.asarray(imgs_block)
    seq_ids_block = np.asarray(seq_ids_block, dtype='int64')
    pSeq = max(1, int(num_images))
    n_frames = int(imgs_block.shape[0])
    n_seqs = int(seq_ids_block.shape[0])
    if n_frames != pSeq * n_seqs:
        logger.warning(
            'append_images_block: %d frame(s) for %d sequence(s) at pSeq=%d '
            '(expected %d) - per-frame ids follow the seq_ids block, so the '
            'frame/seq counts will disagree by %d row(s).',
            n_frames, n_seqs, pSeq, pSeq * n_seqs, n_frames - pSeq * n_seqs)

    with _open_h5_append(image_path) as f:
        # --- orphan self-heal, BEFORE anything is appended ---
        if expected_seq_rows is not None and 'imgs' in f:
            aligned = pSeq * int(expected_seq_rows)
            have = int(f['imgs'].shape[0])
            if have > aligned:
                logger.warning(
                    'append_images_block: trimming %d orphan image row(s) '
                    '(%d present, %d aligned to %d saved sequence(s) at '
                    'pSeq=%d) - a previous data append failed or crashed; '
                    'realigning so positional shot joins cannot shift.',
                    have - aligned, have, aligned, int(expected_seq_rows), pSeq)
                f['imgs'].resize(aligned, axis=0)
                for name, n_keep in (('seq_ids', int(expected_seq_rows)),
                                     ('frame_seq_ids', aligned)):
                    if name in f and f[name].shape[0] > n_keep:
                        f[name].resize(n_keep, axis=0)
                f.attrs['committed_frames'] = int(f['imgs'].shape[0])

        _append_rows(f, 'imgs', imgs_block,
                     chunks=(1,) + tuple(imgs_block.shape[1:]))
        _append_rows(f, 'seq_ids', seq_ids_block, chunks=(64,))
        _append_rows(f, 'frame_seq_ids',
                     np.repeat(seq_ids_block, pSeq), chunks=(256,))
        # LAST: publish the watermark only once every byte above is in.
        f.attrs['committed_frames'] = int(f['imgs'].shape[0])


def append_block(path, imgs_block, logicals_block, intensities_block,
                 seq_ids_block, logicals_img2_block=None,
                 intensities_img2_block=None, proba_img2_block=None,
                 logicals_mid_block=None, intensities_mid_block=None,
                 write_imgs=True):
    """Append a block of data to an existing HDF5 file.

    Parameters
    ----------
    path : str
    imgs_block : ndarray, shape (N, H, W), uint16
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
    logicals_mid_block : ndarray or None
        If non-None, two-array mode (NumImages >= 3): shape (NSeqs, M1), the
        MIDDLE (verify) frame logicals, appended to ``logicals_mid``.
    intensities_mid_block : ndarray or None
        If non-None, two-array mode (NumImages >= 3): shape (NSeqs, M1), the
        middle-frame intensities, appended to ``intensities_mid``.
    write_imgs : bool
        True (default) -> the COMBINED layout: ``imgs_block`` is appended to this
        file's ``/imgs``. False -> the SPLIT layout: skip ``/imgs`` entirely
        (``append_images_block`` already put the frames in the image file);
        everything else is identical.
    """
    if h5py is None:
        raise ImportError("h5py is required for HDF5 storage")

    two_array = logicals_img2_block is not None

    with _open_h5_append(path) as f:
        # Append the imgs block as-is (interleaved frames) -- unless the frames
        # live in the sibling image file (split layout).
        if write_imgs:
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
            if logicals_mid_block is not None:
                pairs.append(('logicals_mid',
                              np.asarray(logicals_mid_block, dtype='bool')))
            if intensities_mid_block is not None:
                pairs.append(('intensities_mid',
                              np.asarray(intensities_mid_block, dtype='float64')))
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

    Notes
    -----
    Unused by the package (the live readers go through
    ``yb_analysis.analysis.load_data``, which is split-aware). Left as-is:
    combined-layout only -- it would KeyError on a split data file's absent
    ``/imgs``, which is the intended loud failure.
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
