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

ROTATION (huge scans) -- lazy virtual-dataset segments
------------------------------------------------------
A multi-day scan can drive the bulk file past any sane single-file size, so when
``image_<stamp>.h5`` grows beyond ``config.IMAGE_FILE_ROTATE_GB`` (env
``YB_IMAGE_FILE_ROTATE_GB``, default 10 GB) the next ``append_images_block``
ROTATES -- between blocks, never mid-handle::

    image_<stamp>.000.h5   frozen segment (the old bulk file, renamed)
    image_<stamp>.001.h5   LIVE segment: /imgs + LOCAL /seq_ids + /frame_seq_ids,
                           attrs segment_index, frame_offset, committed_frames
                           (LOCAL row count of this segment)
    image_<stamp>.h5       small MASTER, same path every reader already opens:
                           /imgs           VIRTUAL dataset stitching the segments
                                           contiguously
                           /seq_ids        REAL, GLOBAL
                           /frame_seq_ids  REAL, GLOBAL
                           attrs layout='images-master', segments=<count>,
                           committed_frames (GLOBAL watermark, written LAST)

Below the threshold -- i.e. for essentially every scan -- nothing above happens
and the layout is bit-identical to the plain SPLIT layout described above.

MASTER ``/imgs`` SHAPE SEMANTICS (why the watermark still rules). The frozen
segments are mapped with their exact extents; the live segment is mapped with
``h5py.h5s.UNLIMITED``, so HDF5's default virtual view (LAST_AVAILABLE)
recomputes the virtual extent from the live segment's CURRENT dataset extent on
every space query -- the shape therefore tracks the live segment's growth (and
shrinkage, e.g. an orphan trim), even on a handle that was already open.
Consequences readers must know:

* rows the writer has resized into but not yet filled show up in ``shape`` and
  read as ZERO fill -- exactly the same torn-read window the single-file layout
  has always had, which is why ``committed_frames`` (GLOBAL on the master), not
  ``shape``, is the live watermark. Reads inside ``[0, committed_frames)`` are
  always correct and contiguous across the seams;
* a missing segment file never raises and never shifts rows: a missing FROZEN
  segment reads as zero fill at exactly its own row range (its extent is baked
  into the mapping), and a missing LIVE segment simply truncates the tail;
* the master's ``/imgs`` is virtual, so it reports ``chunks is None`` and
  ``compression is None`` -- the gzip-1 / 1-frame chunking lives in the segments.
  Nothing in the package inspects those on ``/imgs`` (checked), but a hand-written
  cell that does should look at a segment.

Segment paths are stored RELATIVE in the virtual mapping (HDF5 resolves them
against the master's own directory), so a scan folder stays valid when the
OneDrive tree is moved or copied.
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


def _create_image_file(image_path, data_path, frame_size, num_images_per_seq,
                       segment_index=None, frame_offset=None):
    """Create the SPLIT layout's bulk ``image_<stamp>.h5`` (tmp + os.replace).

    Holds ``/imgs`` (identical dtype/chunking/compression to the combined
    layout's -- the tuning is deliberately unchanged by the split), plus the
    per-sequence ``/seq_ids`` and the per-frame ``/frame_seq_ids`` that make a
    frame row self-describing, and the ``committed_frames`` watermark.

    ``segment_index`` / ``frame_offset`` are set only when this file is a
    ROTATION segment (see the module docstring): the ids and watermark inside a
    segment are LOCAL to it, and ``frame_offset`` is the global row index of its
    first frame. Both None (the default) -> the plain, unrotated bulk file,
    created exactly as it always was.
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
        if segment_index is not None:
            f.attrs['segment_index'] = int(segment_index)
            f.attrs['frame_offset'] = int(frame_offset or 0)
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


# --------------------------------------------------------------------------
# ROTATION: image_<stamp>.h5 -> frozen segments + a small VDS master.
# See the module docstring for the on-disk shape and the reader contract.
# --------------------------------------------------------------------------

MASTER_LAYOUT = 'images-master'     # image_<stamp>.h5 attrs['layout'] once rotated
SEGMENT_LAYOUT = 'images'           # a segment carries the plain images layout tag

# Fallback when config cannot be read (import problem, exotic value): the
# documented default of config.IMAGE_FILE_ROTATE_GB.
_ROTATE_GB_FALLBACK = 10.0

# The rename / master-publish retries are SHORT on purpose: a concurrent reader
# holding the file open on Windows blocks the rename, and postponing rotation to
# the next block (the file overshoots the threshold a little) beats stalling the
# acquisition save thread.
_ROTATE_RETRIES = 5
_ROTATE_BASE_DELAY = 0.05
_ROTATE_MAX_DELAY = 0.4


def _rotate_limit_bytes(rotate_bytes=None):
    """Resolve the rotation threshold in BYTES (None -> rotation disabled).

    ``rotate_bytes`` is the explicit override (tests pass a tiny value); None
    reads ``config.IMAGE_FILE_ROTATE_GB`` LAZILY -- imported inside the function
    so this module keeps its "no package imports" property at import time
    (``yb_analysis.config`` itself imports only os + tempfile, so there is no
    cycle either way). A threshold <= 0 disables rotation entirely.
    """
    if rotate_bytes is not None:
        limit = float(rotate_bytes)
        return limit if limit > 0 else None
    try:
        from yb_analysis import config as _cfg
        gb = float(getattr(_cfg, 'IMAGE_FILE_ROTATE_GB', _ROTATE_GB_FALLBACK))
    except Exception:
        gb = _ROTATE_GB_FALLBACK
    return gb * (1024.0 ** 3) if gb > 0 else None


def _over_rotate_limit(path, rotate_bytes=None):
    """True when ``path`` is bigger than the rotation threshold.

    ``os.path.getsize`` (one stat, no open) is deliberately "close enough": the
    file has HDF5 overhead and an unflushed tail, so the crossing point is
    approximate -- the only thing that matters is that it triggers once per
    threshold's worth of frames. A missing/unreadable file never rotates.
    """
    limit = _rotate_limit_bytes(rotate_bytes)
    if limit is None:
        return False
    try:
        return os.path.getsize(path) > limit
    except OSError:
        return False


def segment_path(image_path, index):
    """``image_X.h5``, 1 -> ``image_X.001.h5`` (the rotation segment naming)."""
    root, ext = os.path.splitext(str(image_path))
    return '%s.%03d%s' % (root, int(index), ext or '.h5')


def existing_segments(image_path, limit=100000):
    """Segment paths of a rotated image file, in order; ``[]`` when unrotated.

    Walks ``.000``, ``.001``, ... until one is missing, so an unrotated scan
    costs exactly ONE ``os.path.exists`` per block. Filesystem-first, like the
    resolver: the presence of ``.000`` -- not an attribute -- is what says
    "this image_<stamp>.h5 is a master".
    """
    segs = []
    for i in range(limit):
        p = segment_path(image_path, i)
        if not os.path.exists(p):
            break
        segs.append(p)
    return segs


def _replace_with_retry(src, dst, what='rename'):
    """``os.replace`` with a short retry loop; False when it never succeeded.

    On Windows a reader with the file open can block the replace
    (ERROR_ACCESS_DENIED / ERROR_SHARING_VIOLATION). The caller postpones
    rotation rather than failing the block -- the file overshoots the threshold
    by one block, which is harmless.
    """
    delay = _ROTATE_BASE_DELAY
    for attempt in range(_ROTATE_RETRIES):
        try:
            os.replace(src, dst)
            return True
        except OSError as e:
            if attempt == _ROTATE_RETRIES - 1:
                logger.warning('image rotation: %s %s -> %s failed after %d '
                               'attempt(s): %s', what, os.path.basename(src),
                               os.path.basename(dst), _ROTATE_RETRIES, e)
                return False
            time.sleep(delay)
            delay = min(_ROTATE_MAX_DELAY, delay * 1.7)
    return False


def _segment_rows(path):
    """Durable row count of a segment: its LOCAL ``committed_frames``."""
    try:
        with h5py.File(path, 'r') as f:
            n = int(f['imgs'].shape[0]) if 'imgs' in f else 0
            cf = f.attrs.get('committed_frames')
            return min(n, int(cf)) if cf is not None else n
    except (OSError, KeyError):
        return 0


def _segment_local_ids(path):
    """``(seq_ids, frame_seq_ids)`` of one segment (empty arrays on failure)."""
    empty = np.zeros((0,), dtype='int64')
    try:
        with h5py.File(path, 'r') as f:
            sid = f['seq_ids'][:] if 'seq_ids' in f else empty
            fsid = f['frame_seq_ids'][:] if 'frame_seq_ids' in f else empty
            return (np.asarray(sid, dtype='int64'),
                    np.asarray(fsid, dtype='int64'))
    except (OSError, KeyError):
        return empty, empty


def _global_ids(read_paths):
    """Concatenate the segments' LOCAL ids into the master's GLOBAL arrays.

    The segments are the source of truth (each holds only its own rows, in
    order), so the master can always be rebuilt from them -- which is what makes
    a crash mid-rotation recoverable.
    """
    sids, fsids = [], []
    for p in read_paths:
        a, b = _segment_local_ids(p)
        sids.append(a)
        fsids.append(b)
    cat = (np.concatenate(sids) if sids else np.zeros((0,), dtype='int64'),
           np.concatenate(fsids) if fsids else np.zeros((0,), dtype='int64'))
    return cat


def _freeze_segment(path, pSeq, index, frame_offset):
    """Stamp a segment's identity and trim it to its durable watermark.

    Called on the OUTGOING live segment at every rotation. Rows past
    ``committed_frames`` are an interrupted append's uncommitted tail (never
    referenced by any data row, and their per-frame ids were never written), so
    dropping them before the extent is frozen into the master's virtual mapping
    keeps the global row arithmetic exactly aligned with the watermark.
    Returns the frozen row count.
    """
    with _open_h5_append(path) as f:
        n = int(f['imgs'].shape[0]) if 'imgs' in f else 0
        cf = f.attrs.get('committed_frames')
        cf = n if cf is None else int(cf)
        if cf < n:
            logger.warning('image rotation: dropping %d uncommitted row(s) '
                           'from segment %s before freezing it (%d present, '
                           'watermark %d).', n - cf, os.path.basename(path),
                           n, cf)
            f['imgs'].resize(cf, axis=0)
            for name, keep in (('seq_ids', cf // pSeq), ('frame_seq_ids', cf)):
                if name in f and f[name].shape[0] > keep:
                    f[name].resize(keep, axis=0)
            n = cf
        f.attrs['layout'] = SEGMENT_LAYOUT
        f.attrs['segment_index'] = int(index)
        f.attrs['frame_offset'] = int(frame_offset)
        f.attrs['committed_frames'] = int(n)
        return n


def _write_master(master_path, mapping, live_rel, live_offset, dtype,
                  frame_size, seq_ids, frame_seq_ids, attrs):
    """Write the VDS master to ``master_path + '.tmp'``; return the tmp path.

    ``mapping`` is ``[(relative_segment_name, rows), ...]`` for the FROZEN
    segments (exact extents); ``live_rel`` is mapped from ``live_offset`` with
    ``h5py.h5s.UNLIMITED`` so the virtual extent follows the live segment's
    growth (module docstring: MASTER /imgs SHAPE SEMANTICS). The declared
    layout shape only has to reach one row INTO the unlimited mapping --
    HDF5 reports the extent it can actually resolve, so a live segment with
    zero rows yields exactly ``live_offset`` rows.

    Relative source names keep the scan folder portable (HDF5 resolves them
    against the master's directory, not the process CWD -- verified).
    """
    H, W = frame_size
    layout = h5py.VirtualLayout(shape=(int(live_offset) + 1, H, W),
                                maxshape=(None, H, W), dtype=dtype)
    off = 0
    for rel, rows in mapping:
        if rows <= 0:
            continue
        layout[off:off + rows] = h5py.VirtualSource(rel, 'imgs',
                                                   shape=(rows, H, W))
        off += rows
    if off != int(live_offset):
        raise ValueError('rotation mapping covers %d rows, expected %d'
                         % (off, live_offset))
    live_src = h5py.VirtualSource(live_rel, 'imgs', shape=(1, H, W),
                                  maxshape=(None, H, W))[0:h5py.h5s.UNLIMITED]
    layout[int(live_offset):h5py.h5s.UNLIMITED] = live_src

    tmp = master_path + '.tmp'
    with h5py.File(tmp, 'w') as f:
        # fillvalue 0: rows the writer resized into but has not filled yet read
        # as zeros, never as stale bytes.
        f.create_virtual_dataset('imgs', layout, fillvalue=0)
        f.create_dataset('seq_ids', data=np.asarray(seq_ids, dtype='int64'),
                         maxshape=(None,), chunks=(64,), dtype='int64')
        f.create_dataset('frame_seq_ids',
                         data=np.asarray(frame_seq_ids, dtype='int64'),
                         maxshape=(None,), chunks=(256,), dtype='int64')
        for k, v in attrs.items():
            if v is not None:
                f.attrs[k] = v
    return tmp


def _master_attrs(src_attrs, n_segments, live_rel, frame_size,
                  committed_frames):
    """Attrs for the master, carrying the bulk file's identity forward."""
    out = {
        'schema_version': 1,
        'layout': MASTER_LAYOUT,
        'segments': int(n_segments),
        'live_segment': live_rel,
        'frame_size': (int(frame_size[0]), int(frame_size[1])),
        'committed_frames': int(committed_frames),
    }
    for k in ('num_images_per_seq', 'data_file', 'scan_id'):
        if k in src_attrs:
            v = src_attrs[k]
            out[k] = int(v) if k == 'num_images_per_seq' else str(v)
    return out


def _image_header(path, pSeq):
    """``(attrs_dict, frame_size, dtype)`` of an image file / segment.

    ``frame_size`` comes from the DATASET, not the attr: it is what the virtual
    mapping has to match exactly.
    """
    with h5py.File(path, 'r') as f:
        attrs = dict(f.attrs)
        ds = f['imgs']
        shape = tuple(int(s) for s in ds.shape)
        dtype = ds.dtype
    if len(shape) != 3:
        raise ValueError('%s: /imgs is not (N,H,W) but %r' % (path, shape))
    attrs.setdefault('num_images_per_seq', int(pSeq))
    return attrs, (shape[1], shape[2]), dtype


def _build_master_tmp(image_path, segments_after, live_path, extents, pSeq,
                      header_path, read_paths):
    """Build the master into ``image_path + '.tmp'``; return that path.

    ``segments_after`` are the FROZEN segment paths (in order) with row counts
    ``extents``; ``live_path`` is the live segment. ``read_paths`` are the files
    to concatenate the global ids from -- the frozen segments as they are named
    RIGHT NOW, because at the first rotation the outgoing bulk file has not been
    renamed yet (building before the rename keeps the window in which
    ``image_<stamp>.h5`` does not exist down to two syscalls).
    """
    attrs, frame_size, dtype = _image_header(header_path, pSeq)
    total = int(sum(extents))
    sids, fsids = _global_ids(read_paths)
    mapping = [(os.path.basename(p), int(n))
               for p, n in zip(segments_after, extents)]
    live_rel = os.path.basename(live_path)
    return _write_master(
        image_path, mapping, live_rel, total, dtype, frame_size, sids, fsids,
        _master_attrs(attrs, len(segments_after) + 1, live_rel, frame_size,
                      total))


def _discard(path):
    try:
        os.remove(path)
    except OSError:
        pass


def _rotate_image_file(image_path, pSeq):
    """Rotate the bulk image file. Returns the segment list (``[]`` = postponed).

    Between-blocks only, and in an order that keeps every failure recoverable:

    1. freeze the outgoing live segment (trim to its watermark, stamp its
       ``segment_index`` / ``frame_offset``);
    2. create the NEW live segment (empty, same schema);
    3. write the master to a ``.tmp`` -- the mapping already names the frozen
       segment by its POST-rename name, so nothing has to exist yet;
    4. (first rotation only) rename ``image_<stamp>.h5`` -> ``...000.h5``;
    5. publish the master with ``os.replace``.

    A blocked rename (step 4) or a blocked publish (step 5) postpones rotation:
    the block still lands in the oversized file and the next block retries. A
    crash between 4 and 5 leaves segments with no master, which the next
    ``append_images_block`` detects and rebuilds from the segments themselves.
    Steps 4+5 are back-to-back so the window in which ``image_<stamp>.h5`` is
    absent (a reader would get FileNotFoundError, or resolve the ``.000`` prefix)
    is two syscalls long.
    """
    segments = existing_segments(image_path)
    first = not segments
    live_now = image_path if first else segments[-1]
    if not os.path.exists(live_now):
        return segments

    # Names the segments will have AFTER this rotation.
    frozen_after = [segment_path(image_path, 0)] if first else list(segments)
    read_paths = [live_now] if first else segments[:-1] + [live_now]

    extents = [_segment_rows(p) for p in frozen_after[:-1]]
    offset_live = int(sum(extents))
    try:
        extents.append(_freeze_segment(live_now, pSeq, len(frozen_after) - 1,
                                       offset_live))
    except (OSError, KeyError) as e:
        logger.warning('image rotation postponed: cannot freeze %s (%s)',
                       os.path.basename(live_now), e)
        return segments

    total = int(sum(extents))
    new_index = len(frozen_after)
    new_path = segment_path(image_path, new_index)
    try:
        attrs, frame_size, _dtype = _image_header(live_now, pSeq)
        data_file = attrs.get('data_file')
        data_hint = os.path.join(os.path.dirname(image_path),
                                 str(data_file) if data_file else
                                 'data_%s.h5' % _scan_id_of(image_path))
        _create_image_file(new_path, data_hint, frame_size,
                           attrs.get('num_images_per_seq'),
                           segment_index=new_index, frame_offset=total)
    except (OSError, KeyError, ValueError) as e:
        logger.warning('image rotation postponed: cannot create segment %s (%s)',
                       os.path.basename(new_path), e)
        return segments

    try:
        tmp = _build_master_tmp(image_path, frozen_after, new_path, extents,
                                pSeq, live_now, read_paths)
    except Exception as e:                                  # noqa: BLE001
        logger.error('image rotation postponed: master build FAILED (%s: %s)',
                     type(e).__name__, e)
        return segments

    if first and not _replace_with_retry(image_path, frozen_after[0],
                                         what='freeze'):
        # The oversized file keeps taking this block; retried next block.
        logger.warning('image rotation postponed: %s is busy (a reader holds '
                       'it); the file overshoots the threshold until the next '
                       'block.', os.path.basename(image_path))
        _discard(tmp)
        _discard(new_path)          # the unused empty segment
        return []

    if not _replace_with_retry(tmp, image_path, what='publish master'):
        _discard(tmp)
        if first:
            # Roll the rename back so the plain layout is intact again.
            if _replace_with_retry(frozen_after[0], image_path,
                                   what='rollback'):
                logger.warning('image rotation postponed: master could not be '
                               'published; rolled back to the single file.')
                _discard(new_path)
                return []
            logger.error('image rotation: master not published AND rollback '
                         'failed; rebuilding the master from the segments.')
            segs = existing_segments(image_path)
            _rebuild_master_if_missing(image_path, segs, pSeq)
            return segs
        # A later rotation whose master could not be published: the OLD master
        # still maps the OLD live segment, so keep writing there (the file
        # overshoots) instead of into a segment nothing points at.
        logger.warning('image rotation postponed: master %s could not be '
                       'republished; staying on segment %s.',
                       os.path.basename(image_path),
                       os.path.basename(segments[-1]))
        _discard(new_path)
        return segments

    logger.info('image file rotated: %s frozen at %d frame(s); live segment is '
                'now %s (master %s stitches %d segment(s)).',
                os.path.basename(frozen_after[-1]), extents[-1],
                os.path.basename(new_path), os.path.basename(image_path),
                len(frozen_after) + 1)
    return existing_segments(image_path)


def _rebuild_master_if_missing(image_path, segments, pSeq):
    """Recreate the master after a crash between the rename and the publish.

    The segments hold every row and every local id, so the master is fully
    derivable from them: the LAST segment is the live one (its extent is
    unlimited in the mapping), the earlier ones are frozen at their watermark.
    """
    if os.path.exists(image_path) or not segments:
        return
    logger.warning('image rotation: master %s is missing; rebuilding it from '
                   '%d segment(s).', os.path.basename(image_path),
                   len(segments))
    frozen = segments[:-1]
    extents = [_segment_rows(p) for p in frozen]
    try:
        tmp = _build_master_tmp(image_path, frozen, segments[-1], extents,
                                pSeq, segments[-1], segments)
    except Exception as e:                                  # noqa: BLE001
        logger.error('image rotation: master rebuild FAILED (%s: %s)',
                     type(e).__name__, e)
        return
    if not _replace_with_retry(tmp, image_path, what='publish master'):
        _discard(tmp)


def _heal_orphans_plain(f, pSeq, expected_seq_rows):
    """Trim orphan image rows in an UNROTATED bulk file (open handle ``f``)."""
    if expected_seq_rows is None or 'imgs' not in f:
        return
    aligned = pSeq * int(expected_seq_rows)
    have = int(f['imgs'].shape[0])
    if have <= aligned:
        return
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


def _heal_orphans_rotated(image_path, segments, pSeq, expected_seq_rows):
    """Orphan self-heal for a rotated layout: LIVE segment + master tail.

    Orphans never span segments -- a rotation happens between blocks and only
    AFTER this trim -- so the rows to drop are always inside the live segment.
    A demand to trim further than the live segment holds would mean cutting a
    FROZEN extent out of the master's mapping, which we refuse (and shout
    about): a hole in the images beats a shift in the images.
    """
    if expected_seq_rows is None:
        return
    aligned = pSeq * int(expected_seq_rows)
    live = segments[-1]
    try:
        with _open_h5_append(live) as f:
            if 'imgs' not in f:
                return
            offset = int(f.attrs.get('frame_offset', 0))
            have_local = int(f['imgs'].shape[0])
            if offset + have_local <= aligned:
                return
            keep_local = aligned - offset
            if keep_local < 0:
                logger.error(
                    'append_images_block: %d image row(s) beyond the %d saved '
                    'sequence(s) live in FROZEN segment(s); refusing to rewrite '
                    'a frozen extent. Shot joins past frame %d may be shifted.',
                    -keep_local, int(expected_seq_rows), aligned)
                keep_local = 0
            logger.warning(
                'append_images_block: trimming %d orphan image row(s) from the '
                'live segment %s (%d present locally, keeping %d to align with '
                '%d saved sequence(s) at pSeq=%d).',
                have_local - keep_local, os.path.basename(live), have_local,
                keep_local, int(expected_seq_rows), pSeq)
            f['imgs'].resize(keep_local, axis=0)
            for name, n_keep in (('seq_ids', max(0, keep_local // pSeq)),
                                 ('frame_seq_ids', keep_local)):
                if name in f and f[name].shape[0] > n_keep:
                    f[name].resize(n_keep, axis=0)
            f.attrs['committed_frames'] = int(f['imgs'].shape[0])
            new_global = offset + int(f['imgs'].shape[0])
    except (OSError, KeyError) as e:
        logger.error('append_images_block: orphan trim on live segment %s '
                     'failed (%s); leaving it to the next block.',
                     os.path.basename(live), e)
        return

    with _open_h5_append(image_path) as f:
        for name, n_keep in (('seq_ids', max(0, new_global // pSeq)),
                             ('frame_seq_ids', new_global)):
            if name in f and f[name].shape[0] > n_keep:
                f[name].resize(n_keep, axis=0)
        f.attrs['committed_frames'] = int(new_global)


def _append_image_rows(f, imgs_block, seq_ids_block, pSeq):
    """Frames + ids + watermark into one open image file / segment handle."""
    _append_rows(f, 'imgs', imgs_block,
                 chunks=(1,) + tuple(imgs_block.shape[1:]))
    _append_rows(f, 'seq_ids', seq_ids_block, chunks=(64,))
    _append_rows(f, 'frame_seq_ids',
                 np.repeat(seq_ids_block, pSeq), chunks=(256,))
    # LAST: publish the watermark only once every byte above is in.
    n = int(f['imgs'].shape[0])
    f.attrs['committed_frames'] = n
    return n


def append_images_block(image_path, imgs_block, seq_ids_block, num_images,
                        expected_seq_rows=None, rotate_bytes=None):
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
    rotate_bytes : int or None
        Rotation threshold override in BYTES (tests). None -> the configured
        ``config.IMAGE_FILE_ROTATE_GB``; <= 0 disables rotation.

    ROTATION. When the live bulk file is already bigger than the threshold, this
    block rotates it first (see ``_rotate_image_file`` and the module docstring):
    the frames + LOCAL ids then go to the live SEGMENT, and the GLOBAL ids plus
    the GLOBAL ``committed_frames`` watermark go to the small VDS master -- the
    master last, so the watermark still trails every durable byte. Below the
    threshold (the normal case) the code path is exactly the one-handle append it
    has always been.
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

    # One stat (".000" present?) says whether this file has ever rotated.
    segments = existing_segments(image_path)

    if not segments:
        if not _over_rotate_limit(image_path, rotate_bytes):
            # --- the ordinary path: self-heal + append in ONE handle ---
            with _open_h5_append(image_path) as f:
                _heal_orphans_plain(f, pSeq, expected_seq_rows)
                _append_image_rows(f, imgs_block, seq_ids_block, pSeq)
            return
        # Over the threshold: heal FIRST (so orphans can never end up straddling
        # a segment boundary), then rotate between blocks. The size check runs
        # before the trim only because it needs no handle -- a trim frees no disk
        # space in an HDF5 file, so it cannot change the verdict.
        if expected_seq_rows is not None:
            with _open_h5_append(image_path) as f:
                _heal_orphans_plain(f, pSeq, expected_seq_rows)
        segments = _rotate_image_file(image_path, pSeq)
        if not segments:
            # Rotation postponed (busy file): this block lands in the oversized
            # file exactly as before, and the next block tries again.
            with _open_h5_append(image_path) as f:
                _append_image_rows(f, imgs_block, seq_ids_block, pSeq)
            return
    else:
        _rebuild_master_if_missing(image_path, segments, pSeq)
        _heal_orphans_rotated(image_path, segments, pSeq, expected_seq_rows)
        if _over_rotate_limit(segments[-1], rotate_bytes):
            segments = _rotate_image_file(image_path, pSeq) or segments

    # --- rotated layout: frames + LOCAL ids into the live segment ... ---
    live = segments[-1]
    with _open_h5_append(live) as f:
        n_local = _append_image_rows(f, imgs_block, seq_ids_block, pSeq)
        offset = int(f.attrs.get('frame_offset', 0))
    # ... then the GLOBAL ids and the GLOBAL watermark into the master, LAST.
    if not os.path.exists(image_path):
        # Only reachable if the master could be neither published nor rebuilt;
        # writing one here from scratch would be a stub with no /imgs at all.
        logger.error('append_images_block: master %s is MISSING; %d frame(s) '
                     'landed in %s but no master points at them.',
                     os.path.basename(image_path), n_frames,
                     os.path.basename(live))
        return
    with _open_h5_append(image_path) as f:
        _append_rows(f, 'seq_ids', seq_ids_block, chunks=(64,))
        _append_rows(f, 'frame_seq_ids',
                     np.repeat(seq_ids_block, pSeq), chunks=(256,))
        f.attrs['committed_frames'] = offset + n_local


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
