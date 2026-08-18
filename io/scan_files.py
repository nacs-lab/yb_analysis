"""Locate a scan's data / image HDF5 files (the split-file resolver).

A scan directory ``data_<YYYYMMDD>_<HHMMSS>/`` holds either

* the LEGACY combined layout -- one ``data_<stamp>.h5`` carrying everything
  (``/imgs`` plus logicals / intensities / seq_ids / scan_config), or
* the SPLIT layout -- a small ``data_<stamp>.h5`` (no ``/imgs``) beside a bulk
  ``image_<stamp>.h5`` holding ``/imgs`` + ``/seq_ids`` + ``/frame_seq_ids``.

Everything that wants image pixels asks this module where they live, so call
sites never branch on the layout: in the combined case ``image_path`` is simply
``data_path``. Detection is filesystem-first (the presence of the image file
wins over any file attribute) so a crash mid-write is resolved the tolerant way,
and ``probe_attrs=False`` keeps hot paths (the runs list) stat-only.

Deliberately stdlib + h5py only -- no imports from ``yb_analysis.analysis`` or
``yb_analysis.acquisition`` -- so the writer (``hdf5_store``), the readers and
the pyctrl tools can all import it without a cycle.
"""

import os
import glob
from typing import NamedTuple, Optional, Tuple

# Layout tags for ``ScanFiles.layout``.
LAYOUT_COMBINED = 'combined'
LAYOUT_SPLIT = 'split'

# Filename prefix of the bulk image file: ``image_<YYYYMMDD>_<HHMMSS>.h5``,
# carrying the same stamp as the scan dir / data file. Lexically distinct from
# ``data_*`` so an existing ``glob('data_*.h5')`` can never pick it up. Change
# the naming here and nowhere else.
IMGS_PREFIX = 'image'


class ScanFiles(NamedTuple):
    """Where one scan's data and image bytes live.

    Attributes
    ----------
    scan_dir : str
        The scan directory that was resolved.
    data_path : str or None
        The small per-scan file (``data_<stamp>.h5``, or the legacy ``.mat``).
        None when the directory holds no scan data file at all.
    image_path : str or None
        Where ``/imgs`` lives. Equal to ``data_path`` in the combined layout,
        the sibling ``image_<stamp>.h5`` in the split layout, and None when
        there is no data file (an image file on its own is not a scan).
    layout : str
        ``LAYOUT_SPLIT`` when a usable image file was found, else
        ``LAYOUT_COMBINED``.
    num_images : int or None
        ``num_images_per_seq`` (frames per shot) read from the file attrs, or
        None when absent / not probed / unreadable.
    frame_size : tuple of (int, int) or None
        ``frame_size`` (H, W) read from the file attrs, or None when absent /
        not probed / unreadable.
    """

    scan_dir: str
    data_path: Optional[str]
    image_path: Optional[str]
    layout: str
    num_images: Optional[int]
    frame_size: Optional[Tuple[int, int]]


def _stamp_of(path):
    """Return the ``<YYYYMMDD>_<HHMMSS>`` stamp of a scan dir / file name.

    Works off the basename: ``data_20260818_101500`` (dir),
    ``data_20260818_101500.h5``, ``image_20260818_101500.h5`` all give
    ``20260818_101500``. Returns '' when the name carries no ``prefix_`` stamp.
    """
    base = os.path.basename(str(path).rstrip('\\/'))
    base = os.path.splitext(base)[0]
    _, sep, stamp = base.partition('_')
    return stamp if sep else ''


def imgs_path_for(data_path):
    """Map ``.../data_<stamp>.h5`` -> ``.../image_<stamp>.h5`` (pure string).

    No filesystem access: this is the name the writer should create and the
    name the resolver looks for. A path whose basename carries no ``prefix_``
    stamp is returned with its prefix swapped anyway (``<IMGS_PREFIX>.h5``),
    which keeps the mapping total.
    """
    data_path = str(data_path)
    d, base = os.path.split(data_path)
    stamp = _stamp_of(base)
    name = f'{IMGS_PREFIX}_{stamp}.h5' if stamp else f'{IMGS_PREFIX}.h5'
    return os.path.join(d, name) if d else name


def image_source(path):
    """Redirect a data-file path to its image file when one exists.

    One ``os.path.exists`` -- cheap enough to call at the top of every image
    reader (``load_images``, ``get_images_shape``), which is exactly the point:
    existing callers keep passing the data path and transparently read the split
    file. Returns ``path`` unchanged for a combined scan, a ``.mat`` file, or a
    path that is already the image file.
    """
    candidate = imgs_path_for(path)
    if candidate != str(path) and os.path.exists(candidate):
        return candidate
    return str(path)


def _find_data_file(scan_dir, base):
    """Locate the scan's data file, mirroring ``load_data.load_scan_from_path``.

    Precedence: the basename convention ``<scan_dir_basename>.h5``, then the
    first ``data_*.h5`` in the directory, then the ``.mat`` fallback.
    """
    h5_path = os.path.join(scan_dir, base + '.h5')
    if os.path.isfile(h5_path):
        return h5_path

    cands = sorted(glob.glob(os.path.join(scan_dir, 'data_*.h5')))
    if cands:
        return cands[0]

    mat_path = os.path.join(scan_dir, base + '.mat')
    if os.path.isfile(mat_path):
        return mat_path
    return None


def _find_image_file(scan_dir, stamp):
    """Locate the bulk image file: exact stamp first, then any ``image_*.h5``."""
    if stamp:
        exact = os.path.join(scan_dir, f'{IMGS_PREFIX}_{stamp}.h5')
        if os.path.isfile(exact):
            return exact

    cands = sorted(glob.glob(os.path.join(scan_dir, f'{IMGS_PREFIX}_*.h5')))
    return cands[0] if cands else None


def _probe_attrs(path):
    """Read ``num_images_per_seq`` / ``frame_size`` from a file's attrs.

    Returns ``(num_images, frame_size)``, either entry None when the attribute
    is absent. Returns ``(None, None)`` -- never raises -- when the file cannot
    be opened at all (corrupt, cloud-only, locked by the live writer): the
    resolver's job is to report paths, and a probe failure must not stop it.
    """
    # hdf5_store disables HDF5's own file locking process-wide for the OneDrive
    # lock-violation failure mode (see bug-hdf5-append-lock-onedrive-silent-loss).
    # We must not import hdf5_store (cycle-free by design), so set it here too --
    # setdefault, so an explicit operator override still wins. HDF5 re-reads the
    # variable per open, so setting it before the first open is enough.
    os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')
    try:
        import h5py
    except ImportError:
        return None, None

    num_images = frame_size = None
    try:
        with h5py.File(path, 'r') as f:
            if 'num_images_per_seq' in f.attrs:
                try:
                    num_images = int(f.attrs['num_images_per_seq'])
                except (TypeError, ValueError):
                    num_images = None
            if 'frame_size' in f.attrs:
                try:
                    fs = tuple(int(v) for v in f.attrs['frame_size'])
                except (TypeError, ValueError):
                    fs = None
                frame_size = fs if fs and len(fs) == 2 else None
    except Exception:
        return None, None
    return num_images, frame_size


def resolve_scan_files(scan_dir, *, probe_attrs=True):
    """Resolve one scan directory to its data + image files.

    Parameters
    ----------
    scan_dir : str
        Path to a scan directory (e.g. ``.../Data/20260818/data_20260818_101500``).
    probe_attrs : bool
        True (default) -> open the small file (and, when split, the image file)
        read-only to fill ``num_images`` / ``frame_size`` from the
        ``num_images_per_seq`` / ``frame_size`` attrs. False -> stat only, ZERO
        file opens, both fields None (for hot paths like the runs list).

    Returns
    -------
    ScanFiles

    Notes
    -----
    * Filesystem presence of ``image_<stamp>.h5`` decides the layout -- attrs
      are never consulted for that -- so a data file whose attrs promise
      external images still resolves to ``combined`` if the image file is gone.
    * No data file at all -> ``data_path is None`` AND ``image_path is None``
      with ``layout='combined'``: an orphan image file (a crash between the two
      creates) is not a scan.
    * Never raises for a missing directory, a corrupt file, or a locked file.
    """
    scan_dir = str(scan_dir)
    base = os.path.basename(scan_dir.rstrip('\\/'))

    data_path = _find_data_file(scan_dir, base)
    if data_path is None:
        # An image file with no data file is an invisible orphan, not a scan.
        return ScanFiles(scan_dir, None, None, LAYOUT_COMBINED, None, None)

    image_path = _find_image_file(scan_dir, _stamp_of(data_path))
    if image_path is None:
        layout = LAYOUT_COMBINED
        image_path = data_path
    else:
        layout = LAYOUT_SPLIT

    num_images = frame_size = None
    if probe_attrs:
        # Probe the SMALL file first (cheap; the new attrs live there), then
        # fall back to the image file's own header for whatever it did not
        # carry -- a legacy data file has neither attr, a split image file has
        # both.
        num_images, frame_size = _probe_attrs(data_path)
        if layout == LAYOUT_SPLIT and (num_images is None or frame_size is None):
            n_img, f_size = _probe_attrs(image_path)
            if num_images is None:
                num_images = n_img
            if frame_size is None:
                frame_size = f_size

    return ScanFiles(scan_dir, data_path, image_path, layout,
                     num_images, frame_size)
