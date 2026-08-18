"""Split image-file layout: the resolver (yb_analysis/io/scan_files.py).

A scan is either COMBINED (one data_<stamp>.h5 with /imgs inside) or SPLIT
(a small data_<stamp>.h5 beside a bulk image_<stamp>.h5). The resolver is the
single seam that tells every reader which it is, so these tests pin:

* combined -> layout='combined' and image_path == data_path (call sites never
  branch -- that equality is the whole trick),
* split    -> both paths + the num_images/frame_size attrs,
* the legacy .mat fallback and the "orphan image file is not a scan" rule,
* probe_attrs=False opens nothing,
* the naming helpers image_source / imgs_path_for.

Writer + reader round-trip tests are added here in a later step.

Run: yb_analysis-env python -m pytest yb_analysis/tests/test_split_image_file.py -v
"""
import os

import pytest

h5py = pytest.importorskip('h5py')

from yb_analysis.io import scan_files as sf


# --------------------------------------------------------------------------
# helpers: build tiny real scan dirs (attrs only, no bulk data)
# --------------------------------------------------------------------------

def _scan_dir(tmp_path, stamp='20260818_101500'):
    d = tmp_path / f'data_{stamp}'
    d.mkdir()
    return str(d)


def _write_data_h5(scan_dir, stamp='20260818_101500', **attrs):
    """Write a minimal data_<stamp>.h5 (a couple of seq_ids + given attrs)."""
    path = os.path.join(scan_dir, f'data_{stamp}.h5')
    with h5py.File(path, 'w') as f:
        f.create_dataset('seq_ids', data=[1, 2], dtype='int64')
        for k, v in attrs.items():
            f.attrs[k] = v
    return path


def _write_image_h5(scan_dir, stamp='20260818_101500',
                    num_images=2, frame_size=(4, 4)):
    """Write a minimal image_<stamp>.h5 carrying the split-layout attrs."""
    path = os.path.join(scan_dir, f'{sf.IMGS_PREFIX}_{stamp}.h5')
    with h5py.File(path, 'w') as f:
        f.create_dataset('imgs', shape=(0,) + tuple(frame_size),
                         maxshape=(None,) + tuple(frame_size), dtype='uint16')
        f.attrs['layout'] = 'images'
        f.attrs['num_images_per_seq'] = num_images
        f.attrs['frame_size'] = tuple(frame_size)
    return path


# --------------------------------------------------------------------------
# resolve_scan_files -- layouts
# --------------------------------------------------------------------------

def test_combined_layout_image_path_equals_data_path(tmp_path):
    """Legacy combined scan: image_path aliases data_path so no reader branches."""
    d = _scan_dir(tmp_path)
    data = _write_data_h5(d)

    r = sf.resolve_scan_files(d)
    assert r.scan_dir == d
    assert r.data_path == data
    assert r.image_path == data           # the minimal-diff aliasing
    assert r.layout == sf.LAYOUT_COMBINED
    assert r.num_images is None           # legacy file carries neither attr
    assert r.frame_size is None


def test_split_layout_paths_and_probed_attrs(tmp_path):
    """Split scan: both paths distinct, attrs filled from the files."""
    d = _scan_dir(tmp_path)
    data = _write_data_h5(d)
    imgs = _write_image_h5(d, num_images=2, frame_size=(4, 4))

    r = sf.resolve_scan_files(d)
    assert r.data_path == data
    assert r.image_path == imgs
    assert r.image_path != r.data_path
    assert r.layout == sf.LAYOUT_SPLIT
    assert r.num_images == 2
    assert r.frame_size == (4, 4)


def test_split_attrs_read_from_data_file_when_present(tmp_path):
    """The small file's own attrs are preferred (one open, no image-file touch)."""
    d = _scan_dir(tmp_path)
    _write_data_h5(d, num_images_per_seq=3, frame_size=(8, 6))
    _write_image_h5(d, num_images=3, frame_size=(8, 6))

    r = sf.resolve_scan_files(d)
    assert r.layout == sf.LAYOUT_SPLIT
    assert r.num_images == 3
    assert r.frame_size == (8, 6)


def test_mat_only_dir_is_combined(tmp_path):
    """A MATLAB-era scan: the .mat is the data file, layout combined."""
    d = _scan_dir(tmp_path)
    mat = os.path.join(d, os.path.basename(d) + '.mat')
    with h5py.File(mat, 'w') as f:          # .mat v7.3 is an HDF5 file
        f.create_dataset('seq_ids', data=[1, 2], dtype='int64')

    r = sf.resolve_scan_files(d)
    assert r.data_path == mat
    assert r.image_path == mat
    assert r.layout == sf.LAYOUT_COMBINED


def test_glob_fallback_when_basename_differs(tmp_path):
    """data_*.h5 glob fallback (mirrors load_data's precedence)."""
    d = str(tmp_path / 'oddly_named_dir')
    os.makedirs(d)
    data = _write_data_h5(d, stamp='20260818_101500')

    r = sf.resolve_scan_files(d)
    assert r.data_path == data
    assert r.layout == sf.LAYOUT_COMBINED


def test_orphan_image_file_is_not_a_scan(tmp_path):
    """A crash between the two creates leaves an image file alone -> not a scan."""
    d = _scan_dir(tmp_path)
    _write_image_h5(d)                      # no data file at all

    r = sf.resolve_scan_files(d)
    assert r.data_path is None
    assert r.image_path is None             # explicitly NOT the orphan
    assert r.layout == sf.LAYOUT_COMBINED
    assert r.num_images is None and r.frame_size is None


def test_empty_dir_resolves_to_nothing(tmp_path):
    """No files (and a missing dir) must resolve, not raise."""
    d = _scan_dir(tmp_path)
    r = sf.resolve_scan_files(d)
    assert (r.data_path, r.image_path, r.layout) == (
        None, None, sf.LAYOUT_COMBINED)

    missing = str(tmp_path / 'data_20990101_000000')
    r2 = sf.resolve_scan_files(missing)
    assert r2.data_path is None and r2.image_path is None


def test_filesystem_presence_beats_attrs(tmp_path):
    """images_file attr promising a split file that is GONE -> combined."""
    d = _scan_dir(tmp_path)
    data = _write_data_h5(d, images_external=True,
                          images_file='image_20260818_101500.h5',
                          num_images_per_seq=2, frame_size=(4, 4))

    r = sf.resolve_scan_files(d)
    assert r.layout == sf.LAYOUT_COMBINED   # crash-tolerant: stat wins
    assert r.image_path == data


def test_corrupt_file_probe_never_raises(tmp_path):
    """A truncated/garbage h5 must yield None attrs, not an exception."""
    d = _scan_dir(tmp_path)
    data = os.path.join(d, os.path.basename(d) + '.h5')
    with open(data, 'wb') as f:
        f.write(b'not an hdf5 file at all')

    r = sf.resolve_scan_files(d)
    assert r.data_path == data
    assert r.layout == sf.LAYOUT_COMBINED
    assert r.num_images is None and r.frame_size is None


# --------------------------------------------------------------------------
# probe_attrs=False -- stat only
# --------------------------------------------------------------------------

def test_probe_attrs_false_opens_no_files(tmp_path, monkeypatch):
    """Hot path (runs list): paths/layout right, attrs None, ZERO h5py opens."""
    d = _scan_dir(tmp_path)
    data = _write_data_h5(d)
    imgs = _write_image_h5(d)

    opens = []
    real_file = h5py.File
    monkeypatch.setattr(h5py, 'File',
                        lambda *a, **k: (opens.append(a[0]),
                                         real_file(*a, **k))[1])

    r = sf.resolve_scan_files(d, probe_attrs=False)
    assert opens == []                      # nothing was opened
    assert r.data_path == data
    assert r.image_path == imgs
    assert r.layout == sf.LAYOUT_SPLIT
    assert r.num_images is None
    assert r.frame_size is None


# --------------------------------------------------------------------------
# naming helpers
# --------------------------------------------------------------------------

def test_imgs_path_for_is_a_pure_mapping(tmp_path):
    """data_<stamp>.h5 -> image_<stamp>.h5, same dir, no filesystem check."""
    p = os.path.join(str(tmp_path), 'data_20260818_101500.h5')
    want = os.path.join(str(tmp_path), 'image_20260818_101500.h5')
    assert sf.imgs_path_for(p) == want
    assert not os.path.exists(want)         # purely a string mapping
    # Bare filename (no directory) still maps.
    assert sf.imgs_path_for('data_20260818_101500.h5') == \
        'image_20260818_101500.h5'


def test_image_source_redirects_only_when_file_exists(tmp_path):
    """image_source: redirect a data path to the image file, else identity."""
    d = _scan_dir(tmp_path)
    data = _write_data_h5(d)

    assert sf.image_source(data) == data    # combined -> identity

    imgs = _write_image_h5(d)
    assert sf.image_source(data) == imgs    # split -> redirected
    assert sf.image_source(imgs) == imgs    # already the image file -> identity

    # A legacy .mat scan (never has an image sibling) -> identity. The redirect
    # is keyed on the STAMP, not the extension, so use a dir of its own.
    md = str(tmp_path / 'mat_only')
    os.makedirs(md)
    mat = os.path.join(md, 'data_20260101_000000.mat')
    open(mat, 'wb').close()
    assert sf.image_source(mat) == mat
