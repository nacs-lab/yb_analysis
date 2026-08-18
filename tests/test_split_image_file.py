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

Step 2 appends the READ-path tests (see the second banner below): the legacy
combined file unchanged through every new reader, and split mode where the
ANALYSIS path must never open the image file. Writer tests come in a later step.

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


# ==========================================================================
# Step 2 -- the READ path (load_data / run_analysis / dashboard / hist_init).
#
# The governing requirement: in the SPLIT layout the ANALYSIS path must read
# ONLY the small data file -- it must never open the multi-GB image file. The
# imgs shape therefore comes from the data file's own attrs, and the image file
# is opened solely on an explicit image request (load_images, avg-image build,
# shot-image popup, focus metrics).
# ==========================================================================

import numpy as np

from yb_analysis.analysis import load_data as ld
from yb_analysis.analysis.run_analysis import _scan_data_h5, _scan_imgs_h5

STAMP = '20260818_101500'


def _track_h5_opens(monkeypatch):
    """Record every path handed to h5py.File. Returns the growing list."""
    opens = []
    real_file = h5py.File

    def _spy(*a, **k):
        opens.append(str(a[0]) if a else str(k.get('name')))
        return real_file(*a, **k)

    monkeypatch.setattr(h5py, 'File', _spy)
    return opens


def _img_stack(n_frames, h=4, w=5):
    """Deterministic distinguishable frames: frame k is filled with k+1."""
    a = np.zeros((n_frames, h, w), dtype=np.uint16)
    for k in range(n_frames):
        a[k] = k + 1
    return a


def _write_combined_scan(scan_dir, n_seq=3, num_images=2, h=4, w=5):
    """A LEGACY combined data_<stamp>.h5, as hdf5_store.create_scan_file
    writes them today: /imgs + two-array logicals + seq_ids + scan_config
    (and no split-layout attrs at all)."""
    path = os.path.join(scan_dir, f'data_{STAMP}.h5')
    imgs = _img_stack(n_seq * num_images, h, w)
    with h5py.File(path, 'w') as f:
        f.attrs['two_array'] = True
        f.create_dataset('imgs', data=imgs, maxshape=(None, h, w),
                         dtype='uint16', chunks=(1, h, w),
                         compression='gzip', compression_opts=1)
        f.create_dataset('logicals_img1', data=np.ones((n_seq, 2), dtype=bool),
                         maxshape=(None, 2))
        f.create_dataset('logicals_img2', data=np.ones((n_seq, 2), dtype=bool),
                         maxshape=(None, 2))
        f.create_dataset('seq_ids', data=np.arange(1, n_seq + 1, dtype='int64'),
                         maxshape=(None,))
        g = f.create_group('scan_config')
        g.attrs['NumImages'] = num_images
    return path, imgs


def _write_split_scan(scan_dir, n_seq=3, num_images=2, h=4, w=5,
                      data_attrs=True):
    """A SPLIT pair: small data_<stamp>.h5 (NO /imgs) + bulk image_<stamp>.h5.

    ``data_attrs=False`` omits num_images_per_seq/frame_size from the data file
    so the attr-fallback path (header open of the image file) can be exercised.
    """
    data = os.path.join(scan_dir, f'data_{STAMP}.h5')
    imgs_path = os.path.join(scan_dir, f'{sf.IMGS_PREFIX}_{STAMP}.h5')
    n_frames = n_seq * num_images
    imgs = _img_stack(n_frames, h, w)
    seq_ids = np.arange(1, n_seq + 1, dtype='int64')

    with h5py.File(data, 'w') as f:
        f.attrs['two_array'] = True
        f.attrs['schema_version'] = 1
        f.attrs['images_external'] = True
        f.attrs['images_file'] = os.path.basename(imgs_path)
        if data_attrs:
            f.attrs['num_images_per_seq'] = num_images
            f.attrs['frame_size'] = (h, w)
        f.create_dataset('logicals_img1', data=np.ones((n_seq, 2), dtype=bool),
                         maxshape=(None, 2))
        f.create_dataset('logicals_img2', data=np.ones((n_seq, 2), dtype=bool),
                         maxshape=(None, 2))
        f.create_dataset('seq_ids', data=seq_ids, maxshape=(None,))
        g = f.create_group('scan_config')
        g.attrs['NumImages'] = num_images

    with h5py.File(imgs_path, 'w') as f:
        f.attrs['schema_version'] = 1
        f.attrs['layout'] = 'images'
        f.attrs['num_images_per_seq'] = num_images
        f.attrs['frame_size'] = (h, w)
        f.attrs['data_file'] = os.path.basename(data)
        f.attrs['committed_frames'] = n_frames
        f.create_dataset('imgs', data=imgs, maxshape=(None, h, w),
                         dtype='uint16', chunks=(1, h, w),
                         compression='gzip', compression_opts=1)
        f.create_dataset('seq_ids', data=seq_ids, maxshape=(None,))
        f.create_dataset('frame_seq_ids',
                         data=np.repeat(seq_ids, num_images), maxshape=(None,))
    return data, imgs_path, imgs


# --------------------------------------------------------------------------
# old combined file -- behavior must be BIT-IDENTICAL to before the split
# --------------------------------------------------------------------------

def test_combined_file_readers_unchanged(tmp_path):
    """The key backwards-compat test: a legacy combined scan through every
    new reader behaves exactly as it did before the split existed."""
    d = _scan_dir(tmp_path, STAMP)
    data, imgs = _write_combined_scan(d, n_seq=3, num_images=2)

    bundle = ld.load_scan_from_path(d)
    assert bundle['path'] == data
    assert bundle['image_path'] == data          # image path == data path
    assert bundle['layout'] == sf.LAYOUT_COMBINED
    assert tuple(bundle['imgs_shape']) == imgs.shape
    assert bundle['logicals_img1'].shape == (3, 2)
    np.testing.assert_array_equal(bundle['seq_ids'], [1, 2, 3])

    # shape + pixels straight off the data path
    assert tuple(ld.get_images_shape(data)) == imgs.shape
    np.testing.assert_array_equal(ld.load_images(data, 3), imgs[3])
    np.testing.assert_array_equal(ld.load_images(data), imgs)

    # both resolvers point at the same (single) file
    from pathlib import Path as _P
    assert _scan_data_h5(_P(d)) == _P(data)
    assert _scan_imgs_h5(_P(d)) == _P(data)


def test_combined_no_imgs_dataset_gives_none_shape(tmp_path):
    """A data file with no /imgs at all -> imgs_shape None (supported state)."""
    d = _scan_dir(tmp_path, STAMP)
    path = os.path.join(d, f'data_{STAMP}.h5')
    with h5py.File(path, 'w') as f:
        f.create_dataset('logicals', data=np.ones((2, 2), dtype=bool))
        f.create_dataset('seq_ids', data=np.array([1, 2], dtype='int64'))

    bundle = ld.load_scan_from_path(d)
    assert bundle['imgs_shape'] is None
    assert bundle['layout'] == sf.LAYOUT_COMBINED


# --------------------------------------------------------------------------
# split mode -- the analysis path must never open the image file
# --------------------------------------------------------------------------

def test_split_load_scan_never_opens_image_file(tmp_path, monkeypatch):
    """load_scan_from_path derives imgs_shape from the DATA file's attrs."""
    d = _scan_dir(tmp_path, STAMP)
    data, imgs_path, imgs = _write_split_scan(d, n_seq=3, num_images=2,
                                              h=4, w=5)

    opens = _track_h5_opens(monkeypatch)
    bundle = ld.load_scan_from_path(d)

    # THE requirement: not one open of the image file.
    assert all(os.path.basename(p) != os.path.basename(imgs_path)
               for p in opens), opens
    assert opens                                # (it did open the data file)

    assert bundle['path'] == data               # scan identity = data file
    assert bundle['image_path'] == imgs_path
    assert bundle['layout'] == sf.LAYOUT_SPLIT
    assert tuple(bundle['imgs_shape']) == (6, 4, 5) == imgs.shape
    assert bundle['logicals_img1'].shape == (3, 2)


def test_split_load_images_redirects_from_data_path(tmp_path):
    """Every unedited caller keeps passing the DATA path and still gets the
    split file's pixels (load_images / get_images_shape redirect)."""
    d = _scan_dir(tmp_path, STAMP)
    data, imgs_path, imgs = _write_split_scan(d, n_seq=3, num_images=2)

    assert tuple(ld.get_images_shape(data)) == imgs.shape
    np.testing.assert_array_equal(ld.load_images(data, 0), imgs[0])
    np.testing.assert_array_equal(ld.load_images(data, 5), imgs[5])
    np.testing.assert_array_equal(ld.load_images(data, slice(1, 4)), imgs[1:4])
    np.testing.assert_array_equal(ld.load_images(data, [4, 1]), imgs[[4, 1]])
    np.testing.assert_array_equal(ld.load_images(data), imgs)

    # bundle['path'] is the data file -> the same redirect applies.
    bundle = ld.load_scan_from_path(d)
    np.testing.assert_array_equal(ld.load_images(bundle['path'], 2), imgs[2])


def test_split_scan_imgs_h5_vs_scan_data_h5(tmp_path):
    """_scan_data_h5 keeps returning the DATA file; _scan_imgs_h5 the image."""
    from pathlib import Path as _P
    d = _scan_dir(tmp_path, STAMP)
    data, imgs_path, _ = _write_split_scan(d)

    assert _scan_data_h5(_P(d)) == _P(data)
    assert _scan_imgs_h5(_P(d)) == _P(imgs_path)


def test_split_hist_init_context_points_at_image_file(tmp_path):
    """hist_init's ctx['data_path'] is the IMAGE file (belt and braces)."""
    from yb_analysis.detection.hist_init import load_scan_context
    d = _scan_dir(tmp_path, STAMP)
    data, imgs_path, imgs = _write_split_scan(d, n_seq=3, num_images=2)

    ctx = load_scan_context(d)
    assert ctx['data_path'] == imgs_path
    assert ctx['total_frames'] == imgs.shape[0]
    assert ctx['num_images'] == 2
    assert ctx['all_first_indices'] == [0, 2, 4]


def test_split_pixel_read_still_opens_the_image_file(tmp_path, monkeypatch):
    """The complement of the no-open rule: an EXPLICIT image request does
    open the image file (and only it)."""
    d = _scan_dir(tmp_path, STAMP)
    data, imgs_path, imgs = _write_split_scan(d)

    opens = _track_h5_opens(monkeypatch)
    got = ld.load_images(data, 1)
    np.testing.assert_array_equal(got, imgs[1])
    assert [os.path.basename(p) for p in opens] == \
        [os.path.basename(imgs_path)]


# --------------------------------------------------------------------------
# attr fallback -- data file lacking the attrs opens the image HEADER
# --------------------------------------------------------------------------

def test_split_imgs_shape_falls_back_to_image_header(tmp_path, monkeypatch):
    """No num_images_per_seq/frame_size on the data file -> header-only open
    of the image file yields the correct shape (never the pixels)."""
    d = _scan_dir(tmp_path, STAMP)
    data, imgs_path, imgs = _write_split_scan(d, n_seq=3, num_images=2,
                                              h=4, w=5, data_attrs=False)

    opens = _track_h5_opens(monkeypatch)
    bundle = ld.load_scan_from_path(d)
    assert tuple(bundle['imgs_shape']) == imgs.shape
    assert bundle['layout'] == sf.LAYOUT_SPLIT
    # It DID have to consult the image file here -- that is the documented
    # fallback, and only in this (writer-impossible) case.
    assert os.path.basename(imgs_path) in [os.path.basename(p) for p in opens]


def test_split_imgs_shape_none_when_image_file_unreadable(tmp_path):
    """Attrs missing AND the image file corrupt -> imgs_shape None, no raise."""
    d = _scan_dir(tmp_path, STAMP)
    data, imgs_path, _ = _write_split_scan(d, data_attrs=False)
    with open(imgs_path, 'wb') as f:
        f.write(b'not an hdf5 file at all')

    bundle = ld.load_scan_from_path(d)
    assert bundle['imgs_shape'] is None
    assert bundle['layout'] == sf.LAYOUT_SPLIT
    assert bundle['image_path'] == imgs_path


def test_split_pseq3_shape_from_attrs(tmp_path):
    """pSeq=3 (rearrangement w/ middle frame): shape math is n_seq*pSeq."""
    d = _scan_dir(tmp_path, STAMP)
    data, imgs_path, imgs = _write_split_scan(d, n_seq=4, num_images=3,
                                              h=3, w=6)
    bundle = ld.load_scan_from_path(d)
    assert tuple(bundle['imgs_shape']) == (12, 3, 6) == imgs.shape


# --------------------------------------------------------------------------
# the end-to-end guarantee: analyze_scan_dir opens the DATA file only
# --------------------------------------------------------------------------

@pytest.mark.parametrize('num_images', [2, 3])
def test_analyze_scan_dir_never_opens_image_file(tmp_path, monkeypatch,
                                                num_images):
    """THE requirement, end to end: a full analysis of a split scan touches
    the small data file and nothing else.

    Parametrized over the multi-image cases (survival pSeq=2, rearrangement
    with a middle frame pSeq=3) -- i.e. every scan whose image file is the
    multi-GB one. NumImages==1 is excluded on purpose: for a single-image
    multi-point SWEEP the analysis also builds the calibration-free focus
    curve, which is an image measurement by definition (it is computed once
    and cached to focus_metrics.json).
    """
    from yb_analysis.analysis.run_analysis import analyze_scan_dir
    d = _scan_dir(tmp_path, STAMP)
    data, imgs_path, imgs = _write_split_scan(d, n_seq=6,
                                              num_images=num_images)

    opens = _track_h5_opens(monkeypatch)
    out = analyze_scan_dir(d)

    img_base = os.path.basename(imgs_path)
    assert [p for p in opens if os.path.basename(p) == img_base] == [], opens
    assert out['images_layout'] == sf.LAYOUT_SPLIT
    # avg_image still reports the frame dims -- from the bundle, not a file.
    assert out['avg_image']['image_shape'] == [4, 5]
    assert out['avg_image']['num_images'] == num_images


def test_analyze_scan_dir_combined_reports_layout(tmp_path):
    """A legacy combined scan reports images_layout='combined' (payload v5)."""
    from yb_analysis.analysis.run_analysis import analyze_scan_dir
    d = _scan_dir(tmp_path, STAMP)
    _write_combined_scan(d, n_seq=6, num_images=2)

    out = analyze_scan_dir(d)
    assert out['images_layout'] == sf.LAYOUT_COMBINED
    assert out['avg_image']['image_shape'] == [4, 5]
