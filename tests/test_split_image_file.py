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


# --------------------------------------------------------------------------
# STEP 3: bootstrap_affine's frames-per-sequence cascade
#
# The highest-severity cross-file bug in the split refactor: the old code
# inferred pSeq as ``len(imgs) // len(intensities_img1)`` -- a division across
# two datasets that post-split live in two DIFFERENT files, silently yielding
# pSeq=1 and averaging img1 together with img2 into the affine fit. A corrupt
# global affine reads out as physics, so the cascade is pinned here.
# --------------------------------------------------------------------------

def _write_image_only_h5(scan_dir, n_seq=3, num_images=2, h=4, w=5,
                         attr=True, frame_ids=True, stamp=STAMP):
    """A bulk image_<stamp>.h5 with the attr and/or frame_seq_ids toggled off."""
    path = os.path.join(scan_dir, f'{sf.IMGS_PREFIX}_{stamp}.h5')
    seq_ids = np.arange(1, n_seq + 1, dtype='int64')
    with h5py.File(path, 'w') as f:
        f.attrs['layout'] = 'images'
        if attr:
            f.attrs['num_images_per_seq'] = num_images
        f.create_dataset('imgs', data=_img_stack(n_seq * num_images, h, w),
                         maxshape=(None, h, w), dtype='uint16')
        f.create_dataset('seq_ids', data=seq_ids, maxshape=(None,))
        if frame_ids:
            f.create_dataset('frame_seq_ids',
                             data=np.repeat(seq_ids, num_images),
                             maxshape=(None,))
    return path


def _pseq(path, config_num_images=None):
    """Run bootstrap_affine's cascade against one h5 file, return (pSeq, src)."""
    from yb_analysis.scripts.bootstrap_affine import _frames_per_seq
    with h5py.File(path, 'r') as f:
        return _frames_per_seq(f, int(f['imgs'].shape[0]), config_num_images)


def test_pseq_attr_wins_when_present(tmp_path):
    """(1) num_images_per_seq on the image file is authoritative."""
    d = _scan_dir(tmp_path, STAMP)
    # frame_seq_ids deliberately absent, so only the attr can answer
    p = _write_image_only_h5(d, n_seq=5, num_images=2, frame_ids=False)
    assert _pseq(p) == (2, 'attr')


def test_pseq_attr_wins_over_frame_ids(tmp_path):
    """The attr is consulted before the frame_seq_ids ratio."""
    d = _scan_dir(tmp_path, STAMP)
    p = _write_image_only_h5(d, n_seq=5, num_images=3)
    assert _pseq(p) == (3, 'attr')


def test_pseq_from_frame_seq_ids_when_attr_stripped(tmp_path):
    """(2) attr gone -> len(frame_seq_ids)//len(seq_ids), inside one file."""
    d = _scan_dir(tmp_path, STAMP)
    p = _write_image_only_h5(d, n_seq=5, num_images=3, attr=False)
    assert _pseq(p) == (3, 'frame_seq_ids')


def test_pseq_combined_file_still_uses_old_division(tmp_path):
    """(3) a LEGACY combined file (no attrs, no frame_seq_ids) keeps working
    via the /imgs vs /intensities_img1 division -- same answer as before."""
    d = _scan_dir(tmp_path, STAMP)
    path = os.path.join(d, f'data_{STAMP}.h5')
    n_seq, num_images = 4, 2
    with h5py.File(path, 'w') as f:
        f.create_dataset('imgs', data=_img_stack(n_seq * num_images))
        f.create_dataset('intensities_img1',
                         data=np.ones((n_seq, 7), dtype='float64'))
        f.create_dataset('seq_ids', data=np.arange(1, n_seq + 1, dtype='int64'))
    assert _pseq(path) == (2, 'intensities')


def test_pseq_split_image_file_never_cross_divides(tmp_path):
    """The regression itself: with the split pair written by the real helper,
    the answer is the true pSeq -- never 1 from a cross-file division."""
    d = _scan_dir(tmp_path, STAMP)
    _data, imgs_path, _imgs = _write_split_scan(d, n_seq=6, num_images=2)
    pSeq, src = _pseq(imgs_path)
    assert pSeq == 2 and src in ('attr', 'frame_seq_ids')


def test_pseq_config_fallback_when_file_says_nothing(tmp_path):
    """(4) no attr, no frame_seq_ids, no intensities -> the config NumImages."""
    d = _scan_dir(tmp_path, STAMP)
    p = _write_image_only_h5(d, n_seq=5, num_images=2, attr=False,
                             frame_ids=False)
    assert _pseq(p, config_num_images=2) == (2, 'config')


def test_pseq_file_attr_beats_a_disagreeing_config(tmp_path):
    """A usable file-derived value is authoritative even when the config
    sidecar disagrees -- no abort, the FILE wins (the sidecar is the guess)."""
    d = _scan_dir(tmp_path, STAMP)
    p = _write_image_only_h5(d, n_seq=6, num_images=2)   # file says 2
    assert _pseq(p, config_num_images=3) == (2, 'attr')  # config says 3


def test_pseq_config_disagreement_aborts_loudly(tmp_path):
    """The guard: falling through to the last-resort config NumImages>1 while a
    file-derived HINT disagrees must abort, not fit a possibly-smeared affine.

    Constructed with an internally inconsistent image file: the attr claims 4
    frames per sequence but does not divide the 10 frames present, so no branch
    of the cascade can use it -- exactly the "something is wrong here" state
    where guessing from the sidecar would smear img1 with img2.
    """
    d = _scan_dir(tmp_path, STAMP)
    p = _write_image_only_h5(d, n_seq=5, num_images=2, frame_ids=False)
    with h5py.File(p, 'r+') as f:
        f.attrs['num_images_per_seq'] = 4        # 4 does not divide 10 frames
    with pytest.raises(RuntimeError) as ei:
        _pseq(p, config_num_images=3)            # config says 3, attr says 4
    assert 'NumImages=3' in str(ei.value)


def test_pseq_config_not_dividing_frames_aborts(tmp_path):
    """Config NumImages that does not divide the frame count, with no
    file-derived hint at all -> abort, not a silent wrong stride."""
    d = _scan_dir(tmp_path, STAMP)
    p = _write_image_only_h5(d, n_seq=5, num_images=2, attr=False,
                             frame_ids=False)           # 10 frames
    with pytest.raises(RuntimeError):
        _pseq(p, config_num_images=3)


def test_pseq_single_image_scan_is_one(tmp_path):
    """A genuine one-frame-per-shot scan: pSeq=1, no abort (NumImages==1 is
    never treated as a disagreement)."""
    d = _scan_dir(tmp_path, STAMP)
    p = _write_image_only_h5(d, n_seq=5, num_images=1, attr=False,
                             frame_ids=False)
    assert _pseq(p, config_num_images=1) == (1, 'default')


# --------------------------------------------------------------------------
# STEP 3: pyctrl's dependency-light mirror (pyctrl/lib/scan_files_lite.py)
# --------------------------------------------------------------------------

def _scan_files_lite():
    """Import pyctrl/lib/scan_files_lite.py by path (pyctrl is a separate repo
    and its lib/ is flat on sys.path at runtime, not a package)."""
    import importlib.util
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    path = os.path.join(repo, 'pyctrl', 'lib', 'scan_files_lite.py')
    spec = importlib.util.spec_from_file_location('scan_files_lite', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_lite_mirrors_resolver_naming(tmp_path):
    """image_source / imgs_path_for must agree with yb_analysis's resolver."""
    lite = _scan_files_lite()
    assert lite.IMGS_PREFIX == sf.IMGS_PREFIX
    d = _scan_dir(tmp_path, STAMP)
    data = os.path.join(d, f'data_{STAMP}.h5')
    assert lite.imgs_path_for(data) == sf.imgs_path_for(data)
    # no image file yet -> identity, both implementations
    assert lite.image_source(data) == data == sf.image_source(data)
    _write_image_only_h5(d)
    assert lite.image_source(data) == sf.image_source(data) \
        == os.path.join(d, f'{sf.IMGS_PREFIX}_{STAMP}.h5')
    # a legacy .mat scan (no image sibling anywhere) -> identity, both
    d2 = _scan_dir(tmp_path, '20260818_101503')
    mat = os.path.join(d2, 'data_20260818_101503.mat')
    assert lite.image_source(mat) == mat == sf.image_source(mat)


def test_lite_frames_per_seq_attr_ratio_default(tmp_path):
    """attr -> ratio -> default, and it accepts a path or an open handle."""
    lite = _scan_files_lite()
    d = _scan_dir(tmp_path, STAMP)
    p_attr = _write_image_only_h5(d, n_seq=4, num_images=3)
    assert lite.frames_per_seq(p_attr) == 3
    with h5py.File(p_attr, 'r') as f:
        assert lite.frames_per_seq(f) == 3

    d2 = _scan_dir(tmp_path, '20260818_101501')
    p_ratio = _write_image_only_h5(d2, n_seq=4, num_images=2, attr=False,
                                   stamp='20260818_101501')
    assert lite.frames_per_seq(p_ratio) == 2

    d3 = _scan_dir(tmp_path, '20260818_101502')
    p_none = _write_image_only_h5(d3, n_seq=4, num_images=2, attr=False,
                                  frame_ids=False, stamp='20260818_101502')
    assert lite.frames_per_seq(p_none) is None
    assert lite.frames_per_seq(p_none, default=7) == 7
    # unreadable / missing file -> the default, never a raise
    assert lite.frames_per_seq(os.path.join(d3, 'nope.h5'), default=5) == 5


# ==========================================================================
# STEP 4: the WRITER (hdf5_store.create_scan_file / append_images_block /
# append_block(write_imgs=), and DataManager.save_data driving them).
#
# The governing invariants, in the order the writer must uphold them:
#   * the image file is created FIRST and completely -- a crash can leave an
#     invisible orphan image file, never a data file promising absent images;
#   * each block appends IMAGES first, then the data rows -- so seq_ids[k]
#     visible implies its frames are durable, never the reverse;
#   * committed_frames is stamped LAST inside the append handle, so the live
#     count min(data seq rows, committed_frames // pSeq) never over-reports;
#   * orphan image rows from a failed data append are TRIMMED by the next
#     append_images_block -- prefer a hole in the images over a shift in them.
# ==========================================================================

import logging
import threading
import time

from yb_analysis.io import hdf5_store as hs
from yb_analysis.acquisition import data_manager as dm_mod

W_STAMP = '20260818_120000'


def _writer_dir(tmp_path, stamp=W_STAMP):
    d = tmp_path / ('data_' + stamp)
    d.mkdir()
    return str(d)


def _writer_paths(scan_dir, stamp=W_STAMP):
    data = os.path.join(scan_dir, 'data_%s.h5' % stamp)
    return data, sf.imgs_path_for(data)


def _create(scan_dir, split, num_images=2, frame_size=(4, 5),
            num_sites=3, stamp=W_STAMP):
    """create_scan_file in either layout; returns (data_path, image_path)."""
    data, imgs = _writer_paths(scan_dir, stamp)
    hs.create_scan_file(
        data, {'NumImages': num_images}, frame_size, num_sites,
        two_array=True, num_sites_img2=num_sites,
        image_path=imgs if split else None,
        num_images_per_seq=num_images)
    return data, imgs


def _append(data, imgs_path, split, imgs_block, sids, num_images,
            num_sites=3, expected_seq_rows=None, rotate_bytes=None):
    """One block through the real writer, images first (as save_data does)."""
    sids = np.asarray(sids, dtype='int64')
    n = len(sids)
    l1 = np.ones((n, num_sites), dtype=bool)
    l2 = np.zeros((n, num_sites), dtype=bool)
    i1 = np.arange(n * num_sites, dtype='float64').reshape(n, num_sites)
    i2 = i1 + 0.5
    if split:
        hs.append_images_block(imgs_path, imgs_block, sids, num_images,
                               expected_seq_rows=expected_seq_rows,
                               rotate_bytes=rotate_bytes)
    hs.append_block(data, imgs_block, l1, i1, sids,
                    logicals_img2_block=l2, intensities_img2_block=i2,
                    write_imgs=not split)


def _tagged_frames(seq_tags, num_images, h=4, w=5):
    """Frames for the given shot tags: every frame of shot t is filled with t."""
    per_shot = np.array(seq_tags, dtype=np.uint16).reshape(-1, 1, 1)
    return np.repeat(per_shot, num_images, axis=0) \
             .repeat(h, axis=1).repeat(w, axis=2)


# --------------------------------------------------------------------------
# test 1: create_scan_file writes the split layout
# --------------------------------------------------------------------------

def test_create_split_writes_two_files_with_attrs(tmp_path):
    d = _writer_dir(tmp_path)
    data, imgs = _create(d, split=True, num_images=2, frame_size=(4, 5))

    assert os.path.isfile(data) and os.path.isfile(imgs)
    assert os.path.basename(imgs) == 'image_%s.h5' % W_STAMP
    # no leftover .tmp from either atomic write
    assert not os.path.exists(data + '.tmp')
    assert not os.path.exists(imgs + '.tmp')

    with h5py.File(data, 'r') as f:
        assert 'imgs' not in f                    # absent, NOT a stub
        assert f.attrs['schema_version'] == 1
        assert bool(f.attrs['images_external']) is True
        assert f.attrs['images_file'] == os.path.basename(imgs)
        assert int(f.attrs['num_images_per_seq']) == 2
        assert tuple(f.attrs['frame_size']) == (4, 5)
        # everything else the data file always had
        for k in ('logicals_img1', 'logicals_img2', 'intensities_img1',
                  'intensities_img2', 'seq_ids', 'scan_config'):
            assert k in f

    with h5py.File(imgs, 'r') as f:
        assert f.attrs['schema_version'] == 1
        assert f.attrs['layout'] == 'images'
        assert int(f.attrs['num_images_per_seq']) == 2
        assert tuple(f.attrs['frame_size']) == (4, 5)
        assert f.attrs['data_file'] == os.path.basename(data)
        assert f.attrs['scan_id'] == W_STAMP
        assert int(f.attrs['committed_frames']) == 0
        assert f['imgs'].shape == (0, 4, 5)
        assert f['seq_ids'].shape == (0,)
        assert f['seq_ids'].maxshape == (None,)
        assert f['seq_ids'].chunks == (64,)
        assert f['frame_seq_ids'].shape == (0,)
        assert f['frame_seq_ids'].maxshape == (None,)
        assert f['frame_seq_ids'].chunks == (256,)

    # and the resolver sees it as a split scan
    r = sf.resolve_scan_files(d)
    assert (r.data_path, r.image_path, r.layout) == (data, imgs,
                                                     sf.LAYOUT_SPLIT)


def test_split_imgs_tuning_identical_to_combined(tmp_path):
    """The /imgs dataset tuning (dtype/chunks/gzip-1/maxshape) is UNCHANGED by
    the split -- pinned so a read-speed experiment can't ride in on this PR."""
    dc = _writer_dir(tmp_path, '20260818_120001')
    ds = _writer_dir(tmp_path, '20260818_120002')
    comb, _ = _create(dc, split=False, stamp='20260818_120001')
    _, imgs = _create(ds, split=True, stamp='20260818_120002')

    def _tuning(path):
        with h5py.File(path, 'r') as f:
            dset = f['imgs']
            return (dset.shape, dset.maxshape, str(dset.dtype), dset.chunks,
                    dset.compression, dset.compression_opts)

    assert _tuning(comb) == _tuning(imgs)
    assert _tuning(imgs)[2] == 'uint16'
    assert _tuning(imgs)[4] == 'gzip' and _tuning(imgs)[5] == 1


def test_create_combined_unchanged_no_split_attrs(tmp_path):
    """image_path=None -> today's behaviour: /imgs present, NONE of the split
    attrs stamped, and no image file created."""
    d = _writer_dir(tmp_path)
    data, imgs = _create(d, split=False)
    assert not os.path.exists(imgs)
    with h5py.File(data, 'r') as f:
        assert 'imgs' in f
        for k in ('schema_version', 'images_external', 'images_file',
                  'num_images_per_seq', 'frame_size'):
            assert k not in f.attrs


def test_create_split_omits_num_images_attr_when_unknown(tmp_path):
    """num_images_per_seq=None -> files still created, attr simply absent."""
    d = _writer_dir(tmp_path)
    data, imgs = _writer_paths(d)
    hs.create_scan_file(data, {}, (4, 5), 3, image_path=imgs,
                        num_images_per_seq=None)
    for p in (data, imgs):
        with h5py.File(p, 'r') as f:
            assert 'num_images_per_seq' not in f.attrs
    with h5py.File(imgs, 'r') as f:
        assert f.attrs['layout'] == 'images'


def test_create_image_file_exists_before_data_file(tmp_path, monkeypatch):
    """ORDER: the data file must not appear while its image file does not."""
    d = _writer_dir(tmp_path)
    data, imgs = _writer_paths(d)

    seen = {}
    real_replace = os.replace

    def _spy(src, dst):
        # at the moment the DATA file is published, the image file must be there
        if os.path.basename(dst) == os.path.basename(data):
            seen['image_ready'] = os.path.isfile(imgs)
        return real_replace(src, dst)

    monkeypatch.setattr(hs.os, 'replace', _spy)
    hs.create_scan_file(data, {}, (4, 5), 3, image_path=imgs,
                        num_images_per_seq=2)
    assert seen.get('image_ready') is True


# --------------------------------------------------------------------------
# tests 3 + 4: round trip -- both layouts produce identical bundles/pixels
# --------------------------------------------------------------------------

@pytest.mark.parametrize('num_images', [1, 2, 3])
def test_round_trip_split_equals_combined(tmp_path, num_images):
    """The regression net for the whole refactor: the same synthetic blocks
    through the real writer in both modes -> equal bundles (bar path /
    image_path / layout) and identical pixels."""
    n_seq_per_block, n_blocks = 3, 2
    out = {}
    for split in (False, True):
        stamp = '2026081%d_120010' % int(split)
        d = _writer_dir(tmp_path, stamp)
        data, imgs_path = _create(d, split, num_images=num_images,
                                  stamp=stamp)
        all_frames = []
        saved_rows = 0
        for b in range(n_blocks):
            tags = [b * n_seq_per_block + k + 1
                    for k in range(n_seq_per_block)]
            block = _tagged_frames(tags, num_images)
            all_frames.append(block)
            _append(data, imgs_path, split, block,
                    np.array(tags, dtype='int64'), num_images,
                    expected_seq_rows=saved_rows)
            saved_rows += n_seq_per_block
        out[split] = (d, data, imgs_path, np.concatenate(all_frames))

    b_comb = ld.load_scan_from_path(out[False][0])
    b_split = ld.load_scan_from_path(out[True][0])

    ignore = {'path', 'image_path', 'layout'}
    assert set(b_comb) == set(b_split)
    for k in set(b_comb) - ignore:
        a, b = b_comb[k], b_split[k]
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            np.testing.assert_array_equal(a, b, err_msg=k)
        elif k == 'Scan':
            assert dict(a) == dict(b)
        else:
            assert a == b, k
    assert b_comb['layout'] == sf.LAYOUT_COMBINED
    assert b_split['layout'] == sf.LAYOUT_SPLIT
    assert tuple(b_split['imgs_shape']) == out[True][3].shape

    # identical pixels, read via the DATA path in both modes (the redirect)
    np.testing.assert_array_equal(ld.load_images(out[False][1]), out[False][3])
    np.testing.assert_array_equal(ld.load_images(out[True][1]), out[True][3])
    np.testing.assert_array_equal(ld.load_images(out[False][1]),
                                  ld.load_images(out[True][1]))


def test_frame_seq_ids_is_repeat_and_seq_ids_match_data(tmp_path):
    """frame_seq_ids == repeat(seq_ids, pSeq), and the image file's /seq_ids is
    the data file's /seq_ids -- values AND order."""
    num_images = 3
    d = _writer_dir(tmp_path)
    data, imgs_path = _create(d, split=True, num_images=num_images)

    saved = 0
    for tags in ([7, 8], [9, 10, 11]):
        _append(data, imgs_path, True, _tagged_frames(tags, num_images),
                np.array(tags, dtype='int64'), num_images,
                expected_seq_rows=saved)
        saved += len(tags)

    with h5py.File(imgs_path, 'r') as fi, h5py.File(data, 'r') as fd:
        img_sids = fi['seq_ids'][:]
        frame_sids = fi['frame_seq_ids'][:]
        data_sids = fd['seq_ids'][:]
        n_frames = fi['imgs'].shape[0]
    np.testing.assert_array_equal(img_sids, data_sids)
    np.testing.assert_array_equal(img_sids, [7, 8, 9, 10, 11])
    np.testing.assert_array_equal(frame_sids, np.repeat(img_sids, num_images))
    assert n_frames == len(frame_sids) == 5 * num_images


def test_append_images_block_lazily_creates_datasets(tmp_path):
    """append_images_block on a file missing the datasets creates them (the
    same lazy-create tolerance append_block has always had)."""
    d = _writer_dir(tmp_path)
    _, imgs_path = _writer_paths(d)
    with h5py.File(imgs_path, 'w') as f:
        f.attrs['layout'] = 'images'
    block = _tagged_frames([1, 2], 2)
    hs.append_images_block(imgs_path, block, np.array([1, 2], dtype='int64'), 2)
    with h5py.File(imgs_path, 'r') as f:
        assert f['imgs'].shape == (4, 4, 5)
        np.testing.assert_array_equal(f['seq_ids'][:], [1, 2])
        np.testing.assert_array_equal(f['frame_seq_ids'][:], [1, 1, 2, 2])
        assert int(f.attrs['committed_frames']) == 4


def test_append_images_block_logs_ragged_block(tmp_path, caplog):
    """A frame count that is not pSeq*len(seq_ids) is LOGGED, not silent."""
    d = _writer_dir(tmp_path)
    data, imgs_path = _create(d, split=True, num_images=2)
    ragged = _tagged_frames([1, 2], 2)[:3]      # 3 frames for 2 shots at pSeq 2
    with caplog.at_level(logging.WARNING, logger='yb_analysis.io.hdf5_store'):
        hs.append_images_block(imgs_path, ragged,
                               np.array([1, 2], dtype='int64'), 2)
    assert any('frame(s) for' in r.message for r in caplog.records)


# --------------------------------------------------------------------------
# test 7: committed_frames watermark + the live min() rule
# --------------------------------------------------------------------------

def _live_count(data, imgs_path, pSeq):
    """The documented live-reader rule."""
    with h5py.File(data, 'r') as f:
        n_seq = f['seq_ids'].shape[0] if 'seq_ids' in f else 0
    with h5py.File(imgs_path, 'r') as f:
        cf = f.attrs.get('committed_frames')
        if cf is None:
            cf = f['imgs'].shape[0]
    return min(int(n_seq), int(cf) // pSeq)


def test_committed_frames_tracks_rows_and_min_rule_is_monotone(tmp_path):
    num_images = 2
    d = _writer_dir(tmp_path)
    data, imgs_path = _create(d, split=True, num_images=num_images)

    counts = [_live_count(data, imgs_path, num_images)]
    saved = 0
    for b in range(4):
        tags = [saved + k + 1 for k in range(2)]
        _append(data, imgs_path, True, _tagged_frames(tags, num_images),
                np.array(tags, dtype='int64'), num_images,
                expected_seq_rows=saved)
        saved += len(tags)
        with h5py.File(imgs_path, 'r') as f:
            assert int(f.attrs['committed_frames']) == f['imgs'].shape[0]
            assert int(f.attrs['committed_frames']) == saved * num_images
        counts.append(_live_count(data, imgs_path, num_images))

    assert counts == [0, 2, 4, 6, 8]
    assert all(b >= a for a, b in zip(counts, counts[1:]))


def test_committed_frames_never_over_reports_mid_append(tmp_path, monkeypatch):
    """The watermark is stamped LAST: interrupt the append after /imgs has been
    resized and committed_frames must still show the PREVIOUS value."""
    num_images = 2
    d = _writer_dir(tmp_path)
    data, imgs_path = _create(d, split=True, num_images=num_images)
    _append(data, imgs_path, True, _tagged_frames([1], num_images),
            np.array([1], dtype='int64'), num_images, expected_seq_rows=0)

    # Build the block BEFORE the patch (np.repeat is the real numpy function,
    # shared with the helpers), then break the frame_seq_ids step -- which runs
    # AFTER /imgs has been resized and written, but BEFORE the watermark.
    block = _tagged_frames([2], num_images)

    def _boom(*a, **k):
        raise RuntimeError('interrupted')

    monkeypatch.setattr(hs.np, 'repeat', _boom)
    with pytest.raises(RuntimeError):
        hs.append_images_block(imgs_path, block,
                               np.array([2], dtype='int64'), num_images,
                               expected_seq_rows=1)
    monkeypatch.undo()

    with h5py.File(imgs_path, 'r') as f:
        assert f['imgs'].shape[0] == 4                     # rows landed
        assert int(f.attrs['committed_frames']) == 2       # watermark did not
    assert _live_count(data, imgs_path, num_images) == 1    # min() rule holds


# --------------------------------------------------------------------------
# test 6: partial failure, both directions (+ the orphan self-heal amendment)
# --------------------------------------------------------------------------

def test_data_append_failure_leaves_orphans_that_next_block_trims(tmp_path,
                                                                  caplog):
    """images land, data append never happens -> orphan image rows; the NEXT
    append_images_block trims them back to alignment (and says so)."""
    num_images = 2
    d = _writer_dir(tmp_path)
    data, imgs_path = _create(d, split=True, num_images=num_images)

    # block 1 lands fully
    _append(data, imgs_path, True, _tagged_frames([1], num_images),
            np.array([1], dtype='int64'), num_images, expected_seq_rows=0)

    # block 2: images land, then the data append blows up (simulated by not
    # running it at all -- what _save_block's failure branch leaves behind)
    hs.append_images_block(imgs_path, _tagged_frames([2], num_images),
                           np.array([2], dtype='int64'), num_images,
                           expected_seq_rows=1)
    with h5py.File(imgs_path, 'r') as f:
        assert f['imgs'].shape[0] == 4                # orphans present
        assert int(f.attrs['committed_frames']) == 4
    with h5py.File(data, 'r') as f:
        assert f['seq_ids'].shape[0] == 1             # data never advanced
    # the min() rule already hides the orphans from a live reader
    assert _live_count(data, imgs_path, num_images) == 1

    # block 3 heals: expected_seq_rows is still 1, so the 2 orphan rows go
    with caplog.at_level(logging.WARNING, logger='yb_analysis.io.hdf5_store'):
        _append(data, imgs_path, True, _tagged_frames([3], num_images),
                np.array([3], dtype='int64'), num_images,
                expected_seq_rows=1)
    assert any('orphan image row' in r.message for r in caplog.records)

    with h5py.File(imgs_path, 'r') as f:
        assert f['imgs'].shape[0] == 4                # 1 kept + 1 new shot
        np.testing.assert_array_equal(f['seq_ids'][:], [1, 3])
        np.testing.assert_array_equal(f['frame_seq_ids'][:], [1, 1, 3, 3])
        assert int(f.attrs['committed_frames']) == 4
        # NO phase shift: the second shot's rows really are shot 3's frames
        assert (f['imgs'][2] == 3).all()
    with h5py.File(data, 'r') as f:
        np.testing.assert_array_equal(f['seq_ids'][:], [1, 3])


# --------------------------------------------------------------------------
# DataManager-driven: save_data writes BOTH files when the toggle is on,
# and the two partial-failure directions through the real save path.
# --------------------------------------------------------------------------

def _split_dm(tmp_path, monkeypatch, pSeq=2, num_sites=3, split=True,
              frame_size=(4, 4)):
    """A bare DataManager wired for the save path only (mirrors
    test_frame_drop_safety._save_dm) with the split toggle forced."""
    monkeypatch.setattr(dm_mod._cfg, 'SPLIT_IMAGE_FILE', split, raising=False)
    dm = dm_mod.DataManager.__new__(dm_mod.DataManager)
    dm.num_images_per_seq = pSeq
    dm.num_sites = num_sites
    dm.num_sites_img2 = num_sites
    dm.is_two_array = True
    dm._save_two_array = True
    dm._save_mid = pSeq >= 3
    dm.frame_size = tuple(frame_size)
    dm.config = {}
    dm.fname = os.path.join(str(tmp_path), 'data_%s.h5' % W_STAMP)
    dm.iname = sf.imgs_path_for(dm.fname)
    dm._split_images = bool(getattr(dm_mod._cfg, 'SPLIT_IMAGE_FILE', False))
    dm._saved_seq_rows = 0
    dm._file_created = True
    dm._save_lock = threading.Lock()
    dm._proba_img2_to_save = []
    hs.create_scan_file(dm.fname, {}, dm.frame_size, num_sites,
                        two_array=True, num_sites_img2=num_sites,
                        save_mid=dm._save_mid,
                        image_path=dm.iname if dm._split_images else None,
                        num_images_per_seq=pSeq)
    return dm, dm.fname, dm.iname


def _dm_push(dm, tags, pSeq, num_sites=3):
    """Fill the save buffers with pSeq frames per shot, frame value == tag."""
    dm._imgs_to_save = [np.full((4, 4), t, dtype=np.uint16)
                        for t in tags for _ in range(pSeq)]
    dm._logicals_to_save = [np.ones(num_sites, dtype=bool)
                            for _ in tags for _ in range(pSeq)]
    dm._intensities_to_save = [np.full(num_sites, float(t))
                               for t in tags for _ in range(pSeq)]
    dm._seq_ids_to_save = list(tags)


def _wait_save_done(dm, want_rows=None, want_state=None, timeout=5.0):
    """Poll until the save thread has recorded the outcome we are waiting for.

    ``want_rows`` -> ``_saved_seq_rows`` reached that value (a SUCCESS);
    ``want_state`` -> ``_save_health['state']`` reached that value (a FAILURE).
    Both are needed because save_health is sticky: a block that fails leaves
    state='fail' behind, so the next block's success can only be recognised by
    the row counter.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if want_rows is not None and \
                getattr(dm, '_saved_seq_rows', 0) >= want_rows:
            return
        sh = getattr(dm, '_save_health', None)
        if want_state is not None and sh is not None and \
                sh.get('state') == want_state:
            return
        time.sleep(0.02)


def _raise_oserror(*a, **k):
    raise OSError('locked forever')


def test_data_manager_split_save_writes_both_files(tmp_path, monkeypatch):
    """The DataManager-driven proof: with config.SPLIT_IMAGE_FILE on, save_data
    puts the frames in the image file and the shot rows in the data file, and
    the data file gets no /imgs at all."""
    pSeq = 2
    dm, data, imgs_path = _split_dm(tmp_path, monkeypatch, pSeq=pSeq)
    _dm_push(dm, tags=[11, 22, 33], pSeq=pSeq)
    dm.save_data()
    _wait_save_done(dm, 3)

    with h5py.File(data, 'r') as f:
        assert 'imgs' not in f
        np.testing.assert_array_equal(f['seq_ids'][:], [11, 22, 33])
        assert f['logicals_img1'].shape[0] == 3
        assert f['logicals_img2'].shape[0] == 3
    with h5py.File(imgs_path, 'r') as f:
        assert f['imgs'].shape == (6, 4, 4)
        assert int(f.attrs['committed_frames']) == 6
        np.testing.assert_array_equal(f['seq_ids'][:], [11, 22, 33])
        np.testing.assert_array_equal(f['frame_seq_ids'][:],
                                      [11, 11, 22, 22, 33, 33])
        for k, tag in enumerate([11, 22, 33]):
            assert (f['imgs'][k * pSeq] == tag).all()
    assert dm._saved_seq_rows == 3
    assert dm._save_health['state'] == 'ok'

    # and the reader sees a normal split scan through the DATA path
    assert sf.image_source(data) == imgs_path
    np.testing.assert_array_equal(ld.load_images(data, 4),
                                  np.full((4, 4), 33, dtype=np.uint16))


def test_data_manager_combined_save_unchanged(tmp_path, monkeypatch):
    """Toggle OFF (the shipped default) -> exactly today's single file."""
    pSeq = 2
    dm, data, imgs_path = _split_dm(tmp_path, monkeypatch, pSeq=pSeq,
                                    split=False)
    assert dm._split_images is False
    _dm_push(dm, tags=[11, 22], pSeq=pSeq)
    dm.save_data()
    _wait_save_done(dm, 2)

    assert not os.path.exists(imgs_path)
    with h5py.File(data, 'r') as f:
        assert f['imgs'].shape == (4, 4, 4)
        np.testing.assert_array_equal(f['seq_ids'][:], [11, 22])


def test_dm_images_append_failure_skips_the_data_append(tmp_path, monkeypatch):
    """append_images_block raises -> the data append is never attempted (the
    block is lost WHOLE, never half)."""
    dm, data, imgs_path = _split_dm(tmp_path, monkeypatch, pSeq=2)

    calls = []
    real_append_block = dm_mod.append_block
    monkeypatch.setattr(dm_mod, 'append_images_block', _raise_oserror)
    monkeypatch.setattr(
        dm_mod, 'append_block',
        lambda *a, **k: (calls.append(1), real_append_block(*a, **k))[1])

    _dm_push(dm, tags=[1, 2], pSeq=2)
    dm.save_data()
    _wait_save_done(dm, want_state='fail')

    assert calls == []                                  # data never attempted
    with h5py.File(data, 'r') as f:
        assert f['seq_ids'].shape[0] == 0
    with h5py.File(imgs_path, 'r') as f:
        assert f['imgs'].shape[0] == 0
    assert dm._save_health['state'] == 'fail'
    assert 'images save failed' in dm._save_health['reason']
    assert not dm._save_health.get('partial')
    assert dm._saved_seq_rows == 0


def test_dm_data_append_failure_marks_partial(tmp_path, monkeypatch):
    """Images landed, data append raised -> save_health says fail + partial and
    names the 'data' stage; _saved_seq_rows does NOT advance, so the next
    block's append_images_block trims the orphans."""
    dm, data, imgs_path = _split_dm(tmp_path, monkeypatch, pSeq=2)
    monkeypatch.setattr(dm_mod, 'append_block', _raise_oserror)

    _dm_push(dm, tags=[1, 2], pSeq=2)
    dm.save_data()
    _wait_save_done(dm, want_state='fail')

    with h5py.File(imgs_path, 'r') as f:
        assert f['imgs'].shape[0] == 4                   # images landed
    with h5py.File(data, 'r') as f:
        assert f['seq_ids'].shape[0] == 0                # data did not
    assert dm._save_health['state'] == 'fail'
    assert dm._save_health.get('partial') is True
    assert 'data save failed' in dm._save_health['reason']
    assert dm._saved_seq_rows == 0                       # not counted

    # the next (healthy) block self-heals the 4 orphan rows away
    monkeypatch.undo()
    dm._split_images = True
    _dm_push(dm, tags=[3, 4], pSeq=2)
    dm.save_data()
    _wait_save_done(dm, 2)

    with h5py.File(imgs_path, 'r') as f:
        assert f['imgs'].shape[0] == 4
        np.testing.assert_array_equal(f['seq_ids'][:], [3, 4])
        assert (f['imgs'][0] == 3).all()                 # no phase shift
    with h5py.File(data, 'r') as f:
        np.testing.assert_array_equal(f['seq_ids'][:], [3, 4])


# --------------------------------------------------------------------------
# test 8: LIVE read while the writer appends (no phase shift, never decreasing)
# --------------------------------------------------------------------------

def test_live_read_during_split_append_never_shifts(tmp_path):
    """A writer thread appends split blocks with small sleeps while a reader
    loop applies the min() rule: the count never decreases, and every counted
    shot's img1 pixels carry that shot's own tag (no positional phase shift).

    Mirrors test_frame_drop_safety's polling idiom.
    """
    num_images, n_blocks, per_block = 2, 6, 2
    d = _writer_dir(tmp_path)
    data, imgs_path = _create(d, split=True, num_images=num_images)

    stop = threading.Event()
    errors = []

    def _writer():
        saved = 0
        try:
            for b in range(n_blocks):
                tags = [saved + k + 1 for k in range(per_block)]
                _append(data, imgs_path, True,
                        _tagged_frames(tags, num_images),
                        np.array(tags, dtype='int64'), num_images,
                        expected_seq_rows=saved)
                saved += per_block
                time.sleep(0.02)
        except Exception as e:              # noqa: BLE001
            errors.append(e)
        finally:
            stop.set()

    th = threading.Thread(target=_writer, daemon=True)
    th.start()

    counts = []
    deadline = time.time() + 20.0
    while time.time() < deadline:
        try:
            n = _live_count(data, imgs_path, num_images)
        except (OSError, KeyError):
            time.sleep(0.005)
            continue
        counts.append(n)
        if n:
            # shot k (1-based) -> its img1 frame row is (k-1)*pSeq; the frame
            # must be filled with the shot's own tag, which IS its seq_id here.
            try:
                with h5py.File(data, 'r') as f:
                    sids_now = f['seq_ids'][:n]
                with h5py.File(imgs_path, 'r') as f:
                    for k in (1, n):
                        frame = f['imgs'][(k - 1) * num_images]
                        assert (frame == sids_now[k - 1]).all(), (k, n)
            except OSError:
                pass
        if stop.is_set() and n >= n_blocks * per_block:
            break
        time.sleep(0.005)
    th.join(timeout=5.0)

    assert not errors, errors
    assert counts and all(b >= a for a, b in zip(counts, counts[1:])), counts
    assert counts[-1] == n_blocks * per_block


# ==========================================================================
# STEP 5: the focus-metric GATE -- the last automatic image read is gone.
#
# The calibration-free focus curve is the only analysis product measured from
# raw /imgs PIXELS, and it used to be computed on EVERY default payload build
# for a NumImages==1 multi-point sweep. Post-split that means opening the
# multi-GB image_<stamp>.h5 (and possibly forcing a OneDrive hydration) just by
# opening the Analysis tab. It is now gated behind compute_focus_metrics=True:
#
#   * default kwargs, no cache -> ZERO image-file opens, panel omitted
#     (seq_specific is None, exactly as for a non-qualifying scan);
#   * compute_focus_metrics=True -> the pixel read happens (image file only)
#     and focus_metrics.json is written;
#   * a later DEFAULT analyze serves that cache with zero image opens again.
# ==========================================================================

F_STAMP = STAMP


def _spot_frame(sigma, h=64, w=64, spacing=16, amp=200.0, bg=10.0, seed=0):
    """One frame of Gaussian spots of width ``sigma`` (mirrors the focus-metric
    test's generator, small enough to keep the split fixture cheap)."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w), bg, dtype=np.float64)
    yy, xx = np.mgrid[0:h, 0:w]
    for cy in range(spacing, h - spacing, spacing):
        for cx in range(spacing, w - spacing, spacing):
            img += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2)
                                / (2 * sigma ** 2))
    return img + rng.normal(0, 1.0, img.shape)


def _write_focus_split_scan(scan_dir, sigmas=(3.0, 1.4, 3.0), n_per=3,
                            h=64, w=64):
    """A SPLIT scan that QUALIFIES for focus metrics: NumImages==1, a multi-point
    sweep, real spot frames in the bulk image file. Returns (data, image, imgs).

    The middle sweep point is the tight focus, so a successful compute is
    checkable (not just "it wrote something").
    """
    data = os.path.join(scan_dir, 'data_%s.h5' % F_STAMP)
    imgs_path = os.path.join(scan_dir, '%s_%s.h5' % (sf.IMGS_PREFIX, F_STAMP))

    frames, sids = [], []
    for p, s in enumerate(sigmas):
        for k in range(n_per):
            frames.append(_spot_frame(s, h=h, w=w, seed=p * 10 + k))
            sids.append(p + 1)
    imgs = np.asarray(frames, dtype=np.uint16)
    seq_ids = np.asarray(sids, dtype='int64')
    n_seq = len(sids)

    with h5py.File(data, 'w') as f:
        f.attrs['schema_version'] = 1
        f.attrs['images_external'] = True
        f.attrs['images_file'] = os.path.basename(imgs_path)
        f.attrs['num_images_per_seq'] = 1
        f.attrs['frame_size'] = (h, w)
        f.create_dataset('logicals', data=np.ones((n_seq, 2), dtype=bool),
                         maxshape=(None, 2))
        f.create_dataset('seq_ids', data=seq_ids, maxshape=(None,))
        g = f.create_group('scan_config')
        g.attrs['NumImages'] = 1
        g.attrs['Params'] = np.array([1, 2, 3], dtype='int64')
        g.attrs['ScanParams'] = np.array([-1.0, 0.0, 1.0])

    with h5py.File(imgs_path, 'w') as f:
        f.attrs['schema_version'] = 1
        f.attrs['layout'] = 'images'
        f.attrs['num_images_per_seq'] = 1
        f.attrs['frame_size'] = (h, w)
        f.attrs['data_file'] = os.path.basename(data)
        f.attrs['committed_frames'] = imgs.shape[0]
        f.create_dataset('imgs', data=imgs, maxshape=(None, h, w),
                         dtype='uint16', chunks=(1, h, w),
                         compression='gzip', compression_opts=1)
        f.create_dataset('seq_ids', data=seq_ids, maxshape=(None,))
        f.create_dataset('frame_seq_ids', data=seq_ids, maxshape=(None,))
    return data, imgs_path, imgs


def _img_opens(opens, imgs_path):
    base = os.path.basename(imgs_path)
    return [p for p in opens if os.path.basename(p) == base]


def test_focus_scan_default_analyze_opens_no_image_file(tmp_path, monkeypatch):
    """THE new requirement: the qualifying NumImages==1 sweep -- the ONE case
    that used to read pixels automatically -- now opens the image file ZERO
    times on a default analyze, and no focus panel is produced."""
    from yb_analysis.analysis.run_analysis import analyze_scan_dir
    d = _scan_dir(tmp_path, F_STAMP)
    data, imgs_path, _imgs = _write_focus_split_scan(d)

    opens = _track_h5_opens(monkeypatch)
    out = analyze_scan_dir(d)

    assert _img_opens(opens, imgs_path) == [], opens
    assert opens                                   # (the data file was read)
    assert out['seq_specific'] is None
    # ...and the un-asked-for compute left no cache behind either.
    assert not os.path.exists(os.path.join(d, 'focus_metrics.json'))


def test_focus_explicit_compute_reads_only_image_file(tmp_path, monkeypatch):
    """compute_focus_metrics=True does the pixel read -- of the IMAGE file --
    and writes the focus_metrics.json cache with a real curve."""
    from yb_analysis.analysis.run_analysis import analyze_scan_dir
    d = _scan_dir(tmp_path, F_STAMP)
    data, imgs_path, _imgs = _write_focus_split_scan(d)

    opens = _track_h5_opens(monkeypatch)
    out = analyze_scan_dir(d, compute_focus_metrics=True)

    # It DID open the image file (that is the point of opting in).
    assert _img_opens(opens, imgs_path), opens
    ss = out['seq_specific']
    assert ss['type'] == 'focus_metrics'
    assert ss['calibration_free'] is True
    vals = np.array(ss['metrics']['spot_width']['values'], dtype=float)
    assert int(np.nanargmin(vals)) == 1            # tight focus = middle point
    assert os.path.isfile(os.path.join(d, 'focus_metrics.json'))


def test_focus_cached_metrics_served_with_zero_image_opens(tmp_path,
                                                          monkeypatch):
    """After an explicit compute, a DEFAULT analyze serves the cached curve --
    still without opening the image file (a JSON read costs nothing)."""
    from yb_analysis.analysis.run_analysis import analyze_scan_dir
    d = _scan_dir(tmp_path, F_STAMP)
    data, imgs_path, _imgs = _write_focus_split_scan(d)

    analyze_scan_dir(d, compute_focus_metrics=True)      # explicit, warms cache
    assert os.path.isfile(os.path.join(d, 'focus_metrics.json'))

    # Bust the payload cache so the default call really rebuilds the payload
    # (otherwise the fast path returns without consulting anything at all).
    payload = os.path.join(d, 'analysis_payload.json')
    if os.path.isfile(payload):
        os.remove(payload)

    opens = _track_h5_opens(monkeypatch)
    out = analyze_scan_dir(d)

    assert _img_opens(opens, imgs_path) == [], opens
    ss = out['seq_specific']
    assert ss['type'] == 'focus_metrics'
    assert ss['metrics']['spot_width']['values']


def test_focus_explicit_compute_refreshes_the_payload_cache(tmp_path):
    """The empty panel must not get PINNED by the payload cache: after the
    explicit compute the cached default payload carries the real curve, so a
    plain re-fetch of the analysis shows it."""
    from yb_analysis.analysis.run_analysis import (
        analyze_scan_dir, ANALYSIS_PAYLOAD_VERSION)
    import json as _json
    d = _scan_dir(tmp_path, F_STAMP)
    _write_focus_split_scan(d)

    first = analyze_scan_dir(d)                     # caches the no-focus payload
    assert first['seq_specific'] is None
    cache_path = os.path.join(d, 'analysis_payload.json')
    with open(cache_path) as f:
        cached = _json.load(f)
    assert cached['_version'] == ANALYSIS_PAYLOAD_VERSION == 6
    assert cached['payload']['seq_specific'] is None

    analyze_scan_dir(d, compute_focus_metrics=True)  # explicit compute

    with open(cache_path) as f:
        cached2 = _json.load(f)
    ss = cached2['payload']['seq_specific']
    assert ss['type'] == 'focus_metrics'
    assert ss['metrics']['spot_width']['values']
    # ...and the plain default call now returns the real curve off that cache.
    again = analyze_scan_dir(d)
    assert again['seq_specific']['metrics']['spot_width']['values']


# ==========================================================================
# STEP 6: image-file ROTATION -- image_<stamp>.h5 past
# config.IMAGE_FILE_ROTATE_GB becomes frozen SEGMENTS behind a small
# virtual-dataset MASTER kept at the same path.
#
# The point of the design is that readers change NOTHING, so that is what
# these tests pin:
#   * the master's /imgs is one contiguous stack whose pixels are right across
#     every seam, read through the ordinary load_images(data_path) redirect;
#   * /seq_ids, /frame_seq_ids and committed_frames on the master are GLOBAL,
#     so min(data seq rows, committed_frames // pSeq) is still THE live rule and
#     stays monotone straight through a rotation;
#   * a segment holds only its own rows and its own LOCAL ids, plus the
#     frame_offset that places them globally;
#   * rotation happens BETWEEN blocks, never mid-handle, and a busy file (a
#     reader blocking the Windows rename) postpones it instead of failing the
#     block;
#   * below the threshold NOTHING changes -- no segments, no master, same attrs.
# ==========================================================================

R_STAMP = '20260820_140000'

# 64x64 uint16 = 8 KB of INCOMPRESSIBLE noise per frame, so a ~150 KB threshold
# is crossed after a couple of blocks. (Uniform frames would gzip to nothing and
# never trigger, which is exactly the trap a "tiny threshold" test can fall in.)
R_H = R_W = 64
R_ROT_BYTES = 150 * 1024


def _rot_frames(tags, pSeq, seed=1, h=R_H, w=R_W):
    """Noise frames, each shot's frames tagged in pixel [0, 0] with its tag."""
    rng = np.random.default_rng(seed)
    blk = rng.integers(0, 65535, size=(len(tags) * pSeq, h, w), dtype=np.uint16)
    for i, t in enumerate(tags):
        blk[i * pSeq:(i + 1) * pSeq, 0, 0] = t
    return blk


def _rot_scan(tmp_path, pSeq=2, stamp=R_STAMP):
    """A split scan sized for rotation; returns (scan_dir, data, image_path)."""
    d = _writer_dir(tmp_path, stamp)
    data, imgs = _writer_paths(d, stamp)
    hs.create_scan_file(data, {'NumImages': pSeq}, (R_H, R_W), 3,
                        two_array=True, num_sites_img2=3, image_path=imgs,
                        num_images_per_seq=pSeq)
    return d, data, imgs


def _rot_blocks(data, imgs, n_blocks, pSeq=2, per_block=3,
                rotate_bytes=R_ROT_BYTES, start=0):
    """Drive n_blocks blocks through the real writer; returns the tag list."""
    tags_all = []
    saved = start
    for b in range(n_blocks):
        tags = [saved + k + 1 for k in range(per_block)]
        _append(data, imgs, True, _rot_frames(tags, pSeq, seed=saved + 1),
                np.array(tags, dtype='int64'), pSeq,
                expected_seq_rows=saved, rotate_bytes=rotate_bytes)
        saved += per_block
        tags_all += tags
    return tags_all


def _segments(imgs_path):
    return [os.path.basename(p) for p in hs.existing_segments(imgs_path)]


def _segment_frames(path):
    with h5py.File(path, 'r') as f:
        return int(f['imgs'].shape[0])


# --------------------------------------------------------------------------
# rotation fires, and what it leaves on disk
# --------------------------------------------------------------------------

def test_rotation_creates_segments_and_a_vds_master(tmp_path):
    """Past the threshold: the old bulk file is renamed to .000, a .001 live
    segment appears, and image_<stamp>.h5 is a small VIRTUAL-dataset master."""
    pSeq = 2
    d, data, imgs = _rot_scan(tmp_path, pSeq)
    tags = _rot_blocks(data, imgs, n_blocks=6, pSeq=pSeq)

    assert _segments(imgs) == ['image_%s.000.h5' % R_STAMP,
                               'image_%s.001.h5' % R_STAMP]
    assert os.path.isfile(imgs)                     # master still at THE path
    assert not os.path.exists(imgs + '.tmp')
    # the master is tiny: no pixels in it, only the mapping + the global ids
    assert os.path.getsize(imgs) < os.path.getsize(hs.segment_path(imgs, 0))

    with h5py.File(imgs, 'r') as f:
        assert f['imgs'].is_virtual
        assert f.attrs['layout'] == 'images-master'
        assert f.attrs['schema_version'] == 1
        assert int(f.attrs['segments']) == 2
        assert f.attrs['live_segment'] == 'image_%s.001.h5' % R_STAMP
        assert int(f.attrs['num_images_per_seq']) == pSeq
        assert tuple(f.attrs['frame_size']) == (R_H, R_W)
        assert f.attrs['data_file'] == os.path.basename(data)
        assert f.attrs['scan_id'] == R_STAMP
        # /seq_ids + /frame_seq_ids are REAL datasets on the master
        assert not f['seq_ids'].is_virtual
        assert not f['frame_seq_ids'].is_virtual
        assert f['imgs'].shape == (len(tags) * pSeq, R_H, R_W)
        # relative source names -> the mapping does not hard-code this tmp path
        srcs = [m.file_name for m in f['imgs'].virtual_sources()]
    assert sorted(srcs) == ['image_%s.000.h5' % R_STAMP,
                            'image_%s.001.h5' % R_STAMP]
    assert not any(os.path.isabs(s) for s in srcs)

    # the frozen segment stops growing; the live one carries the rest
    n0 = _segment_frames(hs.segment_path(imgs, 0))
    n1 = _segment_frames(hs.segment_path(imgs, 1))
    assert n0 + n1 == len(tags) * pSeq
    assert n0 > 0 and n1 > 0


def test_rotated_segments_carry_local_ids_and_frame_offset(tmp_path):
    """A segment describes only ITS rows: local /seq_ids + /frame_seq_ids, a
    LOCAL committed_frames, and the frame_offset that places them globally."""
    pSeq = 2
    d, data, imgs = _rot_scan(tmp_path, pSeq)
    tags = _rot_blocks(data, imgs, n_blocks=6, pSeq=pSeq)

    seg0, seg1 = hs.segment_path(imgs, 0), hs.segment_path(imgs, 1)
    with h5py.File(seg0, 'r') as f:
        n0 = int(f['imgs'].shape[0])
        assert f.attrs['layout'] == 'images'
        assert int(f.attrs['segment_index']) == 0
        assert int(f.attrs['frame_offset']) == 0
        assert int(f.attrs['committed_frames']) == n0
        s0 = f['seq_ids'][:]
        np.testing.assert_array_equal(f['frame_seq_ids'][:],
                                      np.repeat(s0, pSeq))
    with h5py.File(seg1, 'r') as f:
        n1 = int(f['imgs'].shape[0])
        assert int(f.attrs['segment_index']) == 1
        assert int(f.attrs['frame_offset']) == n0       # global placement
        assert int(f.attrs['committed_frames']) == n1   # LOCAL watermark
        s1 = f['seq_ids'][:]
    # local ids concatenate to the global list, in order
    np.testing.assert_array_equal(np.concatenate([s0, s1]), tags)
    assert len(s0) == n0 // pSeq and len(s1) == n1 // pSeq


def test_rotated_master_pixels_are_contiguous_across_the_seam(tmp_path):
    """THE reader guarantee: every frame reads back at its global row through
    the ordinary load_images(data_path) redirect, seam included."""
    pSeq = 2
    d, data, imgs = _rot_scan(tmp_path, pSeq)
    tags = _rot_blocks(data, imgs, n_blocks=7, pSeq=pSeq)
    assert len(_segments(imgs)) >= 2                  # rotation really happened

    expect = np.repeat(np.array(tags, dtype=np.uint16), pSeq)
    got = ld.load_images(data)                        # DATA path, as callers do
    assert got.shape == (len(expect), R_H, R_W)
    np.testing.assert_array_equal(got[:, 0, 0], expect)

    # ... and single-frame / slice / fancy reads straddling the seam agree
    n0 = _segment_frames(hs.segment_path(imgs, 0))
    around = [n0 - 2, n0 - 1, n0, n0 + 1]
    np.testing.assert_array_equal(ld.load_images(data, around)[:, 0, 0],
                                  expect[around])
    np.testing.assert_array_equal(
        ld.load_images(data, slice(n0 - 2, n0 + 2))[:, 0, 0],
        expect[n0 - 2:n0 + 2])
    for i in around:
        np.testing.assert_array_equal(ld.load_images(data, int(i)), got[i])
    # the whole-scan reader path is untouched by rotation
    bundle = ld.load_scan_from_path(d)
    assert bundle['layout'] == sf.LAYOUT_SPLIT
    assert tuple(bundle['imgs_shape']) == (len(expect), R_H, R_W)
    np.testing.assert_array_equal(ld.load_images(bundle['path'])[:, 0, 0],
                                  expect)
    assert ld.get_images_shape(data) == (len(expect), R_H, R_W)


def test_rotated_master_ids_and_watermark_are_global(tmp_path):
    """The master's ids and committed_frames span ALL segments, and the data
    file's /seq_ids still matches the master's row for row."""
    pSeq = 3
    d, data, imgs = _rot_scan(tmp_path, pSeq, stamp='20260820_140001')
    tags = _rot_blocks(data, imgs, n_blocks=6, pSeq=pSeq, per_block=2)
    assert len(_segments(imgs)) >= 2

    with h5py.File(imgs, 'r') as f:
        np.testing.assert_array_equal(f['seq_ids'][:], tags)
        np.testing.assert_array_equal(f['frame_seq_ids'][:],
                                      np.repeat(tags, pSeq))
        assert int(f.attrs['committed_frames']) == len(tags) * pSeq
    with h5py.File(data, 'r') as f:
        np.testing.assert_array_equal(f['seq_ids'][:], tags)
    assert _live_count(data, imgs, pSeq) == len(tags)


def test_second_rotation_adds_a_third_segment(tmp_path):
    """Rotating again needs no rename: .001 freezes, .002 goes live, the master
    is rewritten with two exact extents + one unlimited one."""
    pSeq = 2
    d, data, imgs = _rot_scan(tmp_path, pSeq, stamp='20260820_140002')
    tags = _rot_blocks(data, imgs, n_blocks=14, pSeq=pSeq)

    segs = _segments(imgs)
    assert len(segs) >= 3, segs
    with h5py.File(imgs, 'r') as f:
        assert int(f.attrs['segments']) == len(segs)
        assert f.attrs['live_segment'] == segs[-1]
        assert f['imgs'].shape[0] == len(tags) * pSeq
        assert len(f['imgs'].virtual_sources()) == len(segs)
    # pixels stay right across BOTH seams
    expect = np.repeat(np.array(tags, dtype=np.uint16), pSeq)
    np.testing.assert_array_equal(ld.load_images(data)[:, 0, 0], expect)
    # every frozen segment keeps its own offset
    off = 0
    for i in range(len(segs) - 1):
        with h5py.File(hs.segment_path(imgs, i), 'r') as f:
            assert int(f.attrs['frame_offset']) == off
            off += int(f['imgs'].shape[0])


# --------------------------------------------------------------------------
# the live-reader contract straight through a rotation
# --------------------------------------------------------------------------

def test_min_rule_is_monotone_across_a_rotation_with_a_live_reader(tmp_path):
    """A reader thread polls the documented min() rule while the writer rotates
    underneath it: the count never goes backwards and every counted shot's img1
    row still carries that shot's own tag (no phase shift across the seam)."""
    pSeq, per_block, n_blocks = 2, 2, 10
    d, data, imgs = _rot_scan(tmp_path, pSeq, stamp='20260820_140003')

    stop = threading.Event()
    errors = []

    def _writer():
        try:
            _rot_blocks(data, imgs, n_blocks=n_blocks, pSeq=pSeq,
                        per_block=per_block)
        except Exception as e:                      # noqa: BLE001
            errors.append(e)
        finally:
            stop.set()

    th = threading.Thread(target=_writer, daemon=True)
    th.start()

    counts, seen_rotation = [], False
    deadline = time.time() + 30.0
    while time.time() < deadline:
        try:
            n = _live_count(data, imgs, pSeq)
        except (OSError, KeyError):
            time.sleep(0.005)               # the ms-wide rotation window
            continue
        counts.append(n)
        if hs.existing_segments(imgs):
            seen_rotation = True
        if n:
            try:
                with h5py.File(data, 'r') as f:
                    sids_now = f['seq_ids'][:n]
                with h5py.File(imgs, 'r') as f:
                    for k in (1, max(1, n // 2), n):
                        row = int(f['imgs'][(k - 1) * pSeq, 0, 0])
                        assert row == sids_now[k - 1], (k, n, row)
            except OSError:
                pass
        if stop.is_set() and n >= n_blocks * per_block:
            break
        time.sleep(0.004)
    th.join(timeout=10.0)

    assert not errors, errors
    assert seen_rotation, 'the writer never crossed the threshold'
    assert counts and all(b >= a for a, b in zip(counts, counts[1:])), counts
    assert counts[-1] == n_blocks * per_block


# --------------------------------------------------------------------------
# failure / recovery paths
# --------------------------------------------------------------------------

def test_rotation_postponed_when_the_rename_is_blocked(tmp_path, monkeypatch,
                                                       caplog):
    """A concurrent reader can block the Windows rename. Then rotation is simply
    POSTPONED: a warning, no segment on disk, and the block still lands in the
    (slightly oversized) file -- retried on the next block."""
    pSeq = 2
    d, data, imgs = _rot_scan(tmp_path, pSeq, stamp='20260820_140004')
    tags = _rot_blocks(data, imgs, n_blocks=3, pSeq=pSeq)     # under threshold
    assert _segments(imgs) == []

    seg0 = hs.segment_path(imgs, 0)
    real_replace = os.replace

    def _blocked(src, dst):
        if str(dst) == seg0:
            raise PermissionError(32, 'used by another process')
        return real_replace(src, dst)

    monkeypatch.setattr(hs.os, 'replace', _blocked)
    with caplog.at_level(logging.WARNING, logger='yb_analysis.io.hdf5_store'):
        more = _rot_blocks(data, imgs, n_blocks=3, pSeq=pSeq, start=len(tags))
    monkeypatch.undo()

    assert any('rotation postponed' in r.message for r in caplog.records)
    assert not os.path.exists(seg0)
    assert _segments(imgs) == []
    # nothing leaked: no stray segment, no leftover master tmp
    assert not os.path.exists(hs.segment_path(imgs, 1))
    assert not os.path.exists(imgs + '.tmp')
    # ...and every frame is still there, in the plain (unrotated) file
    all_tags = tags + more
    expect = np.repeat(np.array(all_tags, dtype=np.uint16), pSeq)
    with h5py.File(imgs, 'r') as f:
        assert f.attrs['layout'] == 'images'          # never became a master
        assert not f['imgs'].is_virtual
        assert int(f.attrs['committed_frames']) == len(expect)
    np.testing.assert_array_equal(ld.load_images(data)[:, 0, 0], expect)

    # the NEXT block (unblocked) rotates as normal
    _rot_blocks(data, imgs, n_blocks=1, pSeq=pSeq, start=len(all_tags))
    assert _segments(imgs) == ['image_20260820_140004.000.h5',
                              'image_20260820_140004.001.h5']


def test_self_heal_after_rotation_trims_only_the_live_segment(tmp_path, caplog):
    """A data append that fails right after a rotation leaves orphan rows in the
    LIVE segment; the next block trims exactly those, never a frozen extent, and
    the master's global ids/watermark follow."""
    pSeq = 2
    d, data, imgs = _rot_scan(tmp_path, pSeq, stamp='20260820_140005')
    tags = _rot_blocks(data, imgs, n_blocks=6, pSeq=pSeq)
    assert len(_segments(imgs)) == 2
    seg0, seg1 = hs.segment_path(imgs, 0), hs.segment_path(imgs, 1)
    n0 = _segment_frames(seg0)
    n1_before = _segment_frames(seg1)
    saved = len(tags)

    # images land, data append never happens (what _save_block leaves behind).
    # A huge threshold from here on keeps .001 the live segment, so the trim is
    # observed in isolation from another rotation.
    no_rotate = 1 << 30
    orphan_tags = [saved + 1, saved + 2]
    hs.append_images_block(imgs, _rot_frames(orphan_tags, pSeq, seed=99),
                           np.array(orphan_tags, dtype='int64'), pSeq,
                           expected_seq_rows=saved,
                           rotate_bytes=no_rotate)
    assert _segment_frames(seg1) == n1_before + len(orphan_tags) * pSeq
    assert _live_count(data, imgs, pSeq) == saved      # min() hides them

    # the next healthy block heals the live segment only
    with caplog.at_level(logging.WARNING, logger='yb_analysis.io.hdf5_store'):
        good = _rot_blocks(data, imgs, n_blocks=1, pSeq=pSeq, per_block=2,
                           start=saved, rotate_bytes=no_rotate)
    assert any('live segment' in r.message for r in caplog.records)

    assert _segment_frames(seg0) == n0                 # frozen: untouched
    assert _segment_frames(seg1) == n1_before + len(good) * pSeq
    expect = np.repeat(np.array(tags + good, dtype=np.uint16), pSeq)
    with h5py.File(imgs, 'r') as f:
        assert f['imgs'].shape[0] == len(expect)
        np.testing.assert_array_equal(f['seq_ids'][:], tags + good)
        np.testing.assert_array_equal(f['frame_seq_ids'][:],
                                      np.repeat(tags + good, pSeq))
        assert int(f.attrs['committed_frames']) == len(expect)
    # no phase shift: the healed rows really are the good block's frames
    np.testing.assert_array_equal(ld.load_images(data)[:, 0, 0], expect)


def test_self_heal_with_an_empty_live_segment(tmp_path):
    """A trim demand that arrives when the live segment is EMPTY (rotation just
    happened) must be a no-op, not a raid on the frozen segment."""
    pSeq = 2
    d, data, imgs = _rot_scan(tmp_path, pSeq, stamp='20260820_140006')
    tags = _rot_blocks(data, imgs, n_blocks=6, pSeq=pSeq)
    seg0, seg1 = hs.segment_path(imgs, 0), hs.segment_path(imgs, 1)
    n0, n1 = _segment_frames(seg0), _segment_frames(seg1)

    # Rotate again so the live segment is brand new and empty...
    segs_before = len(_segments(imgs))
    hs._rotate_image_file(imgs, pSeq)
    assert len(_segments(imgs)) == segs_before + 1
    live = hs.segment_path(imgs, segs_before)
    assert _segment_frames(live) == 0

    # ...then heal against the shots that are already saved: nothing to trim.
    hs._heal_orphans_rotated(imgs, hs.existing_segments(imgs), pSeq, len(tags))
    assert _segment_frames(live) == 0
    assert _segment_frames(seg0) == n0 and _segment_frames(seg1) == n1
    with h5py.File(imgs, 'r') as f:
        assert int(f.attrs['committed_frames']) == len(tags) * pSeq
        np.testing.assert_array_equal(f['seq_ids'][:], tags)
        assert f['imgs'].shape[0] == len(tags) * pSeq


def test_master_is_rebuilt_when_it_goes_missing(tmp_path):
    """A crash between the rename and the master publish leaves segments with no
    master. The next append rebuilds it from the segments themselves."""
    pSeq = 2
    d, data, imgs = _rot_scan(tmp_path, pSeq, stamp='20260820_140007')
    tags = _rot_blocks(data, imgs, n_blocks=6, pSeq=pSeq)
    assert len(_segments(imgs)) == 2

    os.remove(imgs)                       # simulate the crash window
    more = _rot_blocks(data, imgs, n_blocks=1, pSeq=pSeq, start=len(tags))

    with h5py.File(imgs, 'r') as f:
        assert f['imgs'].is_virtual
        assert f.attrs['layout'] == 'images-master'
        np.testing.assert_array_equal(f['seq_ids'][:], tags + more)
        assert int(f.attrs['committed_frames']) == \
            (len(tags) + len(more)) * pSeq
    expect = np.repeat(np.array(tags + more, dtype=np.uint16), pSeq)
    np.testing.assert_array_equal(ld.load_images(data)[:, 0, 0], expect)


# --------------------------------------------------------------------------
# below the threshold: nothing whatsoever changes
# --------------------------------------------------------------------------

def test_below_threshold_never_rotates(tmp_path):
    """The 99% case: same single bulk file, real (non-virtual) /imgs, the plain
    'images' layout attr, no segment, no master, no rotation attrs."""
    pSeq = 2
    d, data, imgs = _rot_scan(tmp_path, pSeq, stamp='20260820_140008')
    tags = _rot_blocks(data, imgs, n_blocks=8, pSeq=pSeq,
                       rotate_bytes=64 * 1024 * 1024)      # 64 MB: never hit

    assert _segments(imgs) == []
    assert sorted(os.listdir(d)) == ['data_20260820_140008.h5',
                                     'image_20260820_140008.h5']
    with h5py.File(imgs, 'r') as f:
        assert not f['imgs'].is_virtual
        assert f.attrs['layout'] == 'images'
        assert f['imgs'].chunks == (1, R_H, R_W)
        assert f['imgs'].compression == 'gzip'
        assert f['imgs'].compression_opts == 1
        for k in ('segments', 'live_segment', 'segment_index', 'frame_offset'):
            assert k not in f.attrs
        assert int(f.attrs['committed_frames']) == len(tags) * pSeq
    expect = np.repeat(np.array(tags, dtype=np.uint16), pSeq)
    np.testing.assert_array_equal(ld.load_images(data)[:, 0, 0], expect)


def test_rotation_disabled_by_a_zero_threshold(tmp_path, monkeypatch):
    """IMAGE_FILE_ROTATE_GB <= 0 switches rotation off entirely."""
    from yb_analysis import config as _cfg
    monkeypatch.setattr(_cfg, 'IMAGE_FILE_ROTATE_GB', 0.0, raising=False)
    pSeq = 2
    d, data, imgs = _rot_scan(tmp_path, pSeq, stamp='20260820_140009')
    _rot_blocks(data, imgs, n_blocks=6, pSeq=pSeq, rotate_bytes=None)
    assert _segments(imgs) == []
    assert hs._rotate_limit_bytes(None) is None


def test_config_threshold_default_is_10_gb():
    """The configured default, in bytes, is what an unset override resolves to."""
    from yb_analysis import config as _cfg
    assert _cfg.IMAGE_FILE_ROTATE_GB == 10.0
    assert hs._rotate_limit_bytes(None) == 10.0 * 1024 ** 3
    assert hs._rotate_limit_bytes(2 * 1024 ** 2) == 2 * 1024 ** 2


# --------------------------------------------------------------------------
# ...and the same through a real DataManager save loop
# --------------------------------------------------------------------------

def test_data_manager_save_loop_rotates(tmp_path, monkeypatch):
    """End to end: the DataManager save path (config threshold, no explicit
    rotate_bytes) rotates mid-scan and the scan still reads back as one stack."""
    pSeq = 2
    monkeypatch.setattr(dm_mod._cfg, 'IMAGE_FILE_ROTATE_GB',
                        150 * 1024.0 / 1024 ** 3, raising=False)   # ~150 KB
    # the DataManager default of _split_dm is (4, 4) frames; rotation needs
    # bulk, so build this one around the noise frame size.
    dm, data, imgs = _split_dm(tmp_path, monkeypatch, pSeq=pSeq,
                               frame_size=(R_H, R_W))

    tags_all = []
    for b in range(8):
        tags = [b * 2 + 1, b * 2 + 2]
        blk = _rot_frames(tags, pSeq, seed=b + 1)
        dm._imgs_to_save = [blk[i] for i in range(blk.shape[0])]
        dm._logicals_to_save = [np.ones(3, dtype=bool)
                                for _ in tags for _ in range(pSeq)]
        dm._intensities_to_save = [np.full(3, float(t))
                                   for t in tags for _ in range(pSeq)]
        dm._seq_ids_to_save = list(tags)
        dm.save_data()
        _wait_save_done(dm, len(tags_all) + len(tags))
        tags_all += tags

    assert dm._saved_seq_rows == len(tags_all)
    assert dm._save_health['state'] == 'ok'
    assert len(hs.existing_segments(imgs)) >= 2, \
        sorted(os.listdir(os.path.dirname(imgs)))

    expect = np.repeat(np.array(tags_all, dtype=np.uint16), pSeq)
    with h5py.File(imgs, 'r') as f:
        assert f['imgs'].is_virtual
        assert int(f.attrs['committed_frames']) == len(expect)
        np.testing.assert_array_equal(f['seq_ids'][:], tags_all)
    with h5py.File(data, 'r') as f:
        assert 'imgs' not in f
        np.testing.assert_array_equal(f['seq_ids'][:], tags_all)
    np.testing.assert_array_equal(ld.load_images(data)[:, 0, 0], expect)
    assert _live_count(data, imgs, pSeq) == len(tags_all)
