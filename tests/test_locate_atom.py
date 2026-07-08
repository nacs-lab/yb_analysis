"""Correctness of the vectorized locate_atom_update heatmap.

The vectorized path (FFT correlation + gather) replaced a triple Python loop
that cost ~10 s per call at 1068 sites / R=12 and starved the acquisition
loop at every scan boundary. These tests pin the new path against a verbatim
copy of the original loop: same heatmap (to FFT rounding, ~1e-9 abs), same
argmax peak, and the same final rounded (dy, dx) shift — including the skip
semantics for patches that fall off the image edge.
"""

import numpy as np
import pytest

from yb_analysis.detection.locate_atom import locate_atom_update


def _reference_heatmap(images, grid_locations, search_range, mask_mat):
    """Verbatim copy of the original (pre-vectorization) heatmap loop."""
    images = np.asarray(images, dtype=np.float64)
    grid_locations = np.asarray(grid_locations, dtype=np.float64)
    mask_mat = np.asarray(mask_mat, dtype=np.float64)

    avg_image = images.mean(axis=0)
    R = int(round(search_range))
    H, W = avg_image.shape
    num_sites = grid_locations.shape[0]
    box_size = mask_mat.shape[0]
    half_box = box_size // 2

    heatmap = np.zeros((2 * R + 1, 2 * R + 1))
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            total_intensity = 0.0
            valid_count = 0
            for s in range(num_sites):
                ny = grid_locations[s, 0] + dy
                nx = grid_locations[s, 1] + dx
                y_min = int(round(ny)) - half_box
                y_max = int(round(ny)) + half_box
                x_min = int(round(nx)) - half_box
                x_max = int(round(nx)) + half_box
                if y_min < 0 or x_min < 0 or y_max >= H or x_max >= W:
                    continue
                patch = avg_image[y_min:y_max + 1, x_min:x_max + 1]
                if patch.shape != mask_mat.shape:
                    continue
                total_intensity += np.sum(patch * mask_mat) / mask_mat.size
                valid_count += 1
            if valid_count > 0:
                heatmap[dy + R, dx + R] = total_intensity / valid_count
    return heatmap


def _gaussian_mask(box_size, sigma):
    from scipy.ndimage import gaussian_filter
    mask = np.zeros((box_size, box_size))
    mask[box_size // 2, box_size // 2] = 1.0
    return gaussian_filter(mask, sigma)


def _synthetic_scene(rng, n_sites, H=220, W=260, shift=(2, -1), spacing=14):
    """Pedestal + Gaussian spots on a grid, atoms displaced by `shift`."""
    side = int(np.ceil(np.sqrt(n_sites)))
    ys, xs = np.mgrid[0:side, 0:side]
    grid = np.column_stack([
        20 + ys.ravel()[:n_sites] * spacing + rng.uniform(-0.5, 0.5, n_sites),
        20 + xs.ravel()[:n_sites] * spacing + rng.uniform(-0.5, 0.5, n_sites),
    ])
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.full((H, W), 200.0)
    for (gy, gx) in grid:
        if rng.random() < 0.6:  # loaded site
            img += 30.0 * np.exp(-(((yy - gy - shift[0]) ** 2)
                                   + ((xx - gx - shift[1]) ** 2)) / (2 * 2.0 ** 2))
    imgs = img[None] + rng.normal(0, 2.0, size=(8, H, W))
    return imgs, grid


@pytest.mark.parametrize('seed,R', [(0, 10), (1, 12), (2, 4)])
def test_matches_reference_loop(seed, R):
    rng = np.random.default_rng(seed)
    imgs, grid = _synthetic_scene(rng, n_sites=40)
    mask = _gaussian_mask(11, 2.0)

    ref_heat = _reference_heatmap(imgs, grid, R, mask)
    new_grid, avg, dy, dx, heat = locate_atom_update(imgs, grid, R, mask)

    np.testing.assert_allclose(heat, ref_heat, rtol=0, atol=1e-8)
    assert np.argmax(heat) == np.argmax(ref_heat)
    # The planted shift is (2, -1); the reference pipeline must agree on the
    # final rounded shift, and it should recover the plant.
    assert (dy, dx) == (2, -1)
    np.testing.assert_array_equal(new_grid, np.asarray(grid) + [[dy, dx]])


def test_edge_sites_skip_semantics():
    """Sites whose shifted patch falls off the image must be excluded from
    the sum AND the count for exactly the shifts where they fall off."""
    rng = np.random.default_rng(3)
    imgs, grid = _synthetic_scene(rng, n_sites=30)
    # Push several sites right up against (and past) the edges.
    grid[0] = [3.2, 30.0]        # top edge: OOB for dy <= ~2
    grid[1] = [216.7, 40.0]      # bottom edge
    grid[2] = [50.0, 2.1]        # left edge
    grid[3] = [60.0, 257.9]      # right edge
    grid[4] = [-4.0, 100.0]      # fully outside
    mask = _gaussian_mask(11, 2.0)

    R = 8
    ref_heat = _reference_heatmap(imgs, grid, R, mask)
    _, _, dy, dx, heat = locate_atom_update(imgs, grid, R, mask)
    np.testing.assert_allclose(heat, ref_heat, rtol=0, atol=1e-8)
    assert np.argmax(heat) == np.argmax(ref_heat)


def test_half_integer_rounding_matches():
    """round(g + d) uses half-to-even; it is NOT always round(g) + d.
    Exact-.5 site coordinates exercise that difference."""
    rng = np.random.default_rng(4)
    imgs, grid = _synthetic_scene(rng, n_sites=25)
    grid = np.floor(grid) + 0.5   # every coordinate exactly on .5
    mask = _gaussian_mask(11, 2.0)

    ref_heat = _reference_heatmap(imgs, grid, 6, mask)
    _, _, _, _, heat = locate_atom_update(imgs, grid, 6, mask)
    np.testing.assert_allclose(heat, ref_heat, rtol=0, atol=1e-8)


def test_degenerate_mask_gives_zero_heatmap():
    """Even-sized mask: the original loop's patch-shape check skipped every
    patch, leaving an all-zero heatmap. Preserve that."""
    rng = np.random.default_rng(5)
    imgs, grid = _synthetic_scene(rng, n_sites=10)
    mask = np.ones((10, 10))

    ref_heat = _reference_heatmap(imgs, grid, 5, mask)
    _, _, _, _, heat = locate_atom_update(imgs, grid, 5, mask)
    assert not ref_heat.any()
    np.testing.assert_array_equal(heat, ref_heat)


def test_image_smaller_than_mask():
    """Image smaller than the mask: every patch OOB -> zero heatmap (the
    FFT path must not be reached / must not raise)."""
    imgs = np.full((3, 8, 8), 200.0)
    grid = np.array([[4.0, 4.0]])
    mask = _gaussian_mask(11, 2.0)
    _, _, _, _, heat = locate_atom_update(imgs, grid, 3, mask)
    assert heat.shape == (7, 7)
    assert not heat.any()


def test_no_sites():
    imgs = np.full((2, 60, 60), 200.0)
    grid = np.zeros((0, 2))
    _, _, _, _, heat = locate_atom_update(imgs, grid, 5, _gaussian_mask(11, 2.0))
    assert not heat.any()
