"""Tests for my_library/demosaic.py.  Run with:  uv run pytest tests -q"""
import numpy as np
import pytest

import my_library.demosaic as dm


def bilinear_loop(mosaic):
    """
    Slow, obvious reference: for every pixel and color, average the measured
    same-color sites among its 8 neighbors (or itself if measured), using the
    neighbor sets from the lecture table. Only neighbors inside the frame count.

    For each color, go through each pixel.  Each pixel is missing two of its three colors.
    If we need green, interpolate the green on the left, right, top and bottom of the current pixel to get the green value.
    If we need red, interpolate either:
        a) the left and right pixels to calculate green
        b) the four corner pixels to calculate blue
    If we need blue, interpolate either:
        a) the top and bottom pixels to calculate green
        b) the four corners to calculate red
    If at an edge, use fewer pixels in the interpolation.  Do not assume black outside, do not assume wrap, reflect or extend.
    """
    H, W = mosaic.shape
    masks = dm.rggb_masks(mosaic.shape)
    out = np.zeros((H, W, 3))
    cross = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    diag = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for row in range(H):
        for column in range(W):
            for channel in range(3):
                if masks[row, column, channel]:
                    out[row, column, channel] = mosaic[row, column]
                    continue
                # Green missing -> cross neighbors. R/B missing -> whichever of
                # cross/diagonal ring holds that color (cross for G sites, diag for R/B sites).
                ring = cross if (channel == 1 or masks[row, column, 1]) else diag
                vals = [mosaic[row + dr, column + dc] for dr, dc in ring
                        if 0 <= row + dr < H and 0 <= column + dc < W and masks[row + dr, column + dc, channel]]
                out[row, column, channel] = np.mean(vals)
    return out


def test_constant_image_is_reproduced_exactly():
    """A flat color must come back flat everywhere, border included."""
    ref = np.ones((10, 14, 3)) * np.array([0.2, 0.5, 0.8])
    out = dm.bilinear_demosaic(dm.mosaic_reference(ref))
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-15)


def test_measured_sites_untouched():
    rng = np.random.default_rng(1)
    ref = rng.random((12, 16, 3))
    mos = dm.mosaic_reference(ref)
    out = dm.bilinear_demosaic(mos)
    dm.assert_round_trip(out, mos)          # every measured site exactly preserved


@pytest.mark.parametrize("shape", [(2, 2), (4, 6), (22, 14), (7, 9), (3, 2)])
def test_matches_slow_loop(shape):
    """Vectorized version agrees with the pixel loop, borders and corners included."""
    rng = np.random.default_rng(2)
    mos = dm.mosaic_reference(rng.random((*shape, 3)))
    np.testing.assert_allclose(dm.bilinear_demosaic(mos), bilinear_loop(mos), atol=1e-12)


def test_linear_ramp_exact_in_interior():
    """Bilinear reproduces a linear ramp exactly away from the border."""
    y, x = np.mgrid[0:20, 0:24].astype(float)
    ramp = (0.01 * x + 0.02 * y)[..., None] * np.ones(3)
    out = dm.bilinear_demosaic(dm.mosaic_reference(ramp))
    np.testing.assert_allclose(out[1:-1, 1:-1], ramp[1:-1, 1:-1], atol=1e-12)


def test_cfa_offset_table():
    for pat, off in [("RGGB", (0, 0)), ("GRBG", (0, 1)), ("GBRG", (1, 0)), ("BGGR", (1, 1))]:
        assert dm.rggb_offset(pat) == off
