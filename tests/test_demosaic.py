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


# ---------------------------- MHC ----------------------------
from scipy.ndimage import correlate

# The eight full kernels as printed in Malvar, He & Cutler (2004), x 1/8.
_G_AT_RB = np.array([[0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [-1, 2, 4, 2, -1],
                     [0, 0, 2, 0, 0], [0, 0, -1, 0, 0]]) / 8
_ROW = np.array([[0, 0, .5, 0, 0], [0, -1, 0, -1, 0], [-1, 4, 5, 4, -1],
                 [0, -1, 0, -1, 0], [0, 0, .5, 0, 0]]) / 8
_DIAG = np.array([[0, 0, -1.5, 0, 0], [0, 2, 0, 2, 0], [-1.5, 0, 6, 0, -1.5],
                  [0, 2, 0, 2, 0], [0, 0, -1.5, 0, 0]]) / 8


def mhc_from_paper(mosaic):
    """Independent MHC: apply the paper's full kernels directly (no bilinear split)."""
    s = dm.rggb_site_masks(mosaic.shape)
    f = lambda k: correlate(mosaic, k, mode="constant")
    g, row, col, diag = f(_G_AT_RB), f(_ROW), f(_ROW.T), f(_DIAG)
    R = np.select([s["R"], s["G1"], s["G2"], s["B"]], [mosaic, row, col, diag])
    G = np.select([s["G1"] | s["G2"], s["R"] | s["B"]], [mosaic, g])
    B = np.select([s["B"], s["G2"], s["G1"], s["R"]], [mosaic, row, col, diag])
    return np.stack([R, G, B], axis=-1)


def test_mhc_matches_paper_kernels_in_interior():
    rng = np.random.default_rng(3)
    mos = dm.mosaic_reference(rng.random((30, 34, 3)))
    np.testing.assert_allclose(dm.mhc_demosaic(mos)[2:-2, 2:-2],
                               mhc_from_paper(mos)[2:-2, 2:-2], atol=1e-12)


def test_mhc_border_falls_back_to_bilinear():
    rng = np.random.default_rng(4)
    mos = dm.mosaic_reference(rng.random((16, 18, 3)))
    edge = np.ones(mos.shape, bool); edge[2:-2, 2:-2] = False
    np.testing.assert_array_equal(dm.mhc_demosaic(mos)[edge], dm.bilinear_demosaic(mos)[edge])


@pytest.mark.parametrize("shape", [(10, 14), (7, 9)])
def test_mhc_constant_image_exact(shape):
    ref = np.ones((*shape, 3)) * np.array([0.2, 0.5, 0.8])
    np.testing.assert_allclose(dm.mhc_demosaic(dm.mosaic_reference(ref)), ref, atol=1e-14)


def test_mhc_measured_sites_untouched():
    rng = np.random.default_rng(5)
    mos = dm.mosaic_reference(rng.random((12, 16, 3)))
    dm.assert_round_trip(dm.mhc_demosaic(mos), mos)


def _shared_detail_errors(channel_gains):
    y, x = np.mgrid[0:64, 0:64].astype(float)
    gray = 0.5 + 0.4 * np.sin(x * y / 400.0)
    ref = gray[..., None] * np.asarray(channel_gains, dtype=float)
    mos = dm.mosaic_reference(ref)
    err = lambda out: np.mean((out[2:-2, 2:-2] - ref[2:-2, 2:-2]) ** 2)
    return err(dm.bilinear_demosaic(mos)), err(dm.mhc_demosaic(mos))


def test_mhc_beats_bilinear_on_shared_detail():
    """Gray texture, channels equal (as after white balance): the case MHC is built for.
    Measured: MHC error is about 0.34x bilinear's."""
    bil, mhc = _shared_detail_errors([1.0, 1.0, 1.0])
    assert mhc < 0.5 * bil


def test_mhc_gain_shrinks_without_white_balance():
    """Same texture with unequal channel gains (no white balance): MHC still wins, by less
    (about 0.64x), because the Laplacian it borrows has the wrong amplitude. This is one
    reason white balance goes before demosaicing."""
    bil_wb, mhc_wb = _shared_detail_errors([1.0, 1.0, 1.0])
    bil, mhc = _shared_detail_errors([0.6, 1.0, 0.7])
    assert mhc < bil
    assert mhc / bil > mhc_wb / bil_wb
