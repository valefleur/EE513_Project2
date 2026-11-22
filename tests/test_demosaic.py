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


def test_full_kernels_match_paper_and_rotation():
    k = dm.mhc_full_kernels()
    np.testing.assert_allclose(k["G at R/B"], _G_AT_RB, atol=1e-15)
    np.testing.assert_allclose(k["R/B at G, row"], _ROW, atol=1e-15)
    np.testing.assert_allclose(k["R/B at G, column"], _ROW.T, atol=1e-15)
    np.testing.assert_allclose(k["R at B / B at R"], _DIAG, atol=1e-15)
    np.testing.assert_allclose(k["R/B at G, column"], np.rot90(k["R/B at G, row"]), atol=1e-15)
    for name, kern in k.items():
        assert np.isclose(kern.sum(), 1.0), name


def test_latex_numbers_are_exact():
    import my_library.latex_formatter as lf
    assert [lf.latex_number(v) for v in (4, -1, 0, 0.5, -1.5)] == \
        ["4", "-1", "0", r"\tfrac{1}{2}", r"-\tfrac{3}{2}"]


# ------------- against Professor McNames' reference implementations -------------
def test_matches_lecture7_reference_implementations():
    """make_figures_07.py (Prof. James McNames) is the trusted implementation.
    Bilinear must match everywhere; MHC must match away from the 2-px border, where
    the two border rules differ by design (see hw5.py Problem 10). Theirs clips to
    [0, 1], so mine is clipped before comparing."""
    mf = pytest.importorskip("make_figures_07")
    rng = np.random.default_rng(10)
    mos = dm.mosaic_reference(rng.random((40, 52, 3)))
    bil = dm.compare_demosaic(dm.bilinear_demosaic(mos), mf.bilinear_demosaic(mos))
    assert bil["max_all"] < 1e-12
    mhc = dm.compare_demosaic(dm.mhc_demosaic(mos, clip=True), mf.mhc_demosaic(mos))
    assert mhc["max_interior"] < 1e-12


def test_mhc_border_difference_is_only_the_border_rule():
    """Problem 11: separate a border disagreement from an algorithm disagreement.
    Pad the mosaic by 2 with np.pad 'reflect', which mirrors ABOUT the edge pixel
    (row -1 = row 1, row -2 = row 2) and so keeps every padded site its correct Bayer
    color. Both implementations then see the same, phase-correct context, every
    original pixel is 'interior', and they must agree to rounding everywhere,
    edge rows included."""
    mf = pytest.importorskip("make_figures_07")
    rng = np.random.default_rng(11)
    mos = dm.mosaic_reference(rng.random((40, 52, 3)))
    p = 2
    padded = np.pad(mos, p, mode="reflect")
    crop = (slice(p, -p), slice(p, -p))
    mine = dm.mhc_demosaic(padded, clip=True)[crop]
    theirs = mf.mhc_demosaic(padded)[crop]
    assert dm.compare_demosaic(mine, theirs)["max_all"] < 1e-12
    # And without the shared padding the edge DOES disagree, so the test is not vacuous.
    unpadded = dm.compare_demosaic(dm.mhc_demosaic(mos, clip=True), mf.mhc_demosaic(mos))
    assert unpadded["max_all"] > 1e-3 and unpadded["in_border"]


def test_mhc_clip_flag():
    """Default keeps the overshoot; clip=True is exactly np.clip of the default."""
    rng = np.random.default_rng(12)
    mos = dm.mosaic_reference(rng.random((30, 34, 3)))
    raw = dm.mhc_demosaic(mos)
    assert raw.min() < 0 or raw.max() > 1          # random data overshoots
    np.testing.assert_array_equal(dm.mhc_demosaic(mos, clip=True), np.clip(raw, 0, 1))


def test_measured_site_check_detects_a_change():
    """The checker itself must catch a single altered site, or a PASS means nothing."""
    rng = np.random.default_rng(13)
    mos = dm.mosaic_reference(rng.random((8, 10, 3)))
    out = dm.bilinear_demosaic(mos)
    assert dm.measured_site_check(out, mos)["exact"]
    out[0, 0, 0] += 1e-15                      # R site, top-left corner
    res = dm.measured_site_check(out, mos)
    assert res["changed"] == 1 and res["on_edge"] == 1 and not res["exact"]


def test_latex_sci_and_table_alignment():
    import my_library.latex_formatter as lf
    assert lf.latex_sci(0) == "0"
    assert lf.latex_sci(5.551e-17) == r"$5.55 \times 10^{-17}$"
    t = lf.generate_table([["a"], ["1"]], ["x", "y"], VERBOSE=False, align="lr")
    assert t.startswith(r"\begin{tabular}{l | r}")
    # default unchanged
    assert lf.generate_table([["a"], ["1"]], ["x", "y"], VERBOSE=False).startswith(
        r"\begin{tabular}{l | l}")


def test_psnr_known_values_and_margin():
    ref = np.zeros((10, 10))
    est = np.full((10, 10), 0.1)                     # MSE 0.01 -> 20 dB at peak 1
    assert np.isclose(dm.psnr(est, ref), 20.0)
    assert np.isclose(dm.psnr(est * 0.5, ref), 20.0 + 20 * np.log10(2))   # +6.02 dB
    est2 = est.copy(); est2[0, :] = 1.0              # damage only the top row
    assert np.isclose(dm.psnr(est2, ref, margin=1), 20.0)
    assert dm.psnr(ref, ref) == float("inf")


def test_psnr_matches_lecture_helper_on_in_range_data():
    """Agrees with psu.psnr (Prof. McNames) when nothing needs clipping."""
    import psu_imaging as psu
    rng = np.random.default_rng(14)
    a, b = rng.random((20, 20, 3)), rng.random((20, 20, 3))
    assert np.isclose(dm.psnr(a, b), psu.psnr(a, b))


def test_box_downsample_shape_values_and_noise():
    img = np.arange(5 * 7 * 3, dtype=float).reshape(5, 7, 3)
    out = dm.box_downsample(img)
    assert out.shape == (2, 3, 3)                         # odd row/col dropped
    np.testing.assert_allclose(out[0, 0], img[:2, :2].mean(axis=(0, 1)))
    # Independent noise: std halves (4 samples averaged), within sampling error.
    noise = np.random.default_rng(15).normal(0, 1.0, (800, 800))
    assert abs(dm.box_downsample(noise).std() - 0.5) < 0.01
