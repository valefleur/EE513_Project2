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


def test_estimate_noise_model_recovers_known_model():
    """Smooth scene + known Poisson-Gaussian-like noise: the fit must find a and b."""
    rng = np.random.default_rng(16)
    a_true, b_true = 2e-4, 1e-6
    y, x = np.mgrid[0:800, 0:800]
    clean = 0.05 + 0.8 * (x / 800.0)                     # smooth ramp, many signal levels
    noisy = clean + rng.normal(0, 1, clean.shape) * np.sqrt(a_true * clean + b_true)
    fit = dm.estimate_noise_model(noisy)
    assert abs(fit["a"] - a_true) / a_true < 0.05
    assert abs(fit["b"] - b_true) < 2e-6

    # Same noise on a TEXTURED scene: the gradient-cancelling estimator must still hold.
    textured = clean + 0.1 * np.sin(x * y / 12000.0)
    noisy_t = textured + rng.normal(0, 1, clean.shape) * np.sqrt(a_true * textured + b_true)
    fit_t = dm.estimate_noise_model(noisy_t)
    assert abs(fit_t["a"] - a_true) / a_true < 0.10


def test_restore_single_frame_noise_restores_variance():
    """Box-average independent noise (var/4), restore 3/4: total variance back to var."""
    rng = np.random.default_rng(17)
    model = {"a": 0.0, "b": 1e-4}                         # flat noise for a clean check
    ref = 0.5 + rng.normal(0, 1, (600, 600, 3)) * np.sqrt(model["b"])
    ref[..., 1] = 0.5 + rng.normal(0, 1, (600, 600)) * np.sqrt(model["b"] / 2)
    quiet = dm.box_downsample(ref)
    restored = dm.restore_single_frame_noise(quiet, model, rng)
    np.testing.assert_allclose(restored[..., 0].var(), model["b"], rtol=0.05)
    np.testing.assert_allclose(restored[..., 1].var(), model["b"] / 2, rtol=0.05)


def test_noise_gain_values_and_meaning():
    """Exact sums of squared weights, and that they really predict filtered-noise variance."""
    bil, mhc = dm.bilinear_full_kernels(), dm.mhc_full_kernels()
    expected = {"G at R/B": (1 / 4, 36 / 64), "R/B at G, row": (1 / 2, 63.5 / 64),
                "R/B at G, column": (1 / 2, 63.5 / 64), "R at B / B at R": (1 / 4, 61 / 64)}
    for name, (b, m) in expected.items():
        assert np.isclose(dm.noise_gain(bil[name]), b)
        assert np.isclose(dm.noise_gain(mhc[name]), m)
        assert np.isclose(bil[name].sum(), 1.0)
    # Filter white noise (var 1) with the MHC R-at-B kernel: output variance ~ 61/64.
    noise = np.random.default_rng(18).normal(0, 1, (1000, 1000))
    out = correlate(noise, mhc["R at B / B at R"], mode="constant")[3:-3, 3:-3]
    assert abs(out.var() - 61 / 64) < 0.01


@pytest.mark.parametrize("tilt, expect", [(4.0, "vertical"), (86.0, "horizontal")])
def test_find_straight_edges_orientation_and_tilt(tilt, expect):
    """A single straight edge tilted a few degrees from an axis is found and classified."""
    y, x = np.mgrid[0:200, 0:200].astype(float)
    t = np.radians(tilt)                           # edge angle from vertical
    side = (x - 100) * np.cos(t) - (y - 100) * np.sin(t)
    img = np.repeat((0.2 + 0.6 * (side > 0))[..., None], 3, axis=-1)
    best = dm.find_straight_edges(img, size=48)[0]
    assert best["orientation"] == expect
    assert abs(abs(best["tilt_deg"]) - min(tilt, 90 - tilt)) < 2.0
    # The reported center lies on the edge (within a couple of pixels).
    t_r, t_c = best["center_row"], best["center_col"]
    assert abs((t_c - 100) * np.cos(t) - (t_r - 100) * np.sin(t)) < 2.0


def test_magnify_is_exact_blocks():
    img = np.arange(6, dtype=float).reshape(2, 3)
    big = dm.magnify(img, 4)
    assert big.shape == (8, 12)
    for i in range(2):
        for j in range(3):
            assert np.all(big[4 * i:4 * i + 4, 4 * j:4 * j + 4] == img[i, j])
    assert dm.magnify(np.zeros((2, 2, 3)), 3).shape == (6, 6, 3)


def test_find_straight_edges_polarity_separates_edge_from_stripe():
    """One edge -> polarity near 1; a light stripe (two opposite edges) -> near 0, and
    single_edge=True skips the stripe."""
    y, x = np.mgrid[0:200, 0:200].astype(float)
    edge = np.repeat((0.2 + 0.6 * (x > 100))[..., None], 3, axis=-1)
    stripe = np.repeat((0.2 + 0.6 * ((x > 88) & (x < 112)))[..., None], 3, axis=-1)
    e = dm.find_straight_edges(edge, size=48)[0]
    st = dm.find_straight_edges(stripe, size=48)[0]
    assert e["polarity"] > 0.9 and e["single_edge"]
    assert st["polarity"] < 0.1 and not st["single_edge"]
    assert st["orientation"] == "vertical"          # still found, still classified
    assert all(c["single_edge"] for c in dm.find_straight_edges(stripe, size=48,
                                                                single_edge=True))
