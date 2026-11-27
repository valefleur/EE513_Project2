"""Tests for my_library/burst.py and green_plane_from_raw.  Run: uv run pytest tests -q"""
import types

import cv2
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, shift as nd_shift  # independent shifter

import my_library.burst as bu
import my_library.linear_light as ll

LAYOUTS = {  # color_desc, raw_pattern, indices of the two greens' (row, col) in raster order
    "BGGR": (b"RGBG", [[2, 1], [3, 0]], [(0, 1), (1, 0)]),
    "RGGB": (b"RGBG", [[0, 1], [3, 2]], [(0, 1), (1, 0)]),
    "GRBG": (b"RGBG", [[1, 0], [2, 3]], [(0, 0), (1, 1)]),
}


def fake_raw(layout, shape=(8, 10), black=528, white=4095, seed=0):
    desc, pat, _ = LAYOUTS[layout]
    img = np.random.default_rng(seed).integers(0, 4095, shape).astype(np.uint16)
    return types.SimpleNamespace(color_desc=desc, raw_pattern=np.array(pat), raw_image_visible=img,
                                 black_level_per_channel=[black] * 4, white_level=white)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("which", [0, 1])
def test_green_plane_is_the_hand_sliced_green_site(layout, which):
    raw = fake_raw(layout)
    r, c = LAYOUTS[layout][2][which]
    got = ll.green_plane_from_raw(raw, which, black_level=0, saturation=1, clip=False, verbose=False)
    assert got.shape == (4, 5)
    assert np.array_equal(got, raw.raw_image_visible[r::2, c::2])


def test_defaults_unchanged_declared_black_white_tag_and_clip():
    raw = fake_raw("BGGR", black=528, white=4095)
    got = ll.green_plane_from_raw(raw, verbose=False)
    ref = np.clip((raw.raw_image_visible[0::2, 1::2].astype(float) - 528) / (4095 - 528), 0, 1)
    assert np.allclose(got, ref) and got.min() >= 0 and got.max() <= 1


def test_overrides_use_measured_levels_and_keep_negative_noise():
    raw = fake_raw("BGGR")
    raw.raw_image_visible[:] = 520
    got = ll.green_plane_from_raw(raw, black_level=527.45, saturation=4095, clip=False, verbose=False)
    assert np.allclose(got, (520 - 527.45) / (4095 - 527.45)) and got.max() < 0


def test_load_burst_green_stacks_frames_and_checks_consistency(monkeypatch):
    raws = {f"f{i}.dng": fake_raw("BGGR", seed=i) for i in range(3)}

    class Ctx:
        def __init__(self, raw): self.raw = raw
        def __enter__(self): return self.raw
        def __exit__(self, *a): return False

    monkeypatch.setattr(bu.rawpy, "imread", lambda p: Ctx(raws[p]))
    monkeypatch.setattr(bu.dm, "cfa_pattern", lambda raw: "BGGR")
    stack, info = bu.load_burst_green(list(raws), 4095, black_level=527.45)
    assert stack.shape == (3, 4, 5) and info["n_frames"] == 3 and info["pattern"] == "BGGR"
    raws["f1.dng"].black_level_per_channel = [600] * 4
    with pytest.raises(ValueError):
        bu.load_burst_green(list(raws), 4095, black_level=527.45)


# ---- alignment -------------------------------------------------------------------------


def _scene(seed=0, shape=(240, 320)):
    return gaussian_filter(np.random.default_rng(seed).random(shape), 3).astype(np.float32)


@pytest.mark.parametrize("dx,dy", [(3.3, -2.1), (-0.4, 0.7), (0.0, 0.0), (5.0, 4.0)])
def test_estimate_shift_recovers_known_subpixel_shift(dx, dy):
    ref = _scene()
    img = nd_shift(ref, (dy, dx), order=3, mode="reflect").astype(np.float32)  # content moves +dx,+dy
    ex, ey, resp = bu.estimate_shift(ref, img)
    assert abs(ex - dx) < 0.15 and abs(ey - dy) < 0.15 and resp > 0


def test_estimate_shift_agrees_with_skimage_phase_cross_correlation():
    """Known-good widely used implementation. skimage reports the shift needed to register
    img onto ref (the opposite sign) and needs the same Hann taper to avoid wrap-around."""
    from skimage.registration import phase_cross_correlation
    ref = _scene(1)
    img = nd_shift(ref, (1.6, -2.4), order=3, mode="reflect").astype(np.float32)   # (dy, dx)
    w = np.outer(np.hanning(ref.shape[0]), np.hanning(ref.shape[1])).astype(np.float32)
    sk, _, _ = phase_cross_correlation(ref * w, img * w, upsample_factor=100)      # (row, col)
    ex, ey, _ = bu.estimate_shift(ref, img)
    assert abs(ex - (-sk[1])) < 0.2 and abs(ey - (-sk[0])) < 0.2


def test_align_to_first_undoes_shifts_and_leaves_frame0_alone():
    ref = _scene(2)
    true = [(0, 0), (1.5, -0.8), (-2.2, 1.3)]
    stack = np.stack([nd_shift(ref, (dy, dx), order=3, mode="reflect") for dx, dy in true]).astype(np.float32)
    aligned, shifts, resp = bu.align_to_first(stack)
    assert np.array_equal(aligned[0], stack[0]) and np.all(shifts[0] == 0)
    assert np.allclose(shifts[1:], np.array(true[1:]), atol=0.15)
    m = slice(8, -8), slice(8, -8)                                  # ignore reflected borders
    for i in (1, 2):
        before = np.sqrt(np.mean((stack[i][m] - stack[0][m]) ** 2))
        after = np.sqrt(np.mean((aligned[i][m] - stack[0][m]) ** 2))
        assert after < 0.2 * before   # residual = bilinear resampling error


def test_undo_shift_integer_shift_is_exact():
    img = _scene(3)
    moved = np.roll(img, (2, -3), axis=(0, 1))      # content moves down 2, left 3
    out = bu.undo_shift(moved, -3, 2)               # same (dx, dy) estimate_shift would report
    assert np.allclose(out[10:-10, 10:-10], img[10:-10, 10:-10])


def test_estimate_then_undo_needs_no_sign_juggling():
    ref = _scene(4)
    img = nd_shift(ref, (-1.2, 2.5), order=3, mode="reflect").astype(np.float32)
    dx, dy, _ = bu.estimate_shift(ref, img)
    out = bu.undo_shift(img, dx, dy)
    m = slice(8, -8), slice(8, -8)
    assert np.sqrt(np.mean((out[m] - ref[m]) ** 2)) < 0.2 * np.sqrt(np.mean((img[m] - ref[m]) ** 2))


@pytest.mark.parametrize("f", [0.0, 0.25, 0.5])
def test_fractional_shift_attenuates_white_noise_as_predicted(f):
    """Analytic: bilinear weights (1-f, f) per axis scale white-noise std by
    sqrt((1-f)^2 + f^2) per axis."""
    noise = np.random.default_rng(0).standard_normal((600, 600)).astype(np.float32)
    out = bu.undo_shift(noise, -f, -f)              # content moves +f in x and y
    expected = (1 - f) ** 2 + f ** 2                # per-axis variance factor; 2 axes -> std = factor
    assert abs(out[10:-10, 10:-10].std() - expected) < 0.02


def test_default_border_is_constant_zero_but_ours_is_not():
    """Documents the OpenCV default we deliberately override."""
    a = np.ones((6, 8), np.float32)
    M = np.float32([[1, 0, 2], [0, 1, 0]])
    assert cv2.warpAffine(a, M, (8, 6))[0, 0] == 0                       # OpenCV default
    assert bu.undo_shift(a, -2, 0)[0, 0] == 1                            # reflect keeps values


# ---- averaging, crop choice, real-data checks ----------------------------------------------
def _noisy_stack(n, sigma=0.05, shape=(120, 160), seed=0):
    rng = np.random.default_rng(seed)
    return (_scene(seed, shape)[None] + sigma * rng.standard_normal((n,) + shape)).astype(np.float32)


@pytest.mark.parametrize("n", [4, 16, 64])
def test_average_noise_falls_as_one_over_sqrt_n(n):
    """Analytic: std of the mean of N independent frames is sigma / sqrt(N)."""
    st = _noisy_stack(n, 0.05)
    resid = bu.average_burst(st) - _scene(0, (120, 160))
    assert abs(resid.std() - 0.05 / np.sqrt(n)) < 0.15 * 0.05 / np.sqrt(n)


def test_average_matches_numpy_mean_in_float64():
    st = _noisy_stack(5)
    assert np.allclose(bu.average_burst(st), st.astype(np.float64).mean(0)) and bu.average_burst(st).dtype == np.float64


def test_find_detail_crop_picks_textured_window_and_avoids_clipping():
    img = np.full((200, 300), 0.3, np.float32)
    img[50:114, 180:244] += 0.2 * np.random.default_rng(0).standard_normal((64, 64)).astype(np.float32)
    r, c = bu.find_detail_crop(np.clip(img, 0, 0.9), 64)
    assert 36 <= r <= 64 and 166 <= c <= 194
    img[50:114, 180:244] = 1.0                                         # clipped now
    img[150:180, 20:60] = np.tile([0.2, 0.6], (30, 20))                # fine stripes elsewhere
    r, c = bu.find_detail_crop(img, 32)
    assert r + 32 > 150 and c < 60 and not np.any(img[r:r + 32, c:c + 32] > 0.98)


def test_half_split_ratio_is_one_for_iid_noise_and_flags_drift():
    st = _noisy_stack(32, 0.05)
    r = bu.half_split_noise_check(st)
    assert 0.9 < r["even_odd"] < 1.1 and 0.9 < r["first_second"] < 1.1
    drift = st + (0.05 * np.linspace(0, 1, 32))[:, None, None].astype(np.float32)   # slow brightening
    r = bu.half_split_noise_check(drift)
    assert r["first_second"] > 1.5 and 0.9 < r["even_odd"] < 1.3


def test_crosscheck_shifts_skimage_small_on_clean_known_shifts():
    ref = _scene(5)
    true = [(0, 0), (1.5, -0.8), (-2.2, 1.3)]
    st = np.stack([nd_shift(ref, (dy, dx), order=3, mode="reflect") for dx, dy in true]).astype(np.float32)
    _, shifts, _ = bu.align_to_first(st)
    assert np.abs(bu.crosscheck_shifts_skimage(st, shifts)).max() < 0.25


def test_valid_margin_covers_largest_shift():
    assert bu.valid_margin(np.array([[0, 0], [2.3, -4.6]])) == 6


# ---- third estimator, outcome score, drift -------------------------------------------------
@pytest.mark.parametrize("dx,dy", [(3.3, -2.1), (-0.4, 0.7), (0.5, 0.5)])
def test_ecc_recovers_known_shift_with_same_sign_as_phase_correlation(dx, dy):
    ref = _scene(7)
    ref = (ref - ref.min()) / np.ptp(ref)
    img = nd_shift(ref, (dy, dx), order=3, mode="reflect").astype(np.float32)
    ex, ey = bu.estimate_shift_ecc(ref, img)
    px, py, _ = bu.estimate_shift(ref, img)
    assert abs(ex - dx) < 0.1 and abs(ey - dy) < 0.1
    assert abs(ex - px) < 0.3 and abs(ey - py) < 0.3          # two methods, one sign convention


def test_alignment_residual_drops_with_correct_shifts_and_not_with_wrong_ones():
    ref = _scene(8)
    true = [(0, 0), (1.5, -0.8), (-2.2, 1.3)]
    st = np.stack([nd_shift(ref, (dy, dx), order=3, mode="reflect") for dx, dy in true]).astype(np.float32)
    none = bu.alignment_residuals(st, np.zeros((3, 2)))[1:]
    good = bu.alignment_residuals(st, np.array(true, float))[1:]
    bad = bu.alignment_residuals(st, np.array([(0, 0), (-1.5, 0.8), (2.2, -1.3)], float))[1:]   # wrong sign
    assert np.all(good < 0.3 * none) and np.all(bad > none)


def test_cubic_beats_linear_on_a_fractional_shift():
    """Analytic motivation: bilinear attenuates detail at fractional shifts (cos(pi*nu) at
    half pixel), cubic much less. Both must beat doing nothing."""
    ref = _scene(9, (200, 200))
    img = nd_shift(ref, (0.5, 0.5), order=3, mode="reflect").astype(np.float32)
    err = lambda out: np.sqrt(np.mean((out[8:-8, 8:-8] - ref[8:-8, 8:-8]) ** 2)) # MSE
    lin = err(bu.undo_shift(img, 0.5, 0.5, cv2.INTER_LINEAR))
    cub = err(bu.undo_shift(img, 0.5, 0.5, cv2.INTER_CUBIC))
    assert cub < lin < err(img)


def test_frame_gains_report_scaling_and_trend():
    st = np.stack([np.full((20, 30), 0.5, np.float32) * g for g in (1.0, 1.01, 1.02)])
    assert np.allclose(bu.frame_gains(st), [1.0, 1.01, 1.02])


# ---- cv2.phaseCorrelate quirks (found on real data) -----------------------------------------
def _pc_raw(a, b):
    """cv2.phaseCorrelate exactly as the assignment describes, with no workaround."""
    a, b = a.astype(np.float32).copy(), b.astype(np.float32).copy()     # copies: cv2.phaseCorrelate can modify inputs
    h, w = a.shape
    return cv2.phaseCorrelate(a, b, cv2.createHanningWindow((w, h), cv2.CV_32F))[0]


def test_dft_safe_size_is_even_and_its_own_optimal_size():
    assert bu.dft_safe_size(2016) == 2000 and bu.dft_safe_size(1512) == 1500
    for n in (100, 361, 368, 1000, 1512, 2016):
        m = bu.dft_safe_size(n)
        assert m <= n and m % 2 == 0 and cv2.getOptimalDFTSize(m) == m


@pytest.mark.parametrize("shape", [(256, 368), (368, 256), (368, 368)])
def test_estimate_shift_is_unbiased_where_raw_cv2_is_off_by_half_a_pixel(shape):
    """Sizes 361-375 pad to 375 (odd). Raw cv2 is then off by ~+0.5 px on that axis."""
    ref = _scene(11, shape)
    true = (0.05, -0.05)                                                # (dx, dy)
    img = nd_shift(ref, (true[1], true[0]), order=3, mode="reflect").astype(np.float32)
    rdx, rdy = _pc_raw(ref, img)
    if max(abs(rdx - true[0]), abs(rdy - true[1])) < 0.3:
        pytest.skip("this OpenCV build no longer shows the odd-padded-size bias")
    dx, dy, _ = bu.estimate_shift(ref, img)
    assert abs(dx - true[0]) < 0.1 and abs(dy - true[1]) < 0.1
    assert max(abs(rdx - true[0]), abs(rdy - true[1])) > 0.4            # documents the library quirk


def test_estimate_shift_does_not_modify_its_inputs():
    """OpenCV 5.0's phaseCorrelate multiplies its inputs by the window in place when no
    padding is needed; estimate_shift must protect the caller's frames."""
    ref, img = _scene(12), _scene(13)
    r0, i0 = ref.copy(), img.copy()
    bu.estimate_shift(ref, img)
    bu.estimate_shift(ref, img, crop_to_safe_size=False)
    assert np.array_equal(ref, r0) and np.array_equal(img, i0)
