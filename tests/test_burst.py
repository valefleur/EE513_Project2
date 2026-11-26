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
