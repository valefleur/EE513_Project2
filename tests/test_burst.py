"""Tests for my_library/burst.py and green_plane_from_raw.  Run: uv run pytest tests -q"""
import sys
import types

import numpy as np
import pytest

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

    monkeypatch.setitem(sys.modules, "rawpy", types.SimpleNamespace(imread=lambda p: Ctx(raws[p])))
    monkeypatch.setattr(bu.dm, "cfa_pattern", lambda raw: "BGGR")
    stack, info = bu.load_burst_green(list(raws), 4095, black_level=527.45)
    assert stack.shape == (3, 4, 5) and info["n_frames"] == 3 and info["pattern"] == "BGGR"
    raws["f1.dng"].black_level_per_channel = [600] * 4
    with pytest.raises(ValueError):
        bu.load_burst_green(list(raws), 4095, black_level=527.45)
