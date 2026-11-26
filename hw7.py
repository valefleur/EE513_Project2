"""HW7: Training Without Clean Images

Run:  uv run python hw7.py                       # all frames of the burst set below
      uv run python hw7.py --max-frames 16       # only the first 16 (sorted by filename)
      uv run python hw7.py --burst-dir images/iPhone16/Lightroom/Project2/Still-Life/stable/set3_ISO2000

Stands alone from Project 2 but calls the same my_library functions the pipeline will use.
Bayer slicing and burst-averaging ideas follow Professor James McNames' EE 513 demos
(11_deep_learning_on_ramp_b.ipynb section 1; 12_learned_restoration_and_isps_a.ipynb).
"""
import argparse
import os

import numpy as np

import my_library.burst as bu
import my_library.image_metadata as im
from my_library.camera_specs import IPHONE_16_PRO_MAX as cam_spec

INPUT_PATH = "images/iPhone16/Lightroom/"
OUTPUT_PATH = "output/hw7/"

# All bursts live under one folder; change only the last directory to switch ISO sets.
hw7_input_path = f"{INPUT_PATH}Project2/Still-Life/stable/"
default_burst_path = f"{hw7_input_path}set3_ISO2000"

# Saturation and black level MEASURED in Project 1 (results.json), not the file tags.
# Saturation: R, G1, G2 all clip at 4095 in the histogram tails (same as WhiteLevel here).
# Black level: 527.45 +/- 0.05 DN from the zero-exposure intercept; the files declare 528.
SATURATION_DN = cam_spec.saturation_dn          # 4095
BLACK_LEVEL_DN = 527.45       # Measured in Project 1

SHOW = False

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--burst-dir", default=default_burst_path)
    ap.add_argument("--black_level", default=BLACK_LEVEL_DN, type=float, help="black level in DN")
    ap.add_argument("--max-frames", type=int, default=None, help="limit to the first N frames")
    a = ap.parse_args()
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    burst_path = a.burst_dir
    max_frames = a.max_frames
    black_level = a.black_level

    problem = 1
    print(f"\n*~*~*~*~*~*~*~*~*~*~ Problem {problem} ~*~*~*~*~*~*~*~*~*~*")
    """Item 1: load a burst (all frames, or the first max_frames) and take the same green
    sites from each."""
    paths = sorted(im.get_image_paths(burst_path))
    if not paths:
        raise SystemExit(f"{burst_path}: no raw files found")
    if max_frames is not None:
        paths = paths[:max_frames]
    im.assert_one_camera(paths)                      # same lens_id as Project 1
    isos = sorted({ int(im.raw_metadata(p).iso) for p in paths })

    stack, info = bu.load_burst_green(paths, SATURATION_DN, which=0, black_level=black_level)

    print(f"burst {burst_path}: ISO {isos}, CFA {info['pattern']}, "
          f"black {info['black_level']} DN, saturation {info['saturation']} DN")
    print(f"DELIVERABLE {problem:02d}: {info['n_frames']} frames; one half-size green image is "
          f"{info['shape'][0]} rows x {info['shape'][1]} columns (raw mosaic {info['raw_shape']}). "
          f"Value range {stack.min():.4f} to {stack.max():.4f}.")
    problem = 2
    print(f"\n*~*~*~*~*~*~*~*~*~*~ Problem {problem} ~*~*~*~*~*~*~*~*~*~*")
    """Item 2: align every frame to the first (phase correlation on a Hann-windowed pair,
    then warpAffine back). Report the median and largest shift."""
    aligned, shifts, responses = bu.align_to_first(stack)

    # Frame 0 is the reference, so its zero shift is not a measurement; leave it out.
    mags = np.hypot(shifts[1:, 0], shifts[1:, 1])
    for i in range(1, len(stack)):
        print(f"  frame {i:2d}: dx={shifts[i, 0]:+.3f} dy={shifts[i, 1]:+.3f} px "
              f"|shift|={mags[i - 1]:.3f} px  peak={responses[i]:.3f}")
    print(f"DELIVERABLE {problem:02d}: median shift {np.median(mags):.3f} px, "
          f"largest shift {mags.max():.3f} px (magnitude sqrt(dx^2+dy^2), "
          f"frames 1..{len(stack) - 1} relative to frame 0). "
          f"Weakest correlation peak {np.nanmin(responses):.3f} vs median {np.nanmedian(responses):.3f}.")
