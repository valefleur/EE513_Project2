"""HW7: Training Without Clean Images

Run:  uv run python hw7.py                       # all frames of the burst set below
      uv run python hw7.py --max-frames 16       # only the first 16 (sorted by filename)
      uv run python hw7.py --burst-dir images/iPhone16/Lightroom/Project2/Still-Life/stable/set3_ISO2000

Stands alone from Project 2 but calls the same my_library functions the pipeline will use.
Methods for problems 1-3 (green sites, phase-correlation alignment, averaging) are specified
by Professor James McNames' EE 513 HW7 assignment; see my_library/burst.py for per-function
attribution. Notebook code from his Lecture 11/12 demos is   credited where it is used.
"""
import argparse
import os

import cv2
import numpy as np

import my_library.burst as bu
import my_library.burst_figures as bf
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
    # Real-data cross-checks: two independent estimators, then the outcome score.
    sk_diff = bu.crosscheck_shifts_skimage(stack, shifts)[1:]
    sk_shifts = shifts[1:] - sk_diff                      # skimage's own (dx, dy)
    ecc_shifts = np.array([bu.estimate_shift_ecc(stack[0], stack[i]) for i in range(1, len(stack))])
    d = lambda p, q: np.abs(p - q)
    print(f"CHECK 02a: shift disagreement between estimators (px, over dx and dy, frames 1..): "
          f"cv2-vs-skimage median {np.median(d(shifts[1:], sk_shifts)):.3f} max {d(shifts[1:], sk_shifts).max():.3f}; "
          f"cv2-vs-ECC median {np.nanmedian(d(shifts[1:], ecc_shifts)):.3f} max {np.nanmax(d(shifts[1:], ecc_shifts)):.3f}; "
          f"skimage-vs-ECC median {np.nanmedian(d(sk_shifts, ecc_shifts)):.3f} max {np.nanmax(d(sk_shifts, ecc_shifts)):.3f}.")
    print(f"          ECC |shift| median {np.nanmedian(np.hypot(*ecc_shifts.T)):.3f} px, "
          f"skimage |shift| median {np.median(np.hypot(*sk_shifts.T)):.3f} px, "
          f"cv2 |shift| median {np.median(mags):.3f} px.")
    # Which shifts actually line frames up? Score by outcome (RMS to frame 0 on smoothed images).
    # The same cv2 call on the full 1512 x 2016 frame (pads to 1536 x 2025, odd width) for contrast.
    full = np.array([bu.estimate_shift(stack[0], stack[i], crop_to_safe_size=False)[:2] for i in range(1, len(stack))])
    all_shifts = {"cv2 full frame (biased)": np.vstack([[0, 0], full]), "cv2 phase corr": shifts, "skimage": np.vstack([[0, 0], sk_shifts]),
                  "ECC": np.vstack([[0, 0], np.nan_to_num(ecc_shifts)])}
    none = np.nanmedian(bu.alignment_residuals(stack, np.zeros_like(shifts)))
    print(f"CHECK 02b: median smoothed RMS difference to frame 0 (lower = better aligned); unaligned {none:.5f}")
    for name, sh in all_shifts.items():
        for iname, interp in (("linear", cv2.INTER_LINEAR), ("cubic", cv2.INTER_CUBIC), ("lanczos4", cv2.INTER_LANCZOS4)):
            r = np.nanmedian(bu.alignment_residuals(stack, sh, interp))
            print(f"          {name:23s} + {iname:8s}: {r:.5f}  ({r / none:.3f} x unaligned)")
    print(f"CHECK 02c: cv2 on the full frame (odd padded width) median dx {np.median(full[:, 0]):+.3f} px vs "
          f"on the DFT-safe center crop {np.median(shifts[1:, 0]):+.3f} px vs scikit-image {np.median(sk_shifts[:, 0]):+.3f} px; "
          f"the +0.5 px difference is cv2's odd-padding bias.")
    print(f"DELIVERABLE {problem:02d}: median shift {np.median(mags):.3f} px, "
          f"largest shift {mags.max():.3f} px (magnitude sqrt(dx^2+dy^2), "
          f"frames 1..{len(stack) - 1} relative to frame 0). "
          f"Weakest correlation peak {np.nanmin(responses):.3f} vs median {np.nanmedian(responses):.3f}.")

    problem = 3
    print(f"\n*~*~*~*~*~*~*~*~*~*~ Problem {problem} ~*~*~*~*~*~*~*~*~*~*")
    """Item 3: average the aligned frames (the clean image) and show one frame next to the
    average, same crop, same display scale, somewhere with fine detail."""
    average = bu.average_burst(aligned)
    margin = bu.valid_margin(shifts)                 # keep reflected-border fill out of the crop
    crop_size = 256
    row, col = bu.find_detail_crop(average, crop_size, margin)
    lo, hi = bf.plot_frame_vs_average(
        stack[0], average, row, col, crop_size,
        f"{OUTPUT_PATH}p{problem:02d}_frame_vs_average.png",
        f"HW7 problem 3: one frame vs the average of {len(stack)} aligned frames "
        f"({burst_path.rstrip('/').split('/')[-1]})", show=SHOW)

    # Real-data checks (not just synthetic unit tests).
    sl = (slice(row, row + crop_size), slice(col, col + crop_size))
    sigma_t = np.sqrt(stack[:, sl[0], sl[1]].astype(np.float64).var(axis=0, ddof=1)).mean() # average of the sqrt of the variation on the crop
    resid = stack[0][sl] - average[sl]
    print(f"CHECK 03a: crop noise std of (frame 0 - average) {resid.std():.5f}; prediction from the "
          f"temporal std of the raw frames {sigma_t:.5f} x sqrt(1-1/N) = {sigma_t * np.sqrt(1 - 1 / len(stack)):.5f}. "
          f"(Temporal std also contains drift and misregistration, so it can only over-predict.)")
    ratios = bu.half_split_noise_check(aligned, margin)
    print(f"CHECK 03b: measured/predicted variance of (mean of half A - mean of half B), 1.0 = pure "
          f"independent noise: even/odd {ratios['even_odd']:.2f}, first/second half {ratios['first_second']:.2f}. "
          f"A first/second value well above even/odd means the scene or exposure drifted during the burst.")
    gains = bu.frame_gains(stack, margin)
    trend = np.polyfit(np.arange(len(gains)), gains, 1)[0] * (len(gains) - 1)
    print(f"CHECK 03c: mean level of each frame relative to frame 0: min {gains.min():.4f}, max {gains.max():.4f}, "
          f"linear trend over the burst {trend:+.4f}. Per frame: " + " ".join(f"{g:.4f}" for g in gains))
    print(f"DELIVERABLE {problem:02d}: figure {OUTPUT_PATH}p{problem:02d}_frame_vs_average.png "
          f"(crop rows {row}:{row + crop_size}, cols {col}:{col + crop_size}; display window {lo:.4f}-{hi:.4f}, "
          f"gamma 2.2; N={len(stack)}).")
