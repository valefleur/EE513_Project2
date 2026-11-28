"""HW7: Training Without Clean Images

Run:  uv run python hw7.py                       # all frames of the burst set below
      uv run python hw7.py --max-frames 16       # only the first 16 (sorted by filename)
      uv run python hw7.py --burst-dir images/iPhone16/Lightroom/Project2/Still-Life/stable/set3_ISO2000

Stands alone from Project 2 but calls the same my_library functions the pipeline will use.
Methods for problems 1-3 (green sites, phase-correlation alignment, averaging) are specified
by Professor James McNames' EE 513 HW7 assignment; see my_library/burst.py for per-function
attribution. Notebook code from his Lecture 11/12 demos is credited where it is used.
"""
import argparse
import os

import cv2
import numpy as np

import my_library.burst as bu
import my_library.burst_figures as bf
import my_library.image_metadata as im
import my_library.report as report
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

    report.say(f"Burst folder: {burst_path}")
    report.key_values([("ISO", ", ".join(map(str, isos)), ""),
                   ("Bayer pattern (CFA)", info["pattern"], ""),
                   ("black level (measured, Project 1)", f"{info['black_level']}", "DN"),
                   ("saturation (measured, Project 1)", f"{info['saturation']:.0f}", "DN")])
    report.banner("DELIVERABLE", f"{problem:02d}", "frames and half-size green image")
    report.key_values([("frames in the burst", f"{info['n_frames']}", ""),
                   ("height of one half-size image", f"{info['shape'][0]}", "rows"),
                   ("width of one half-size image", f"{info['shape'][1]}", "columns"),
                   ("raw mosaic (height x width)", f"{info['raw_shape'][0]} x {info['raw_shape'][1]}", "pixels"),
                   ("value range after normalization", f"{stack.min():.4f} to {stack.max():.4f}", "(0 = black, 1 = saturation)")])

    problem = 2
    print(f"\n*~*~*~*~*~*~*~*~*~*~ Problem {problem} ~*~*~*~*~*~*~*~*~*~*")
    """Item 2: align every frame to the first (phase correlation on a Hann-windowed pair,
    then warpAffine back). Report the median and largest shift."""
    n_frames = len(stack)
    aligned, shifts, responses = bu.align_to_first(stack)           # cv2 on the DFT-safe center crop
    cv2_shifts = shifts[1:]                                          # frame 0 is the reference
    mag = lambda v: np.hypot(v[:, 0], v[:, 1])

    # Independent estimators for cross-checks, plus cv2 on the full frame to expose its bias.
    sk_shifts = cv2_shifts - bu.crosscheck_shifts_skimage(stack, shifts)[1:]
    ecc_shifts = np.array([bu.estimate_shift_ecc(stack[0], stack[i]) for i in range(1, n_frames)])
    full = np.array([bu.estimate_shift(stack[0], stack[i], crop_to_safe_size=False)[:2]
                     for i in range(1, n_frames)])
    # Per-frame uncertainty: how far the other two estimators land from cv2 (vector length, px).
    disagree = np.nanmax([mag(cv2_shifts - sk_shifts), mag(cv2_shifts - ecc_shifts)], axis=0)

    report.say("Shift of each frame relative to frame 0, in pixels (+dx = content moved right, "
           "+dy = content moved down). cv2 is the assignment's method (DFT-safe crop); scikit-image "
           "and ECC are independent cross-checks.")
    rows = [[i + 1, f"{cv2_shifts[i, 0]:+.3f}", f"{cv2_shifts[i, 1]:+.3f}",
             f"{sk_shifts[i, 0]:+.3f}", f"{sk_shifts[i, 1]:+.3f}",
             f"{ecc_shifts[i, 0]:+.3f}", f"{ecc_shifts[i, 1]:+.3f}",
             f"{mag(cv2_shifts[i:i + 1])[0]:.3f}", f"{disagree[i]:.3f}", f"{responses[i + 1]:.3f}"]
            for i in range(n_frames - 1)]
    report.table(["frame", "cv2 dx", "cv2 dy", "sk dx", "sk dy", "ECC dx", "ECC dy", "|cv2|", "spread", "peak"],
             rows, indent=2, left=())
    report.say("Units: px. |cv2| = length of cv2's shift; spread = largest disagreement of scikit-image or "
           "ECC from cv2; peak = cv2 correlation peak (unitless).", indent=2)

    report.banner("CHECK", "02a", "do three independent estimators agree?")
    report.table(["estimator pair", "median |difference| [px]", "largest |difference| [px]"], [
        ["cv2 vs scikit-image", f"{np.median(mag(cv2_shifts - sk_shifts)):.3f}", f"{mag(cv2_shifts - sk_shifts).max():.3f}"],
        ["cv2 vs ECC", f"{np.nanmedian(mag(cv2_shifts - ecc_shifts)):.3f}", f"{np.nanmax(mag(cv2_shifts - ecc_shifts)):.3f}"],
        ["scikit-image vs ECC", f"{np.nanmedian(mag(sk_shifts - ecc_shifts)):.3f}", f"{np.nanmax(mag(sk_shifts - ecc_shifts)):.3f}"]])
    print()
    report.table(["estimator", "median |shift| " + report.PM + " half-IQR over frames [px]"], [
        ["cv2 (DFT-safe crop)", report.pm(np.median(mag(cv2_shifts)), bu.half_iqr(mag(cv2_shifts)))],
        ["scikit-image", report.pm(np.median(mag(sk_shifts)), bu.half_iqr(mag(sk_shifts)))],
        ["ECC", report.pm(np.nanmedian(mag(ecc_shifts)), bu.half_iqr(mag(ecc_shifts)))]])

    report.banner("CHECK", "02b", "which shifts line the frames up best?")
    report.say("Score = RMS difference to frame 0 after undoing the shifts, on Gaussian-smoothed images "
           "(linear light, 0-1 scale). Lower is better; the " + report.PM + " is half the interquartile range "
           "over frames. 'Biased' = cv2 on the full frame without the DFT-safe crop.", indent=2)
    none = bu.alignment_residuals(stack, np.zeros_like(shifts))
    rows = [["no alignment", "-", report.pm(np.nanmedian(none), bu.half_iqr(none), 5), "1.000"]]
    all_shifts = {"cv2, full frame (biased)": np.vstack([[0, 0], full]),
                  "cv2, DFT-safe crop": shifts,
                  "scikit-image": np.vstack([[0, 0], sk_shifts]),
                  "ECC": np.vstack([[0, 0], np.nan_to_num(ecc_shifts)])}
    for name, sh in all_shifts.items():
        for iname, interp in (("linear", cv2.INTER_LINEAR), ("cubic", cv2.INTER_CUBIC),
                              ("lanczos4", cv2.INTER_LANCZOS4)):
            r = bu.alignment_residuals(stack, sh, interp)
            rows.append([name, iname, report.pm(np.nanmedian(r), bu.half_iqr(r), 5),
                         f"{np.nanmedian(r) / np.nanmedian(none):.3f}"])
        rows.append(None)
    report.table(["shifts from", "interpolation", "RMS difference " + report.PM + " spread", "vs. unaligned"],
             rows[:-1], indent=4, left=(0, 1))

    report.banner("CHECK", "02c", "cv2's odd-padded-size bias on this burst")
    rows = [[name, report.pm(np.median(s[:, 0]), bu.half_iqr(s[:, 0]), signed=True),
             report.pm(np.median(s[:, 1]), bu.half_iqr(s[:, 1]), signed=True)]
            for name, s in (("cv2, full frame (pads to odd width)", full), ("cv2, DFT-safe crop", cv2_shifts),
                            ("scikit-image", sk_shifts), ("ECC", ecc_shifts))]
    report.table(["method", "median dx " + report.PM + " half-IQR [px]", "median dy " + report.PM + " half-IQR [px]"], rows)
    bias = np.median(full[:, 0]) - np.median(cv2_shifts[:, 0])
    report.say(f"On the full frame cv2's dx is {bias:+.3f} px away from its own DFT-safe-crop dx; a difference near "
           "+0.5 px is cv2's odd-padding bias (the half-size green width 2016 pads to 2025), not motion.", indent=2)

    report.banner("DELIVERABLE", f"{problem:02d}", f"shift of frames 1..{n_frames - 1} relative to frame 0")
    worst = int(np.argmax(mag(cv2_shifts)))
    report.key_values([
        ("median shift", report.pm(np.median(mag(cv2_shifts)), float(np.median(disagree))), "px"),
        ("largest shift", report.pm(mag(cv2_shifts)[worst], float(disagree[worst])), f"px (frame {worst + 1})"),
        ("weakest cv2 correlation peak", f"{np.nanmin(responses):.3f}", f"(median {np.nanmedian(responses):.3f}, unitless)")])
    report.say(f"Shift = length sqrt(dx^2 + dy^2) of cv2's estimate on the DFT-safe crop. " + report.PM + " = "
           "disagreement of scikit-image and ECC with cv2 (median over frames for the median; at that "
           "frame for the largest). It is an estimator spread, not a statistical error bar.", indent=2)

    problem = 3
    print(f"\n*~*~*~*~*~*~*~*~*~*~ Problem {problem} ~*~*~*~*~*~*~*~*~*~*")
    """Item 3: average the aligned frames (the clean image) and show one frame next to the
    average, same crop, same display scale, somewhere with fine detail."""
    average = bu.average_burst(aligned)
    biased_shifts = np.vstack([[0, 0], full])                        # cv2 on the full frame
    average_biased = bu.average_aligned(stack, biased_shifts)
    margin = max(bu.valid_margin(shifts), bu.valid_margin(biased_shifts))   # same crop valid for both
    crop_size = 256
    row, col = bu.find_detail_crop(average, crop_size, margin)
    burst_name = burst_path.rstrip("/").split("/")[-1]
    fig1 = f"{OUTPUT_PATH}p{problem:02d}_frame_vs_average.png"
    fig2 = f"{OUTPUT_PATH}p{problem:02d}_dft_safe_crop_comparison.png"
    lo, hi = bf.plot_frame_vs_average(
        stack[0], average, row, col, crop_size, fig1,
        f"HW7 problem 3: one frame vs the average of {n_frames} aligned frames ({burst_name})", show=SHOW)
    _, _, dmax = bf.plot_crop_panels(
        [stack[0], average, average_biased],
        ["single frame (frame 0)", "average, DFT-safe cv2 shifts", "average, full-frame cv2 shifts (biased)"],
        row, col, crop_size, fig2,
        f"HW7 problem 3: effect of the DFT-safe crop on the {n_frames}-frame average ({burst_name})",
        window_from=1, diff_pair=(1, 2), show=SHOW)

    sl = (slice(row, row + crop_size), slice(col, col + crop_size))
    n_pix = crop_size ** 2
    rel = 1 / np.sqrt(2 * (n_pix - 1))                               # relative error of a std, indep. pixels
    sigma_t = np.sqrt(stack[:, sl[0], sl[1]].astype(np.float64).var(axis=0, ddof=1)).mean()
    meas = (stack[0][sl] - average[sl]).std()
    pred = sigma_t * np.sqrt(1 - 1 / n_frames)

    report.banner("CHECK", "03a", "noise left after averaging, in the detail crop")
    report.table(["quantity (linear light, 0-1)", "value " + report.PM + " stat. [unitless]"], [
        ["std of (frame 0 - average)", report.pm(meas, meas * rel, 5)],
        ["predicted: temporal std x sqrt(1 - 1/N)", report.pm(pred, pred * rel, 5)],
        ["measured / predicted", f"{meas / pred:.2f}"]])
    report.say(f"Temporal std = mean over crop pixels of the std across the {n_frames} raw frames. " + report.PM +
           f" counts only the {n_pix} crop pixels (assumes independent pixels). A ratio above 1 means frame 0 "
           "differs from the average by more than noise (e.g. frame 0 is the only unwarped frame, and this crop "
           "was chosen for fine detail).", indent=2)

    ratios = bu.half_split_noise_check(aligned, margin)
    n_in = (aligned.shape[1] - 2 * margin) * (aligned.shape[2] - 2 * margin)
    rel_v = np.sqrt(2 / n_in)
    report.banner("CHECK", "03b", "does averaging behave like independent noise?")
    report.table(["split of the burst into two halves", "measured / predicted variance " + report.PM + " stat."], [
        ["even frames vs odd frames", report.pm(ratios["even_odd"], ratios["even_odd"] * rel_v, 2)],
        ["first half vs second half", report.pm(ratios["first_second"], ratios["first_second"] * rel_v, 2)]])
    gap = ratios["first_second"] > 1.3 * ratios["even_odd"]
    report.say("1.00 = pure independent noise. The " + report.PM + " counts only image pixels, so it is tiny; the real "
           "uncertainty is systematic. " + (
               "The first half differs from the second more than noise explains (the first/second ratio is well "
               "above even/odd), so something changed during the burst. Not investigated further; cite it in the "
               "write-up." if gap else "Both splits agree, so no change during the burst is detected."), indent=2)

    gains = bu.frame_gains(stack, margin)
    (slope, _), cov = np.polyfit(np.arange(n_frames), gains, 1, cov=True)
    report.banner("CHECK", "03c", "mean level of each frame relative to frame 0 (1.0000 = same brightness)")
    report.grid("frame", list(gains), per_row=10)
    report.key_values([("lowest", f"{gains.min():.4f}", ""), ("highest", f"{gains.max():.4f}", ""),
                   ("linear trend over the burst", report.pm(slope * (n_frames - 1), np.sqrt(cov[0, 0]) * (n_frames - 1), 4, True),
                    "(fraction; " + report.PM + " from straight-line fit scatter)")], indent=4)

    d_rms = np.sqrt(np.mean((average[sl] - average_biased[sl]) ** 2))
    report.banner("DELIVERABLE", f"{problem:02d}", "frame vs average figures")
    print(f"    figure 1 (frame vs average)               {fig1}")
    print(f"    figure 2 (DFT-safe vs full-frame crop)    {fig2}\n")
    report.key_values([("crop (rows, columns)", f"{row}:{row + crop_size}, {col}:{col + crop_size}", "half-size green pixels"),
                   ("display window", f"{lo:.4f} to {hi:.4f}", "linear light (gamma 2.2)"),
                   ("frames averaged", f"{n_frames}", ""),
                   ("RMS difference between the two averages", f"{d_rms:.5f}", "linear light, in the crop"),
                   ("figure 2 difference-panel scale", f"+/-{dmax:.5f}", "linear light")])
