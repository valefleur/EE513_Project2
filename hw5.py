# hw5.py -- Demosaicing: bilinear vs Malvar-He-Cutler
import os
import numpy as np
import rawpy
import matplotlib.pyplot as plt

import psu_imaging as psu
import my_library.demosaic as dm
import my_library.linear_light as ll
import my_library.latex_formatter as lf
# Reference demosaicers from the Lecture 7 slides, by Professor James McNames (Canvas,
# Slides folder). Importing it creates an empty Figures/ folder next to it and calls
# np.seterr(divide/over/invalid='ignore') for the whole process.
import make_figures_07 as mf
from my_library.camera_specs import IPHONE_16_PRO_MAX as cam_spec

SHOW = True

# A raw file with fine, high-contrast detail (checkerboard from HW2,
# slanted edge, brick, text, foliage). The same file is used for every item.
image_path = 'images/iPhone16/Lightroom/'
output_path = 'output/hw5_demosaic/'

raw_file = f'{image_path}hw2_slanted/Nov_08/set3/center_501mm/APC_0658.dng'

'''
def part1(raw_file=RAW_FILE, SHOW=False):
    """Items 1-3. Returns (reference, mosaic) for the rest of the assignment."""
    os.makedirs(output_path, exist_ok=True)
    name = os.path.basename(raw_file)

    # ---- Item 1: CFA pattern and offset ----
    with rawpy.imread(str(raw_file)) as raw:
        pattern = dm.cfa_pattern(raw)
        blacks = set(raw.black_level_per_channel)
        assert len(blacks) == 1, f"per-channel black levels differ: {blacks}"
        black = blacks.pop()
        raw_dn = raw.raw_image_visible.copy()      # copy: the buffer dies when the file closes
    offset = dm.rggb_offset(pattern)
    rggb = dm.align_to_rggb(raw_dn, offset)
    print(f"DELIVERABLE 01: {name}: CFA pattern {pattern}; offset {offset[0]} row, "
          f"{offset[1]} column to put red at the top left. "
          f"Mosaic {raw_dn.shape} -> {rggb.shape}.")

    # ---- Item 2: ground truth ----
    saturation = cam_spec.saturation_dn
    reference = dm.reference_from_mosaic(rggb, black, saturation)
    print(f"\tblack {black} DN, saturation {saturation} DN, reference {reference.shape}, "
          f"range {reference.min():.4f} to {reference.max():.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    ax[0].imshow(dm.display_rgb(reference))
    ax[0].set_title("Reference (linear, sRGB-encoded for display)")
    ax[1].imshow(dm.display_rgb(reference, balance=True))
    ax[1].set_title("Same, gray-world balanced for viewing only")
    for a in ax: a.axis("off")
    fig.suptitle(f"Quarter-resolution ground truth from {name}: "
                 f"{reference.shape[1]} x {reference.shape[0]}, no interpolation")
    fig.tight_layout()
    fig.savefig(f"{output_path}02_reference.png", dpi=150)
    print(f"DELIVERABLE 02: {output_path}02_reference.png")
    if SHOW: plt.show()
    plt.close(fig)

    # ---- Item 3: mosaic it back down, confirm the round trip ----
    mosaic = dm.mosaic_reference(reference)
    n = dm.assert_round_trip(reference, mosaic)
    print(f"DELIVERABLE 03: round trip PASSED, {n:,} of {n:,} sampled sites equal the "
          f"reference exactly (mosaic {mosaic.shape}, RGGB).")
    return reference, mosaic
'''

if __name__ == "__main__":
    # reference, mosaic = part1()

    os.makedirs(output_path, exist_ok=True)
    name = os.path.basename(raw_file)

    # Problem 1
    problem = 1
    print(f"\n*~*~*~*~*~*~*~*~*~*~ Problem {problem} ~*~*~*~*~*~*~*~*~*~*")
    # CFA pattern and offset
    with rawpy.imread(str(raw_file)) as raw:
        pattern = dm.cfa_pattern(raw)
        blacks = set(raw.black_level_per_channel)
        assert len(blacks) == 1, f"per-channel black levels differ: {blacks}"
        black = blacks.pop()
        raw_dn = raw.raw_image_visible.copy()      # copy: the buffer dies when the file closes
    offset = dm.rggb_offset(pattern)
    rggb = dm.align_to_rggb(raw_dn, offset)
    print(f"DELIVERABLE {problem:02d}: {name}: CFA pattern {pattern}; offset {offset[0]} row, "
          f"{offset[1]} column to put red at the top left. "
          f"Mosaic {raw_dn.shape} -> {rggb.shape}.")

    # Problem 2
    problem = 2
    print(f"\n*~*~*~*~*~*~*~*~*~*~ Problem {problem} ~*~*~*~*~*~*~*~*~*~*")
    # Generate a ground truth
    saturation = cam_spec.saturation_dn
    reference = dm.reference_from_mosaic(rggb, black, saturation)
    print(f"\tblack {black} DN, saturation {saturation} DN, reference {reference.shape}, "
          f"range {reference.min():.4f} to {reference.max():.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    ax[0].imshow(dm.display_rgb(reference))
    ax[0].set_title("Reference (linear, sRGB-encoded for display)")
    ax[1].imshow(dm.display_rgb(reference, balance=True))
    ax[1].set_title("Same, gray-world balanced for viewing only")
    for a in ax:
        a.axis("off")
    fig.suptitle(f"Quarter-resolution ground truth from {name}: "
                 f"{reference.shape[1]} x {reference.shape[0]}, no interpolation")
    fig.tight_layout()
    fig.savefig(f"{output_path}{problem:02d}_reference.png", dpi=150)
    print(f"DELIVERABLE {problem:02d}: {output_path}{problem:02d}_reference.png")
    if SHOW:
        plt.show()
    plt.close(fig)


    # Problem 3
    problem = 3
    print(f"\n*~*~*~*~*~*~*~*~*~*~ Problem {problem} ~*~*~*~*~*~*~*~*~*~*")
    # Mosaic it back down, confirm the round trip
    mosaic = dm.mosaic_reference(reference)
    n = dm.assert_round_trip(reference, mosaic)
    print(f"DELIVERABLE {problem:02d}: round trip PASSED, {n:,} of {n:,} sampled sites equal the "
          f"reference exactly (mosaic {mosaic.shape}, RGGB).")


    # Problem 6
    problem = 6
    print(f"\n*~*~*~*~*~*~*~*~*~*~ Problem {problem} ~*~*~*~*~*~*~*~*~*~*")
    '''
    6. Implement bilinear demosaicing yourself. Interpolate each channel from its own measured
    neighbors, and take care at the frame border, where a pixel has fewer neighbors than it does
    in the middle.
    '''

    # Bilinear demosaic of the re-mosaicked reference.
    crop_size = 96
    bilinear = dm.bilinear_demosaic(mosaic)
    dm.assert_round_trip(bilinear, mosaic)       # measured sites must survive untouched
    r, c = dm.busiest_crop(reference, crop_size)
    sl = (slice(r, r + crop_size), slice(c, c + crop_size))

    # Colorbar gets its own narrow column so all three image panels are the same size.
    fig = plt.figure(figsize=(13.5, 4.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.08)
    ax = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])
    panels = [("Reference (truth)", reference),
              ("Bilinear", bilinear),
              ("|error|, mean over R,G,B", None)]
    for a, (title, img) in zip(ax, panels):
        if img is None:
            im = a.imshow(np.abs(bilinear - reference)[sl].mean(axis=2), cmap="magma")
            fig.colorbar(im, cax=cax, label="|error|, linear light")
        else:
            a.imshow(dm.display_rgb(img, balance=True)[sl], interpolation="nearest")
        a.set_title(title); a.axis("off")
    fig.suptitle(f"Item {problem}: bilinear demosaic, {crop_size}x{crop_size} crop at row {r}, col {c} "
                 f"(display balanced, numbers are not)")
    # let problem have a leading zero so the files sort in order
    fig.savefig(f"{output_path}{problem:02d}_bilinear.png", bbox_inches="tight", dpi=150)
    print(f"DELIVERABLE {problem:02d}: bilinear demosaic {bilinear.shape}; measured sites preserved; "
          f"figure {output_path}{problem:02d}_bilinear.png")
    if SHOW:
        plt.show()
    plt.close(fig)


    # Problem 7
    problem = 7
    print(f"\n*~*~*~*~*~*~*~*~*~*~ Problem {problem} ~*~*~*~*~*~*~*~*~*~*")
    '''
    7. Implement Malvar, He and Cutler yourself. Start from bilinear and add the correction term
    for each of the eight cases in the lecture 7 table. Write it as fixed linear filters, with no
    branching and no per-pixel decisions, because that's the property that makes it cheap.
    '''
    mhc = dm.mhc_demosaic(mosaic)
    dm.assert_round_trip(mhc, mosaic)            # measured sites must survive untouched
    print(f"\tMHC range {mhc.min():.4f} to {mhc.max():.4f} (outside [0, 1] = overshoot "
          f"from the negative taps; clipped only for display)")

    # Same crop as item 4. Shared error scale so the two error maps compare by eye.
    err_b = np.abs(bilinear - reference)[sl].mean(axis=2)
    err_m = np.abs(mhc - reference)[sl].mean(axis=2)
    vmax = np.percentile(np.concatenate([err_b.ravel(), err_m.ravel()]), 99)
    fig = plt.figure(figsize=(17, 4.6))
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.05], wspace=0.08)
    ax = [fig.add_subplot(gs[0, i]) for i in range(5)]
    cax = fig.add_subplot(gs[0, 5])
    for a, (title, img) in zip(ax[:3], [("Reference (truth)", reference),
                                        ("Bilinear", bilinear), ("MHC", mhc)]):
        a.imshow(dm.display_rgb(img, balance=True)[sl], interpolation="nearest")
        a.set_title(title)
    for a, (title, e) in zip(ax[3:], [("|error| bilinear", err_b), ("|error| MHC", err_m)]):
        im = a.imshow(e, cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
        a.set_title(title)
    for a in ax:
        a.axis("off")
    fig.colorbar(im, cax=cax, label="|error|, linear light (shared scale)")
    fig.suptitle(f"Item {problem:02d}: bilinear vs Malvar-He-Cutler, {crop_size}x{crop_size} crop at "
                 f"row {r}, col {c} (display balanced, numbers are not)")
    fig.savefig(f"{output_path}{problem:02d}_mhc.png", bbox_inches="tight", dpi=150)
    print(f"DELIVERABLE {problem:02d}: MHC demosaic {mhc.shape}; measured sites preserved; "
          f"figure {output_path}{problem:02d}_mhc.png")
    if SHOW:
        plt.show()
    plt.close(fig)


    # Problem 8
    problem = 8
    print(f"\n*~*~*~*~*~*~*~*~*~*~ Problem {problem} ~*~*~*~*~*~*~*~*~*~*")
    '''
    8. The eight cases need only four filter patterns between them, because two of them are
    each other rotated by ninety degrees. Print the four you used.
    '''
    kernels = dm.mhc_full_kernels()
    assert np.allclose(kernels["R/B at G, column"], np.rot90(kernels["R/B at G, row"]))
    print(f"DELIVERABLE {problem:02d}: the four MHC filters, shown x 8 (divide by 8 to apply). "
          "Center tap at [2, 2].")
    for name, k in kernels.items():
        print(f"\n{name}   (sum = {k.sum():.3f})")
        print(np.array2string(8 * k, precision=1, suppress_small=True, floatmode="maxprec"))
    print("\n'R/B at G, column' == np.rot90('R/B at G, row'): True")

    # Same four filters as LaTeX, for pasting into the report.
    latex_labels = {"G at R/B": r"K_{\text{G at R, B}}",
                    "R/B at G, row": r"K_{\text{R, B at G, row}}",
                    "R/B at G, column": r"K_{\text{R, B at G, col}}",
                    "R at B / B at R": r"K_{\text{R at B, B at R}}"}
    latex_file = f"{output_path}{problem:02d}_mhc_kernels.txt"
    lf.generate_matrix_grid({latex_labels[n]: k for n, k in kernels.items()},
                            scale=8, per_row=2, output_path=latex_file)
    print(f"\tLaTeX for the report: {latex_file}")


    # Problem 9
    problem = 9
    print(f"\n*~*~*~*~*~*~*~*~*~*~ Problem {problem} ~*~*~*~*~*~*~*~*~*~*")
    '''
    9. Show what the two algorithms produced, next to the reference.
    '''
    # Same crop as Problems 6 and 7 so the figures line up. Every panel gets the SAME
    # display gains (gray-world from the reference), so a color difference between
    # panels is a real demosaicing difference, not a per-panel balance choice.
    gains = reference.reshape(-1, 3).mean(axis=0)
    gains = gains[1] / gains
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 5.1), layout="constrained")
    for a, (title, img) in zip(ax, [("Reference (truth)", reference),
                                    ("Bilinear", bilinear),
                                    ("Malvar-He-Cutler", mhc)]):
        a.imshow(dm.display_rgb(img[sl] * gains), interpolation="nearest")
        a.set_title(title)
        a.axis("off")
    fig.suptitle(f"Item {problem:02d}: demosaicing results, {crop_size}x{crop_size} crop at "
                 f"row {r}, col {c} of {reference.shape[0]}x{reference.shape[1]}\n"
                 f"same display balance on all three panels")
    fig.savefig(f"{output_path}{problem:02d}_three_panel.png", bbox_inches="tight", dpi=150)
    print(f"DELIVERABLE {problem:02d}: figure {output_path}{problem:02d}_three_panel.png")
    if SHOW:
        plt.show()
    plt.close(fig)


    # Problem 10
    problem = 10
    print(f"\n*~*~*~*~*~*~*~*~*~*~ Problem {problem} ~*~*~*~*~*~*~*~*~*~*")
    '''
    10. Check both of your implementations against the reference ones in make_figures.py from
    the lecture 7 slides, which are bilinear_demosaic and mhc_demosaic. They assume an RGGB
    mosaic, so feed them the aligned version from item 1.
    '''
    # Input: the item-1 aligned RGGB mosaic, full resolution, in linear light [0, 1]
    # (The professor's functions expect normalized floats, not DN).
    linear_rggb = ll.raw_to_linear_light(rggb, black, saturation, clip=True)
    print(f"\tinput: aligned RGGB mosaic from item 1, {linear_rggb.shape}, linear light")

    mine_bil, professors_bil = dm.bilinear_demosaic(linear_rggb), mf.bilinear_demosaic(linear_rggb)
    mine_mhc, professors_mhc = dm.mhc_demosaic(linear_rggb), mf.mhc_demosaic(linear_rggb)
    mine_mhc_clipped = dm.mhc_demosaic(linear_rggb, clip=True)   # reused in Problem 12
    # Professor's MHC clips to [0, 1] on return and mine does not, so clip mine to compare
    # like with like; the unclipped figure shows how much of the gap is just that.
    results = {"bilinear": dm.compare_demosaic(mine_bil, professors_bil),
               "MHC (mine clipped to [0, 1])": dm.compare_demosaic(mine_mhc_clipped,
                                                                   professors_mhc),
               "MHC (mine unclipped)": dm.compare_demosaic(mine_mhc, professors_mhc)}
    print(f"DELIVERABLE {problem:02d}: largest |mine - McNames'| (linear light, full scale = 1)")
    print(f"\t{'method':30s} {'whole frame':>12s} {'at (row, col, ch)':>20s} "
          f"{'interior*':>12s}")
    for name, res in results.items():
        loc = f"{res['where']}{' border' if res['in_border'] else ''}"
        print(f"\t{name:30s} {res['max_all']:12.3e} {loc:>20s} {res['max_interior']:12.3e}")
    print("\t* interior excludes the 2-pixel frame. Inside it the methods differ only in the\n"
          "\t  border rule: mine falls back to bilinear there; professors uses cv2.BORDER_REFLECT,\n"
          "\t  which copies row 0 into row -1 and so breaks the Bayer phase for 5x5 kernels.")

    # Same table as LaTeX for the report. Labels shortened to fit a 1-inch-margin page
    # (the caption spells them out); numbers right-aligned.
    short = {"MHC (mine clipped to [0, 1])": r"MHC, mine clipped to $[0, 1]$",
             "MHC (mine unclipped)": "MHC, mine unclipped"}
    names = list(results)
    columns = [[short.get(n, n) for n in names],
               [lf.latex_sci(results[n]["max_all"]) for n in names],
               ["({}, {}, {})".format(*results[n]["where"]) for n in names],
               ["border" if results[n]["in_border"] else "interior" for n in names],
               [lf.latex_sci(results[n]["max_interior"]) for n in names]]
    latex_file = f"{output_path}{problem:02d}_compare_to_reference.txt"
    lf.generate_table(columns,
                      ["Method", "Whole frame", "Where (row, col, ch)", "Region",
                       "Interior"],
                      caption="Largest absolute difference between my demosaicers and the "
                              "Lecture 7 reference implementations (Prof.\\ McNames), on the "
                              "aligned RGGB raw mosaic in linear light (full scale 1). Where: "
                              "location of the whole-frame maximum; Region: whether it lies "
                              "within 2 px of the edge. Interior: the 2-px frame excluded. The "
                              "reference clips MHC to $[0, 1]$, so mine is shown both clipped "
                              "and unclipped.",
                      output_path=latex_file, VERBOSE=False, align="lrrlr")
    print(f"\tLaTeX for the report: {latex_file}")


    # Problem 12
    problem = 12
    print(f"\n*~*~*~*~*~*~*~*~*~*~ Problem {problem} ~*~*~*~*~*~*~*~*~*~*")
    '''
    12. Both reference implementations leave the measured sites exactly as they found them.
    That's a property worth testing rather than trusting, and your own code should have it too.
    '''
    # Every output was computed in Problems 6, 7 and 10; nothing is recomputed here.
    P3, P1 = "re-mosaicked reference (P3)", "aligned raw mosaic (P1)"
    checks = [(P3, "bilinear (mine)", bilinear, mosaic),
              (P3, "MHC (mine)", mhc, mosaic),
              (P1, "bilinear (mine)", mine_bil, linear_rggb),
              (P1, "MHC (mine)", mine_mhc, linear_rggb),
              (P1, "MHC (mine, clip=True)", mine_mhc_clipped, linear_rggb),
              (P1, "bilinear (McNames)", professors_bil, linear_rggb),
              (P1, "MHC (McNames)", professors_mhc, linear_rggb)]
    print(f"DELIVERABLE {problem:02d}: measured sites returned unchanged? (exact, bit for bit)")
    print(f"\t{'input':29s} {'implementation':23s} {'sites':>11s} {'changed':>8s} "
          f"{'on edge':>8s} {'max |diff|':>11s}  result")
    rows = []
    for input_name, impl, rgb, mos in checks:
        res = dm.measured_site_check(rgb, mos)
        verdict = "PASS" if res["exact"] else "within rounding" if res["max_abs"] < 1e-12 else "FAIL"
        print(f"\t{input_name:29s} {impl:23s} {res['sites']:11,d} {res['changed']:8,d} "
              f"{res['on_edge']:8,d} {res['max_abs']:11.2e}  {verdict}")
        rows.append((input_name, impl, res, verdict))
        if "mine" in impl:
            assert res["exact"], f"{impl} changed measured sites on {input_name}"
    print("\tMine restore each measured value explicitly (np.where), so they are bit-exact.\n"
          "\tAny McNames changes sit on the outermost ring: cv2.BORDER_REFLECT duplicates the\n"
          "\tedge pixel next to itself, so the normalized average there is (6v)/6 or (9v)/9,\n"
          "\twhich can round off by one unit in the last place.")

    # Same table as LaTeX for the report. Labels shortened to fit a 1-inch-margin page
    # (the caption spells them out); numbers right-aligned.
    short = {P3: "P3 reference", P1: "P1 raw mosaic", "within rounding": "rounding"}
    tex = lambda t: short.get(t, t).replace("clip=True", r"\texttt{clip=True}")
    columns = [[tex(r[0]) for r in rows],
               [tex(r[1]) for r in rows],
               [f"{r[2]['sites']:,}" for r in rows],
               [f"{r[2]['changed']:,}" for r in rows],
               [f"{r[2]['on_edge']:,}" for r in rows],
               [lf.latex_sci(r[2]["max_abs"]) for r in rows],
               [tex(r[3]) for r in rows]]
    latex_file = f"{output_path}{problem:02d}_measured_sites.txt"
    lf.generate_table(columns,
                      ["Input", "Implementation", "Sites", "Changed", "On edge",
                       r"max $|\Delta|$", "Result"],
                      caption="Measured sites returned by each demosaicer, compared bit for bit "
                              "with the mosaic it was given. P3 reference: the re-mosaicked "
                              "quarter-resolution reference (Problem 3). P1 raw mosaic: the "
                              "aligned RGGB raw mosaic in linear light (Problem 1). On edge: "
                              "outermost row or column. Rounding: largest change is below "
                              r"$10^{-12}$ (one unit in the last place).",
                      output_path=latex_file, VERBOSE=False, align="llrrrrl")
    print(f"\tLaTeX for the report: {latex_file}")


    # Problem 13
    problem = 13
    print(f"\n*~*~*~*~*~*~*~*~*~*~ Problem {problem} ~*~*~*~*~*~*~*~*~*~*")
    '''
    13. Report PSNR against your reference for both methods, over the whole frame, and then
    again over the middle of the frame only, with a margin of a few pixels excluded on every
    side. State the peak value you used in the caption.
    '''
    PEAK = 1.0      # reference is linear light normalized so saturation = 1 (Problem 2)
    MARGIN = 4      # px; twice the 2-px reach of the 5x5 MHC kernels
    # bilinear and mhc are the Problem 6 / 7 outputs on the re-mosaicked reference.
    psnr_table = {name: (dm.psnr(img, reference, PEAK), dm.psnr(img, reference, PEAK, MARGIN))
                  for name, img in [("Bilinear", bilinear), ("Malvar-He-Cutler", mhc)]}
    print(f"DELIVERABLE {problem:02d}: PSNR vs reference (dB), peak {PEAK}, "
          f"{reference.shape[1]}x{reference.shape[0]} reference")
    print(f"\t{'method':20s} {'whole frame':>12s} {f'middle ({MARGIN}-px margin)':>22s}")
    for name, (whole, middle) in psnr_table.items():
        print(f"\t{name:20s} {whole:12.2f} {middle:22.2f}")
    gain_whole = psnr_table["Malvar-He-Cutler"][0] - psnr_table["Bilinear"][0]
    gain_middle = psnr_table["Malvar-He-Cutler"][1] - psnr_table["Bilinear"][1]
    print(f"\tMHC - bilinear: {gain_whole:+.2f} dB whole frame, {gain_middle:+.2f} dB middle")
    print(f"\t(MHC unclipped; clipped to [0, 1] it scores "
          f"{dm.psnr(np.clip(mhc, 0, 1), reference, PEAK):.2f} dB whole frame)")

    latex_file = f"{output_path}{problem:02d}_psnr.txt"
    names = list(psnr_table)
    lf.generate_table([names,
                       [f"{psnr_table[n][0]:.2f}" for n in names],
                       [f"{psnr_table[n][1]:.2f}" for n in names]],
                      ["Method", "Whole frame (dB)", f"Middle, {MARGIN}-px margin (dB)"],
                      caption=f"PSNR of each demosaicer against the quarter-resolution "
                              f"reference ({reference.shape[1]} $\\times$ {reference.shape[0]}), "
                              f"computed in linear light over all three channels with peak "
                              f"value {PEAK:g} (the saturation level after normalization). "
                              f"Middle: {MARGIN} pixels excluded on every side. MHC output "
                              f"is not clipped.",
                      output_path=latex_file, VERBOSE=False, align="lrr")
    print(f"\tLaTeX for the report: {latex_file}")
