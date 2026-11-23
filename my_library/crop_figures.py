"""crop_figures.py -- magnified crop figures for comparing demosaicers (HW5 items 17-18).

Layout after fig_zippering() in make_figures_07.py (Lecture 7, Professor James McNames).
Written with help from Claude Opus 5.5 (Anthropic).
"""
import numpy as np
import matplotlib.pyplot as plt

import my_library.demosaic as dm


def plot_crop_comparison(reference, bilinear, mhc, row, col, size, path, heading,
                         zoom=8, dpi=100, show=False):
    """Save a figure of one magnified crop: reference, bilinear, MHC, and both |error| maps.

    Top row: the three images. Bottom row: a thumbnail with the crop's location,
    then |bilinear - reference| and |MHC - reference| (mean over R, G, B, linear
    light) on one shared color scale.

    Display choices, so what you see is what was computed:
      - One display balance (gray-world gains from the reference) for every panel,
        so color differences between panels are real.
      - Each crop pixel becomes an exact zoom x zoom block (dm.magnify) placed with
        fig.figimage at native size, so nothing is resampled for display.
      - Saved at the same dpi with savefig.bbox="standard": make_figures_07 sets
        "tight" globally, which would crop and shift the canvas.

    Parameters
    ----------
    reference, bilinear, mhc : float arrays (H, W, 3), linear light.
    row, col : int, top-left corner of the crop.
    size : int, crop side in pixels.
    path : str, output PNG path.
    heading : str, first part of the figure title (e.g. "Item 17: vertical edge").
    zoom : int, display pixels per image pixel.
    dpi : int, figure dpi (also the save dpi).
    show : bool, call plt.show() before closing.

    Returns
    -------
    dict with "mae_bilinear", "mae_mhc": mean |error| over the crop.
    """
    sl = (slice(row, row + size), slice(col, col + size))
    display_gains = reference.reshape(-1, 3).mean(axis=0)
    display_gains = display_gains[1] / display_gains
    view = lambda img: dm.display_rgb(img[sl] * display_gains)
    err_bil = np.abs(bilinear - reference)[sl].mean(axis=2)
    err_mhc = np.abs(mhc - reference)[sl].mean(axis=2)
    vmax = np.percentile(np.concatenate([err_bil.ravel(), err_mhc.ravel()]), 99.5)
    err_norm = plt.Normalize(vmin=0, vmax=vmax)
    err_rgb = lambda e: plt.get_cmap("magma")(err_norm(e))[..., :3]

    P = size * zoom                                    # panel side, figure pixels
    GAP, VGAP, LEFT, TOP, TITLE, CBAR, BOTTOM = 30, 24, 20, 80, 28, 130, 20
    W = LEFT + 3 * P + 2 * GAP + CBAR
    H = TOP + 2 * (TITLE + P) + VGAP + BOTTOM
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    panel_y = [H - TOP - TITLE - P, BOTTOM]            # bottom edges of the two rows
    panel_x = [LEFT + k * (P + GAP) for k in range(3)]
    title_at = lambda x, y, text: fig.text((x + P / 2) / W, (y + P + 6) / H, text,
                                           ha="center", va="bottom", fontsize=11)
    for x, (title, img) in zip(panel_x, [("Reference (truth)", reference),
                                         ("Bilinear", bilinear), ("Malvar-He-Cutler", mhc)]):
        fig.figimage(dm.magnify(view(img), zoom), xo=x, yo=panel_y[0], origin="upper")
        title_at(x, panel_y[0], title)
    for x, (title, e) in zip(panel_x[1:], [("|error| bilinear", err_bil),
                                           ("|error| MHC", err_mhc)]):
        fig.figimage(dm.magnify(err_rgb(e), zoom), xo=x, yo=panel_y[1], origin="upper")
        title_at(x, panel_y[1], f"{title}, mean {e.mean():.2e}")
    # Bottom left: where the crop is (ordinary axes; the thumbnail may be resampled).
    thumb = fig.add_axes([panel_x[0] / W, panel_y[1] / H, P / W, P / H])
    thumb.imshow(dm.display_rgb(reference * display_gains))
    thumb.add_patch(plt.Rectangle((col, row), size, size, fill=False,
                                  edgecolor="red", linewidth=1.5))
    thumb.axis("off")
    title_at(panel_x[0], panel_y[1], "Crop location (red box)")
    cax = fig.add_axes([(panel_x[2] + P + 15) / W, panel_y[1] / H, 15 / W, P / H])
    fig.colorbar(plt.cm.ScalarMappable(norm=err_norm, cmap="magma"), cax=cax,
                 label="|error|, linear light (shared scale)")
    fig.suptitle(f"{heading}, {size}x{size} crop at row {row}, "
                 f"col {col}, each pixel shown as a {zoom}x{zoom} block\n"
                 f"same display balance on all panels; errors are mean |error| over R, G, B",
                 y=1 - 12 / H, va="top")
    with plt.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(path, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)
    return {"mae_bilinear": float(err_bil.mean()), "mae_mhc": float(err_mhc.mean())}
