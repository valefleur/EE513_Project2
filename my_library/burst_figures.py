"""burst_figures.py -- figures for the burst-averaging problems (HW7).

Attribution: plot_frame_vs_average follows the layout idea of crop_figures.plot_crop_comparison,
which itself follows fig_zippering() in make_figures_07.py by Professor James McNames
(EE 513 Lecture 7). No other code here is borrowed. Written with help from Claude (Anthropic).
"""
import matplotlib.pyplot as plt
import numpy as np


def display_map(img, lo, hi, gamma=2.2):
    """Linear light -> display: clip to [lo, hi], rescale to [0, 1], apply 1/gamma.

    The same (lo, hi, gamma) must be used for every panel being compared, or brightness and
    noise differences are partly an artifact of display. Noise below `lo` is clipped (it
    cannot be shown on a gray scale), which is why lo should sit at or below black.
    """
    return np.clip((img - lo) / (hi - lo), 0, 1) ** (1 / gamma)


def plot_frame_vs_average(frame, average, row, col, size, path, title, gamma=2.2, show=False):
    """Save a side-by-side figure of one crop: a single frame and the burst average.

    Attribution: layout based on crop_figures.plot_crop_comparison / Professor James McNames'
    fig_zippering() (Lecture 7); the display mapping and content are ours.

    Both panels use the same crop, the same display window (0.5th to 99.5th percentile of
    the AVERAGE's crop) and the same gamma, drawn with nearest-neighbor so each data pixel
    is one block (nothing smoothed). A thumbnail shows where the crop sits.

    Parameters
    ----------
    frame, average : (H, W) linear-light float arrays, already aligned to each other.
    row, col, size : crop's top-left corner and side, in pixels.
    path : output PNG path.  title : figure title.  gamma : display gamma.  show : plt.show().

    Returns
    -------
    (lo, hi) display window used, in linear-light units.
    """
    sl = (slice(row, row + size), slice(col, col + size))
    lo, hi = np.percentile(average[sl], [0.5, 99.5])
    fig = plt.figure(figsize=(13, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 3, 3])
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(display_map(average, lo, hi, gamma), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax0.add_patch(plt.Rectangle((col - 0.5, row - 0.5), size, size, fill=False, ec="red", lw=1.5))
    ax0.set_title("average (crop in red)", fontsize=9)
    ax0.axis("off")
    for k, (img, name) in enumerate(((frame, "single frame"), (average, "average of aligned frames"))):
        ax = fig.add_subplot(gs[1 + k])
        ax.imshow(display_map(img[sl], lo, hi, gamma), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(f"{title}\ncrop rows {row}:{row + size}, cols {col}:{col + size} of the half-size green image; "
                 f"linear {lo:.4f}-{hi:.4f} -> gray, gamma {gamma}; same window on both panels", fontsize=10)
    if show:
        plt.show()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return float(lo), float(hi)
