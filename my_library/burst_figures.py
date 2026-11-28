"""burst_figures.py -- figures for the burst-averaging problems (HW7).

Attribution: plot_crop_panels follows the layout idea of crop_figures.plot_crop_comparison,
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


def plot_crop_panels(images, names, row, col, size, path, title, window_from=-1, diff_pair=None,
                     gamma=2.2, show=False):
    """Save a figure of one crop shown for several full-size images, side by side.

    Attribution: layout after crop_figures.plot_crop_comparison / Professor James McNames'
    fig_zippering() (Lecture 7).

    All panels share the crop, the display window (0.5th to 99.5th percentile of
    images[window_from] inside the crop) and the gamma, drawn nearest-neighbor so each data
    pixel is one block. A thumbnail of images[window_from] marks where the crop sits.
    `diff_pair=(i, j)` adds a panel of images[i] - images[j] (panels are numbered from 1 in
    the title) in the crop on a linear gray scale centered at mid-gray and scaled so the
    99.5th percentile of |difference| reaches the ends (the scale is stated in the panel title).

    Parameters
    ----------
    images : list of (H, W) linear-light float arrays, mutually aligned.
    names : list of str, one per image.
    row, col, size : crop's top-left corner and side, in pixels.
    path : output PNG path.  title : figure title.  gamma : display gamma.  show : plt.show().

    Returns
    -------
    (lo, hi, dmax) : display window in linear-light units, and the |difference| scale (None
    when diff_pair is None).
    """
    sl = (slice(row, row + size), slice(col, col + size))
    ref = images[window_from]
    lo, hi = np.percentile(ref[sl], [0.5, 99.5])
    n = len(images) + (diff_pair is not None)
    fig = plt.figure(figsize=(1.8 + 4.2 * n, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(1, n + 1, width_ratios=[1] + [3] * n)
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(display_map(ref, lo, hi, gamma), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax0.add_patch(plt.Rectangle((col - 0.5, row - 0.5), size, size, fill=False, ec="red", lw=1.5))
    ax0.set_title("where the crop is\n(red box)", fontsize=9)
    ax0.axis("off")
    for k, (img, name) in enumerate(zip(images, names)):
        ax = fig.add_subplot(gs[1 + k])
        ax.imshow(display_map(img[sl], lo, hi, gamma), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    dmax = None
    if diff_pair is not None:
        i, j = diff_pair
        d = images[i][sl].astype(np.float64) - images[j][sl]
        dmax = float(np.percentile(np.abs(d), 99.5)) or 1e-12
        ax = fig.add_subplot(gs[n])
        ax.imshow(0.5 + 0.5 * np.clip(d / dmax, -1, 1), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"difference: panel {i + 1} minus panel {j + 1}\n"
                     f"gray = 0, white/black = +/-{dmax:.4f} (linear)", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{title}\ncrop rows {row}:{row + size}, cols {col}:{col + size} of the half-size green image; "
                 f"linear {lo:.4f}-{hi:.4f} -> gray, gamma {gamma}; same window on every image panel", fontsize=10)
    if show:
        plt.show()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return float(lo), float(hi), dmax


def plot_frame_vs_average(frame, average, row, col, size, path, title, gamma=2.2, show=False):
    """Side-by-side crop of a single frame and the burst average (see plot_crop_panels).

    Returns (lo, hi), the display window in linear-light units.
    """
    lo, hi, _ = plot_crop_panels([frame, average], ["single frame", "average of aligned frames"],
                                 row, col, size, path, title, window_from=1, gamma=gamma, show=show)
    return lo, hi
