"""Burst loading: the same green sites from every raw frame, as linear light.

Reuses ll.green_plane_from_raw (slicing + black/saturation normalization) and
dm.cfa_pattern; this module only adds the loop over frames and the consistency checks.
Burst-averaging idea follows Professor James McNames' EE 513 Lecture 11/12 demos
(11_deep_learning_on_ramp_b.ipynb, 12_learned_restoration_and_isps_a.ipynb).
"""
import cv2
import numpy as np
import rawpy
from skimage.registration import phase_cross_correlation

import my_library.demosaic as dm
import my_library.linear_light as ll


def load_burst_green(paths, saturation, which=0, black_level=None, clip=False):
    """Load N raw frames and return the same green sites from each, as linear light.

    Parameters
    ----------
    paths : list of path-like   the burst, one raw file per frame.
    saturation : float          measured saturation in DN (NOT the WhiteLevel tag).
    which : 0 or 1              which green of the tile.
    black_level : float|None    DN to subtract; None uses each file's declared level.
    clip : bool                 False (default) keeps noise below black; the burst will
                                be averaged and clipping would bias dark pixels upward.

    Returns
    -------
    stack : (N, H/2, W/2) float32
    info : dict   pattern, black_level, saturation, n_frames, shape, raw_shape.
    Raises ValueError if any frame's CFA pattern, black level or shape differs from frame 0.
    """
    stack, first = [], None
    for p in paths:
        with rawpy.imread(str(p)) as raw:
            plane = ll.green_plane_from_raw(raw, which, black_level, saturation, clip, verbose=False)
            key = (dm.cfa_pattern(raw), tuple(raw.black_level_per_channel), raw.raw_image_visible.shape)
        first = first or key
        if key != first:
            raise ValueError(f"{p}: (pattern, black, shape) {key} differs from first frame {first}")
        stack.append(plane.astype(np.float32))
    stack = np.stack(stack)
    return stack, dict(pattern=first[0], black_level=black_level, saturation=float(saturation),
                       n_frames=len(stack), shape=stack.shape[1:], raw_shape=first[2])


def _hann_window(shape):
    """Hann window for an image of NumPy `shape` = (H, W).

    OpenCV's size arguments (createHanningWindow, warpAffine's dsize) are (width, height),
    the opposite of NumPy's (rows, cols) = (H, W). The swap happens here and nowhere else.
    """
    h, w = shape
    return cv2.createHanningWindow((w, h), cv2.CV_32F)


def estimate_shift(ref, img, window=None):
    """Sub-pixel translation of `img` relative to `ref` by phase correlation.

    Math: if img(x) = ref(x - d), their Fourier transforms differ only by a phase ramp,
    F_img(k) = F_ref(k) exp(-i 2 pi k.d). The normalized cross-power spectrum
    F_ref* F_img / |F_ref* F_img| is therefore a pure complex exponential whose inverse
    FFT is a sharp peak at d; cv2 interpolates around the peak to get a fraction of a
    pixel. The Hann window tapers the borders to zero so the FFT's implied periodic
    wrap-around does not create a false strong edge (spectral leakage).

    Parameters
    ----------
    ref, img : (H, W) float32 arrays, same shape.
    window : optional precomputed _hann_window(ref.shape).

    Returns
    -------
    (dx, dy, response) : how far img's content sits right (+dx) and down (+dy) of ref's,
        in pixels, and the correlation peak height. With a window the peak is not
        normalized to 1, so compare it ACROSS frames; a frame far below its siblings has
        an unreliable shift.
    """
    if window is None:
        window = _hann_window(ref.shape)
    (dx, dy), response = cv2.phaseCorrelate(ref.astype(np.float32), img.astype(np.float32), window)
    return dx, dy, response


def undo_shift(img, dx, dy):
    """Move img back by the shift that estimate_shift reported, so it lines up with ref.

    Pass estimate_shift's (dx, dy) straight in; the sign reversal lives here. estimate_shift
    says "img's content sits dx right / dy down of ref's", and cv2.warpAffine with
    M = [[1, 0, t_x], [0, 1, t_y]] moves content by +t, so undoing needs t = -(dx, dy).

    Resampling details (cv2.warpAffine):
      INTER_LINEAR (also OpenCV's default, written out for clarity): each output pixel is
        a weighted mean of its 4 nearest inputs. Cubic/Lanczos sharpen better but overshoot
        and ring near edges, which would put values below black and above saturation.
      BORDER_REFLECT: the strip the shift uncovers needs *some* value. OpenCV's default,
        BORDER_CONSTANT, would fill it with 0 (a dark band that biases the average); reflection
        fills it with plausible values. The Hann window is irrelevant here: it only tapers the
        FFT used to MEASURE the shift. Either way, crop ceil(max|shift|) pixels at the edges
        before measuring anything.

    Cost: a fractional shift low-pass filters the frame. For white noise the standard
    deviation is scaled by sqrt((1-f)^2 + f^2) per axis (f = fractional part), down to 0.707 at
    a half pixel, and neighboring pixels become correlated. Aligned frames are therefore
    quieter than raw frames; use unwarped frames when measuring a raw frame's noise.
    """
    M = np.array([[1, 0, -dx], [0, 1, -dy]], dtype=np.float64)
    h, w = img.shape
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,   # dsize is (width, height)
                          borderMode=cv2.BORDER_REFLECT)


def align_to_first(stack):
    """Align every frame of a (N, H, W) float stack to frame 0.

    Each frame's shift is measured with estimate_shift and undone with undo_shift. Frame 0 is returned untouched.

    Returns
    -------
    aligned : (N, H, W) float32
    shifts : (N, 2) array of (dx, dy) in pixels, row 0 = (0, 0)
    responses : (N,) phase-correlation peak heights (row 0 = NaN, nothing was measured)
    """
    n, h, w = stack.shape
    window = _hann_window((h, w))
    aligned = np.empty(stack.shape, np.float32)
    aligned[0] = stack[0]
    shifts, responses = np.zeros((n, 2)), np.full(n, np.nan)
    for i in range(1, n):
        dx, dy, resp = estimate_shift(stack[0], stack[i], window)
        shifts[i], responses[i] = (dx, dy), resp
        aligned[i] = undo_shift(stack[i], dx, dy)
    return aligned, shifts, responses


def valid_margin(shifts):
    """Pixels to crop from every edge so no reflected-border fill remains after alignment:
    ceil of the largest |dx| or |dy|, plus 1 for the bilinear kernel's footprint."""
    return int(np.ceil(np.abs(shifts).max())) + 1


def average_burst(aligned):
    """Per-pixel mean over frames of an aligned (N, H, W) stack, accumulated in float64.

    For N frames with independent zero-mean noise of standard deviation sigma, the mean has
    noise sigma / sqrt(N): 16 frames -> 1/4, 64 frames -> 1/8. (Residual noise of the
    reference is quantified in a later problem.)
    """
    return aligned.mean(axis=0, dtype=np.float64)


def find_detail_crop(img, size, margin=0, max_value=0.98):
    """Top-left (row, col) of the size x size window with the most fine detail.

    Detail = mean gradient magnitude (Sobel) of `img` averaged over each window with a box
    filter. Windows must lie inside `margin` pixels of the edge and contain no pixel above
    `max_value` (clipped highlights have no detail to show). Intended for the burst AVERAGE:
    on a single noisy frame, noise would score as detail.
    """
    gx = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    score = cv2.boxFilter(np.hypot(gx, gy), -1, (size, size), borderType=cv2.BORDER_CONSTANT)
    clipped = cv2.dilate((img > max_value).astype(np.uint8), np.ones((size, size), np.uint8))
    h, w = img.shape
    # boxFilter puts the window mean at its CENTER; convert to top-left corners.
    ok = np.zeros((h, w), bool)
    r0, r1, c0, c1 = margin + size // 2, h - margin - size + size // 2, margin + size // 2, w - margin - size + size // 2
    ok[r0:r1 + 1, c0:c1 + 1] = True
    ok &= clipped == 0
    score = np.where(ok, score, -np.inf)
    cr, cc = np.unravel_index(np.argmax(score), score.shape)
    return int(cr - size // 2), int(cc - size // 2)


def half_split_noise_check(aligned, margin=0):
    """Real-data check that averaging behaves as independent noise predicts.

    Split the N aligned frames into two halves A and B (N1 and N2 frames) and average each.
    With temporal variance s^2 per pixel (ddof=1), Var(mean_A - mean_B) = s^2 (1/N1 + 1/N2).
    Returns measured / predicted for two splits:
      'even_odd'     interleaved frames: insensitive to slow drift, tests independence.
      'first_second' first half vs second half: inflated if brightness or position drifts
                     during the burst (lighting flicker, tripod creep).
    Ratios near 1 mean the 1/sqrt(N) rule holds for this burst; well above 1 means
    something besides independent noise varies between frames.
    """
    a = aligned[:, margin:aligned.shape[1] - margin, margin:aligned.shape[2] - margin].astype(np.float64)
    n = a.shape[0]
    s2 = a.var(axis=0, ddof=1).mean()
    out = {}
    for name, ia, ib in (("even_odd", np.arange(0, n, 2), np.arange(1, n, 2)),
                         ("first_second", np.arange(0, n // 2), np.arange(n // 2, n))):
        d = a[ia].mean(axis=0) - a[ib].mean(axis=0)
        out[name] = float((d ** 2).mean() / (s2 * (1 / len(ia) + 1 / len(ib))))
    return out


def crosscheck_shifts_skimage(stack, shifts, upsample_factor=100):
    """Re-measure each frame's shift with scikit-image and compare to cv2's.

    skimage.registration.phase_cross_correlation is an independent implementation (its
    sub-pixel step is a matrix-multiply DFT around the peak, not cv2's centroid), and needs
    the same Hann taper to avoid wrap-around. It reports the shift that registers img onto
    ref, the opposite sign of estimate_shift, so we negate it.

    Returns (N, 2) array of (cv2 - skimage) differences in (dx, dy) pixels; row 0 is 0.
    """
    h, w = stack.shape[1:]
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    ref = stack[0] * win
    diff = np.zeros_like(shifts)
    for i in range(1, len(stack)):
        sk, _, _ = phase_cross_correlation(ref, stack[i] * win, upsample_factor=upsample_factor)
        diff[i] = shifts[i] - (-sk[1], -sk[0])          # skimage is (row, col) = (dy, dx)
    return diff
