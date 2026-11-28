"""Burst loading, alignment and averaging (HW7).

Attribution: the METHODS in this module (green-site extraction, Hann-windowed phase
correlation with cv2.phaseCorrelate, undoing the shift with cv2.warpAffine, averaging the
aligned frames) are specified by Professor James McNames' EE 513 HW7 assignment. His
notebooks (11_*, 12_*) contain Bayer mosaicking from RGB and DnCNN training code but no
burst alignment or averaging, so no code here is copied from them.
"""
import functools

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

    Attribution: green-site slicing and normalization per Professor James McNames' HW7
    assignment (Part 1); built on this repo's green_plane_from_raw.

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


@functools.lru_cache(maxsize=8)
def _hann_window(shape):
    """Hann window for an image of NumPy `shape` = (H, W).

    OpenCV's size arguments (createHanningWindow, warpAffine's dsize) are (width, height),
    the opposite of NumPy's (rows, cols) = (H, W). The swap happens here and nowhere else.
    """
    h, w = shape
    return cv2.createHanningWindow((w, h), cv2.CV_32F)


def dft_safe_size(n):
    """Largest m <= n with cv2.getOptimalDFTSize(m) == m and m even.

    cv2.phaseCorrelate zero-pads every axis up to getOptimalDFTSize(n). When that padded
    size is ODD, the returned shift on that axis is wrong by +0.5 px (verified on synthetic
    data with a known shift; see tests and DECISIONS_AND_VERIFICATION.md). Note the odd
    size is the PADDED size, not the image size: 2016 is even but pads to 2025. 2000 needs
    no padding and is even. Cropping to a size that equals its own optimal DFT size avoids
    both the padding and the bias.
    """
    for m in range(n, 1, -1):
        if m % 2 == 0 and cv2.getOptimalDFTSize(m) == m:
            return m
    raise ValueError(f"no DFT-safe size at or below {n}")


def _center_crop(shape):
    """Slices that center-crop an image of NumPy `shape` = (H, W) to (dft_safe_size(H), dft_safe_size(W))."""
    h, w = (dft_safe_size(n) for n in shape)
    r0, c0 = (shape[0] - h) // 2, (shape[1] - w) // 2
    return slice(r0, r0 + h), slice(c0, c0 + w)


def estimate_shift(ref, img, crop_to_safe_size=True):
    """Sub-pixel translation of `img` relative to `ref` by phase correlation.

    Math: if img(x) = ref(x - d), their Fourier transforms differ only by a phase ramp,
    F_img(k) = F_ref(k) exp(-i 2 pi k.d). The normalized cross-power spectrum
    F_ref* F_img / |F_ref* F_img| is therefore a pure complex exponential whose inverse
    FFT is a sharp peak at d; cv2 interpolates around the peak to get a fraction of a
    pixel. The Hann window tapers the borders to zero so the FFT's implied periodic
    wrap-around does not create a false strong edge (spectral leakage).

    The shift is measured on a center crop whose sides are DFT-safe (dft_safe_size) because
    cv2.phaseCorrelate is off by +0.5 px on any axis that pads to an odd size. A translation
    measured on the crop is the translation of the whole image. For the 1512 x 2016 green
    plane the crop is 1500 x 2000.

    Inputs are copied before the call: in OpenCV 5.0 phaseCorrelate multiplies its input
    arrays by the window IN PLACE when no padding is needed, which would silently corrupt
    the caller's frames.

    Attribution: method (cv2.phaseCorrelate on a Hann-windowed pair) per Professor James
    McNames' HW7 assignment (Part 1, item 2).

    Parameters
    ----------
    ref, img : (H, W) float arrays, same shape.
    crop_to_safe_size : set False only to demonstrate the odd-padding bias on the full frame.

    Returns
    -------
    (dx, dy, response) : how far img's content sits right (+dx) and down (+dy) of ref's,
        in pixels, and the correlation peak height. With a window the peak is not
        normalized to 1, so compare it ACROSS frames; a frame far below its siblings has
        an unreliable shift.
    """
    sl = _center_crop(ref.shape) if crop_to_safe_size else (slice(None), slice(None))
    a = np.array(ref[sl], dtype=np.float32, copy=True)
    b = np.array(img[sl], dtype=np.float32, copy=True)
    (dx, dy), response = cv2.phaseCorrelate(a, b, _hann_window(a.shape))
    return dx, dy, response


def undo_shift(img, dx, dy, interpolation=cv2.INTER_LINEAR):
    """Move img back by the shift that estimate_shift reported, so it lines up with ref.

    Pass estimate_shift's (dx, dy) straight in; the sign reversal lives here. estimate_shift
    says "img's content sits dx right / dy down of ref's", and cv2.warpAffine with
    M = [[1, 0, t_x], [0, 1, t_y]] moves content by +t, so undoing needs t = -(dx, dy).

    Attribution: cv2.warpAffine step per Professor James McNames' HW7 assignment.

    Resampling details (cv2.warpAffine); `interpolation` selects the kernel:
      INTER_LINEAR (default, also OpenCV's default): each output pixel is
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
    return cv2.warpAffine(img, M, (w, h), flags=interpolation,   # dsize is (width, height)
                          borderMode=cv2.BORDER_REFLECT)


def apply_shifts(stack, shifts, interpolation=cv2.INTER_LINEAR):
    """Undo per-frame (dx, dy) shifts (as reported by estimate_shift) on an (N, H, W) stack.

    Frames whose shift is exactly (0, 0) (frame 0) are copied untouched.
    """
    aligned = np.empty(stack.shape, np.float32)
    for i, (dx, dy) in enumerate(shifts):
        aligned[i] = stack[i] if (dx == 0 and dy == 0) else undo_shift(stack[i], dx, dy, interpolation)
    return aligned


def average_aligned(stack, shifts, interpolation=cv2.INTER_LINEAR):
    """Per-pixel mean of the frames after undoing `shifts`, without storing the aligned stack.

    Same result as average_burst(apply_shifts(stack, shifts)) but accumulates frame by frame
    in float64, so a second alignment variant costs one image of memory instead of N.
    """
    acc = np.zeros(stack.shape[1:], np.float64)
    for frame, (dx, dy) in zip(stack, shifts):
        acc += frame if (dx == 0 and dy == 0) else undo_shift(frame, dx, dy, interpolation)
    return acc / len(stack)


def half_iqr(x):
    """Half the interquartile range of x, ignoring NaNs: a robust 'typical spread' for
    frame-to-frame variability. For Gaussian data it equals 0.6745 standard deviations
    (multiply by 1.4826 to estimate sigma)."""
    q1, q3 = np.nanpercentile(x, [25, 75])
    return float((q3 - q1) / 2)


def align_to_first(stack, interpolation=cv2.INTER_LINEAR):
    """Align every frame of a (N, H, W) float stack to frame 0.

    Each frame's shift is measured with estimate_shift and undone with undo_shift.
    Frame 0 is returned untouched. Method per Professor James McNames' HW7 assignment
    (Part 1, item 2).

    Returns
    -------
    aligned : (N, H, W) float32
    shifts : (N, 2) array of (dx, dy) in pixels, row 0 = (0, 0)
    responses : (N,) phase-correlation peak heights (row 0 = NaN, nothing was measured)
    """
    shifts = np.zeros((len(stack), 2))
    responses = np.full(len(stack), np.nan)
    for i in range(1, len(stack)):
        dx, dy, resp = estimate_shift(stack[0], stack[i])
        shifts[i], responses[i] = (dx, dy), resp
    return apply_shifts(stack, shifts, interpolation), shifts, responses


def valid_margin(shifts):
    """Pixels to crop from every edge so no reflected-border fill remains after alignment:
    ceil of the largest |dx| or |dy|, plus 1 for the bilinear kernel's footprint."""
    return int(np.ceil(np.abs(shifts).max())) + 1


def average_burst(aligned):
    """Per-pixel mean over frames of an aligned (N, H, W) stack, accumulated in float64.

    Method per Professor James McNames' HW7 assignment (Part 1, item 3).

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


def estimate_shift_ecc(ref, img, gauss_filt_size=5, iterations=200, eps=1e-6):
    """Third, independent shift estimate: ECC (enhanced correlation coefficient) maximization.

    Uses OpenCV's cv2.findTransformECC.

    Unlike phase correlation (a Fourier-phase method that weights every frequency equally,
    including noise-dominated ones), ECC searches for the translation that maximizes the
    normalized correlation of the *intensities* by Gauss-Newton iteration, after a Gaussian
    smoothing (`gauss_filt_size`) that suppresses noise. Agreement between two methods that
    fail in different ways is better evidence than either alone. For MOTION_TRANSLATION the
    returned translation has the same sign convention as estimate_shift (checked by test).

    Returns
    -------
    (dx, dy) in pixels, or (nan, nan) if ECC does not converge.
    """
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, eps)
    try:
        _, warp = cv2.findTransformECC(ref.astype(np.float32), img.astype(np.float32), warp,
                                       cv2.MOTION_TRANSLATION, crit, None, gauss_filt_size)
    except cv2.error:
        return np.nan, np.nan
    return float(warp[0, 2]), float(warp[1, 2])


def alignment_residuals(stack, shifts, interpolation=cv2.INTER_LINEAR, margin=4, sigma=2.0):
    """RMS difference to frame 0 of each frame after undoing `shifts`, on Gaussian-smoothed images.

    This scores an alignment by its OUTCOME, needing no ground-truth shift: if the shifts are right,
    the aligned frame should match frame 0 better than the unaligned one. Smoothing (Gaussian,
    `sigma` pixels) keeps noise from dominating, and the same smoothing is applied to every candidate,
    so methods are compared on equal footing. Note the warp itself also smooths; at equal `sigma` that
    is part of what is being scored (a method that blurs away detail can look 'better').

    Returns (N,) array; entry 0 is NaN.
    """
    ref = cv2.GaussianBlur(stack[0].astype(np.float32), (0, 0), sigma)
    m = slice(margin, -margin), slice(margin, -margin)
    out = np.full(len(stack), np.nan)
    for i in range(1, len(stack)):
        img = undo_shift(stack[i], *shifts[i], interpolation) if np.any(shifts[i]) else stack[i]
        img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma)
        out[i] = np.sqrt(np.mean((img[m] - ref[m]) ** 2))
    return out


def frame_gains(stack, margin=0):
    """Mean level of each frame relative to frame 0 (interior only). 1.0 means no brightness
    change; a steady trend means lighting or exposure drifted during the burst."""
    inner = stack[:, margin:stack.shape[1] - margin, margin:stack.shape[2] - margin]
    m = inner.reshape(len(stack), -1).mean(axis=1, dtype=np.float64)
    return m / m[0]
