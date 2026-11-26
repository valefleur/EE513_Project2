"""Burst loading: the same green sites from every raw frame, as linear light.

Reuses ll.green_plane_from_raw (slicing + black/saturation normalization) and
dm.cfa_pattern; this module only adds the loop over frames and the consistency checks.
Burst-averaging idea follows Professor James McNames' EE 513 Lecture 11/12 demos
(11_deep_learning_on_ramp_b.ipynb, 12_learned_restoration_and_isps_a.ipynb).
"""
import cv2
import numpy as np
import rawpy

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
