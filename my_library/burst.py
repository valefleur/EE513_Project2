"""Burst loading: the same green sites from every raw frame, as linear light.

Reuses ll.green_plane_from_raw (slicing + black/saturation normalization) and
dm.cfa_pattern; this module only adds the loop over frames and the consistency checks.
Burst-averaging idea follows Professor James McNames' EE 513 Lecture 11/12 demos
(11_deep_learning_on_ramp_b.ipynb, 12_learned_restoration_and_isps_a.ipynb).
"""
import numpy as np

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
    import rawpy
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
