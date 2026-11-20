"""demosaic.py -- HW5 / Project 2: demosaicing and the reference to judge it against.

Part 1 (this file so far): build a ground truth that was never interpolated.

Reuses, rather than rewrites:
  my_library.linear_light.raw_to_linear_light   (HW4: black subtraction + normalization)
  my_library.linear_light.quarter_rgb           (HW4: 2x2 block -> one RGB pixel)
  psu_imaging.mosaic_rggb                       (Professor James McNames, psu_imaging.py)

The idea of testing against a known truth by re-mosaicing follows Professor McNames'
Lecture 7 demo notebook, 07_build_an_isp.ipynb (setup cell and section 2).
Written with help from Claude (Anthropic).
"""
import numpy as np

import psu_imaging as psu
import my_library.linear_light as ll

# After align_to_rggb() every mosaic in this module is RGGB, described the rawpy way:
# color_desc "RGBG" indexes as R=0, G=1, B=2, G=3, and raw_pattern holds those indices.
RGGB_PATTERN = np.array([[0, 1], [3, 2]])
RGGB_DESC = "RGBG"


def cfa_pattern(raw):
    """Read the color filter array (CFA) layout from an open rawpy object.

    rawpy stores the layout in two pieces: ``raw.color_desc`` names the colors
    (e.g. b"RGBG") and ``raw.raw_pattern`` is a 2x2 array of indices into that
    string, one per site of the repeating tile. Looking each index up gives the
    tile in reading order (top-left, top-right, bottom-left, bottom-right).

    Parameters
    ----------
    raw : rawpy.RawPy
        An open raw file.

    Returns
    -------
    str
        Four letters, e.g. "RGGB" or "BGGR".
    """
    desc = raw.color_desc.decode() if isinstance(raw.color_desc, bytes) else raw.color_desc
    return "".join(desc[i] for i in np.asarray(raw.raw_pattern).flatten())


def rggb_offset(pattern):
    """(row, col) offset that puts a red site at the top left of the mosaic.

    A Bayer tile repeats every 2 pixels in each direction, so dropping one row
    and/or one column slides the tile under the origin. The offset needed is
    just where "R" sits inside the tile:

        RGGB -> (0, 0)    GRBG -> (0, 1)    GBRG -> (1, 0)    BGGR -> (1, 1)

    Parameters
    ----------
    pattern : str
        Four-letter tile from cfa_pattern().

    Returns
    -------
    (int, int)
        Rows and columns to drop from the top and left edges.
    """
    assert sorted(pattern) == list("BGGR"), f"not a Bayer tile: {pattern!r}"
    return divmod(pattern.index("R"), 2)


def align_to_rggb(mosaic, offset):
    """Crop a Bayer mosaic so that it is RGGB with even height and width.

    Drops ``offset`` rows/columns from the top/left (which makes the top-left
    site red), then drops at most one row/column from the bottom/right so that
    every 2x2 tile is complete. No pixel value is changed; at most a one-pixel
    border is lost.

    Parameters
    ----------
    mosaic : 2-D array
        Raw Bayer data, any dtype.
    offset : (int, int)
        From rggb_offset().

    Returns
    -------
    2-D array (a view of the input), RGGB, even dimensions.
    """
    dr, dc = offset
    m = mosaic[dr:, dc:]
    h, w = (m.shape[0] // 2) * 2, (m.shape[1] // 2) * 2
    return m[:h, :w]


def reference_from_mosaic(mosaic_rggb_dn, black_level, saturation_level, clip=True):
    """Ground-truth RGB at quarter resolution. Nothing in it is interpolated.

    Two steps:

    1. Linear light:  x = (DN - black) / (saturation - black)
       0.0 is "no light", 1.0 is "pixel full".
    2. Binning: each 2x2 tile  [R  G1]  becomes ONE output pixel
                               [G2 B ]
           ref[i, j] = ( R,  (G1 + G2) / 2,  B )

    Every output pixel therefore holds three *measured* colors. The price is
    half the width and half the height (a quarter of the pixel count).

    Parameters
    ----------
    mosaic_rggb_dn : 2-D array
        RGGB mosaic in raw digital numbers, from align_to_rggb().
    black_level : float
        Sensor pedestal in DN.
    saturation_level : float
        DN at which the pixel clips (measured in HW4, not the declared white level).
    clip : bool
        Limit the result to [0, 1]. True here: the reference is used as an
        image, and PSNR later assumes a peak of 1.0.

    Returns
    -------
    float array, shape (H/2, W/2, 3), linear light, channel order R, G, B.
    """
    linear = ll.raw_to_linear_light(mosaic_rggb_dn, black_level, saturation_level, clip=clip)
    return ll.quarter_rgb(linear, RGGB_PATTERN, RGGB_DESC)


def mosaic_reference(reference):
    """Throw away two of the three colors at every pixel, in an RGGB layout.

    This simulates a Bayer sensor photographing the reference. Because we kept
    the reference, we know exactly what a perfect demosaicer should return.
    Thin wrapper around psu.mosaic_rggb (Professor James McNames).

    Parameters
    ----------
    reference : float array (H, W, 3)

    Returns
    -------
    float array (H, W): R at (even, even), G at (even, odd) and (odd, even),
    B at (odd, odd).
    """
    return psu.mosaic_rggb(reference)


def assert_round_trip(reference, mosaic):
    """Check that the mosaic's sampled sites still equal the reference, exactly.

    Mosaicing only *selects* values, so reading each site back must return the
    reference value bit for bit (np.array_equal, not np.allclose). A failure
    means the sampling grid and the reading grid disagree, i.e. a CFA phase bug.

    Parameters
    ----------
    reference : float array (H, W, 3)
    mosaic : float array (H, W), from mosaic_reference()

    Returns
    -------
    int : number of sites checked (H * W). Raises AssertionError on mismatch.
    """
    sites = {"R  (even row, even col)": (slice(0, None, 2), slice(0, None, 2), 0),
             "G1 (even row, odd col)":  (slice(0, None, 2), slice(1, None, 2), 1),
             "G2 (odd row, even col)":  (slice(1, None, 2), slice(0, None, 2), 1),
             "B  (odd row, odd col)":   (slice(1, None, 2), slice(1, None, 2), 2)}
    assert mosaic.shape == reference.shape[:2], "mosaic and reference sizes differ"
    for name, (rows, cols, ch) in sites.items():
        assert np.array_equal(mosaic[rows, cols], reference[rows, cols, ch]), \
            f"round trip failed at {name} sites"
    return mosaic.size


def display_rgb(linear_rgb, balance=False):
    """Linear RGB -> sRGB-encoded image for imshow. For LOOKING only, never for numbers.

    balance=True applies gray-world gains (scale R and B so their means match
    green) so the picture is not green. The reference itself is left untouched.
    """
    x = np.asarray(linear_rgb, dtype=float)
    if balance:
        means = x.reshape(-1, 3).mean(axis=0)
        x = x * (means[1] / means)
    return psu.linear_to_srgb(np.clip(x, 0, 1))
