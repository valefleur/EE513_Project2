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
from scipy.ndimage import correlate

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



# Red and blue live on a square lattice (every other row AND column): the 2-D
# triangle (tent) filter, the outer product of the 1-D linear interpolator [1, 2, 1]/2.
K_RB = np.array([[1, 2, 1],
                 [2, 4, 2],
                 [1, 2, 1]], dtype=float) / 4.0
# Green lives on a checkerboard (quincunx) lattice: its nearest neighbors are the
# four up/down/left/right sites, so the tent becomes a diamond.
K_G = np.array([[0, 1, 0],
                [1, 4, 1],
                [0, 1, 0]], dtype=float) / 4.0


def rggb_masks(shape):
    """Boolean masks of where each color was MEASURED in an RGGB mosaic.

    Parameters
    ----------
    shape : (int, int)
        Mosaic height and width.

    Returns
    -------
    bool array (H, W, 3): [..., 0] True at R sites (even, even),
    [..., 1] at both G sites, [..., 2] at B sites (odd, odd).
    """
    m = np.zeros((*shape, 3), dtype=bool)
    m[0::2, 0::2, 0] = True # Red
    m[0::2, 1::2, 1] = True # Green (on red rows)
    m[1::2, 0::2, 1] = True # Green (on blue rows)
    m[1::2, 1::2, 2] = True # Blue
    return m


def normalized_interpolate(values, mask, kernel):
    """Weighted average of only the MEASURED neighbors (normalized convolution).

        est(x) = sum_k K(k) * M(x+k) * V(x+k)  /  sum_k K(k) * M(x+k)

    V holds the samples, M is 1 where a sample was measured and 0 elsewhere.
    Outside the frame both are zero-padded, so a border pixel divides by the
    weight of the neighbors that really exist rather than inventing any: no
    black border, no wrap, no reflection. In the interior of a Bayer mosaic
    the denominator is always 4, so this is exactly ordinary bilinear.

    Parameters
    ----------
    values : float array (H, W)   mosaic (only the masked sites are read)
    mask   : bool array (H, W)    True where this color was measured
    kernel : float array (3, 3)   K_RB or K_G

    Returns
    -------
    float array (H, W): measured values kept exactly, gaps filled.
    """
    m = mask.astype(float)
    num = correlate(values * m, kernel, mode="constant", cval=0.0)
    den = correlate(m, kernel, mode="constant", cval=0.0)
    est = num / den                       # den >= 1 everywhere for a 2x2-tiled mosaic
    return np.where(mask, values, est)    # measured sites returned bit for bit; others estimated


def bilinear_demosaic(mosaic):
    """Bilinear demosaic of an RGGB mosaic; each channel from its own neighbors only.

    Missing value at a site, by where you stand:
        R site : G = mean of 4 cross neighbors,  B = mean of 4 diagonals
        G1 site (red row) : R = mean(left, right),  B = mean(up, down)
        G2 site (blue row): R = mean(up, down),     B = mean(left, right)
        B site : G = mean of 4 cross neighbors,  R = mean of 4 diagonals
    At the frame border, the mean is over whichever of those neighbors exist.

    Frequency view: each color plane is a zero-stuffed (decimated) signal, and
    bilinear is a fixed low-pass (tent) filter on it. It blurs fine detail and
    smooths noise; it never looks at the other channels.

    Parameters
    ----------
    mosaic : float array (H, W), RGGB, linear light. Odd H or W is fine
        (the quarter-res reference from a 3022 x 4030 mosaic is 1511 x 2015).

    Returns
    -------
    float array (H, W, 3), channel order R, G, B.
    """
    mosaic = np.asarray(mosaic, dtype=float)
    masks = rggb_masks(mosaic.shape)
    kernels = (K_RB, K_G, K_RB)
    return np.stack([normalized_interpolate(mosaic, masks[..., c], kernels[c])
                     for c in range(3)], axis=-1)


def busiest_crop(rgb, size=96):
    """(row, col) of the size x size window with the most green-channel detail.

    Uses the standard deviation of tile means of the gradient magnitude, a cheap
    "where are the edges" finder, so figures show a region where demosaicing matters.
    """
    g = rgb[..., 1]
    grad = np.hypot(*np.gradient(g))
    h, w = (g.shape[0] // size) * size, (g.shape[1] // size) * size
    tiles = grad[:h, :w].reshape(h // size, size, w // size, size).mean(axis=(1, 3))
    i, j = np.unravel_index(np.argmax(tiles), tiles.shape)
    return i * size, j * size

# ---------------------------------------------------------------------------
# Item 5: Malvar-He-Cutler (MHC) gradient-corrected linear demosaicing
#   H. S. Malvar, L.-W. He, R. Cutler, "High-quality linear interpolation for
#   demosaicing of Bayer-patterned color images," Proc. IEEE ICASSP, 2004.
# ---------------------------------------------------------------------------
# Every MHC kernel is   (bilinear kernel)  +  gain * (correction kernel).
# The correction kernels read ONLY the channel measured at the center pixel
# (all their taps land on same-color sites), so each one is a high-pass
# "how much does this pixel stand out from its same-color neighbors" detector.
# Each sums to zero: on flat regions the correction vanishes.

# Measured channel is R or B, spacing 2: center minus mean of the 4 at distance 2.
LAP_RB = np.array([[0,     0, -1/4, 0,     0],
                   [0,     0,  0,   0,     0],
                   [-1/4,  0,  1,   0, -1/4],
                   [0,     0,  0,   0,     0],
                   [0,     0, -1/4, 0,     0]])
# Measured channel is G, and the missing color's neighbors are LEFT/RIGHT.
# Green sits on a checkerboard, so its nearest same-color neighbors are the
# diagonals; the shape (weaker, positive vertical taps) is from the paper.
LAP_G_ROW = np.array([[0,    0,   1/10, 0,    0],
                      [0,   -1/5, 0,   -1/5,  0],
                      [-1/5, 0,   1,    0,   -1/5],
                      [0,   -1/5, 0,   -1/5,  0],
                      [0,    0,   1/10, 0,    0]])
LAP_G_COL = LAP_G_ROW.T          # same, when the missing color's neighbors are UP/DOWN

MHC_ALPHA = 1 / 2   # G at an R or B site
MHC_BETA  = 5 / 8   # R or B at a G site
MHC_GAMMA = 3 / 4   # R at a B site, B at an R site


def rggb_site_masks(shape):
    """Boolean masks of the four RGGB site types (G1 and G2 kept separate).

    G1 is green on a red row (red neighbors left/right, blue up/down);
    G2 is green on a blue row (blue neighbors left/right, red up/down).
    MHC needs the distinction; bilinear does not.

    Returns
    -------
    dict str -> bool array (H, W), keys "R", "G1", "G2", "B".
    """
    keys = {"R": (0, 0), "G1": (0, 1), "G2": (1, 0), "B": (1, 1)}
    out = {}
    for k, (r, c) in keys.items():
        m = np.zeros(shape, dtype=bool)
        m[r::2, c::2] = True
        out[k] = m
    return out


def mhc_correction_planes(mosaic, border=2):
    """The three gain-scaled correction images MHC adds on top of bilinear.

    Each is a convolution of the WHOLE mosaic with one correction kernel. It
    is only meaningful at the sites where it gets used (where every tap lands
    on the center's own color), and the caller selects those sites with fixed
    masks, so nothing here depends on the data.

    The 5x5 kernels need 2 pixels of context. Within ``border`` pixels of the
    frame edge the correction is set to 0, i.e. MHC falls back to the
    border-safe bilinear there, rather than inventing pixels outside the frame.

    Parameters
    ----------
    mosaic : float array (H, W), RGGB.
    border : int, width of the bilinear-only frame (2 for 5x5 kernels).

    Returns
    -------
    dict with "rb" (alpha * LAP_RB), "row" (beta * LAP_G_ROW),
    "col" (beta * LAP_G_COL), "diag" (gamma * LAP_RB), each float (H, W).
    """
    planes = {
        "rb":   MHC_ALPHA * correlate(mosaic, LAP_RB, mode="constant"),
        "row":  MHC_BETA * correlate(mosaic, LAP_G_ROW, mode="constant"),
        "col":  MHC_BETA * correlate(mosaic, LAP_G_COL, mode="constant"),
        "diag": MHC_GAMMA * correlate(mosaic, LAP_RB, mode="constant"),
    }
    if border:
        inner = np.zeros(mosaic.shape, dtype=bool)
        inner[border:-border, border:-border] = True
        for k in planes:
            planes[k] = np.where(inner, planes[k], 0.0)
    return planes


def mhc_demosaic(mosaic, border=2, clip=False):
    """Malvar-He-Cutler demosaic: bilinear plus a scaled Laplacian of the measured channel.

    Idea: within a few pixels, the DIFFERENCE between color channels (e.g. R - G)
    changes slowly, while each channel's fine detail is shared, since edges and
    texture are mostly brightness. So the amount by which the measured channel
    stands out from its own neighbors (its Laplacian, a high-pass) is a good
    guess for how much the missing channel stands out too:

        est_missing = bilinear_missing + gain * Laplacian(measured channel)

    The eight cases (missing color, at which site) -> correction used:
        G at R  : alpha * LAP_RB          G at B  : alpha * LAP_RB
        R at G1 : beta  * LAP_G_ROW       R at G2 : beta  * LAP_G_COL
        B at G2 : beta  * LAP_G_ROW       B at G1 : beta  * LAP_G_COL
        R at B  : gamma * LAP_RB          B at R  : gamma * LAP_RB
    with alpha = 1/2, beta = 5/8, gamma = 3/4 (fitted by the authors on the
    Kodak image set, then rounded to multiples of 1/8 for cheap hardware).

    Implementation: 3 bilinear convolutions + 4 correction convolutions, then
    a fixed selection by site type. No branching on pixel values, which is why
    it costs no more than one fixed linear filter per output channel.

    Parameters
    ----------
    mosaic : float array (H, W), RGGB, linear light.
    border : int, frame width that falls back to bilinear (see mhc_correction_planes).
    clip : bool, default False. The negative taps overshoot past strong edges and
        next to clipped highlights, so outputs can fall outside [0, 1]. False keeps
        the overshoot, which is what later linear stages (color matrix, denoise)
        should see; clip once at display/encode time. True clips to [0, 1], matching
        the Lecture 7 reference mhc_demosaic in make_figures_07.py.

    Returns
    -------
    float array (H, W, 3), R, G, B; in [0, 1] only if clip=True.
    """
    mosaic = np.asarray(mosaic, dtype=float)
    base = bilinear_demosaic(mosaic)
    s = rggb_site_masks(mosaic.shape)
    c = mhc_correction_planes(mosaic, border)

    zero = np.zeros(mosaic.shape)
    r_corr = np.where(s["G1"], c["row"], zero) + np.where(s["G2"], c["col"], zero) \
        + np.where(s["B"], c["diag"], zero)
    g_corr = np.where(s["R"] | s["B"], c["rb"], zero)
    b_corr = np.where(s["G2"], c["row"], zero) + np.where(s["G1"], c["col"], zero) \
        + np.where(s["R"], c["diag"], zero)
    out = base + np.stack([r_corr, g_corr, b_corr], axis=-1)
    return np.clip(out, 0.0, 1.0) if clip else out


def _pad5(k3):
    """Embed a 3x3 kernel in the center of a 5x5."""
    k = np.zeros((5, 5))
    k[1:4, 1:4] = k3
    return k


def mhc_full_kernels():
    """The four complete MHC filters, assembled from the pieces mhc_demosaic() uses.

    Each is (bilinear taps on the missing channel) + gain * (correction on the
    measured channel), on one 5x5 grid. Built from the module constants rather
    than retyped, so what is printed is what runs.

        "G at R/B"           : bilinear cross        + alpha * LAP_RB
        "R/B at G, row"      : bilinear left/right   + beta  * LAP_G_ROW
        "R/B at G, column"   : bilinear up/down      + beta  * LAP_G_COL
                               (= "row" rotated 90 degrees)
        "R at B / B at R"    : bilinear 4 diagonals  + gamma * LAP_RB

    These four cover all eight cases: G at R and G at B share one filter,
    R at B and B at R share one, and the four "color at green" cases use the
    row filter or its rotation depending on which way the missing color's
    neighbors lie.

    Returns
    -------
    dict str -> float array (5, 5). Each sums to 1 (flat areas pass unchanged).
    """
    bil_cross = _pad5([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4
    bil_row = _pad5([[0, 0, 0], [1, 0, 1], [0, 0, 0]]) / 2
    bil_diag = _pad5([[1, 0, 1], [0, 0, 0], [1, 0, 1]]) / 4
    # The center tap belongs to the correction: bilinear has none, because the
    # missing color was not measured there.
    return {
        "G at R/B": bil_cross + MHC_ALPHA * LAP_RB,
        "R/B at G, row": bil_row + MHC_BETA * LAP_G_ROW,
        "R/B at G, column": bil_row.T + MHC_BETA * LAP_G_COL,
        "R at B / B at R": bil_diag + MHC_GAMMA * LAP_RB,
    }


def compare_demosaic(mine, theirs, border=2):
    """Largest absolute difference between two demosaiced images, overall and in the interior.

    Demosaicers usually agree in the interior and differ only in how they treat
    the frame edge, so the two numbers separate "the algorithm differs" from
    "the border rule differs".

    Parameters
    ----------
    mine, theirs : float arrays (H, W, 3), same shape.
    border : int, width of the frame excluded from the interior figure
        (2 for 5x5 MHC kernels).

    Returns
    -------
    dict with
        "max_all"      : max |mine - theirs| over the whole image
        "where"        : (row, col, channel) of that maximum
        "in_border"    : True if that maximum lies within ``border`` of the edge
        "max_interior" : max |mine - theirs| with the border frame excluded
    """
    d = np.abs(np.asarray(mine, float) - np.asarray(theirs, float))
    where = np.unravel_index(int(np.argmax(d)), d.shape)
    h, w = d.shape[:2]
    in_border = not (border <= where[0] < h - border and border <= where[1] < w - border)
    return {"max_all": float(d.max()),
            "where": tuple(int(i) for i in where),
            "in_border": in_border,
            "max_interior": float(d[border:h - border, border:w - border].max())}


def measured_site_check(rgb, mosaic, edge=1):
    """How much a demosaicer changed the values it was given (it should change none).

    At every site, the color the sensor measured there is already known, so a
    demosaicer should fill in only the two missing colors and hand the measured
    one back untouched. This compares output[site, measured color] with the
    mosaic value, bit for bit.

    Parameters
    ----------
    rgb : float array (H, W, 3), demosaiced output.
    mosaic : float array (H, W), the RGGB mosaic that was demosaiced.
    edge : int, width of the outer frame reported separately (1 = outermost ring),
        because border handling is where implementations differ.

    Returns
    -------
    dict with
        "sites"      : number of measured sites checked (H * W)
        "changed"    : how many came back different (exact comparison)
        "on_edge"    : how many of those lie within ``edge`` of the frame
        "max_abs"    : largest |output - mosaic| over measured sites
        "exact"      : True if every measured site is bit-identical
    """
    rgb = np.asarray(rgb, float)
    mosaic = np.asarray(mosaic, float)
    masks = rggb_masks(mosaic.shape)
    measured = np.take_along_axis(rgb, np.argmax(masks, axis=-1)[..., None], axis=-1)[..., 0]
    changed = measured != mosaic
    h, w = mosaic.shape
    ring = np.ones((h, w), bool)
    ring[edge:h - edge, edge:w - edge] = False
    return {"sites": int(mosaic.size),
            "changed": int(changed.sum()),
            "on_edge": int((changed & ring).sum()),
            "max_abs": float(np.abs(measured - mosaic).max()),
            "exact": not changed.any()}
