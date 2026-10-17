"""Thanh Long (azure dragon): pixelize config + hand touch-ups (512px sprite coords).

The automatic pass gives the body, mane and tail. touchup() then
  * remode(): re-picks the automatic colours from the full-resolution reference (no blend
    colours along colour edges), and silhouette() drops pixelize's extra 1px outer ring and
    trims the outer line to the original's ~4px weight;
  * re-picks the small, high-contrast features (eyes, mouth, scarf knot, ear frill, tail curl,
    feet) from the full-resolution reference with a short hand-chosen palette per feature;
  * redraws the "IQX" lettering, the diamond pendant and the left whisker as clean geometry;
  * seal_edge() keeps the outermost ring navy.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from pixelize import TRANSPARENT, grow4, hex2rgb, majority, merge_islands, nearest, placement, reference, rgb2lab

SLUG = "thanh-long"
SIZE = 512

CONFIG = {
    "colors": 56,
    "ink_l": 22.0,      # only the real navy line art counts as ink (scarf/iris shading stays blue)
    "ink_frac": 0.45,
    "smooth": 3,
    "min_island": 28,
    "seed": 0,
}

# ---------------------------------------------------------------- helpers ----
_REF: dict = {}


def ref_small():
    """The reference box-downscaled onto the sprite grid exactly like pixelize: (rgba, lab)."""
    if "rgba" not in _REF:
        im = reference(SLUG)
        box, (ox, oy, nw, nh) = placement(SLUG, SIZE, 6)
        small = im.crop(box).convert("RGBa").resize((nw, nh), Image.BOX).convert("RGBA")
        c = np.zeros((SIZE, SIZE, 4))
        c[oy:oy + nh, ox:ox + nw] = np.array(small, np.float64)
        _REF["rgba"] = c
        _REF["lab"] = rgb2lab(c[..., :3])
    return _REF["rgba"], _REF["lab"]


def _rgb(c):
    return hex2rgb(c) if isinstance(c, str) else tuple(int(v) for v in c)


def requant(art, box, pairs, where=None, smooth=1):
    """Re-pick the colours inside box=(x0, y0, x1, y1) (inclusive) from the reference.

    pairs: [(sample_rgb, out)]. Each opaque pixel takes the `out` of the sample nearest
    (in Lab) to the reference colour there; out=None leaves the pixel as it is. A small
    majority filter on the label map keeps the areas flat.
    """
    x0, y0, x1, y1 = box
    _, lab = ref_small()
    sub = lab[y0:y1 + 1, x0:x1 + 1]
    pal = rgb2lab(np.array([_rgb(s) for s, _ in pairs], np.float64))
    lbl = nearest(sub.reshape(-1, 3), pal).reshape(sub.shape[:2])
    region = art.idx[y0:y1 + 1, x0:x1 + 1]
    lbl[region == TRANSPARENT] = TRANSPARENT
    if smooth:
        lbl = majority(lbl, len(pairs), smooth)
    m = region != TRANSPARENT
    if where is not None:
        m &= where[y0:y1 + 1, x0:x1 + 1]
    for j, (_, out) in enumerate(pairs):
        if out is not None:
            region[m & (lbl == j)] = art.color(out)


def ref_hi():
    """The reference crop at full resolution (rgba float, lab) and its sprite placement."""
    if "hi" not in _REF:
        im = reference(SLUG)
        box, place = placement(SLUG, SIZE, 6)
        hi = np.array(im.crop(box), np.float64)
        _REF["hi"] = (hi, rgb2lab(hi[..., :3]), place)
    return _REF["hi"]


def mode_requant(art, box, pairs, where=None, smooth=1, min_cover=0.0):
    """Like requant, but classify the full-resolution reference first and give every sprite
    pixel the class that covers most of it. Edges come out crisper than when classifying
    the box-averaged colour, because blends between two colours never become a third one."""
    hi, hlab, (ox, oy, nw, nh) = ref_hi()
    x0, y0 = max(box[0], ox), max(box[1], oy)              # stay inside the placed reference
    x1, y1 = min(box[2], ox + nw - 1), min(box[3], oy + nh - 1)
    sx, sy = hi.shape[1] / nw, hi.shape[0] / nh
    hx0, hy0 = max(0, int(np.floor((x0 - ox) * sx))), max(0, int(np.floor((y0 - oy) * sy)))
    hx1, hy1 = int(np.ceil((x1 + 1 - ox) * sx)), int(np.ceil((y1 + 1 - oy) * sy))
    sub = hlab[hy0:hy1, hx0:hx1]
    pal = rgb2lab(np.array([_rgb(p) for p, _ in pairs], np.float64))
    lbl = nearest(sub.reshape(-1, 3), pal).reshape(sub.shape[:2])
    w, h = x1 - x0 + 1, y1 - y0 + 1
    # exact sub-box of the hi-res crop that maps onto the sprite box
    fb = ((x0 - ox) * sx - hx0, (y0 - oy) * sy - hy0, (x1 + 1 - ox) * sx - hx0, (y1 + 1 - oy) * sy - hy0)
    cover = np.stack([np.array(Image.fromarray((lbl == j).astype(np.float32)).resize((w, h), Image.BOX, box=fb))
                      for j in range(len(pairs))])
    best = cover.argmax(0)
    best[cover.max(0) < min_cover] = len(pairs)        # no clear winner: keep the automatic colour
    region = art.idx[y0:y1 + 1, x0:x1 + 1]
    best[region == TRANSPARENT] = TRANSPARENT
    if smooth:
        best = majority(best, len(pairs) + 1, smooth)
    m = region != TRANSPARENT
    if where is not None:
        m &= where[y0:y1 + 1, x0:x1 + 1]
    for j, (_, out) in enumerate(pairs):
        if out is not None:
            region[m & (best == j)] = art.color(out)
    return best


def ellipse_mask(cx, cy, rx, ry):
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    return ((xx + 0.5 - cx) / rx) ** 2 + ((yy + 0.5 - cy) / ry) ** 2 <= 1.0


def poly_mask(pts, ss=8):
    """Pixels at least half covered by the polygon (supersampled ss x ss)."""
    m = Image.new("L", (SIZE * ss, SIZE * ss), 0)
    ImageDraw.Draw(m).polygon([(x * ss, y * ss) for x, y in pts], fill=255)
    return np.array(m.resize((SIZE, SIZE), Image.BOX)) >= 128


def poly_dist(pts):
    """Signed distance of every pixel centre to a convex polygon (positive inside)."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE] + 0.5
    pts = np.array(pts, np.float64)
    c = pts.mean(0)
    d = np.full((SIZE, SIZE), np.inf)
    for (ax, ay), (bx, by) in zip(pts, np.roll(pts, -1, 0)):
        nx, ny = by - ay, ax - bx
        n = np.hypot(nx, ny)
        nx, ny = nx / n, ny / n
        if (c[0] - ax) * nx + (c[1] - ay) * ny < 0:
            nx, ny = -nx, -ny
        d = np.minimum(d, (xx - ax) * nx + (yy - ay) * ny)
    return d


def stroke_cover(shapes, box, ss=8):
    """Coverage of capsules / elliptic rings inside box, supersampled; returns a full mask."""
    x0, y0, x1, y1 = box
    w, h = (x1 - x0 + 1) * ss, (y1 - y0 + 1) * ss
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    xx = x0 + (xx + 0.5) / ss
    yy = y0 + (yy + 0.5) / ss
    m = np.zeros((h, w), bool)
    for kind, *a in shapes:
        if kind == "cap":
            ax, ay, bx, by, r = a
            dx, dy = bx - ax, by - ay
            t = np.clip(((xx - ax) * dx + (yy - ay) * dy) / (dx * dx + dy * dy), 0, 1)
            m |= (xx - ax - t * dx) ** 2 + (yy - ay - t * dy) ** 2 <= r * r
        elif kind == "ring":                                       # outer and inner ellipse
            cx, cy, rx, ry, icx, icy, ix, iy = a
            m |= ((((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1) & \
                 ((((xx - icx) / ix) ** 2 + ((yy - icy) / iy) ** 2) > 1)
    cov = m.reshape(h // ss, ss, w // ss, ss).mean((1, 3))
    out = np.zeros((SIZE, SIZE), bool)
    out[y0:y1 + 1, x0:x1 + 1] = cov >= 0.5
    return out


# ---------------------------------------------------------------- palette ----
BLACK = (6, 8, 20)          # pupils, eye rims (the original eyes are black, the line art navy)
NAVY = (5, 23, 74)          # = the automatic outline colour
WHITE = (255, 255, 255)
CREAM = (248, 242, 226)

IRIS_D = (8, 36, 128)
IRIS_M = (6, 76, 178)
IRIS_B = (10, 120, 210)
IRIS_L = (16, 168, 234)
SCLERA = (242, 244, 248)
SCLERA_S = (176, 182, 210)


SCLERA_L = (214, 216, 236)
EYE_PAIRS_HI = [                 # classes of the full-resolution reference
    ((5, 7, 14), BLACK),
    ((4, 22, 68), BLACK),
    ((5, 34, 124), IRIS_D),
    ((5, 79, 175), IRIS_M),
    ((33, 84, 134), IRIS_M),
    ((10, 124, 208), IRIS_B),
    ((12, 168, 232), IRIS_L),
    ((161, 169, 190), SCLERA_S),
    ((205, 206, 228), SCLERA_L),
    ((239, 241, 238), SCLERA),
    ((56, 82, 91), None),
    ((72, 151, 146), None),
    ((103, 220, 199), None),
    ((248, 242, 226), None),
]

R_IRIS = (22, 84, 136)

R_CRES = (226, 222, 208)
RIGHT_EYE_HI = [
    ((3, 7, 23), BLACK),
    ((6, 20, 60), BLACK),
    ((29, 59, 90), BLACK),
    ((54, 102, 129), R_IRIS),
    ((144, 145, 149), None),
    ((219, 220, 205), R_CRES),
    ((255, 255, 255), WHITE),
    ((97, 169, 180), None),
    ((123, 228, 211), None),
    ((248, 242, 226), None),
]

NOSTRIL = (4, 58, 54)
M_DARK = (44, 8, 12)
M_RED = (176, 60, 58)
TONGUE = (236, 130, 121)
M_LINE = (96, 26, 28)
TONGUE_L = (244, 152, 142)
MOUTH_HI = [
    ((20, 15, 14), M_DARK),
    ((40, 9, 10), M_DARK),
    ((88, 28, 27), M_LINE),
    ((152, 60, 59), M_RED),
    ((180, 59, 58), M_RED),
    ((163, 102, 97), M_LINE),
    ((233, 123, 114), TONGUE),
    ((240, 145, 136), TONGUE_L),
    ((74, 101, 90), None),
    ((135, 209, 193), None),
    ((109, 233, 210), None),
    ((246, 241, 228), None),
    ((240, 230, 214), None),
]

SC_D2 = (4, 46, 136)
SC_D1 = (8, 64, 182)
SC = (26, 84, 224)
SC_L = (60, 112, 232)
SCARF_PAIRS = [
    ((4, 15, 60), NAVY),
    ((3, 46, 133), SC_D2),
    ((6, 64, 179), SC_D1),
    ((24, 83, 223), SC),
    ((85, 129, 232), SC_L),
    ((158, 187, 244), SC),
    ((231, 241, 253), None),
    ((244, 241, 230), None),
    ((192, 187, 172), None),
    ((62, 68, 77), None),
    ((15, 133, 161), None),
    ((95, 217, 197), None),
    ((40, 200, 200), None),
    ((30, 190, 240), None),
]
SCARF_POLY = [(238, 334), (246, 320), (262, 320), (270, 308), (290, 311), (330, 316), (380, 317),
              (410, 311), (420, 305), (428, 310), (428, 330), (416, 352), (404, 362), (380, 368),
              (360, 372), (330, 371), (300, 361), (280, 352), (268, 362), (258, 374), (246, 372),
              (236, 356)]

# "IQX": rounded sans, ~5.5px strokes, measured on the reference lettering
LETTERS = [
    ("cap", 333.4, 336.6, 333.4, 352.4, 2.6),                      # I
    ("ring", 351.5, 345.6, 11.5, 10.7, 351.3, 345.0, 6.3, 6.0),    # Q bowl
    ("cap", 355.6, 350.4, 359.4, 356.6, 2.3),                      # Q tail
    ("cap", 367.6, 337.8, 380.8, 352.6, 2.5),                      # X
    ("cap", 381.6, 337.0, 367.8, 353.8, 2.5),
]
LETTER_BOX = (326, 330, 392, 362)

W_EDGE = (30, 92, 90)
W_DARK = (52, 128, 112)
W_MID = (92, 180, 150)
W_FILL = (116, 210, 166)
W_HI = (160, 240, 208)
WHISKER_PAIRS = [
    ((3, 11, 57), NAVY),
    ((0, 17, 48), NAVY),
    ((12, 35, 69), NAVY),
    ((20, 60, 100), NAVY),
    ((36, 96, 95), W_EDGE),
    ((17, 54, 70), W_EDGE),
    ((20, 71, 76), W_EDGE),
    ((62, 129, 130), W_DARK),
    ((49, 134, 106), W_DARK),
    ((72, 139, 120), W_DARK),
    ((101, 184, 167), W_MID),
    ((83, 181, 143), W_MID),
    ((115, 210, 165), W_FILL),
    ((122, 211, 176), W_FILL),
    ((165, 244, 214), W_HI),
    ((240, 245, 238), None),
    ((229, 244, 235), None),
    ((125, 176, 177), None),
    ((127, 160, 148), None),
    ((187, 205, 197), None),
    ((149, 167, 174), None),
    ((104, 120, 134), None),
    ((95, 217, 197), None),
    ((106, 232, 208), None),
    ((248, 242, 226), None),
]
WHISKER_R = [(436, 253), (452, 258), (470, 272), (494, 274), (494, 294), (470, 293), (452, 282), (436, 268)]

CLAW = (36, 62, 100)
TOE_LINE = (128, 124, 96)
TOE_S = (180, 172, 130)
TOE_Y = (232, 222, 160)
TOE = (244, 240, 210)
FEET = [(128, 472, 208, 508), (272, 470, 336, 508), (392, 472, 454, 508)]

G_OUT = (14, 44, 130)
G1 = (8, 98, 209)
G3 = (14, 160, 234)
G5 = (34, 204, 247)
G6 = (90, 232, 250)
G8 = (230, 253, 252)
G_RIM = (240, 248, 244)
GT, GR, GB, GL = (368.3, 362.2), (395.4, 389.9), (370.8, 425.9), (343.1, 392.0)
GC = (369.8, 390.9)                                                 # middle of the girdle


# ---------------------------------------------------------------- features ---
def eyes(art):
    mode_requant(art, (286, 186, 376, 264), EYE_PAIRS_HI, where=ellipse_mask(331, 225, 41.5, 39.5))
    art.ellipse(352.5, 216.5, 3.9, 3.9, WHITE)                    # catch light, left eye
    # crisp rim: outside the rim (left and below, where the skin is cream) the pale eye
    # colours are only blends -> skin; the lower lid becomes a clean 2px black arc
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    outer = ellipse_mask(333, 225, 40, 37)
    zone = ellipse_mask(331, 225, 43, 41) & (xx < 346) & (yy > 196)
    pale = np.isin(art.idx, [art.color(c) for c in (SCLERA, SCLERA_L, SCLERA_S)])
    art.idx[zone & ~outer & pale] = art.closest(CREAM)
    art.idx[outer & ~ellipse_mask(333, 225, 37.9, 34.9) & (yy >= 226) & (xx <= 346)] = art.color(BLACK)
    # grey leftovers just inside the lower lid -> sclera shading; a stray grey dash left of the rim
    lid_in = outer & (yy >= 226)
    art.idx[lid_in & (art.idx == art.closest((106, 116, 122)))] = art.color(SCLERA_S)
    art.idx[lid_in & (art.idx == art.closest((166, 179, 170)))] = art.color(SCLERA_L)
    art.replace(art.closest((148, 142, 120)), art.closest(CREAM), (282, 208, 291, 228))
    # eyelash flick at the outer corner (clear the auto pass's grey fringe first)
    sub = art.idx[200:215, 284:294]
    sub[~np.isin(sub, [art.color(BLACK), art.closest(CREAM)])] = art.closest(CREAM)
    art.replace(SCLERA_S, art.closest((240, 245, 242)), (284, 206, 291, 224))
    art.idx[poly_mask([(299, 199.5), (292, 202.5), (287.6, 205.8), (288.6, 208.4), (294, 209.4), (298.5, 213)])] = art.color(BLACK)
    mode_requant(art, (418, 186, 462, 256), RIGHT_EYE_HI, where=ellipse_mask(438, 221, 22, 34))
    despeckle(art, (418, 190, 462, 252), 6)
    art.replace(WHITE, BLACK, (424, 206, 442, 224))
    art.replace(R_IRIS, BLACK, (416, 196, 425, 238))               # iris tint only low and by the crescent
    art.ellipse(430.2, 215.6, 3.3, 3.3, WHITE)                    # catch light, right eye


def mouth(art):
    mode_requant(art, (358, 264, 426, 310), MOUTH_HI, where=ellipse_mask(392, 287, 34, 22))
    # nostril: a crisp dark slanted dash
    sub = art.idx[250:261, 393:408]
    sub[sub == art.closest((26, 101, 107))] = art.closest((105, 190, 168))
    art.idx[stroke_cover([("cap", 397.6, 253.7, 402.6, 256.4, 1.75)], (393, 250, 407, 260))] = art.color(NOSTRIL)


def scarf(art):
    band = poly_mask(SCARF_POLY)
    requant(art, (232, 300, 430, 378), SCARF_PAIRS, where=band)
    # around the letters every non-blue pixel is lettering halo: make it band blue first
    halo = [(s, SC if o is None and s in ((231, 241, 253), (244, 241, 230)) else o) for s, o in SCARF_PAIRS]
    lb = np.zeros((SIZE, SIZE), bool)
    x0, y0, x1, y1 = LETTER_BOX
    lb[y0:y1 + 1, x0:x1 + 1] = True
    requant(art, LETTER_BOX, halo, where=lb & band)
    txt = stroke_cover(LETTERS, LETTER_BOX)
    near = txt.copy()
    for _ in range(2):
        near = near | np.roll(near, 1, 0) | np.roll(near, -1, 0) | np.roll(near, 1, 1) | np.roll(near, -1, 1)
    ring = near & ~txt & lb
    art.idx[ring & np.isin(art.idx, [art.color(SC_L), art.color(SC_D1)])] = art.color(SC)
    art.idx[txt] = art.color(WHITE)
    art.replace(SC_L, SC, LETTER_BOX)                              # flat band behind the letters
    despeckle(art, LETTER_BOX, 6)


KNOT_D = (4, 34, 104)
KNOT_HI = [
    ((3, 20, 62), NAVY),
    ((2, 31, 95), KNOT_D),
    ((2, 44, 125), SC_D2),
    ((3, 62, 162), SC_D1),
    ((7, 71, 120), None),
    ((55, 104, 107), None),
    ((9, 122, 157), None),
    ((22, 154, 173), None),
    ((34, 189, 191), None),
    ((196, 202, 175), None),
]


def knot(art):
    """Scarf knot at the back: thin crisp navy folds instead of the auto pass's thick bands."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    where = poly_mask(SCARF_POLY) & (xx < 300)
    mode_requant(art, (232, 300, 300, 380), KNOT_HI, where=where)
    despeckle(art, (232, 300, 300, 380), 8)


def gem(art):
    poly = [GT, GR, GB, GL]
    d = poly_dist(poly)
    inside = d > 0
    yy = np.mgrid[0:SIZE, 0:SIZE][0]

    def tri(*pts):
        return poly_mask(list(pts)) & inside

    # leftovers of the automatic gem around the new outline -> chest cream, then a pale rim
    chest = ~poly_mask(SCARF_POLY) & (yy >= 364)
    pal = np.array(art.palette + [(0, 0, 0)], np.float64)
    cur = rgb2lab(pal[art.idx])
    around = (d <= -1.4) & (d > -5) & chest & (cur[..., 2] < -12) & (art.idx != TRANSPARENT)
    art.idx[around] = art.color(CREAM)
    art.idx[(d <= 0) & (d > -1.4) & chest] = art.color(G_RIM)

    art.idx[tri(GT, GC, GL)] = art.color(G5)                      # upper left facet
    art.idx[tri(GT, GR, GC)] = art.color(G6)                      # upper right facet
    art.idx[tri((368.5, 366.0), (378.6, 390.6), (369.6, 390.6))] = art.color(G8)   # light streak
    art.idx[tri(GL, GC, GB)] = art.color(G3)                      # lower left facet
    art.idx[tri(GL, GC, (365.0, 410.0))] = art.color(G1)          # shadowed part of it
    art.idx[tri(GC, GR, GB)] = art.color(G3)                      # lower right facet
    art.idx[tri(GC, (374.6, 390.7), GB)] = art.color(G5)          # light vertical edge
    rim_in = inside & (d <= 3.6) & poly_mask([GT, GC, GL]) & (d > 2.3)
    art.idx[rim_in] = art.color(G6)                                # lit upper-left edge
    art.idx[inside & (d <= 2.3)] = art.color(G_OUT)


def whiskers(art):
    whisker_left(art)
    requant(art, (432, 248, 496, 296), WHISKER_PAIRS, where=poly_mask(WHISKER_R))
    despeckle(art, (432, 248, 496, 296), 5)


def grow8(m, n=1):
    for _ in range(n):
        p = np.pad(m, 1)
        m = m.copy()
        for dy in range(3):
            for dx in range(3):
                m |= p[dy:dy + m.shape[0], dx:dx + m.shape[1]]
    return m


WL_LINE = (22, 74, 88)
WL_FILL = (156, 228, 196)
WL_HI = (196, 236, 192)
WL_MID = (104, 196, 168)


def ribbon(points, half, line_w, box, ss=4):
    """A stroked centre line with per-point half widths: (fill mask, edge mask), each pixel
    taken when at least half of it is covered (supersampled ss x ss)."""
    x0, y0, x1, y1 = box
    w, h = (x1 - x0 + 1) * ss, (y1 - y0 + 1) * ss
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    xx = x0 + (xx + 0.5) / ss
    yy = y0 + (yy + 0.5) / ss
    best = np.full((h, w), np.inf)               # distance minus local half width
    for (ax, ay), (bx, by), ha, hb in zip(points[:-1], points[1:], half[:-1], half[1:]):
        dx, dy = bx - ax, by - ay
        t = np.clip(((xx - ax) * dx + (yy - ay) * dy) / (dx * dx + dy * dy), 0, 1)
        d = np.hypot(xx - ax - t * dx, yy - ay - t * dy) - (ha + (hb - ha) * t)
        best = np.minimum(best, d)

    def down(m):
        out = np.zeros((SIZE, SIZE), bool)
        out[y0:y1 + 1, x0:x1 + 1] = m.reshape(h // ss, ss, w // ss, ss).mean((1, 3)) >= 0.5
        return out
    fill = down(best <= 0)
    edge = down(best <= line_w) & ~fill
    return fill, edge


# centre line of the left whisker, measured on the reference's mint fill (tip by the snout ->
# down-left -> along the bottom of the curl -> up the left arm)
WL_PTS = [(313.5, 261.2), (308, 259.8), (300, 259.4), (293, 259.6), (287, 261.0), (281, 263.6), (276, 267.0),
          (272, 270.4), (268, 274.2), (264.5, 278.2), (261.5, 282.2), (258.5, 286.2), (256, 290.0), (252, 294.2),
          (247.5, 298.2), (241.5, 302.0), (234, 305.0), (226, 306.4), (218, 305.6), (211, 303.0), (205, 299.0),
          (200.6, 294.0), (198.2, 288.0), (197.8, 281.0), (199.2, 274.0), (201.6, 267.4), (205, 263.2), (208.6, 261.0)]
WL_HALF = [0.4, 1.6, 2.5, 3.0, 3.3, 3.5, 3.6,
           3.6, 3.6, 3.6, 3.6, 3.6, 3.6, 3.6,
           3.6, 3.6, 3.5, 3.4, 3.2, 3.0, 2.8,
           2.6, 2.4, 2.3, 2.2, 2.0, 1.7, 1.2]


def whisker_left(art):
    """The curly left whisker: a continuous pale-mint ribbon with a dark teal edge, drawn along
    the centre line of the reference's whisker (the automatic pass broke it into teal blotches)."""
    box = (188, 248, 320, 318)
    fill, edge = ribbon(WL_PTS, WL_HALF, 1.9, box)
    old = grow8(fill | edge, 6)
    # leftovers of the automatic whisker around the new one -> the neighbouring colours
    teal = [art.closest(c) for c in ((105, 190, 168), (60, 145, 138), (26, 101, 107), (136, 233, 213),
                                     (172, 229, 188), (92, 180, 150), (54, 67, 76), (106, 116, 122),
                                     (148, 142, 120), (206, 191, 160), (186, 231, 216))]
    stray = old & ~(fill | edge) & np.isin(art.idx, teal)
    # fill tones from the reference
    tones = [((156, 228, 196), WL_FILL), ((196, 236, 192), WL_HI), ((104, 196, 168), WL_MID)]
    yy, xx = np.nonzero(fill)
    pal = rgb2lab(np.array([t for t, _ in tones], np.float64))
    _, lab = ref_small()
    k = nearest(lab[yy, xx], pal)
    tmp = np.full((SIZE, SIZE), TRANSPARENT, int)
    tmp[yy, xx] = k
    tmp = majority(tmp, len(tones), 2)
    tmp[fill & (tmp == TRANSPARENT)] = 0
    for j, (_, out) in enumerate(tones):
        art.idx[fill & (tmp == j)] = art.color(out)
    art.idx[edge] = art.color(WL_LINE)
    fill_from_neighbours(art, stray, avoid=teal + [art.color(c) for c in (WL_FILL, WL_HI, WL_MID, WL_LINE)])


TOE_HI = [
    ((11, 25, 52), NAVY),
    ((10, 28, 71), NAVY),
    ((60, 71, 77), CLAW),
    ((44, 74, 104), CLAW),
    ((110, 119, 115), TOE_LINE),
    ((141, 138, 105), TOE_LINE),
    ((189, 183, 143), TOE_S),
    ((235, 228, 165), TOE_Y),
    ((242, 240, 206), TOE),
    ((50, 172, 187), None),
    ((149, 202, 167), None),
    ((88, 216, 193), None),
]


def feet(art):
    for box in FEET:
        x0, y0, x1, y1 = box
        mode_requant(art, (x0, y0 + 4, x1, y1), TOE_HI)
        despeckle(art, box, 5)


def despeckle(art, box, min_area):
    x0, y0, x1, y1 = box
    art.idx[y0:y1 + 1, x0:x1 + 1] = merge_islands(art.idx[y0:y1 + 1, x0:x1 + 1], min_area)


def body(art):
    # leg / belly outline: mend the broken soft-grey segment into a clean navy line
    box = (310, 434, 322, 458)
    for c in ((54, 67, 76), (106, 116, 122), (148, 142, 120)):
        art.replace(art.closest(c), art.closest((225, 212, 193)), box)
    art.idx[stroke_cover([("cap", 318.2, 437.0, 316.8, 446.0, 1.0), ("cap", 316.8, 446.0, 316.2, 458.0, 1.1)],
                         box)] = art.color(NAVY)
    art.replace(art.closest((225, 212, 193)), art.closest((22, 172, 192)), (310, 440, 315, 458))
    # hip gloss: the auto pass gives it a muddy grey-green; the original is a cool pale grey
    art.replace(art.closest((166, 179, 170)), (184, 202, 210), (200, 384, 222, 410))


def dist4(m):
    """City-block distance of every pixel of mask m to the outside of m (1 on its edge)."""
    d = np.zeros(m.shape, int)
    cur = m.copy()
    k = 0
    while cur.any():
        k += 1
        d[cur] = k
        p = np.pad(cur, 1)
        cur = cur & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return d


def fill_from_neighbours(art, hole, avoid=()):
    """Give every pixel of `hole` the most common colour among its already-settled
    8-neighbours (not in hole, not transparent, not in `avoid`), growing inwards."""
    idx = art.idx
    bad = set(int(a) for a in avoid) | {TRANSPARENT}
    hole = hole & (idx != TRANSPARENT)
    n = len(art.palette)
    for _ in range(40):
        if not hole.any():
            break
        ok = ~hole & ~np.isin(idx, list(bad))
        p_idx = np.pad(idx, 1, constant_values=TRANSPARENT)
        p_ok = np.pad(ok, 1)
        h, w = idx.shape
        counts = np.zeros((n, h, w), np.int16)
        for dy in range(3):
            for dx in range(3):
                if dy == 1 and dx == 1:
                    continue
                v = p_idx[dy:dy + h, dx:dx + w]
                o = p_ok[dy:dy + h, dx:dx + w]
                for c in np.unique(v[o & hole]) if (o & hole).any() else []:
                    counts[c] += (v == c) & o
        best = counts.argmax(0)
        got = hole & (counts.max(0) > 0)
        idx[got] = best[got]
        hole = hole & ~got


def remode(art, smooth=3, min_island=40):
    """Re-pick the automatic colours from the full-resolution reference: every hi-res pixel
    takes its nearest automatic palette colour and every sprite pixel the colour covering most
    of it. Same palette and flat areas as pixelize, but without the in-between colours that
    box-averaging leaves along every colour edge. Ink (outline) pixels stay as they are."""
    hi, hlab, (ox, oy, nw, nh) = ref_hi()
    k_out = art.names["outline"]
    plab = rgb2lab(np.array(art.palette[:k_out], np.float64))
    lbl = nearest(hlab.reshape(-1, 3), plab).reshape(hlab.shape[:2])
    cover = np.stack([np.array(Image.fromarray((lbl == j).astype(np.float32)).resize((nw, nh), Image.BOX))
                      for j in range(k_out)])
    best = np.full((SIZE, SIZE), TRANSPARENT, int)
    best[oy:oy + nh, ox:ox + nw] = cover.argmax(0)
    idx = art.idx
    m = (idx != TRANSPARENT) & (idx != k_out)
    new = idx.copy()
    new[m] = best[m]
    new = majority(new, len(art.palette), smooth)
    new[idx == k_out] = k_out
    new[idx == TRANSPARENT] = TRANSPARENT
    art.idx[:] = merge_islands(new, min_island)


WHISKER_TIP = (484, 268, 500, 296)          # the original's whisker end has no line across it


def silhouette(art):
    """Outer line: drop pixelize's extra 1px ring so the silhouette matches the original,
    make the edge a continuous navy line and trim the line to the original's ~4px weight."""
    rgba, lab = ref_small()
    L = lab[..., 0]
    inside = rgba[..., 3] >= 128
    art.idx[~inside] = TRANSPARENT
    # nubs: lone pixels sticking out of / notched into the edge
    for _ in range(2):
        op = art.idx != TRANSPARENT
        p = np.pad(op, 1)
        nb = p[:-2, 1:-1].astype(int) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
        art.idx[op & (nb <= 1)] = TRANSPARENT
        art.idx[~op & (nb >= 3) & inside] = art.color(NAVY)
    op = art.idx != TRANSPARENT
    d = dist4(op)
    navy = art.color(NAVY)
    tip = np.zeros_like(op)
    x0, y0, x1, y1 = WHISKER_TIP
    tip[y0:y1 + 1, x0:x1 + 1] = True
    is_n = art.idx == navy
    want = (d == 1) | (d == 2) | (((d == 3) | (d == 4)) & (L < 45)) | ((d == 3) & is_n) | ((d == 4) & is_n)
    want |= (d == 5) & is_n & (L < 24)
    want |= (d >= 6) & (d <= 7) & is_n & (L < 19)
    want &= ~tip
    # clean the inner edge of the band: binary 3x3 majority inside the band zone
    zone = (d >= 3) & (d <= 7) & ~tip
    for _ in range(2):
        p = np.pad(want, 1)
        cnt = sum(p[dy:dy + SIZE, dx:dx + SIZE].astype(int) for dy in range(3) for dx in range(3))
        want = np.where(zone, np.where(cnt >= 6, True, np.where(cnt <= 3, False, want)), want)
    drop = is_n & ~want & (d <= 7) & ~tip
    art.idx[want & op] = navy
    fill_from_neighbours(art, drop, avoid=[navy])


TUFT_R_HI = [(4, 8, 20), (7, 25, 78), (50, 66, 93), (5, 86, 200), (18, 112, 185), (83, 124, 146),
             (163, 164, 159), (111, 184, 195), (226, 216, 189), (247, 242, 223)]


def tuft_right(art):
    """The far ear frill: grey and teal specks from the automatic pass -> cream, tan, navy."""
    out = [BLACK, NAVY, NAVY, art.closest((5, 86, 200)), art.closest((18, 112, 185)), None,
           art.closest((206, 191, 160)), None, art.closest((225, 212, 193)), art.closest(CREAM)]
    pairs = list(zip(TUFT_R_HI, out))
    where = poly_mask([(426, 140), (446, 136), (462, 140), (464, 198), (446, 194), (428, 186)])
    mode_requant(art, (424, 134, 466, 200), pairs, where=where)
    despeckle(art, (424, 134, 466, 200), 5)
    art.replace(art.closest((5, 86, 200)), NAVY, (448, 160, 462, 196))


CURL_HI = [(3, 22, 66), (11, 64, 126), (18, 73, 116), (40, 116, 166), (9, 137, 173), (19, 172, 197),
           (94, 172, 213), (34, 199, 209), (188, 229, 241)]


def tail_curl(art):
    """Where the tail's inner edge curls against the back, the blurred reference merged two
    lines into one navy blob; re-pick it from the full-resolution reference to get the two thin
    lines and the pale glint between them back."""
    pairs = [(c, NAVY if i == 0 else art.closest(c)) for i, c in enumerate(CURL_HI)]
    mode_requant(art, (132, 336, 166, 392), pairs, where=ellipse_mask(149, 364, 17, 28))
    art.ellipse(145.6, 361.2, 1.4, 3.6, art.closest((188, 229, 241)), only=[art.closest(c) for c in CURL_HI[3:]])


def seal_edge(art):
    """Feature re-picks run after silhouette() and may leave a light pixel on the outer edge:
    keep the outermost ring navy (black where the eye touches it, open at the whisker tip)."""
    op = art.idx != TRANSPARENT
    edge = op & ~(np.pad(op, 1)[:-2, 1:-1] & np.pad(op, 1)[2:, 1:-1] & np.pad(op, 1)[1:-1, :-2] & np.pad(op, 1)[1:-1, 2:])
    x0, y0, x1, y1 = WHISKER_TIP
    edge[y0:y1 + 1, x0:x1 + 1] = False
    art.idx[edge & ~np.isin(art.idx, [art.color(NAVY), art.color(BLACK)])] = art.color(NAVY)


def touchup(art):
    art.name("black", BLACK)
    art.name("white", WHITE)
    remode(art)
    silhouette(art)
    eyes(art)
    mouth(art)
    scarf(art)
    knot(art)
    gem(art)
    whiskers(art)
    feet(art)
    body(art)
    tail_curl(art)
    tuft_right(art)
    seal_edge(art)
