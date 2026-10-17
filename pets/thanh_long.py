"""Thanh Long (azure dragon): pixelize config + hand touch-ups (512px sprite coords).

The automatic pass gives the body, mane and tail. The touch-ups re-pick the colours of
the small, high-contrast features (eyes, mouth, scarf) from the reference with a short
hand-chosen palette per feature, so those areas come out flat and crisp, then redraw
the "IQX" lettering and the diamond pendant as clean geometric shapes.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from pixelize import TRANSPARENT, hex2rgb, majority, merge_islands, nearest, placement, reference, rgb2lab

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
    if not _REF:
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
LID = (40, 46, 66)

EYE_PAIRS = [
    ((2, 5, 14), BLACK),
    ((3, 21, 67), BLACK),
    ((4, 32, 121), IRIS_D),
    ((4, 73, 171), IRIS_M),
    ((30, 79, 135), IRIS_M),
    ((9, 118, 205), IRIS_B),
    ((12, 166, 231), IRIS_L),
    ((168, 176, 199), SCLERA_S),
    ((240, 243, 244), SCLERA),
    ((255, 255, 255), SCLERA),
    ((38, 64, 73), BLACK),
    # skin and soft blends: keep the automatic result
    ((95, 217, 197), None),
    ((106, 232, 208), None),
    ((250, 241, 226), None),
    ((212, 214, 180), None),
    ((88, 120, 133), None),
]

R_IRIS = (22, 84, 136)
RIGHT_EYE_PAIRS = [
    ((3, 5, 15), BLACK),
    ((6, 13, 39), BLACK),
    ((7, 25, 68), BLACK),
    ((37, 79, 110), R_IRIS),
    ((212, 208, 201), (226, 222, 208)),
    ((255, 255, 255), WHITE),
    ((125, 128, 136), None),
    ((73, 152, 162), None),
    ((136, 230, 212), None),
    ((95, 217, 197), None),
]

NOSTRIL = (4, 58, 54)
M_DARK = (44, 8, 12)
M_RED = (176, 60, 58)
TONGUE = (236, 130, 121)
MOUTH_PAIRS = [
    ((33, 4, 6), M_DARK),
    ((35, 9, 10), M_DARK),
    ((123, 49, 49), M_RED),
    ((181, 61, 60), M_RED),
    ((158, 53, 49), M_RED),
    ((172, 118, 113), TONGUE),
    ((235, 129, 120), TONGUE),
    ((115, 232, 212), None),
    ((95, 217, 197), None),
    ((244, 241, 230), None),
    ((240, 230, 214), None),
    ((216, 230, 219), None),
    ((59, 73, 66), None),
    ((95, 148, 133), None),
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
TOE_PAIRS = [
    ((13, 30, 65), NAVY),
    ((7, 15, 45), NAVY),
    ((15, 33, 73), NAVY),
    ((49, 74, 95), CLAW),
    ((54, 68, 84), CLAW),
    ((58, 86, 125), CLAW),
    ((115, 115, 102), TOE_LINE),
    ((111, 118, 112), TOE_LINE),
    ((166, 162, 125), TOE_S),
    ((228, 219, 164), TOE_Y),
    ((242, 239, 205), TOE),
    ((240, 236, 196), TOE),
    ((85, 217, 200), None),
    ((104, 213, 187), None),
    ((150, 201, 170), None),
    ((40, 190, 200), None),
    ((20, 150, 190), None),
]
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
    requant(art, (286, 186, 376, 264), EYE_PAIRS, where=ellipse_mask(331, 225, 41.5, 39.5))
    art.ellipse(352.5, 216.5, 3.9, 3.9, WHITE)                    # catch light, left eye
    # lower lid: the soft grey band becomes a thin crisp dark line along the sclera
    x0, y0, x1, y1 = 288, 232, 330, 262
    sub = art.idx[y0:y1 + 1, x0:x1 + 1]
    grey = np.isin(sub, [art.closest(c) for c in ((106, 116, 122), (54, 67, 76), (148, 142, 120), (166, 179, 170))])
    eye = np.isin(sub, [art.color(c) for c in (SCLERA, SCLERA_S, BLACK)])
    near = eye.copy()
    p = np.pad(eye, 1)
    for dy in range(3):
        for dx in range(3):
            near |= p[dy:dy + sub.shape[0], dx:dx + sub.shape[1]]
    sub[grey & near] = art.color(LID)
    sub[grey & ~near] = art.color(CREAM)
    # eyelash flick at the outer corner (clear the auto pass's grey fringe first)
    sub = art.idx[200:215, 284:294]
    sub[~np.isin(sub, [art.color(BLACK), art.closest(CREAM)])] = art.closest(CREAM)
    art.replace(SCLERA_S, art.closest((240, 245, 242)), (284, 206, 291, 224))
    art.idx[poly_mask([(299, 199.5), (292, 202.5), (287.6, 205.8), (288.6, 208.4), (294, 209.4), (298.5, 213)])] = art.color(BLACK)
    requant(art, (422, 186, 462, 256), RIGHT_EYE_PAIRS, where=ellipse_mask(440, 221, 20, 34))
    despeckle(art, (422, 190, 462, 252), 6)
    art.replace(WHITE, BLACK, (424, 206, 442, 224))
    art.ellipse(430.5, 215.5, 3.7, 3.7, WHITE)                    # catch light, right eye


def mouth(art):
    requant(art, (358, 264, 426, 310), MOUTH_PAIRS, where=ellipse_mask(392, 287, 34, 22))
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
    requant(art, (432, 248, 496, 296), WHISKER_PAIRS, where=poly_mask(WHISKER_R))
    despeckle(art, (432, 248, 496, 296), 5)


def feet(art):
    for box in FEET:
        requant(art, box, TOE_PAIRS)
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


def touchup(art):
    art.name("black", BLACK)
    art.name("white", WHITE)
    eyes(art)
    mouth(art)
    scarf(art)
    despeckle(art, (232, 300, 300, 380), 8)                       # scarf knot
    gem(art)
    whiskers(art)
    feet(art)
    body(art)
