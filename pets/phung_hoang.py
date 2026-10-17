"""Phung Hoang (phoenix): pixelize config and hand touch-ups (512x512 sprite coords)."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from pixelize import TRANSPARENT, majority, merge_islands, nearest, placement, reference, rgb2lab

SLUG = "phung-hoang"

CONFIG = {
    "colors": 56,
    "ink_l": 27.0,
    "ink_frac": 0.6,
    "smooth": 2,
    "min_island": 12,
    "seed": 0,
}

# ---------------------------------------------------------------- helpers

_CACHE = {}


def ref_small(size=512, margin=6):
    """Reference box-downscaled onto the sprite grid (what pixelize sees), RGBA float."""
    if "small" not in _CACHE:
        im = reference(SLUG)
        box, (ox, oy, nw, nh) = placement(SLUG, size, margin)
        small = np.array(im.crop(box).convert("RGBa").resize((nw, nh), Image.BOX).convert("RGBA"), np.float64)
        canvas = np.zeros((size, size, 4))
        canvas[oy:oy + nh, ox:ox + nw] = small
        _CACHE["small"] = canvas
        _CACHE["lab"] = rgb2lab(canvas[..., :3])
    return _CACHE["small"]


def ref_lab():
    ref_small()
    return _CACHE["lab"]


def poly(points, shape=(512, 512)):
    """Boolean mask of a filled polygon (sprite coords, pixel corners)."""
    img = Image.new("L", (shape[1], shape[0]), 0)
    ImageDraw.Draw(img).polygon([tuple(p) for p in points], fill=1, outline=1)
    return np.array(img, bool)


def ell(cx, cy, rx, ry, shape=(512, 512)):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return ((xx + 0.5 - cx) / rx) ** 2 + ((yy + 0.5 - cy) / ry) ** 2 <= 1.0


def smooth_region(art, mask, passes=1):
    """Majority-filter art.idx inside mask only (other pixels are context)."""
    if passes <= 0 or not mask.any():
        return
    ys, xs = np.where(mask)
    y0, y1 = max(ys.min() - 1, 0), min(ys.max() + 2, art.idx.shape[0])
    x0, x1 = max(xs.min() - 1, 0), min(xs.max() + 2, art.idx.shape[1])
    sub = art.idx[y0:y1, x0:x1]
    uniq = np.unique(sub[sub >= 0])
    lut = np.full(max(uniq.max() + 2, 1), -1)
    lut[uniq] = np.arange(len(uniq))
    comp = np.where(sub >= 0, lut[np.clip(sub, 0, None)], -1)
    comp = majority(comp, len(uniq), passes)
    back = np.where(comp >= 0, uniq[np.clip(comp, 0, None)], -1)
    m = mask[y0:y1, x0:x1]
    sub[m] = back[m]


def requantize(art, mask, colors, smooth=1):
    """Re-map masked pixels to the nearest (Lab) of `colors`, from the reference."""
    ids = [art.color(c) for c in colors]
    pal = rgb2lab(np.array([art.palette[i] for i in ids], np.float64))
    lab = ref_lab()[mask]
    art.idx[mask] = np.array(ids)[nearest(lab, pal)]
    smooth_region(art, mask, smooth)


def dilate(m, r=1):
    """Square (8-connected) dilation by r pixels."""
    out = m.copy()
    for _ in range(r):
        p = np.pad(out, 1)
        out = (p[:-2, :-2] | p[:-2, 1:-1] | p[:-2, 2:] | p[1:-1, :-2] | p[1:-1, 1:-1] | p[1:-1, 2:]
               | p[2:, :-2] | p[2:, 1:-1] | p[2:, 2:])
    return out


def fill_holes(mask):
    """Fill enclosed holes of a boolean mask (flood fill from the border)."""
    reach = np.zeros_like(mask)
    reach[0, :] = ~mask[0, :]
    reach[-1, :] = ~mask[-1, :]
    reach[:, 0] |= ~mask[:, 0]
    reach[:, -1] |= ~mask[:, -1]
    while True:
        p = np.pad(reach, 1)
        grown = (reach | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]) & ~mask
        if (grown == reach).all():
            return mask | ~reach
        reach = grown


def merge_region(art, mask, min_area):
    """merge_islands() applied only to pixels inside mask."""
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = art.idx[y0:y1, x0:x1]
    merged = merge_islands(sub, min_area)
    mm = mask[y0:y1, x0:x1]
    sub[mm] = merged[mm]


def fold_colors(art, ids, region=None, exclude=()):
    """Repaint pixels of the given palette ids with the most common other
    colour (not in `exclude`) among their 8 neighbours, repeated so clusters
    fill from their edge."""
    ids = list(ids)
    bad = ids + list(exclude)
    if not ids:
        return
    idx = art.idx
    n = len(art.palette)
    for _ in range(10):
        todo = np.isin(idx, ids)
        if region is not None:
            todo &= region
        if not todo.any():
            return
        counts = np.zeros((n,) + idx.shape, np.int16)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    si = shift(idx, dy, dx, -1)
                    ok = (si >= 0) & ~np.isin(si, bad)
                    np.add.at(counts, (si[ok], *np.nonzero(ok)), 1)
        best, has = counts.argmax(0), counts.max(0) > 0
        idx[todo & has] = best[todo & has]


def cleanup_auto_colors(art, n_auto):
    """The k-means palette leaves a few stray tones: near-black specks that are
    really outline, greys on feather tips and colours used by only a handful
    of pixels. Fold them into the outline / their neighbours."""
    pal = rgb2lab(np.array(art.palette[:n_auto], np.float64))
    o = art.names["outline"]
    chroma = np.hypot(pal[:, 1], pal[:, 2])
    count = np.bincount(art.idx[art.idx >= 0], minlength=len(art.palette))
    dark = [i for i in range(n_auto) if i != o and pal[i, 0] < 20]
    art.idx[np.isin(art.idx, dark)] = o
    odd = [i for i in range(n_auto) if i != o and ((chroma[i] < 14 and 20 <= pal[i, 0] < 88)
                                                  or 0 < count[i] < 12)]
    fold_colors(art, odd)


def shift(a, dy, dx, fill):
    """a shifted so that out[y, x] = a[y + dy, x + dx]."""
    h, w = a.shape
    n = max(abs(dy), abs(dx), 1)
    p = np.pad(a, n, constant_values=fill)
    return p[n + dy:n + dy + h, n + dx:n + dx + w]


def edt(solid, R=8):
    """Euclidean distance of each pixel centre to the nearest non-solid pixel (capped at R+1)."""
    dist = np.where(solid, float(R + 1), 0.0)
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            r2 = dy * dy + dx * dx
            if 0 < r2 <= R * R:
                hole = shift(~solid, dy, dx, True)
                dist = np.where(hole & (dist > r2 ** 0.5), r2 ** 0.5, dist)
    return dist


def fill_from_neighbours(art, holes, avoid=()):
    """Give each hole pixel the colour of a solid 4-neighbour (not in `avoid`), growing inward."""
    idx = art.idx
    bad = set(avoid)
    idx[holes] = -2
    for _ in range(12):
        todo = idx == -2
        if not todo.any():
            break
        new = np.full(idx.shape, -3)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            si = shift(idx, dy, dx, -1)
            ok = todo & (new == -3) & (si >= 0) & ~np.isin(si, list(bad))
            new[ok] = si[ok]
        idx[new >= 0] = new[new >= 0]
    idx[idx == -2] = art.names["outline"]


def thin_outline(art):
    """pixelize adds a 1px ring of ink outside the alpha>=50% mask. Where the
    mask edge is already inked (the original's own dark contour), that ring only
    makes the line heavier than in the original, so drop it there. The ring is
    kept wherever the edge pixel is not ink, so the contour stays closed."""
    o = art.names["outline"]
    idx = art.idx
    mask = ref_small()[..., 3] >= 128
    ring = (idx == o) & ~mask
    ok = np.ones(idx.shape, bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                si, sm = shift(idx, dy, dx, -1), shift(mask, dy, dx, False)
                ok &= ~sm | (si == o)
    idx[ring & ok] = TRANSPARENT


def trim_band(art, keep=3.0, zone=5.5, reach=3):
    """Silhouette ink deeper than `keep` px is trimmed back to the fill colour,
    unless it belongs to an interior line: ink pixels connected (within `reach`
    steps through ink) to ink deeper than `zone` are left alone."""
    o = art.names["outline"]
    ink = art.idx == o
    e = edt(art.idx >= 0)
    cand = ink & (e > keep + 0.01) & (e <= zone)
    line = ink & (e > zone)
    for _ in range(reach):
        p = np.pad(line, 1)
        line = line | ((p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]) & cand)
    fill_from_neighbours(art, cand & ~line, avoid=[o])


def solidify_edge(art, n_auto, depth=3.5):
    """Dark brown k-means tones that sit right against the silhouette ink are
    the original's soft line edge: make them ink so the contour is one clean
    colour instead of an outline with brown smudges."""
    o = art.names["outline"]
    pal = rgb2lab(np.array(art.palette[:n_auto], np.float64))
    chroma = np.hypot(pal[:, 1], pal[:, 2])
    browns = [i for i in range(n_auto) if i != o and pal[i, 0] < 40 and chroma[i] < 50 and pal[i, 2] > 5]
    e = edt(art.idx >= 0, 5)
    for _ in range(2):
        near = dilate(art.idx == o, 1)
        art.idx[np.isin(art.idx, browns) & (e <= depth) & near] = o
    # lighter muddy browns/greys along the contour belong to the feather fill
    muddy = [i for i in range(n_auto) if i != o and 35 <= pal[i, 0] < 56 and chroma[i] < 42]
    region = (e <= depth + 1) & dilate(art.idx == o, 2)
    region &= ~dilate(FEET, 2)
    fold_colors(art, muddy, region, exclude=[o])


# ---------------------------------------------------------------- palette

INK = (51, 21, 38)          # outline (auto colour, kept)
BLACK = (14, 9, 12)         # pupils / lashes are neutral black in the original
RBROWN = (112, 26, 10)      # iris rim
RORANGE = (168, 52, 8)
IORANGE = (226, 98, 8)
AMBER = (240, 140, 14)
YELLOW = (250, 186, 16)
WHITE = (255, 255, 255)
SCLERA = (244, 243, 240)
SCLERA_G = (196, 184, 186)
CREAM = (249, 240, 226)
CREAM2 = (249, 233, 210)
CREAM3 = (239, 209, 183)


NAVY = (8, 30, 88)
DBLUE = (9, 46, 130)
MBLUE = (13, 68, 170)
BLUE = (22, 94, 204)
LETTER = (244, 250, 255)


def scarf(art):
    lab = ref_lab()
    a = ref_small()[..., 3]
    box = np.zeros(lab.shape[:2], bool)
    box[268:352, 256:452] = True
    blue = box & (a >= 128) & (lab[..., 2] < -18) & (lab[..., 0] < 80)
    blue = fill_holes(blue) & box             # swallow the letters
    near = box & dilate(blue, 2) & ~blue & (lab[..., 0] < 36) & (a >= 64)
    m = blue | near
    requantize(art, m, [INK, NAVY, DBLUE, MBLUE, BLUE], smooth=0)
    smooth_region(art, m, 2)
    merge_region(art, m, 16)
    # clean band behind the lettering, then the letters themselves
    letters = blue & (lab[..., 0] > 55)
    art.idx[dilate(letters, 2) & blue] = art.color(BLUE)
    art.idx[letter_mask()] = art.color(LETTER)


def scarf_smooth(art):
    """Second pass on the scarf: the blue shading still had small ragged
    patches; widen the majority filter so it reads as a few soft folds, and
    give the lettering one even blue behind it (the Q's counter included)."""
    o = art.names["outline"]
    blues = [art.color(c) for c in (NAVY, DBLUE, MBLUE, BLUE)]
    m = np.isin(art.idx, blues)
    majority_wide(art, m, 2, 1)
    majority_wide(art, m, 1, 1)
    merge_in(art, m, 30, avoid=[o, art.color(LETTER)])
    letters = letter_mask()
    band = dilate(letters, 2) | (ell(372.3, 324.0, 6.0, 6.4) & ~letters)
    art.idx[band & m] = art.color(BLUE)
    art.idx[letters] = art.color(LETTER)


def letter_mask(shape=(512, 512)):
    """'IQX' as on the original: ~20px tall, 4px strokes, slightly slanted
    with the scarf, I at x354-359, Q at x363-381, X at x382-400 (y312-333).
    The original's Q tail is barely visible; a short tail under the bowl
    keeps it reading as IQX without crowding the X."""
    m = np.zeros(shape, bool)
    for y in range(312, 332):                      # I: 4px bar, two 1px steps
        x0 = 356 if y <= 317 else (355 if y <= 324 else 354)
        m[y, x0:x0 + 4] = True
    m |= ell(372.3, 324.0, 9.6, 10.0) & ~ell(372.3, 324.0, 5.6, 6.0)   # Q bowl
    for x, y in ((376, 333), (377, 333), (377, 334), (378, 334), (379, 334)):
        m[y, x] = True                             # Q tail
    for y in range(315, 334):                      # X: two diagonal strokes
        t = y - 315
        xa = int(round(386.3 + 0.54 * t - 2.0))
        xb = int(round(398.4 - 0.79 * t - 2.0))
        m[y, xa:xa + 4] = True
        m[y, xb:xb + 4] = True
    return m


GEM_LINE = (122, 32, 16)
GEM_DEEP = (232, 82, 10)
GEM_OR = (246, 116, 24)
GEM_AMB = (248, 162, 46)
GEM_YEL = (250, 212, 96)
GEM_PALE = (252, 242, 190)


GEM_POLY = poly([(378.0, 341.0), (401.0, 370.0), (377.0, 399.0), (354.0, 367.0)])


def gem(art):
    """Orange diamond pendant: clean 2px red-brown outline, facets from the original."""
    P = GEM_POLY
    # the auto pass smeared orange past the gem edge: the original has a pale
    # glow around the pendant, so repaint a few px around it in creams
    yy = np.mgrid[0:512, 0:512][0]
    halo = dilate(P, 4) & ~P & (yy >= 347) & (art.idx != art.names["outline"])
    requantize(art, halo, [CREAM, CREAM2, CREAM3, (253, 248, 238)], smooth=1)
    e = edt(P, 4)
    inner = P & (e > 2.9)
    requantize(art, inner, [GEM_DEEP, GEM_OR, GEM_AMB, GEM_YEL, GEM_PALE], smooth=1)
    merge_region(art, inner, 6)
    art.idx[P & ~inner] = art.color(GEM_LINE)


BEAK_LINE = (84, 12, 14)
BEAK_RED = (190, 58, 20)
BEAK_SH = (232, 116, 40)
BEAK_OR = (246, 146, 52)
BEAK_LT = (250, 200, 116)
PINK = (238, 116, 112)
PINK_D = (196, 80, 74)


BEAK_GLOW = (253, 248, 238)


def beak(art):
    m = BEAK
    requantize(art, m, [BEAK_LINE, BEAK_RED, BEAK_SH, BEAK_OR, BEAK_LT, PINK, PINK_D,
                        CREAM, CREAM2, CREAM3], smooth=1)
    merge_region(art, m, 5)


def beak_edge(art):
    """The original beak has a thin dark rust edge on top and a white glow
    around it; quantised, the edge was missing and the glow came out beige."""
    pal = rgb2lab(np.array(art.palette, np.float64))
    warm = [i for i in range(len(art.palette))
            if pal[i, 0] > 45 and np.hypot(pal[i, 1], pal[i, 2]) > 40 and pal[i, 2] > 0]
    beak_ids = [art.color(c) for c in (BEAK_LINE, BEAK_RED, PINK_D)] + warm
    shape = dilate(BEAK, 3) & np.isin(art.idx, beak_ids)
    p = np.pad(shape, 1)
    inner = p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    edge = shape & ~inner
    pink = np.isin(art.idx, [art.color(PINK), art.color(PINK_D)])
    art.idx[edge & np.isin(art.idx, warm) & ~pink] = art.color(BEAK_RED)
    creams = cream_ids(art)
    ring = dilate(shape, 1) & ~shape & np.isin(art.idx, creams)
    art.idx[ring] = art.color(BEAK_GLOW)
    ring2 = dilate(shape, 4) & ~dilate(shape, 1) & np.isin(art.idx, creams) & ~(EYE_L | EYE_R)
    art.idx[ring2] = art.color(CREAM)


F_DARK = (92, 40, 36)
F_RED = (138, 62, 40)
F_MID = (154, 90, 62)
F_LT = (186, 126, 90)
F_HL = (214, 172, 134)
CLAW = (110, 98, 122)
CLAW_L = (172, 162, 178)


FOOT_L = poly([(240, 446), (258, 444), (262, 450), (276, 450), (284, 444), (300, 446), (300, 470),
               (312, 476), (312, 506), (228, 506), (228, 470), (238, 466)])
FOOT_R = poly([(330, 446), (346, 440), (350, 452), (366, 452), (386, 446), (388, 506), (310, 506),
               (312, 470), (326, 466)])
FEET = FOOT_L | FOOT_R


def feet(art):
    a = ref_small()[..., 3] >= 128
    left, right = FOOT_L, FOOT_R
    cols = [INK, F_DARK, F_RED, F_MID, F_LT, F_HL, CLAW, CLAW_L]
    lab = ref_lab()
    orange = (lab[..., 0] > 55) & (lab[..., 2] > 45)      # leg feathers stay as they are
    for m in (left, right):
        m = m & a & (art.idx >= 0) & ~dilate(orange, 1)
        requantize(art, m, cols, smooth=2)
        merge_region(art, m, 8)
    # toe highlights and claw glints of the original
    toes = (left | right) & (art.idx >= 0) & (art.idx != art.names["outline"])
    art.idx[ell(271.5, 483.0, 2.6, 5.0) & toes] = art.color(F_LT)
    art.idx[ell(271.5, 480.0, 1.6, 2.6) & toes] = art.color(F_HL)
    art.idx[ell(291.5, 476.5, 3.2, 1.8) & toes] = art.color(F_HL)
    for (x0, y0), (x1, y1) in (((301, 480), (302, 489)), ((377, 482), (378, 490)), ((243, 492), (241, 495))):
        art.line(x0, y0, x1, y1, CLAW_L)


FOOT_R_ALL = poly([(330, 446), (346, 440), (350, 452), (366, 452), (386, 446), (404, 460), (404, 506),
                   (310, 506), (312, 470), (326, 466)])


_YY, _XX = np.mgrid[0:512, 0:512]
FOOT_INNER = (FOOT_L & (_YY >= 452) & (_XX >= 236)) | (FOOT_R_ALL & (_YY >= 456) & (_XX >= 316))


def feet_redraw(art):
    """The original feet are blurry; quantised they turned into brown blobs
    with a 5px contour, no clear toes and a dark mass where the right foot's
    outer toe is. Redraw their insides inside the clean contour: three
    browns from the original's shading, a dark junction line under the leg
    feathers, two lines splitting each foot into three toes, pale claw
    highlights and blue-grey claw tips where the original has them."""
    o = art.names["outline"]
    yy, xx = _YY, _XX
    solid = art.idx >= 0
    e = edt(solid, 4)
    region = FOOT_INNER & solid & (e > OUT_W + 0.01)
    ids = [art.color(c) for c in (F_RED, F_MID, F_LT)]
    pal = rgb2lab(np.array([art.palette[i] for i in ids], np.float64))
    lab = rgb2lab(ref_soft(1.0)[region])
    art.idx[region] = np.array(ids)[nearest(lab, pal)]
    majority_wide(art, region, 2, 1)
    majority_wide(art, region, 1, 2)
    merge_in(art, region, 16, avoid=[o])

    def ln(pts, c, w=2):
        """Polyline painted only on the body, never on the background."""
        before = art.idx.copy()
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            art.line(x0, y0, x1, y1, c, width=w)
        art.idx[~solid] = before[~solid]
    # leg tops: dark line under the feather fringe
    ln([(257, 451), (278, 451)], o)
    ln([(337, 456), (357, 455)], o)
    # toe splits
    ln([(262, 471), (262, 480), (263, 490)], o)
    ln([(282, 471), (282, 481), (281, 491)], o)
    ln([(346, 473), (346, 482), (347, 490)], o)
    ln([(370, 470), (371, 480), (372, 489)], o)
    # pale claw highlights
    hl, lt = art.color(F_HL), art.color(F_LT)
    toes = region & (art.idx != o)
    for cx, cy, rx, ry, c in ((272.5, 482.0, 2.6, 7.5, lt), (272.5, 480.0, 1.6, 5.0, hl),
                              (248.5, 472.5, 2.5, 1.6, hl), (290.5, 475.5, 3.0, 1.6, hl),
                              (329.0, 482.5, 1.6, 4.5, hl), (365.5, 474.5, 3.5, 1.6, hl),
                              (384.0, 471.5, 3.6, 1.6, hl)):
        art.idx[ell(cx, cy, rx, ry) & toes] = c
    # claw crescents hugging the contour of the outer toes
    ink = art.idx == o
    toes = region & (art.idx != o)
    art.idx[toes & shift(ink, 0, -1, False) & (xx <= 247) & (yy >= 478) & (yy <= 493)] = hl
    art.idx[toes & shift(ink, 0, 1, False) & (xx >= 294) & (xx <= 308) & (yy >= 480) & (yy <= 494)] = hl
    art.idx[toes & shift(ink, 0, 1, False) & (xx >= 388) & (yy >= 474) & (yy <= 484)] = hl
    # blue-grey claw tips on the right foot
    for pts in (((377, 490), (378, 494)), ((395, 486), (397, 491))):
        ln(list(pts), CLAW, 2)

SHADE = (232, 190, 152)


def cream_ids(art):
    pal = rgb2lab(np.array(art.palette, np.float64))
    chroma = np.hypot(pal[:, 1], pal[:, 2])
    return [i for i in range(len(art.palette)) if pal[i, 0] >= 74 and pal[i, 2] >= 0 and chroma[i] < 36]


def flatten_cream(art, region, colors, passes=2, min_island=24):
    """Re-shade the cream feathers of a region with a few tones: no blotches."""
    m = region & np.isin(art.idx, cream_ids(art))
    requantize(art, m, colors, smooth=0)
    smooth_region(art, m, passes)
    merge_region(art, m, min_island)


FACE_POLY = poly([(286, 214), (300, 172), (318, 160), (350, 160), (378, 176), (400, 196), (420, 182),
                  (446, 176), (466, 196), (472, 240), (468, 270), (446, 286), (400, 296), (340, 296),
                  (300, 284), (282, 256)])
CHEST_POLY = poly([(306, 340), (330, 336), (420, 330), (430, 360), (424, 410), (404, 430), (360, 440),
                   (318, 438), (300, 410), (296, 370)])
PEACH = (226, 172, 134)


def face(art):
    region = FACE_POLY & ~EYE_L & ~EYE_R & ~BEAK
    flatten_cream(art, region, [CREAM, CREAM3, SHADE])
    flatten_cream(art, CHEST_POLY, [CREAM, CREAM2, CREAM3, SHADE])


def reshade_cream(art):
    """Second, smoother pass over the cream face and chest: the first pass
    still left wavy beige blotches, and the flame pass turned the peach
    shadow at the bottom of the chest into a muddy brown. Re-shade from a
    well-blurred original with a wide majority filter so the shading forms
    soft crescents as in the original."""
    o = art.names["outline"]
    pal = rgb2lab(np.array(art.palette, np.float64))
    chroma = np.hypot(pal[:, 1], pal[:, 2])
    peachy = [i for i in range(len(art.palette))
              if i != o and pal[i, 0] >= 54 and chroma[i] < 46 and 0 <= pal[i, 2] < 40 and pal[i, 1] < 32]
    eye_core = np.isin(art.idx, [art.color(c) for c in (BLACK, RBROWN, RORANGE, IORANGE, AMBER, YELLOW,
                                                        SCLERA, SCLERA_G, WHITE)]) & (EYE_L | EYE_R)
    keep_out = dilate(eye_core, 2) | dilate(BEAK, 1) | dilate(GEM_POLY, 5)
    rgb = ref_soft(1.5)
    for region, cols in ((FACE_POLY | EYE_L | EYE_R, [CREAM, CREAM3, SHADE]),
                         (CHEST_POLY, [CREAM, CREAM2, CREAM3, SHADE, PEACH])):
        m = region & np.isin(art.idx, peachy) & ~keep_out
        ids = [art.color(c) for c in cols]
        cl = rgb2lab(np.array([art.palette[i] for i in ids], np.float64))
        art.idx[m] = np.array(ids)[nearest(rgb2lab(rgb[m]), cl)]
        majority_wide(art, m, 2, 2)
        majority_wide(art, m, 1, 1)
        merge_in(art, m, 40, avoid=[o])


TAIL_TIP = (190, 446, 232, 476)     # the original's grey-brown feather tip by the left foot


def demud(art, n_auto):
    """Leftover automatic dark/muddy tones (dark red-brown, mauve, grey-brown)
    sit as specks and second lines next to the contour and feather lines.
    The darkest become outline, the rest fold into the neighbouring fill."""
    o = art.names["outline"]
    pal = rgb2lab(np.array(art.palette[:n_auto], np.float64))
    chroma = np.hypot(pal[:, 1], pal[:, 2])
    mud = [i for i in range(n_auto) if i != o and (pal[i, 0] < 30 or (pal[i, 0] < 58 and chroma[i] < 40))]
    yy, xx = np.mgrid[0:512, 0:512]
    x0, y0, x1, y1 = TAIL_TIP
    protect = dilate(EYE_L | EYE_R | BEAK | GEM_POLY, 2) | FOOT_INNER
    protect |= (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
    m = np.isin(art.idx, mud) & ~protect
    dark = [i for i in mud if pal[i, 0] < 26]
    art.idx[m & np.isin(art.idx, dark)] = o
    m &= art.idx != o
    fold_colors(art, mud, m, exclude=[o])
    art.idx[m & np.isin(art.idx, mud)] = o


def belly_line(art):
    """The line under the belly came out as two dark lines with a brown
    stripe between them: repaint it as one line where the original has it."""
    o = art.names["outline"]
    L = ref_lab()[..., 0]
    a = ref_small()[..., 3]
    box = np.zeros(L.shape, bool)
    box[428:443, 284:347] = True
    art.idx[box & (L < 30) & (a >= 128) & (art.idx >= 0)] = o
    # below it the soft edge of the line made the silhouette 3-5px too deep
    gap = np.zeros(L.shape, bool)
    gap[437:453, 286:333] = True
    art.idx[gap & (a < 200)] = TRANSPARENT


EYE_L = poly([(286, 186), (296, 176), (312, 168), (336, 166), (358, 172), (370, 186), (372, 206),
              (368, 226), (356, 240), (336, 244), (316, 240), (300, 226), (292, 204)])
EYE_R = poly([(412, 222), (418, 204), (430, 192), (446, 188), (462, 194), (472, 210), (472, 230),
              (466, 248), (452, 260), (432, 262), (418, 254), (410, 240)])
BEAK = poly([(362, 252), (372, 242), (386, 236), (400, 232), (410, 236), (418, 246), (418, 258),
             (412, 266), (402, 272), (396, 280), (388, 284), (380, 278), (372, 266), (364, 260)])


def eyes(art):
    left, right = EYE_L, EYE_R
    cols = [BLACK, RBROWN, RORANGE, IORANGE, AMBER, YELLOW, SCLERA, CREAM, CREAM2, CREAM3]
    for m in (left, right):
        requantize(art, m, cols, smooth=1)
    # grey shading only on the top of the left sclera, under the lash
    shade = poly([(298, 192), (306, 185), (318, 181), (332, 180), (344, 182), (342, 188),
                  (330, 188), (318, 191), (308, 197), (300, 203)])
    requantize(art, shade, [BLACK, RBROWN, SCLERA, SCLERA_G], smooth=0)
    # the original has a pale rim around each eye, not the beige that the
    # black -> cream transition quantises to
    rim = dilate(art.idx == art.color(BLACK), 2) & (EYE_L | EYE_R)
    for c in (CREAM3, CREAM2, SHADE):
        art.idx[rim & (art.idx == art.color(c))] = art.color(CREAM)
    # the lash line round the left sclera is black in the original; the
    # quantiser made its thin part red-brown (only the iris rim is brown)
    iris = ell(334.5, 212.0, 21.0, 22.0)
    art.idx[EYE_L & ~iris & (art.idx == art.color(RBROWN))] = art.color(BLACK)
    # crisp white catch-lights
    art.idx[ell(333.6, 198.6, 3.6, 3.6)] = art.color(WHITE)
    art.idx[ell(430.6, 219.8, 2.8, 3.4)] = art.color(WHITE)


# ---------------------------------------------------------------- flames

def box_count(m, r):
    """Number of True pixels in the (2r+1)^2 window around each pixel."""
    k = 2 * r + 1
    p = np.pad(m.astype(np.int32), r)
    ii = np.pad(p.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    return ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]


def majority_wide(art, mask, r=1, passes=1):
    """Majority filter with a (2r+1)^2 window that only repaints pixels in
    mask and only with colours already used inside mask (so the outline and
    neighbouring regions never leak in)."""
    idx = art.idx
    need = (2 * r + 1) ** 2 // 2 + 1
    for _ in range(passes):
        best = np.full(idx.shape, -1)
        bestc = np.zeros(idx.shape, np.int32)
        for c in np.unique(idx[mask]):
            n = box_count(idx == c, r)
            up = n > bestc
            best[up], bestc[up] = c, n[up]
        ch = mask & (bestc >= need)
        idx[ch] = best[ch]


def merge_in(art, mask, min_area, avoid=()):
    """Fold same-colour regions inside mask smaller than min_area into their
    most common 4-neighbour colour, never into a colour of `avoid`."""
    from pixelize import components
    idx = art.idx
    h, w = idx.shape
    flat = idx.ravel()
    sub = np.where(mask, idx, TRANSPARENT)
    bad = set(avoid) | {TRANSPARENT}
    for c, region in list(components(sub)):
        if len(region) >= min_area:
            continue
        ys, xs = np.divmod(region, w)
        nb = []
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            yy, xx = ys + dy, xs + dx
            ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
            nb.append(flat[yy[ok] * w + xx[ok]])
        nb = np.concatenate(nb)
        nb = nb[(nb != c) & ~np.isin(nb, list(bad))]
        if len(nb):
            vals, cnt = np.unique(nb, return_counts=True)
            flat[region] = vals[cnt.argmax()]


def ref_soft(blur=0.8, ink_l=30.0):
    """Reference on the sprite grid, blurred without letting the dark line
    art bleed into the colours next to it (normalised convolution)."""
    key = ("soft", blur, ink_l)
    if key not in _CACHE:
        small = ref_small()
        w = ((ref_lab()[..., 0] >= ink_l) & (small[..., 3] >= 64)).astype(np.float64)
        r = int(np.ceil(blur * 3))
        ker = np.exp(-0.5 * (np.arange(-r, r + 1) / blur) ** 2)
        ker /= ker.sum()

        def g(ch):
            p = np.pad(ch, r)
            p = sum(ker[i] * p[i:i + p.shape[0] - 2 * r, :] for i in range(2 * r + 1))
            return sum(ker[i] * p[:, i:i + p.shape[1] - 2 * r] for i in range(2 * r + 1))
        wb = g(w)
        rgb = np.stack([g(small[..., i] * w) for i in range(3)], -1) / np.maximum(wb, 1e-3)[..., None]
        rgb = np.where(wb[..., None] > 0.05, rgb, small[..., :3])
        _CACHE[key] = rgb.clip(0, 255)
    return _CACHE[key]


FLAME = {"k": 18, "blur": 0.7, "wide": 1, "passes": 2, "island": 20, "min_l": 36.0}


def flame(art, n_auto):
    """The 56-colour k-means bands left the flame feathers wavy and speckled.
    Re-quantise every flame pixel (all automatic warm colours outside the
    face, eyes, beak, scarf, gem and feet) from a softened reference to one
    flame ramp, then clean it with a wide majority filter so each colour
    band is a smooth, flat stroke like in the original. Dark crimson crease
    tones are left alone so the feather separations stay."""
    from pixelize import kmeans
    f = FLAME
    o = art.names["outline"]
    pal = rgb2lab(np.array(art.palette, np.float64))
    chroma = np.hypot(pal[:, 1], pal[:, 2])
    creams = set(cream_ids(art))
    warm = [i for i in range(n_auto)
            if i != o and i not in creams and chroma[i] >= 28 and pal[i, 0] >= f["min_l"]]
    special = dilate(EYE_L | EYE_R | BEAK | GEM_POLY, 1)
    m = np.isin(art.idx, warm) & ~special
    rgb = ref_soft(f["blur"])
    lab_all = rgb2lab(rgb)
    # the feet pass also browned the orange leg-feather tufts above the feet
    x0, y0, x1, y1 = TAIL_TIP
    yy, xx = _YY, _XX
    tip = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
    browns = [art.color(c) for c in (F_DARK, F_RED, F_MID, F_LT, F_HL, CLAW, CLAW_L)]
    m |= (np.isin(art.idx, browns) & ~dilate(FOOT_INNER, 1) & ~tip
          & (np.hypot(lab_all[..., 1], lab_all[..., 2]) >= 40))
    lab = lab_all[m]
    # fit the ramp away from the line art so edge mixing does not add muddy tones
    fit = m & ~dilate(art.idx == o, 2) & (lab_all[..., 0] >= 42)
    cent = kmeans(lab_all[fit], f["k"], 0)
    # a dull brown cluster only shows up as a 1px muddy rim inside the
    # contour (edge blending); the original goes straight from flame to line
    cch = np.hypot(cent[:, 1], cent[:, 2])
    cent = cent[~((cch < 45) & (cent[:, 2] > 20))]
    k = len(cent)
    lbl = nearest(lab, cent)
    cols = []
    for j in range(k):
        sel = rgb[fit][nearest(lab_all[fit], cent) == j]
        cols.append(tuple(int(round(v)) for v in sel.mean(0)) if len(sel) else tuple(int(v) for v in rgb[m][lbl == j].mean(0)))
    ids = np.array([art.color(c) for c in cols])
    # the pale pink rim tone belongs only where the original has its pink
    # rim light (top edges of the red feathers), not on blended edge pixels
    rl = ref_lab()
    rim = dilate((rl[..., 1] > 40) & (rl[..., 2] < 20) & (rl[..., 0] > 45), 1)
    pink = cent[:, 2] < 20
    if pink.any() and (~pink).any():
        alt = np.flatnonzero(~pink)[nearest(lab, cent[~pink])]
        off = pink[lbl] & ~rim[m]
        lbl = np.where(off, alt, lbl)
    art.idx[m] = ids[lbl]
    if f["wide"]:
        majority_wide(art, m, 2, f["wide"])
    majority_wide(art, m, 1, f["passes"])
    merge_in(art, m, f["island"], avoid=[o])
    return m


# ---------------------------------------------------------------- outline

def clean_silhouette(art):
    """Drop 1px spurs and isolated specks from the silhouette and fill 1px
    notches, so the contour line runs smoothly."""
    o = art.names["outline"]
    idx = art.idx
    for _ in range(2):
        s = idx >= 0
        n8 = box_count(s, 1) - s
        spur = s & (n8 <= 2)
        idx[spur] = TRANSPARENT
        s = idx >= 0
        n8 = box_count(s, 1) - s
        n4 = shift(s, 0, 1, False).astype(int) + shift(s, 0, -1, False) + shift(s, 1, 0, False) + shift(s, -1, 0, False)
        notch = ~s & (n8 >= 6) & (n4 >= 3)
        idx[notch] = o
    # specks: tiny separate bits of silhouette
    from pixelize import components
    solid = np.where(idx >= 0, 0, TRANSPARENT)
    for _, region in components(solid):
        if len(region) < 30:
            idx.ravel()[region] = TRANSPARENT


OUT_W = 3.0       # contour width (edt units): 2.3 -> 2px, 3.0 -> 3px lines


# places where the original's soft purple edge shadow is as dark as the line
# itself, so pixelize made a 6-9px blob of ink; trim them back to the band
SHADOW_BOXES = [(88, 236, 116, 266)]


def outline_band(art, width=None):
    """One clean contour of even width all round: every silhouette pixel
    within `width` of the background is outline (it was 1-2px in places and
    5-9px in others). Interior feather lines are kept."""
    width = OUT_W if width is None else width
    o = art.names["outline"]
    e = edt(art.idx >= 0, 8)
    ink = art.idx == o
    for x0, y0, x1, y1 in SHADOW_BOXES:
        box = np.zeros(ink.shape, bool)
        box[y0:y1 + 1, x0:x1 + 1] = True
        fill_from_neighbours(art, box & ink & (e > width + 0.01), avoid=[o])
    art.idx[(art.idx >= 0) & (e <= width + 0.01)] = o


def fold_rare(art, min_count=16):
    """Colours left on only a handful of pixels (edges of the eye, beak and
    gem areas that the flame pass skipped) fold into their neighbours."""
    count = np.bincount(art.idx[art.idx >= 0], minlength=len(art.palette))
    keep = {art.color(WHITE), art.names["outline"]}
    rare = [i for i in range(len(art.palette)) if 0 < count[i] < min_count and i not in keep]
    fold_colors(art, rare)


def edge_halo(art):
    """The original was anti-aliased against white, so the first pixels
    inside the contour quantised one or two tones lighter than the feather
    (a pale peach rim along the tail and wings). Give those pixels the
    colour of the feather just inside them. The real pink rim light on the
    red feathers, the creams and the hand-drawn parts are left alone."""
    o = art.names["outline"]
    pal = rgb2lab(np.array(art.palette, np.float64))
    e = edt(art.idx >= 0, 6)
    rl = ref_lab()
    rim = dilate((rl[..., 1] > 40) & (rl[..., 2] < 20) & (rl[..., 0] > 45), 1)
    protect = (dilate(EYE_L | EYE_R | BEAK | GEM_POLY, 2) | FOOT_INNER | FACE_POLY | CHEST_POLY | rim)
    protect |= np.isin(art.idx, cream_ids(art) + [o, art.color(LETTER)])
    for depth in (OUT_W + 1.0, OUT_W + 2.0):
        ring = (art.idx >= 0) & (e > depth - 1.0) & (e <= depth) & ~protect
        ys, xs = np.nonzero(ring)
        new = art.idx.copy()
        for y, x in zip(ys, xs):
            c = art.idx[y, x]
            y0, y1, x0, x1 = max(y - 2, 0), y + 3, max(x - 2, 0), x + 3
            win, ew = art.idx[y0:y1, x0:x1], e[y0:y1, x0:x1]
            cand = win[(ew >= e[y, x] + 1.0) & (win >= 0) & (win != o)]
            if not len(cand):
                continue
            vals, cnt = np.unique(cand, return_counts=True)
            b = vals[cnt.argmax()]
            if b != c and pal[c, 0] > pal[b, 0] + 4:
                new[y, x] = b
        art.idx[:] = new


def merge_specks(art, max_size=3):
    """Final cleanup: colour specks of up to `max_size` pixels (8-connected,
    so thin diagonal lines are not counted as specks) take the most common
    colour around them. Outline, pupils, catch-lights and letters stay."""
    idx = art.idx
    h, w = idx.shape
    keep = {TRANSPARENT, art.names["outline"], art.color(BLACK), art.color(WHITE), art.color(LETTER)}
    flat = idx.ravel()
    seen = np.zeros(h * w, bool)
    seen[np.isin(flat, list(keep))] = True
    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for start in np.flatnonzero(~seen):
        if seen[start]:
            continue
        c = flat[start]
        stack, region = [start], [start]
        seen[start] = True
        while stack and len(region) <= max_size:
            q = stack.pop()
            y, x = divmod(q, w)
            for dy, dx in nbrs:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w:
                    r = yy * w + xx
                    if not seen[r] and flat[r] == c:
                        seen[r] = True
                        stack.append(r)
                        region.append(r)
        if len(region) > max_size:
            # finish marking this big region so it is not revisited
            while stack:
                q = stack.pop()
                y, x = divmod(q, w)
                for dy, dx in nbrs:
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w:
                        r = yy * w + xx
                        if not seen[r] and flat[r] == c:
                            seen[r] = True
                            stack.append(r)
            continue
        votes = {}
        for q in region:
            y, x = divmod(q, w)
            for dy, dx in nbrs:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w:
                    v = flat[yy * w + xx]
                    if v != c and v != TRANSPARENT:
                        votes[v] = votes.get(v, 0) + 1
        if votes:
            best = max(sorted(votes), key=lambda v: votes[v])
            flat[region] = best


def touchup(art):
    n_auto = len(art.palette)
    art.name("black", BLACK)
    cleanup_auto_colors(art, n_auto)
    trim_band(art, keep=3.5)
    thin_outline(art)
    face(art)
    eyes(art)
    beak(art)
    scarf(art)
    gem(art)
    feet(art)
    trim_band(art, keep=3.0)      # the feet re-quantisation re-thickens their contour
    solidify_edge(art, n_auto)
    flame(art, n_auto)
    reshade_cream(art)
    beak_edge(art)
    scarf_smooth(art)
    demud(art, n_auto)
    belly_line(art)
    clean_silhouette(art)
    outline_band(art)
    feet_redraw(art)
    edge_halo(art)
    merge_specks(art)
    fold_rare(art)
