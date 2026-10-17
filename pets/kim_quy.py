"""Kim Quy (turtle): pixelize config and hand touch-ups (512x512 sprite coords)."""
from __future__ import annotations

import numpy as np
from PIL import Image

from pixelize import TRANSPARENT, majority, nearest, placement, reference, rgb2lab

SLUG = "kim-quy"
SIZE, MARGIN = 512, 6

CONFIG = {
    "colors": 44,
    "ink_l": 22.0,      # the royal-blue scarf is L*~28: keep it out of the ink
    "ink_frac": 0.5,
    "smooth": 3,
    "min_island": 20,
    "seed": 0,
}

YY, XX = np.mgrid[0:SIZE, 0:SIZE]
PX, PY = XX + 0.5, YY + 0.5
_CACHE = {}


# ---------------------------------------------------------------- reference --
def ref_data():
    """Hi-res crop (RGBA, Lab), its box-downscale on the sprite grid (RGBA, Lab)."""
    if "ref" not in _CACHE:
        im = reference(SLUG)
        box, (ox, oy, nw, nh) = placement(SLUG, SIZE, MARGIN)
        crop = im.crop(box)
        hi = np.array(crop, np.float64)
        small = np.array(crop.convert("RGBa").resize((nw, nh), Image.BOX).convert("RGBA"), np.float64)
        canvas = np.zeros((SIZE, SIZE, 4))
        canvas[oy:oy + nh, ox:ox + nw] = small
        _CACHE["ref"] = dict(hi=hi, hi_lab=rgb2lab(hi[..., :3]), small=canvas,
                             lab=rgb2lab(canvas[..., :3]), place=(ox, oy, nw, nh))
    return _CACHE["ref"]


# ------------------------------------------------------------------ helpers --
def ell(cx, cy, rx, ry, angle=0.0):
    """Boolean ellipse (pixel centres), optional rotation in degrees."""
    t = np.deg2rad(angle)
    dx, dy = PX - cx, PY - cy
    u = dx * np.cos(t) + dy * np.sin(t)
    v = -dx * np.sin(t) + dy * np.cos(t)
    return (u / rx) ** 2 + (v / ry) ** 2 <= 1.0


def box(x0, y0, x1, y1):
    m = np.zeros((SIZE, SIZE), bool)
    m[y0:y1, x0:x1] = True
    return m


def poly(pts):
    """Boolean convex/concave polygon (even-odd rule) over pixel centres."""
    m = np.zeros((SIZE, SIZE), bool)
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        if y0 == y1:
            continue
        cond = ((y0 <= PY) & (PY < y1)) | ((y1 <= PY) & (PY < y0))
        xint = x0 + (PY - y0) * (x1 - x0) / (y1 - y0)
        m ^= cond & (PX < xint)
    return m


def grow(m, n=1, diag=False):
    for _ in range(n):
        p = np.pad(m, 1)
        g = m | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
        if diag:
            g |= p[:-2, :-2] | p[:-2, 2:] | p[2:, :-2] | p[2:, 2:]
        m = g
    return m


def seg(p0, p1, r):
    """Pixels whose centre lies within r of the segment p0-p1."""
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    t = np.clip(((PX - x0) * dx + (PY - y0) * dy) / float(dx * dx + dy * dy), 0, 1)
    return (PX - x0 - t * dx) ** 2 + (PY - y0 - t * dy) ** 2 <= r * r


def shrink(m, n=1):
    return ~grow(~m, n)


def edge(m):
    """Pixels of m that touch a non-m pixel (4-neighbourhood)."""
    return m & ~shrink(m)


def smooth_mask(m, passes=1):
    """Binary 3x3 majority filter."""
    for _ in range(passes):
        p = np.pad(m.astype(np.int16), 1)
        s = sum(p[dy:dy + SIZE, dx:dx + SIZE] for dy in range(3) for dx in range(3))
        m = s >= 5
    return m


def requant(art, mask, colors, smooth=1, lab=None):
    """Map masked pixels to the nearest of `colors` (Lab, from the reference), then majority-clean."""
    if not mask.any():
        return
    lab = ref_data()["lab"] if lab is None else lab
    ids = np.array([art.color(c) for c in colors])
    pal = rgb2lab(np.array([art.palette[i] for i in ids], np.float64))
    art.idx[mask] = ids[nearest(lab[mask], pal)]
    if smooth:
        ys, xs = np.where(mask)
        y0, y1, x0, x1 = max(ys.min() - 1, 0), ys.max() + 2, max(xs.min() - 1, 0), xs.max() + 2
        sub = art.idx[y0:y1, x0:x1]
        uniq = np.unique(sub[sub >= 0])
        lut = np.full(len(art.palette) + 1, -1)
        lut[uniq] = np.arange(len(uniq))
        comp = np.where(sub >= 0, lut[np.clip(sub, 0, None)], -1)
        comp = majority(comp, len(uniq), smooth)
        back = np.where(comp >= 0, uniq[np.clip(comp, 0, None)], -1)
        m = mask[y0:y1, x0:x1]
        sub[m] = back[m]


# ------------------------------------------------------------------ colours --
EYE_TEAL, EYE_G1, EYE_G2, EYE_G3, EYE_G4 = (8, 62, 44), (12, 100, 43), (16, 137, 53), (24, 171, 65), (61, 211, 89)
WHITE, SCLERA_S, SCLERA_M = (248, 250, 250), (193, 205, 206), (146, 160, 166)
MAROON_D, MAROON, RED, TONGUE, TONGUE_L = (80, 10, 10), (113, 19, 20), (182, 53, 50), (212, 82, 72), (241, 109, 93)
NAVY, BLUE_D, BLUE_M, BLUE_B = (2, 22, 71), (4, 48, 117), (3, 63, 151), (5, 86, 185)
GEM_RIM, GEM_D, GEM_M, GEM_C, GEM_L, GEM_W = (10, 62, 150), (8, 112, 212), (13, 163, 237), (42, 207, 247), (130, 234, 250), (236, 252, 255)


def heal(art, mask, avoid=(), ring=(2, 6), smooth=1):
    """Re-paint masked pixels from the reference, using only colours found just around them."""
    if not mask.any():
        return
    idx = art.idx
    avoid_ids = [art.color(c) for c in avoid] + [art.names["outline"], TRANSPARENT]
    zone = grow(mask, ring[1], diag=True) & ~grow(mask, ring[0], diag=True)
    cols = [int(c) for c in np.unique(idx[zone]) if c not in avoid_ids]
    if cols:
        requant(art, mask, cols, smooth=smooth)


def blur_lab(r=1):
    """Box-blurred sprite-grid Lab of the reference (softens banding before re-quantizing).

    Ink and background pixels (L* < 24, or transparent) are left out of the average,
    so colour areas next to an outline are not dragged towards grey-olive.
    """
    key = ("blur", r)
    if key not in _CACHE:
        d = ref_data()
        lab = d["lab"]
        w = ((lab[..., 0] >= 24) & (d["small"][..., 3] >= 128)).astype(np.float64)
        n = 2 * r + 1
        p = np.pad(lab * w[..., None], ((r, r), (r, r), (0, 0)), mode="edge")
        pw = np.pad(w, r, mode="edge")
        acc = sum(p[dy:dy + SIZE, dx:dx + SIZE] for dy in range(n) for dx in range(n))
        cnt = sum(pw[dy:dy + SIZE, dx:dx + SIZE] for dy in range(n) for dx in range(n))
        _CACHE[key] = np.where(cnt[..., None] > 0, acc / np.maximum(cnt, 1e-9)[..., None], lab)
    return _CACHE[key]


def draw_eye(art, outer, inner, iris, sclera, pupil, bands, lights, shade_y, icx, irx, iry):
    """Anime eye: black rim, white sclera (grey under the lid), banded green iris, pupil, glints."""
    idx = art.idx
    out = art.names["outline"]
    W = art.color(WHITE)
    idx[outer] = out
    sclera = sclera & inner & ~iris
    idx[sclera] = W
    # grey shadow of the upper lid: a band under the rim, plus a thin line along the iris
    idx[sclera & ~shrink(inner, 4) & (PY < shade_y + 10)] = art.color(SCLERA_S)
    idx[sclera & ~shrink(inner, 2) & (PY < shade_y + 2)] = art.color(SCLERA_M)
    idx[sclera & grow(iris & inner) & (PY < shade_y + 20)] = art.color(SCLERA_S)
    ir = iris & inner
    # bands follow the bottom curve of the iris (bright crescent at the bottom)
    dx = np.clip(np.abs(PX - icx) / irx, 0, 1)
    yeff = PY + 0.55 * iry * (1 - np.sqrt(1 - dx * dx))
    for y0, c in bands:
        idx[ir & (yeff >= y0)] = art.color(c)
    # darker ring round the iris: mid green low down, teal higher up
    ring = ir & ~shrink(iris)
    idx[ring & (PY >= bands[2][0])] = art.color(EYE_G2)
    idx[ring & (PY < bands[2][0])] = art.color(EYE_TEAL)
    idx[pupil & inner] = out
    for (cx, cy, rx, ry) in lights:
        idx[ell(cx, cy, rx, ry)] = W


def fix_eyes(art):
    bandsR = [(0, EYE_TEAL), (233, EYE_G1), (241, EYE_G2), (247, EYE_G3), (252, EYE_G4)]
    draw_eye(art,
             outer=ell(231.0, 226.5, 35.5, 36.0),
             inner=ell(231.5, 229.0, 32.0, 31.0),
             iris=ell(222.5, 234.5, 23.5, 28.5),
             sclera=(PX > 224) & (PY > 201),
             pupil=ell(218.5, 229.5, 12.5, 18.5),
             bands=bandsR, lights=[(227.5, 217.0, 5.6, 5.6)], shade_y=208,
             icx=222.5, irx=23.5, iry=28.5)
    bandsL = [(0, EYE_TEAL), (230, EYE_G1), (237, EYE_G2), (243, EYE_G3), (248, EYE_G4)]
    draw_eye(art,
             outer=ell(83.5, 223.0, 21.8, 36.2),
             inner=ell(83.8, 225.5, 19.0, 32.5),
             iris=ell(88.5, 228.5, 16.5, 30.0),
             sclera=(PX < 88) & (PY > 197),
             pupil=ell(92.5, 223.5, 10.5, 20.0),
             bands=bandsL, lights=[(85.5, 212.5, 3.8, 4.0), (99.6, 215.8, 2.6, 2.8)], shade_y=203,
             icx=88.5, irx=16.5, iry=30.0)
    # auto-pass eye colours that fall outside the redrawn eyes -> skin
    eyecols = [EYE_TEAL, EYE_G1, EYE_G2, EYE_G3, EYE_G4, WHITE, SCLERA_S, SCLERA_M]
    outers = ell(231.0, 226.5, 35.5, 36.0) | ell(83.5, 223.0, 21.8, 36.2)
    around = grow(outers, 5, diag=True) & ~outers & (art.idx != TRANSPARENT)
    pal = rgb2lab(np.array(art.palette, np.float64))
    cur = np.clip(art.idx, 0, None)
    stray = around & (np.isin(art.idx, [art.color(c) for c in eyecols]) | ((pal[cur, 0] < 60) & (art.idx != art.names["outline"])))
    stray &= ~box(40, 180, 62, 270)          # the head outline left of the eye
    heal(art, stray, avoid=eyecols, ring=(1, 6))


def fix_mouth(art):
    d = ref_data()
    L, a, b = d["lab"][..., 0], d["lab"][..., 1], d["lab"][..., 2]
    out = art.names["outline"]
    m = box(110, 250, 182, 300) & ((a > 20) | (L < 40))
    mouth = [out, MAROON_D, MAROON, RED, TONGUE, TONGUE_L]
    requant(art, m, mouth, smooth=1)
    # anti-alias greys/khakis left round the lips -> face cream
    near = box(108, 248, 184, 302) & ~np.isin(art.idx, [art.color(c) for c in mouth])
    pal = rgb2lab(np.array(art.palette, np.float64))
    cur = np.clip(art.idx, 0, None)
    odd = near & ((pal[cur, 0] < 86) | (np.hypot(pal[cur, 1], pal[cur, 2]) < 12))
    heal(art, odd, avoid=mouth + [WHITE, SCLERA_S, SCLERA_M], ring=(1, 8))
    # the original's lips sit in a thin white rim
    lips = np.isin(art.idx, [art.color(c) for c in mouth]) & box(108, 248, 184, 302)
    lips = smooth_mask(lips, 1) | (lips & box(108, 248, 184, 262))
    rim = grow(lips, 2, diag=True) & ~lips & box(108, 248, 184, 302) & (art.idx != TRANSPARENT)
    rim &= ~np.isin(art.idx, [art.color(c) for c in mouth])
    art.idx[rim] = art.color((253, 251, 238))


def fix_scarf(art):
    d = ref_data()
    L, a, b = d["lab"][..., 0], d["lab"][..., 1], d["lab"][..., 2]
    out = art.names["outline"]
    scarf = (box(100, 285, 352, 400) & (b < -20) & (a > 8)) | (box(100, 285, 352, 400) & (art.idx == out) & (b < -10))
    scarf &= art.idx != TRANSPARENT
    requant(art, scarf, [out, NAVY, BLUE_D, BLUE_M, BLUE_B], smooth=2)
    # lettering: evenly stroked letters fitted to the original's white core (5-6px strokes)
    letters = box(158, 351, 163, 371)                                                     # I
    letters |= ell(178.5, 361.0, 12.2, 11.2) & ~ell(178.2, 361.5, 5.6, 6.0)                # Q
    letters |= seg((181.5, 364.5), (188.5, 372.0), 2.8)                                   # Q tail
    letters |= seg((197.0, 350.5), (215.0, 366.5), 2.8) | seg((212.0, 349.0), (198.0, 369.5), 2.8)  # X
    lb = box(150, 342, 232, 378) & (b < -15) & ~letters & (art.idx != out)
    requant(art, lb, [BLUE_D, BLUE_M, BLUE_B], smooth=1)
    art.idx[letters] = art.color(WHITE)
    return scarf | letters


def fix_gem(art):
    """Faceted rhombus: cyan upper facets, white kite highlight, deeper lower facets, dark rim."""
    idx = art.idx
    out = art.names["outline"]
    T, Lv, Rv, B = (184, 385), (160, 414), (218, 414), (188, 452)
    c = lambda p: (p[0] + 0.5, p[1] + 0.5)
    gem = poly([c(T), c(Lv), c(B), c(Rv)])
    halo = grow(gem, 6, diag=True) & ~gem & ~box(0, 0, SIZE, 389)
    # wipe the auto pass's blue fringe around the old pendant: re-map it to the belly creams
    pal = rgb2lab(np.array(art.palette, np.float64))
    cur = np.clip(idx, 0, None)
    bluish = (pal[cur, 2] < -4) | (np.hypot(pal[cur, 1], pal[cur, 2]) < 8)
    ring = grow(gem, 10, diag=True) & ~grow(gem, 6, diag=True) & (idx != out) & ~bluish & (idx != TRANSPARENT)
    belly = sorted(set(idx[ring].tolist()))
    fringe = halo & bluish & (idx != out) & (idx != TRANSPARENT)
    if belly:
        requant(art, fringe, belly, smooth=1)

    C = (187.0, 415.5)
    T_, Lv_, Rv_, B_ = c(T), c(Lv), c(Rv), c(B)
    upper_l, upper_r = poly([T_, Lv_, C]), poly([T_, C, Rv_])
    lower_l, lower_r = poly([Lv_, B_, C]), poly([C, B_, Rv_])
    idx[gem & upper_l] = art.color(GEM_C)
    idx[gem & upper_r] = art.color(GEM_C)
    idx[gem & lower_l] = art.color(GEM_D)
    idx[gem & lower_r] = art.color(GEM_M)
    # lighter strips next to the centre line, a dark facet under the right girdle
    idx[gem & upper_r & poly([T_, C, (199.0, 413.0)])] = art.color(GEM_L)
    idx[gem & lower_r & poly([C, (200.0, 415.5), (189.5, 449.0)])] = art.color(GEM_C)
    idx[gem & lower_l & poly([(175.0, 415.5), C, (188.0, 449.0)])] = art.color(GEM_M)
    idx[gem & lower_r & poly([(204.0, 415.5), (216.0, 415.5), (207.0, 430.0)])] = art.color(GEM_D)
    # central streaks and the white kite
    idx[gem & poly([(185.5, 416.0), (189.5, 416.0), (189.5, 450.0), (188.5, 450.0)])] = art.color(GEM_L)
    idx[gem & poly([(184.0, 389.0), (186.0, 389.0), (186.5, 404.0), (183.5, 404.0)])] = art.color(GEM_W)
    idx[gem & poly([(184.5, 399.0), (175.5, 413.0), (186.5, 418.5), (198.0, 413.0)])] = art.color(GEM_W)
    # glint along the inner lower-left edge
    idx[gem & poly([(165.5, 418.0), (168.0, 418.0), (186.0, 446.0), (185.0, 448.0)])] = art.color(GEM_C)
    # rim: deep-blue line on the edge, navy line just outside (same slope, shifted 1px outward)
    edges_ = ((T, Lv, -1), (Lv, B, -1), (B, Rv, 1), (Rv, T, 1))
    for p0, p1, sx in edges_:
        art.line(p0[0] + sx, p0[1], p1[0] + sx, p1[1], NAVY)
    art.put(B[0], B[1] + 1, NAVY)
    for p0, p1, sx in edges_:
        art.line(p0[0], p0[1], p1[0], p1[1], GEM_RIM)
    whole = gem | grow(gem, 2) & np.isin(idx, [art.color(NAVY), art.color(GEM_RIM)])
    # soft cream glow round the stone instead of grey anti-alias
    pal = rgb2lab(np.array(art.palette, np.float64))
    cur = np.clip(idx, 0, None)
    # blunt the side points by a pixel, as in the original
    art.put(Lv[0] - 1, Lv[1], BELLY[0])
    art.put(Rv[0] + 1, Rv[1], BELLY[0])
    whole &= ~box(Lv[0] - 1, Lv[1], Lv[0], Lv[1] + 1) & ~box(Rv[0] + 1, Rv[1], Rv[0] + 2, Rv[1] + 1)
    soft = np.isin(idx, [art.color(c) for c in BELLY]) | ((np.hypot(pal[cur, 1], pal[cur, 2]) < 20) & (pal[cur, 0] > 50))
    soft &= (idx != out) & (idx != TRANSPARENT) & (PY >= 390) & ~whole
    glow1 = grow(whole, 1, diag=True) & soft
    idx[glow1] = art.color(BELLY[0])
    glow2 = grow(whole, 4, diag=True) & ~grow(whole, 1, diag=True) & soft & ~np.isin(idx, [art.color(c) for c in BELLY])
    requant(art, glow2, BELLY[:3], smooth=0)
    # the top tucks under the scarf: give the scarf its edge back there
    top = box(170, 376, 200, 390) & np.isin(idx, [art.color(GEM_RIM), art.color(NAVY), art.color(GEM_C), art.color(GEM_L), art.color(GEM_W)])
    idx[top & (PY < 387)] = out
    return gem


SHELL = [(17, 79, 80), (13, 107, 93), (24, 137, 108), (38, 164, 124), (91, 205, 155), (174, 228, 216)]


def fix_shell(art):
    """Shell plates: one clean teal ramp, dark grooves, pale rim light."""
    d = ref_data()
    L, a, b = d["lab"][..., 0], d["lab"][..., 1], d["lab"][..., 2]
    out = art.names["outline"]
    P = poly([(320, 140), (360, 150), (420, 180), (470, 260), (472, 380), (445, 420), (390, 420),
              (372, 395), (350, 365), (320, 345), (318, 300), (326, 240)])
    m = P & (b > -15) & (b < 26) & (L > 22) & (a < -8) & (art.idx != out) & (art.idx != TRANSPARENT)
    requant(art, m, SHELL, smooth=0, lab=blur_lab(1))
    requant(art, m, SHELL, smooth=2, lab=blur_lab(1))


LEG = [(56, 186, 122), (93, 155, 96), (96, 163, 134), (148, 177, 100), (112, 194, 113), (163, 208, 111), (199, 217, 123),
       (217, 226, 140)]


def fix_legs(art):
    """Legs: clean yellow-green with round green spots (the auto pass muddies them)."""
    d = ref_data()
    L, a, b = d["lab"][..., 0], d["lab"][..., 1], d["lab"][..., 2]
    out = art.names["outline"]
    legs = (poly([(56, 365), (140, 365), (165, 470), (56, 470)])
            | poly([(262, 360), (340, 345), (372, 400), (372, 470), (262, 470)])
            | poly([(372, 400), (446, 392), (446, 470), (372, 470)]))
    m = legs & (L > 45) & (a < -8) & (art.idx != out) & (art.idx != TRANSPARENT)
    requant(art, m, LEG, smooth=0, lab=blur_lab(1))
    requant(art, m, LEG, smooth=2, lab=blur_lab(1))


BELLY = [(248, 243, 207), (238, 228, 178), (218, 204, 146), (190, 180, 122), (150, 146, 98)]


def fix_belly(art):
    """Flatten the cream plastron into a few clean steps of shading."""
    d = ref_data()
    L, a, b = d["lab"][..., 0], d["lab"][..., 1], d["lab"][..., 2]
    out = art.names["outline"]
    m = poly([(138, 380), (290, 380), (282, 470), (140, 470)]) & (a > -11) & (b > 12) & (L > 52)
    m &= (art.idx != out) & (art.idx != TRANSPARENT)
    requant(art, m, BELLY, smooth=0, lab=blur_lab(1))
    requant(art, m, BELLY, smooth=2, lab=blur_lab(1))


def fix_face(art):
    d = ref_data()
    L, a, b = d["lab"][..., 0], d["lab"][..., 1], d["lab"][..., 2]
    idx = art.idx
    out = art.names["outline"]
    # nostrils: two small dark dots
    dot = art.color((52, 66, 48))
    art.blit(122, 230, ["##", "##", "#."], {"#": dot})
    art.blit(147, 230, [".##", "###", "##."], {"#": dot})
    # cheek blush: two soft peach steps
    for cx, cy in ((72, 270), (256, 276)):
        m = ell(cx, cy, 30, 16) & (L > 80) & (a > 2.0) & (idx != out)
        requant(art, m, [(247, 226, 182), (248, 210, 165)], smooth=2)


CLAW_W, CLAW_C, CLAW_Y, CLAW_O, CLAW_D = (250, 248, 228), (234, 226, 176), (205, 188, 96), (140, 142, 88), (70, 86, 70)


def fix_claws(art):
    d = ref_data()
    L, a, b = d["lab"][..., 0], d["lab"][..., 1], d["lab"][..., 2]
    idx = art.idx
    out = art.names["outline"]
    pal = rgb2lab(np.array(art.palette, np.float64))
    for (x0, y0, x1, y1) in ((64, 458, 132, 492), (266, 462, 348, 500), (380, 450, 440, 486)):
        above = idx[y0 - 10:y0, x0:x1]
        skin = [int(c) for c in np.unique(above) if c >= 0 and c != out and pal[c, 0] > 60]
        m = box(x0, y0, x1, y1) & (L > 25) & (idx != TRANSPARENT) & (idx != out)
        requant(art, m, skin + [CLAW_D, CLAW_O, CLAW_Y, CLAW_C], smooth=2)
    # claws: white ovals with a golden rim and a cream shade low down (cx, cy, rx, ry, lit)
    claws = ((75.3, 471.5, 3.0, 8.2, True), (97.7, 474.5, 8.3, 11.2, True), (123.3, 474.0, 5.4, 9.5, True),
             (289.0, 481.0, 9.5, 14.0, True), (328.5, 481.0, 9.5, 13.5, True),
             (404.5, 464.0, 7.0, 7.5, True), (421.0, 463.5, 6.5, 7.0, False))
    room = (L > 40) & (idx != TRANSPARENT) & (idx != out)
    for cx, cy, rx, ry, lit in claws:
        rim = ell(cx, cy, rx + 1.8, ry + 1.8) & room
        core = ell(cx, cy, rx, ry) & room
        idx[rim & ~core] = art.color(CLAW_Y)
        idx[core] = art.color(CLAW_W if lit else CLAW_C)
        idx[core & ~ell(cx - 1.0, cy - 2.5, rx, ry)] = art.color(CLAW_C if lit else CLAW_Y)


def despeckle(art, max_area, protect):
    """Fold 8-connected colour specks of <= max_area px into their most common 4-neighbour."""
    idx = art.idx
    h, w = idx.shape
    flat = idx.ravel()
    prot = protect.ravel()
    seen = flat == TRANSPARENT
    nb8 = (-w - 1, -w, -w + 1, -1, 1, w - 1, w, w + 1)
    for start in np.flatnonzero(~seen):
        if seen[start]:
            continue
        c = flat[start]
        stack, region = [start], []
        seen[start] = True
        big = False
        while stack:
            q = stack.pop()
            region.append(q)
            if len(region) > max_area:
                big = True
            y, x = divmod(q, w)
            for dq in nb8:
                r = q + dq
                ry, rx = divmod(r, w)
                if 0 <= r < h * w and abs(rx - x) <= 1 and not seen[r] and flat[r] == c:
                    seen[r] = True
                    stack.append(r)
        if big or prot[region].any():
            continue
        votes = {}
        for q in region:
            y, x = divmod(q, w)
            for r, ok in ((q - w, y > 0), (q + w, y < h - 1), (q - 1, x > 0), (q + 1, x < w - 1)):
                if ok and flat[r] != c and flat[r] != TRANSPARENT:
                    votes[flat[r]] = votes.get(flat[r], 0) + 1
        if votes:
            best = max(sorted(votes), key=lambda v: votes[v])
            flat[region] = best


def trim_ring(art):
    """Drop the pipeline's extra outer ring where the outline behind it is already >=2px."""
    idx = art.idx
    out = art.names["outline"]
    solid = idx != TRANSPARENT
    rim = edge(solid) & (idx == out)
    inner = shrink(solid)
    # keep a rim pixel unless all of its inner 4-neighbours are outline too
    p = np.pad(idx, 1, constant_values=TRANSPARENT)
    pin = np.pad(inner, 1)
    ok = np.ones_like(rim)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nb = p[1 + dy:1 + dy + SIZE, 1 + dx:1 + dx + SIZE]
        nin = pin[1 + dy:1 + dy + SIZE, 1 + dx:1 + dx + SIZE]
        ok &= ~nin | (nb == out)
    idx[rim & ok] = TRANSPARENT


# ======================================================== second polish pass ==
CREAM, CREAM_G, FACE_SH = (247, 242, 204), (231, 236, 169), (194, 202, 158)
HEAD_L, BLUSH, BLUSH_L = (176, 223, 148), (248, 210, 165), (239, 216, 164)


def eye_requant(art, region, outer):
    """Re-derive an eye straight from the reference with a fixed eye palette.

    Each zone gets its own colours so nothing bleeds: greens only where the
    reference iris is, greys only in the sclera, and the rim edge and the skin
    round the eye use black plus skin tones (no grey anti-alias fringe).
    """
    d = ref_data()
    L, a, b = d["lab"][..., 0], d["lab"][..., 1], d["lab"][..., 2]
    out = art.names["outline"]
    region = region & (art.idx != TRANSPARENT)
    iris = smooth_mask((a < -14) & (L < 80) & outer, 2)
    iris = grow(iris, 1, diag=True) & outer
    sclera = smooth_mask((L > 52) & (np.hypot(a, b) < 13) & outer & ~iris, 1)
    sclera = grow(sclera, 1) & outer & ~iris
    skin = [CREAM, CREAM_G, HEAD_L, FACE_SH, BLUSH, BLUSH_L]
    requant(art, region & iris, [out, EYE_TEAL, EYE_G1, EYE_G2, EYE_G3, EYE_G4, WHITE], smooth=0)
    requant(art, region & sclera, [out, WHITE, SCLERA_S, SCLERA_M], smooth=0)
    requant(art, region & ~iris & ~sclera, [out] + [c for c in skin if c != FACE_SH], smooth=0)
    ys, xs = np.where(region)
    y0, y1, x0, x1 = ys.min() - 1, ys.max() + 2, xs.min() - 1, xs.max() + 2
    blk = art.idx[y0:y1, x0:x1]
    uniq = np.unique(blk[blk >= 0])
    lut = np.full(len(art.palette) + 1, -1)
    lut[uniq] = np.arange(len(uniq))
    comp = np.where(blk >= 0, lut[np.clip(blk, 0, None)], -1)
    comp = majority(comp, len(uniq), 2)
    back = np.where(comp >= 0, uniq[np.clip(comp, 0, None)], -1)
    m = region[y0:y1, x0:x1]
    blk[m] = back[m]


R_EYE_OUT = ell(231.0, 228.0, 37.5, 37.5)
L_EYE_OUT = ell(83.0, 223.0, 23.0, 37.5)
R_LIGHTS = [(231.6, 217.6, 5.6, 6.0)]
L_LIGHTS = [(81.3, 213.2, 3.0, 4.2), (98.9, 216.9, 1.9, 2.4)]


def fix_eyes2(art):
    """Second eye pass: open-bottomed anime eyes as in the original (no lower rim)."""
    eye_requant(art, grow(R_EYE_OUT, 3, diag=True), R_EYE_OUT)
    eye_requant(art, grow(L_EYE_OUT, 3, diag=True) & (PX > 58), L_EYE_OUT)
    W = art.color(WHITE)
    for cx, cy, rx, ry in R_LIGHTS + L_LIGHTS:
        art.idx[ell(cx, cy, rx, ry)] = W


NOSE = (40, 56, 40)
JAW = [CREAM, (236, 235, 188), CREAM_G, (216, 222, 170), (196, 210, 154), (168, 196, 140), (142, 186, 150)]


def fix_face2(art):
    """Nostrils as the original's small dark ovals; a clean shade ramp along the jaw."""
    d = ref_data()
    L, a, b = d["lab"][..., 0], d["lab"][..., 1], d["lab"][..., 2]
    idx = art.idx
    out = art.names["outline"]
    # jaw / chin: cream -> pale yellow-green -> mint rim, instead of grey-olive bands
    near = grow(idx == out, 14, diag=True)
    zone = box(44, 245, 312, 342) & near & (L > 55) & (b > 5) & (a > -30) & (idx != out) & (idx != TRANSPARENT)
    zone &= ~grow(R_EYE_OUT | L_EYE_OUT, 2, diag=True) & ~box(106, 246, 186, 304)
    zone &= ~(ell(72, 270, 30, 16) | ell(256, 276, 30, 16))
    requant(art, zone, JAW, smooth=0, lab=blur_lab(1))
    requant(art, zone, JAW, smooth=2, lab=blur_lab(1))
    # muzzle: round off the stair-stepped edge between the pale-green brow and the cream
    two = np.isin(idx, [art.color(CREAM), art.color(CREAM_G)]) & box(98, 196, 200, 256)
    g = smooth_mask(two & (idx == art.color(CREAM_G)), 3)
    idx[two & g] = art.color(CREAM_G)
    idx[two & ~g] = art.color(CREAM)
    # nostrils
    nose = box(119, 227, 156, 238)
    heal(art, nose & (idx != out), avoid=[(52, 66, 48), NOSE], ring=(1, 5), smooth=0)
    idx[nose & np.isin(idx, [art.color((52, 66, 48))])] = art.color(CREAM)
    art.blit(122, 231, [".##.", "####", "####", ".##."], {"#": NOSE})
    art.blit(146, 230, ["....##", "..####", ".####.", "####..", ".##..."], {"#": NOSE})


LEAF = [(40, 112, 28), (52, 129, 26), (78, 154, 23), (102, 177, 17), (118, 193, 16), (160, 215, 85)]
LEAF_INK = (12, 40, 30)


def fix_leaves(art):
    """Leaves: clean green ramp, midribs and side veins plus the pale rim light, all
    taken from the reference's local light/dark detail; dark-green ink round them."""
    d = ref_data()
    lab = d["lab"]
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    idx = art.idx
    out = art.names["outline"]
    area = poly([(128, 4), (352, 4), (352, 150), (300, 150), (290, 128), (270, 113), (250, 105),
                 (200, 105), (150, 108), (128, 110)])
    leafref = area & (a < -20) & (b > 40) & (L > 30) & (d["small"][..., 3] > 200)
    pal = rgb2lab(np.array(art.palette, np.float64))
    cur = np.clip(idx, 0, None)
    leafspr = area & (pal[cur, 1] < -20) & (pal[cur, 2] > 40) & (idx != TRANSPARENT)
    # the pale rim light is low in chroma, so it falls outside `leafref`: take every
    # pixel inside the leaf ink that is not head, bump or the blue glint between leaves
    notleaf = ((b < 35) & (a < -15) & (PY > 92)) | ((a > -12) & (b > 25) & (PY > 92)) | (b < -5)
    inside = area & (idx != out) & (idx != TRANSPARENT) & ~notleaf & grow(leafref, 3, diag=True)
    zone = (leafref | (leafspr & grow(leafref, 2, diag=True)) | inside) & (idx != out) & (idx != TRANSPARENT)
    ramp = LEAF[1:5]
    requant(art, zone, ramp, smooth=0, lab=blur_lab(1))
    requant(art, zone, ramp, smooth=2, lab=blur_lab(1))
    idx[zone & (L > 80) & (a < -5)] = art.color(LEAF[5])
    # local detail: darker than the 7x7 mean -> vein (one step darker), lighter -> rim light
    Lb = blur_lab(3)[..., 0]
    core = shrink(leafref, 2)
    vein = core & (L < Lb - 5)
    rim = core & (L > Lb + 5)
    ids = [art.color(c) for c in LEAF]
    step = {ids[i]: (ids[max(i - 1, 0)], ids[min(i + 1, len(ids) - 1)]) for i in range(len(ids))}
    for (y, x) in zip(*np.where(zone & (vein | rim))):
        c = idx[y, x]
        if c in step:
            idx[y, x] = step[c][0] if vein[y, x] else step[c][1]
    # leaf ink: the original outlines the leaves in very dark green, not navy
    other = (idx != out) & (idx != TRANSPARENT) & ~zone
    ink = (idx == out) & grow(zone, 5, diag=True) & ~grow(other, 3, diag=True) & area
    idx[ink] = art.color(LEAF_INK)
    return zone


def detail_steps(art, zone, ramp, dark_mask, light_mask):
    """Move zone pixels one step down (dark_mask) or up (light_mask) the ramp."""
    idx = art.idx
    ids = [art.color(c) for c in ramp]
    pos = {c: i for i, c in enumerate(ids)}
    for (y, x) in zip(*np.where(zone & (dark_mask | light_mask))):
        i = pos.get(int(idx[y, x]))
        if i is None:
            continue
        idx[y, x] = ids[max(i - 1, 0)] if dark_mask[y, x] else ids[min(i + 1, len(ids) - 1)]


def fix_shell2(art):
    """Shell: smoother plates (wider blur), then the original's dark grooves and pale plate rims."""
    d = ref_data()
    L, a, b = d["lab"][..., 0], d["lab"][..., 1], d["lab"][..., 2]
    idx = art.idx
    out = art.names["outline"]
    P = poly([(320, 140), (360, 150), (420, 180), (470, 260), (472, 380), (445, 420), (390, 420),
              (372, 395), (350, 365), (320, 345), (318, 300), (326, 240)])
    zone = P & np.isin(idx, [art.color(c) for c in SHELL])
    requant(art, zone, SHELL, smooth=0, lab=blur_lab(2))
    requant(art, zone, SHELL, smooth=3, lab=blur_lab(2))
    Lb = blur_lab(4)[..., 0]
    core = shrink(zone, 1)
    detail_steps(art, zone, SHELL, core & (L < Lb - 6), core & (L > Lb + 6))
    # the deepest part of each groove goes one more step down
    detail_steps(art, zone, SHELL, core & (L < Lb - 11), np.zeros_like(zone))
    return zone


BELLY2 = [(248, 243, 207), (240, 229, 182), (226, 211, 157), (210, 195, 136), (190, 180, 122), (150, 146, 98)]
GEM_POLY = [(184.5, 385.5), (160.5, 414.5), (188.5, 452.5), (218.5, 414.5)]


def fix_belly2(art):
    """Belly: one more in-between tan and a wider blur, so the shading reads as a few
    smooth steps instead of stacked contour bands."""
    idx = art.idx
    zone = np.isin(idx, [art.color(c) for c in BELLY]) & poly([(136, 378), (292, 378), (284, 472), (138, 472)])
    zone &= ~grow(poly(GEM_POLY), 2, diag=True)
    requant(art, zone, BELLY2, smooth=0, lab=blur_lab(2))
    requant(art, zone, BELLY2, smooth=3, lab=blur_lab(2))


HEAD = [(40, 148, 106), (47, 144, 136), (48, 173, 111), (68, 199, 132), (130, 214, 136), (129, 216, 195),
        (183, 224, 149)]


def fix_head(art):
    """Green top of the head: the original's spots on a light-green dome, one clean ramp."""
    d = ref_data()
    L, a, b = d["lab"][..., 0], d["lab"][..., 1], d["lab"][..., 2]
    idx = art.idx
    out = art.names["outline"]
    head = poly([(60, 175), (80, 140), (120, 115), (170, 108), (250, 108), (300, 125), (325, 160),
                 (330, 230), (300, 250), (60, 250)])
    zone = head & (a < -18) & (L > 40) & (b < 45) & (idx != out) & (idx != TRANSPARENT)
    zone &= ~grow(R_EYE_OUT | L_EYE_OUT, 2, diag=True) & ~box(118, 226, 156, 238)   # eyes, nostrils
    requant(art, zone, HEAD, smooth=0, lab=blur_lab(1))
    requant(art, zone, HEAD, smooth=2, lab=blur_lab(1))
    return zone


def plain_blur(r=1):
    key = ("plain", r)
    if key not in _CACHE:
        lab = ref_data()["lab"]
        n = 2 * r + 1
        p = np.pad(lab, ((r, r), (r, r), (0, 0)), mode="edge")
        _CACHE[key] = sum(p[dy:dy + SIZE, dx:dx + SIZE] for dy in range(n) for dx in range(n)) / n / n
    return _CACHE[key]


def fix_scarf2(art):
    """Scarf band: fewer, larger flat blue areas (the original is an even royal blue
    with a darker band under the chin), keeping the letters and the gem untouched."""
    idx = art.idx
    blues = [NAVY, BLUE_D, BLUE_M, BLUE_B]
    zone = np.isin(idx, [art.color(c) for c in blues]) & box(96, 280, 356, 392)
    zone &= ~grow(poly(GEM_POLY), 2, diag=True)
    requant(art, zone, blues, smooth=0, lab=plain_blur(1))
    requant(art, zone, blues, smooth=3, lab=plain_blur(1))


def smooth_silhouette(art, passes=2):
    """Clean the outer edge: drop 1px outline spurs, fill 1px notches, round off
    stair-step bumps. Only ink pixels are removed and only ink is added."""
    idx = art.idx
    out = art.names["outline"]
    leaf_ink = art.color(LEAF_INK)
    for _ in range(passes):
        solid = idx != TRANSPARENT
        ink = np.isin(idx, [out, leaf_ink])
        p = np.pad(solid.astype(np.int16), 1)
        n4 = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
        n8 = sum(p[dy:dy + SIZE, dx:dx + SIZE] for dy in range(3) for dx in range(3)) - p[1:-1, 1:-1]
        drop = solid & ink & ((n4 <= 1) | (n8 <= 2))
        fill = ~solid & ((n4 >= 3) | (n8 >= 6))
        near_leaf = grow(idx == leaf_ink, 1)
        idx[drop] = TRANSPARENT
        idx[fill & near_leaf] = leaf_ink
        idx[fill & ~near_leaf] = out
    # anything left exposed on the edge must be ink
    solid = idx != TRANSPARENT
    bare = edge(solid) & ~np.isin(idx, [out, leaf_ink])
    idx[bare] = out


def fix_gem2(art):
    """Gem facets re-derived from the reference with the gem palette (the hand-drawn
    facets had the lower-left facet far too dark); the crisp rim lines stay."""
    idx = art.idx
    rim = np.isin(idx, [art.color(NAVY), art.color(GEM_RIM)])
    inner = poly(GEM_POLY) & ~rim & np.isin(idx, [art.color(c) for c in (GEM_D, GEM_M, GEM_C, GEM_L, GEM_W)])
    requant(art, inner, [GEM_D, GEM_M, GEM_C, GEM_L, GEM_W], smooth=0)
    requant(art, inner, [GEM_D, GEM_M, GEM_C, GEM_L, GEM_W], smooth=2)


BUMP = [(199, 185, 86), (236, 217, 128), (247, 230, 160)]


def fix_bump(art):
    """Yellow bump on the head: base, darker lower rim and the original's soft pale highlight."""
    d = ref_data()
    L, a, b = d["lab"][..., 0], d["lab"][..., 1], d["lab"][..., 2]
    idx = art.idx
    zone = box(192, 104, 274, 144) & np.isin(idx, [art.color(c) for c in BUMP[:2]])
    inner = shrink(zone, 2)
    idx[inner & ell(229.0, 117.5, 18.0, 4.6)] = art.color(BUMP[2])
    idx[inner & ell(218.5, 117.0, 3.6, 2.4)] = art.color((253, 244, 205))


def fix_mouth2(art):
    """Mouth re-derived at full size: the first pass's mask lost the dark lip line all
    round (it then sat under a 2px white rim), so the mouth came out ~2px too small,
    V-shaped and without its cat-like upper lip."""
    d = ref_data()
    L, a, b = d["lab"][..., 0], d["lab"][..., 1], d["lab"][..., 2]
    idx = art.idx
    out = art.names["outline"]
    mbox = box(106, 246, 186, 302)
    rimc = art.color((253, 251, 238))
    idx[mbox & (idx == rimc)] = art.color(CREAM)
    m = mbox & ((a > 12) | (L < 62))
    mouth = [out, MAROON_D, MAROON, RED, TONGUE, TONGUE_L]
    requant(art, m, mouth, smooth=0)
    requant(art, m, mouth, smooth=1)
    # pixels the smoothing pushed out of the mask go back to face cream
    stray = mbox & ~m & np.isin(idx, [art.color(c) for c in mouth])
    idx[stray] = art.color(CREAM)
    # the original's lips sit in a thin white halo
    lips = mbox & np.isin(idx, [art.color(c) for c in mouth])
    rim = grow(smooth_mask(lips, 1) | lips, 2, diag=True) & ~lips & mbox & (idx != TRANSPARENT)
    idx[rim] = rimc
    # an even lip line all round: near-black on the upper lip and sides, dark maroon low down
    line = lips & grow(rim, 1)
    idx[line & (PY < 272)] = out
    idx[line & (PY >= 272) & ~np.isin(idx, [out])] = art.color(MAROON_D)


# -------------------------------------------------------------------- main --
def touchup(art):
    trim_ring(art)
    fix_eyes(art)
    fix_mouth(art)
    fix_face(art)
    fix_scarf(art)
    fix_belly(art)
    fix_shell(art)
    fix_gem(art)
    fix_legs(art)
    fix_claws(art)
    # second pass: re-derive what the first pass got wrong, smooth the rest
    fix_eyes2(art)
    fix_face2(art)
    fix_leaves(art)
    fix_shell2(art)
    fix_belly2(art)
    fix_head(art)
    fix_scarf2(art)
    fix_gem2(art)
    fix_bump(art)
    fix_mouth2(art)
    smooth_silhouette(art)
    # last: fold stray 1-3px specks, but never inside the hand-drawn features
    protect = (grow(R_EYE_OUT | L_EYE_OUT, 2, diag=True)                   # eyes
               | box(118, 226, 156, 238)                                    # nostrils
               | box(150, 342, 232, 378)                                    # IQX
               | grow(poly(GEM_POLY), 3, diag=True))                          # gem
    despeckle(art, 3, protect)
