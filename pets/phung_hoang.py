"""Phung Hoang (phoenix): pixelize config and hand touch-ups (512x512 sprite coords)."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from pixelize import TRANSPARENT, majority, nearest, placement, reference, rgb2lab

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


def seg(shape, pts, r):
    """Pixels whose centre lies within r of the polyline through pts."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    px, py = xx + 0.5, yy + 0.5
    m = np.zeros(shape, bool)
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        dx, dy = x1 - x0, y1 - y0
        L2 = float(dx * dx + dy * dy) or 1.0
        t = np.clip(((px - x0) * dx + (py - y0) * dy) / L2, 0, 1)
        m |= (px - x0 - t * dx) ** 2 + (py - y0 - t * dy) ** 2 <= r * r
    return m


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


def requantize(art, mask, colors, smooth=1, weights=None):
    """Re-map masked pixels to the nearest (Lab) of `colors`, from the reference."""
    ids = [art.color(c) for c in colors]
    pal = rgb2lab(np.array([art.palette[i] for i in ids], np.float64))
    lab = ref_lab()[mask]
    if weights is not None:
        lab = lab * np.asarray(weights)
        pal = pal * np.asarray(weights)
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


def recolor(art, mask, src, dst):
    sel = mask & np.isin(art.idx, [art.color(s) for s in src])
    art.idx[sel] = art.color(dst)


def depth(idx):
    """4-connected distance (in px) of each solid pixel from the nearest transparent pixel."""
    solid = idx >= 0
    d = np.zeros(idx.shape, int)
    reach = ~solid
    k = 0
    while True:
        k += 1
        p = np.pad(reach, 1)
        g = (p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]) & ~reach
        if not g.any():
            return d
        d[g] = k
        reach |= g


def shift(a, dy, dx, fill):
    """a shifted so that out[y, x] = a[y + dy, x + dx]."""
    h, w = a.shape
    p = np.pad(a, 1, constant_values=fill)
    return p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]


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


def disk(r2):
    r = int(r2 ** 0.5)
    return [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1) if dy * dy + dx * dx <= r2]


def erode(m, offs):
    out = m.copy()
    for dy, dx in offs:
        out &= shift(m, dy, dx, False)
    return out


def dilate_by(m, offs):
    out = m.copy()
    for dy, dx in offs:
        out |= shift(m, dy, dx, False)
    return out


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


def thin_outline(art, keep=3.0, r2=5):
    """Where the silhouette ink band is thick (it survives an opening with a
    disk), trim it to `keep` px from the edge; thin interior lines are untouched."""
    o = art.names["outline"]
    ink = art.idx == o
    e = edt(art.idx >= 0)
    offs = disk(r2)
    thick = dilate_by(erode(ink, offs), offs) & ink
    rem = thick & (e > keep)
    fill_from_neighbours(art, rem, avoid=[o])


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


def eyes(art):
    left = poly([(286, 186), (296, 176), (312, 168), (336, 166), (358, 172), (370, 186), (372, 206),
                 (368, 226), (356, 240), (336, 244), (316, 240), (300, 226), (292, 204)])
    right = poly([(412, 222), (418, 204), (430, 192), (446, 188), (462, 194), (472, 210), (472, 230),
                  (466, 248), (452, 260), (432, 262), (418, 254), (410, 240)])
    cols = [BLACK, RBROWN, RORANGE, IORANGE, AMBER, YELLOW, SCLERA, CREAM, CREAM2, CREAM3]
    for m in (left, right):
        requantize(art, m, cols, smooth=1)
    # grey shading only on the top of the left sclera, under the lash
    shade = poly([(298, 192), (306, 185), (318, 181), (332, 180), (344, 182), (342, 188),
                  (330, 188), (318, 191), (308, 197), (300, 203)])
    requantize(art, shade, [BLACK, RBROWN, SCLERA, SCLERA_G], smooth=0)
    # crisp white catch-lights
    art.idx[ell(333.6, 198.6, 3.6, 3.6)] = art.color(WHITE)
    art.idx[ell(430.6, 219.8, 2.8, 3.4)] = art.color(WHITE)


def touchup(art):
    art.name("black", BLACK)
    thin_outline(art)
    eyes(art)
