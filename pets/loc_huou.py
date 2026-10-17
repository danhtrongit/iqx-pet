"""Loc Huou (golden deer): pixelize config and hand touch-ups (512x512 sprite coords)."""
from __future__ import annotations

import numpy as np
from PIL import Image

from pixelize import TRANSPARENT, majority, nearest, placement, reference, rgb2lab

SLUG = "loc-huou"
ALPHA_CUT = 180

CONFIG = {
    "colors": 56,
    "ink_l": 24.0,
    "ink_frac": 0.5,
    "smooth": 3,
    "min_island": 20,
    "seed": 0,
}


def ref_small(size=512, margin=6):
    """The reference box-downscaled onto the sprite grid (same as pixelize sees it)."""
    im = reference(SLUG)
    box, (ox, oy, nw, nh) = placement(SLUG, size, margin)
    small = np.array(im.crop(box).convert("RGBa").resize((nw, nh), Image.BOX).convert("RGBA"), np.float64)
    canvas = np.zeros((size, size, 4))
    canvas[oy:oy + nh, ox:ox + nw] = small
    return canvas


def edges(solid):
    p = np.pad(solid, 1)
    return solid & ~(p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:])


def fill_holes(mask):
    """Fill enclosed holes of a boolean mask (flood fill from the border)."""
    h, w = mask.shape
    reach = np.zeros_like(mask)
    reach[0, :] = ~mask[0, :]; reach[-1, :] = ~mask[-1, :]
    reach[:, 0] |= ~mask[:, 0]; reach[:, -1] |= ~mask[:, -1]
    while True:
        p = np.pad(reach, 1)
        grown = (reach | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]) & ~mask
        if (grown == reach).all():
            return mask | ~reach
        reach = grown


def requantize(art, lab, mask, colors, smooth=1):
    """Re-map masked pixels to the nearest of `colors` (palette refs), then clean them up."""
    ids = [art.color(c) for c in colors]
    pal = rgb2lab(np.array([art.palette[i] for i in ids], np.float64))
    lbl = nearest(lab[mask], pal)
    art.idx[mask] = np.array(ids)[lbl]
    if smooth:
        ys, xs = np.where(mask)
        y0, y1, x0, x1 = ys.min() - 1, ys.max() + 2, xs.min() - 1, xs.max() + 2
        sub = art.idx[y0:y1, x0:x1]
        # majority() works on palette indices < n; remap to a compact range
        uniq = np.unique(sub)
        uniq = uniq[uniq >= 0]
        lut = {v: i for i, v in enumerate(uniq)}
        comp = np.vectorize(lambda v: lut.get(v, -1))(sub)
        comp = majority(comp, len(uniq), smooth)
        back = np.where(comp >= 0, uniq[np.clip(comp, 0, None)], -1)
        m = mask[y0:y1, x0:x1]
        sub[m] = back[m]


def seg_mask(shape, pts, r):
    """Pixels whose centre lies within r of the polyline through pts (x, y)."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    px, py = xx + 0.5, yy + 0.5
    m = np.zeros(shape, bool)
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        dx, dy = x1 - x0, y1 - y0
        t = np.clip(((px - x0) * dx + (py - y0) * dy) / float(dx * dx + dy * dy), 0, 1)
        m |= (px - x0 - t * dx) ** 2 + (py - y0 - t * dy) ** 2 <= r * r
    return m


def despeckle(art, max_area, protect):
    """Fold 8-connected colour specks of <= max_area px into their most common neighbour."""
    idx = art.idx
    h, w = idx.shape
    seen = np.zeros((h, w), bool)
    seen[idx == TRANSPARENT] = True
    nb8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for y0, x0 in zip(*np.where(~seen)):
        if seen[y0, x0]:
            continue
        c = idx[y0, x0]
        stack, region = [(y0, x0)], []
        seen[y0, x0] = True
        while stack and len(region) <= max_area:
            y, x = stack.pop()
            region.append((y, x))
            for dy, dx in nb8:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and not seen[yy, xx] and idx[yy, xx] == c:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        if stack or len(region) > max_area:
            # big region: mark the rest as seen without collecting
            while stack:
                y, x = stack.pop()
                for dy, dx in nb8:
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and not seen[yy, xx] and idx[yy, xx] == c:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
            continue
        if any(protect[y, x] for y, x in region):
            continue
        votes = {}
        for y, x in region:
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w:
                    v = idx[yy, xx]
                    if v != c and v != TRANSPARENT:
                        votes[v] = votes.get(v, 0) + 1
        if votes:
            best = max(sorted(votes), key=lambda v: votes[v])
            for y, x in region:
                idx[y, x] = best


def touchup(art):
    idx = art.idx
    out = art.names["outline"]
    ref = ref_small()
    lab = rgb2lab(ref[..., :3])
    L = lab[..., 0]
    yy, xx = np.mgrid[0:512, 0:512]
    px, py = xx + 0.5, yy + 0.5

    # --- outline weight --------------------------------------------------
    # The pipeline adds a 1px outline ring outside the silhouette; the original's
    # soft outer edge already gives a dark rim, so the ring makes it look too heavy.
    solid = idx != TRANSPARENT
    idx[edges(solid) & (idx == out)] = TRANSPARENT
    # The original's outer edge is a soft alpha ramp; cutting it a little higher
    # than 50% keeps the silhouette smooth and the outline closer to its ~3px.
    cut = ref[..., 3] < ALPHA_CUT
    cut[312:342, 168:200] = False          # semi-transparent gap, handled below
    idx[cut] = TRANSPARENT
    idx[edges(idx != TRANSPARENT)] = out

    # --- eyes: re-map with an eye palette so the brown/amber iris survives --
    black = out
    dbrown, brown, amber, gold = (60, 27, 8), (112, 58, 10), (176, 108, 16), (239, 176, 14)
    shade, white = (191, 181, 172), (250, 248, 246)
    nose = ((xx + 0.5 - 333) / 13) ** 2 + ((yy + 0.5 - 254) / 8) ** 2 <= 1
    eyes = np.zeros((512, 512), bool)
    for (x0, y0, x1, y1) in ((208, 196, 290, 264), (334, 194, 366, 264)):
        box = np.s_[y0:y1, x0:x1]
        ring = (L[box] < 55) & ~nose[box]
        if x0 < 300:
            # the lower-left lid is soft in the original: close it so the white is inside
            ring |= seg_mask(idx.shape, [(225.5, 238.5), (233.5, 252.0), (246.0, 258.0), (258.0, 259.0)], 0.8)[box]
        m = np.zeros((512, 512), bool)
        m[box] = fill_holes(ring)
        m &= idx != TRANSPARENT
        requantize(art, lab, m, [black, dbrown, brown, amber, gold, shade, white], smooth=1)
        eyes |= m
        # anti-aliased grey where the lid meets the skin -> the skin colour next to it
        sh = art.color(shade)
        for y, x in zip(*np.where(m & (idx == sh))):
            nb = [idx[y + dy, x + dx] for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
                  if not m[y + dy, x + dx] and idx[y + dy, x + dx] not in (out, TRANSPARENT)]
            if nb:
                idx[y, x] = max(set(nb), key=nb.count)
        if x0 < 300:
            # auto-pass gold that leaked below the eye belongs to the lower lid
            prgb = np.array(art.palette, np.float64)
            plab = rgb2lab(prgb)
            pc = np.hypot(plab[:, 1], plab[:, 2])
            below = np.zeros_like(m)
            below[236:262, 228:285] = True
            cur = np.clip(idx, 0, None)
            leak = below & ~m & (idx != out) & (idx != TRANSPARENT) & (pc[cur] > 45) & (plab[cur, 0] < 84)
            m |= leak
            eyes |= leak
            # lower lid: thin brown line under the iris, grey-brown under the white
            for x in range(229, 283):
                col = np.where(m[:, x])[0]
                if len(col) and col.max() > 236:
                    yb = col.max()
                    if idx[yb, x] != out:
                        idx[yb, x] = art.color(brown if x >= 243 else shade)
    # orange glint sits on the black of the iris
    glint = (((xx + 0.5 - 247.5) ** 2 + (yy + 0.5 - 221.5) ** 2) <= 16) & eyes
    idx[glint & np.isin(idx, [art.color(c) for c in (dbrown, brown, amber)])] = out
    # crisp catch-lights
    art.ellipse(266.0, 220.5, 2.7, 2.7, (255, 255, 255))
    art.ellipse(350.5, 213.5, 2.5, 2.5, (255, 255, 255))
    art.blit(246, 220, [".o.", "ooo", ".o."], {"o": (232, 160, 50)})

    # --- nose and mouth: crisp dark line, flat pink nose/tongue, clean cream --
    creams = [(249, 238, 221), (248, 243, 235), (249, 235, 213), (249, 229, 195)]
    B = np.zeros((512, 512), bool)
    B[240:284, 298:354] = True
    B &= ~eyes & (idx != TRANSPARENT)
    dark = B & (L < 42) & (yy < 259)
    pink = B & ~dark & (lab[..., 1] > 20) & (L < 82)
    rest = B & ~dark & ~pink
    requantize(art, lab, rest, creams, smooth=0)
    idx[dark] = out
    idx[pink & (yy < 262)] = art.color((206, 100, 78))
    # the nose rim is a deep red-brown in the original, not the black ink
    nbox = np.zeros((512, 512), bool)
    nbox[244:261, 312:346] = True
    idx[nbox & (idx == out) & ~eyes] = art.color((84, 34, 28))
    # mouth: hand-drawn from the reference (2px smile line, tongue below the left half)
    tongue = art.color((212, 92, 86))
    mouth = {
        263: [(346, 348, out)],
        264: [(344, 347, out)],
        265: [(329, 330, out), (342, 346, out)],
        266: [(302, 304, out), (326, 331, out), (340, 344, out)],
        267: [(302, 305, out), (324, 333, out), (338, 342, out)],
        268: [(303, 307, out), (322, 340, out)],
        269: [(305, 309, out), (320, 337, out)],
        270: [(306, 312, out), (318, 332, out)],
        271: [(309, 331, out)],
        272: [(312, 318, out), (319, 327, tongue), (328, 330, out)],
        273: [(312, 312, out), (313, 327, tongue), (328, 329, out)],
        274: [(312, 328, tongue)],
        275: [(313, 328, tongue)],
        276: [(314, 326, tongue)],
        277: [(316, 325, tongue)],
        278: [(318, 323, tongue)],
    }
    for y, runs in mouth.items():
        for x0, x1, c in runs:
            idx[y, x0:x1 + 1] = c

    # --- face: drop grey anti-alias specks the auto pass left around eyes/mouth --
    face = np.zeros((512, 512), bool)
    face[188:292, 196:376] = True
    face &= ~eyes
    for _ in range(3):
        plab = rgb2lab(np.array(art.palette, np.float64))
        pc = np.hypot(plab[:, 1], plab[:, 2])
        cur = np.clip(idx, 0, None)
        grey = face & (idx != TRANSPARENT) & (idx != out) & (pc[cur] < 10) & (plab[cur, 0] > 35) & (plab[cur, 0] < 88)
        for y, x in zip(*np.where(grey)):
            nb = [idx[y + dy, x + dx] for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))]
            nb = [v for v in nb if v not in (out, TRANSPARENT) and not grey[y, x] == (pc[v] < 10 and 35 < plab[v, 0] < 88)]
            if nb:
                idx[y, x] = max(set(nb), key=nb.count)

    # --- scarf: re-map the blue area with a blue ramp (no black ink on the folds) --
    navy, dblue, mblue, blue = (13, 23, 57), (6, 36, 99), (5, 59, 154), (9, 86, 203)
    lblue, pblue = (87, 142, 214), (167, 204, 240)
    scarf = np.zeros((512, 512), bool)
    scarf[280:362, 150:372] = True
    scarf &= (lab[..., 2] < -12) & (ref[..., 3] >= 128) & (idx != TRANSPARENT)
    requantize(art, lab, scarf, [navy, dblue, mblue, blue, lblue, pblue], smooth=1)

    # --- "IQX" lettering: clean, evenly stroked white letters on the blue band --
    lettering = np.zeros((512, 512), bool)
    lettering[312:335, 278:325] = True
    ramp = [art.color(c) for c in (navy, dblue, mblue, blue)]
    idx[lettering & (lab[..., 2] < 0) & ~np.isin(idx, ramp)] = art.color(blue)
    W = art.color((255, 255, 255))
    art.rect(281, 315, 284, 330, W)                                   # I
    art.blit(288, 315, [                                              # Q (round, 4px stroke)
        "......#####......",
        "....#########....",
        "..#############..",
        "..#############..",
        ".#####.....#####.",
        ".####.......####.",
        "####.........####",
        "####.........####",
        "####.........####",
        "####.........####",
        "####.....##..####",
        ".####....#######.",
        ".#####....######.",
        "..#############..",
        "..##############.",
        "....#########.###",
        "......#####....##",
    ], {"#": W})
    art.blit(307, 315, [                                              # X
        "####......####",
        ".####....####.",
        ".####....####.",
        "..####..####..",
        "...########...",
        "...########...",
        "....######....",
        "....######....",
        ".....####.....",
        "....######....",
        "....######....",
        "...########...",
        "...########...",
        "..####..####..",
        ".####....####.",
        ".####....####.",
        "####......####",
    ], {"#": W})

    # --- gem pendant: crisp faceted rhombus, dark rim, white glow ------------
    gcx, gty, ggy, gby, ghw = 304.5, 341.0, 366.5, 394.5, 19.5
    top = (py <= ggy) & (np.abs(px - gcx) <= ghw * (py - gty) / (ggy - gty))
    bot = (py > ggy) & (np.abs(px - gcx) <= ghw * (gby - py) / (gby - ggy))
    gem = top | bot
    amber, ggold, gyellow, gpale = (230, 160, 16), (247, 183, 21), (253, 219, 51), (252, 238, 176)
    gemrim = art.color((150, 95, 18))
    cream = art.color((255, 252, 243))
    # stray gem-yellow pixels the auto pass spilled around the old pendant
    near = np.zeros_like(gem)
    near[336:402, 278:334] = True
    spill = near & ~gem & ~scarf & (lab[..., 2] > 30) & (L > 60) & (xx < 318) & (idx != out)
    requantize(art, lab, spill, creams + [(255, 252, 243)], smooth=0)
    requantize(art, lab, gem, [amber, ggold, gyellow, gpale], smooth=2)
    grow = gem.copy()
    for _ in range(3):
        p = np.pad(grow, 1)
        grow = grow | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    glow = grow & ~gem & (idx != TRANSPARENT) & (yy > 341) & ~scarf & (idx != out)
    idx[glow] = cream
    p = np.pad(gem, 1)
    ring8 = gem & ~(p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
                    & p[:-2, :-2] & p[:-2, 2:] & p[2:, :-2] & p[2:, 2:])
    idx[ring8] = gemrim
    # the gem's top tucks under the scarf: no light-blue halo there
    tuck = np.zeros_like(gem)
    tuck[332:343, 296:316] = True
    idx[tuck & np.isin(idx, [art.color(lblue), art.color(pblue)])] = art.color(dblue)

    # --- hooves: dark brown, not swallowed by the ink outline -----------------
    hoof = np.zeros((512, 512), bool)
    hoof[462:507, 110:345] = True
    hoof &= (ref[..., 3] >= 128) & (idx != TRANSPARENT) & (L >= 9) & (L < 38)
    hoof &= ~((idx == out) & (L < 12)) & ~edges(idx != TRANSPARENT)
    hdark, hmid, hbrown = (60, 34, 22), (80, 45, 25), (102, 59, 30)
    requantize(art, lab, hoof, [hdark, hmid, hbrown], smooth=1)

    # --- inner ears: restore the soft pink ------------------------------------
    pinks = [(250, 229, 195), (248, 209, 179), (248, 189, 158), (242, 170, 140)]
    for (x0, y0, x1, y1) in ((108, 132, 218, 204), (358, 116, 420, 196)):
        m = np.zeros((512, 512), bool)
        m[y0:y1, x0:x1] = True
        m &= (L > 72) & (lab[..., 2] < 33) & (lab[..., 1] > 3) & (idx != out) & (idx != TRANSPARENT)
        requantize(art, lab, m, pinks, smooth=2)

    # --- white spots on the back ----------------------------------------------
    back = np.zeros((512, 512), bool)
    back[338:392, 155:215] = True
    back &= (idx != out) & ~scarf
    idx[back & (L > 83.5)] = art.color((248, 218, 166))
    idx[back & (L > 87.5)] = art.color((250, 240, 226))

    # --- the soft gap between tail, scarf end and back -------------------------
    # The reference is semi-transparent here (it reads as pale grey fur on white);
    # un-premultiplied it turns muddy brown, so re-map it from the on-white look.
    a = ref[..., 3:4] / 255.0
    onwhite = rgb2lab(ref[..., :3] * a + 255.0 * (1 - a))
    gap = np.zeros((512, 512), bool)
    gap[312:342, 168:200] = True
    solid = idx != TRANSPARENT
    hole = gap & fill_holes(solid) & ~solid
    idx[hole] = art.color((206, 198, 204))
    gap &= (idx != TRANSPARENT) & ~edges(idx != TRANSPARENT) & ~scarf & (onwhite[..., 2] <= 20)
    gl = onwhite[..., 0]
    idx[gap] = np.where(gl[gap] >= 62, art.color((206, 198, 204)),
                        np.where(gl[gap] >= 40, art.color((150, 140, 146)), out))

    # --- white tail: navy outline and blue-grey fluff strokes like the original --
    tnavy, tslate, tgrey, twhite = (17, 27, 61), (56, 60, 81), (171, 170, 174), (244, 239, 230)
    tail = np.zeros((512, 512), bool)
    tail[286:402, 98:152] = True
    tail &= (idx != TRANSPARENT) & (lab[..., 2] < 8)
    rim = tail & edges(idx != TRANSPARENT)
    requantize(art, lab, tail & ~rim, [out, tnavy, tslate, twhite], smooth=1)
    idx[rim] = art.color(tnavy)
    idx[tail & (idx == out)] = art.color(tnavy)

    # navy-tinted bits of the line art (e.g. the gap between left antler and ear)
    navyink = (idx == out) & (lab[..., 2] < -15) & (L > 8) & (ref[..., 3] >= 128)
    idx[navyink] = art.color(tnavy)

    # --- final clean-up: fold lone 1-3px colour specks into their surroundings --
    protect = (idx == out) | eyes | lettering | grow | (idx == W)
    protect[240:284, 296:356] = True       # nose and mouth
    despeckle(art, 3, protect)
