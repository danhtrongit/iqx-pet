"""Phụng Hoàng: evolution line for the game sprites (see tools/gamekit.py).

Big shapes (wings, crest, tail, recolours) are built at illustration level from the original's own
painted parts, so every stage keeps the same brushwork; small details (eyes, brows, gem, IQX
lettering, talons, crown, gorget, aura, sparks) are placed pixel by pixel in touchup().

  1 Tiểu Phụng             the original baby phoenix, faithfully
  2 Hỏa Phụng              wings opened wide, taller flame crest, longer tail, stern eyes, embers
  3 Thần Hỏa Phượng Hoàng  massive raised wings in a crimson -> orange -> gold fire gradient,
                           white-hot crest under a golden crown with a royal-blue gem, golden
                           gorget carrying the gem, slit-pupil glare, crimson fire aura

"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw

_TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

import drawkit as dk  # noqa: E402
import gamekit as gk  # noqa: E402
from pixelize import TRANSPARENT, components, grow4, rgb2lab  # noqa: E402

PET = {
    "name": "Phụng Hoàng",
    "layer": "Tin tức",
    "element": "news_coral",
    "accent": "#ff6b4a",
    "tagline": "Bắt nhịp tin tức nóng trước cả đám đông.",
}

INK = "#3a1426"
CFG = {"colors": 30, "ink_l": 30.0, "ink_frac": 0.45, "smooth": 1, "min_island": 3, "outline": INK}

STAGES = [
    {"rank": "Ấu thể", "name": "Tiểu Phụng", "level": 1, "height": 104,
     "config": dict(CFG),
     "desc": "Chú phượng non nở từ tro tàn của bản tin sáng, mắt hổ phách dõi theo từng dòng tít. "
             "Tin vừa lên sóng là Tiểu Phụng đã ríu rít báo cả bầy.",
     "skill": {"name": "Tia Lửa Tin Nóng",
               "desc": "Phóng một đốm lửa đánh dấu tin mới nhất, cả đội phản ứng nhanh hơn 10% ở lượt kế tiếp."},
     "stats": {"hp": 38, "atk": 36, "def": 26, "spd": 46}},
    {"rank": "Trưởng thành", "name": "Hỏa Phụng", "level": 10, "height": 128,
     "config": dict(CFG),
     "desc": "Cánh lửa đã xòe rộng, mào lửa vút cao. Hỏa Phụng lướt qua nghìn nguồn tin mỗi phút "
             "và thiêu rụi tin đồn trước khi nó kịp lan.",
     "skill": {"name": "Lửa Thanh Lọc",
               "desc": "Quét một luồng lửa qua chiến trường, xóa hiệu ứng tin đồn của đối thủ "
                       "và gây bỏng trong 2 lượt."},
     "stats": {"hp": 62, "atk": 66, "def": 44, "spd": 74}},
    {"rank": "Thần thú", "name": "Thần Hỏa Phượng Hoàng", "level": 25, "height": 146,
     "config": dict(CFG, colors=32),
     "desc": "Thần điểu đội vương miện vàng, sải cánh lửa che kín trời, tái sinh sau mọi cơn bão tin. "
             "Tiếng hót của Người báo trước biến động lớn của thị trường.",
     "skill": {"name": "Niết Bàn Thần Hỏa",
               "desc": "Hóa biển lửa thiêng soi rõ mọi dòng tin: tái sinh đầy máu một lần mỗi trận, "
                       "cả đội tăng 30% công trong 3 lượt."},
     "stats": {"hp": 86, "atk": 95, "def": 68, "spd": 97}},
]

# Fixed detail colours (hand-placed pixels).
P = {
    "ink": INK, "white": "#ffffff", "sclera": "#fff8ee",
    "iris_d": "#a8400e", "iris": "#e8801a", "iris_l": "#ffbe3a", "pupil": "#24080e",
    "glow_d": "#e86a10", "glow": "#ffc22e", "glow_l": "#fff1a0",
    "gem_hi": "#fff4c8", "gem_l": "#ffc040", "gem": "#ff8a1e", "gem_m": "#f05a1c", "gem_d": "#b8301a",
    "blue_d": "#103e9c", "blue": "#1a56c4", "blue_l": "#5d93e0", "blue_hi": "#cfe2ff",
    "au_d": "#b0701a", "au": "#f5bf30", "au_l": "#ffe066", "au_hi": "#fff6c8",
    "mouth": "#e8707a", "mouth_d": "#b84858",
    "tal_d": "#5c2c1c", "tal": "#8e5032", "tal_l": "#c07a4c",
    "red": "#e8382a", "flame": "#ff8a24", "spark": "#ffe680",
}


# ------------------------------------------------------- art-level helpers ----
def R(x, y):
    """Reference poster px -> art canvas px."""
    return x + gk.PAD, y + gk.PAD


def poly_mask(pts):
    """Boolean art-canvas mask of a polygon given in reference px."""
    m = Image.new("L", (gk.ART, gk.ART), 0)
    ImageDraw.Draw(m).polygon([tuple(map(float, R(*q))) for q in pts], fill=255)
    return np.array(m) > 127


def opaque(layer):
    return np.array(layer)[..., 3] > 0


# Parts of the original illustration reused (reference px).
WING_POLY = [(330, 285), (295, 330), (262, 410), (252, 470), (258, 560), (285, 620), (335, 668), (420, 698),
             (500, 702), (530, 680), (535, 630), (515, 580), (480, 500), (445, 430), (410, 370), (375, 315)]
CREST_POLY = [(400, 140), (470, 135), (600, 150), (700, 180), (780, 235), (805, 300), (790, 380),
              (740, 378), (680, 372), (620, 380), (560, 400), (505, 425), (460, 430), (415, 385), (390, 300)]
TAIL_POLY = [(180, 545), (135, 610), (122, 700), (148, 800), (190, 880), (250, 925), (350, 938), (450, 930),
             (495, 885), (480, 800), (430, 745), (340, 700), (280, 655), (235, 590)]


def paste_part(layer, src, poly, a_src, a_dst, scale=1.0, angle=0.0, flip=False):
    """Copy the polygon (reference px) of src, scale ((sx, sy) or one number) / rotate (degrees,
    counter-clockwise) / mirror it about a_src, and composite it onto layer with a_src on a_dst."""
    m = poly_mask(poly)
    part = np.array(src).copy()
    part[..., 3] = np.where(m, part[..., 3], 0)
    th = np.radians(angle)
    fx = -1.0 if flip else 1.0
    sx, sy = (scale, scale) if np.isscalar(scale) else scale
    A = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]]) @ np.diag([fx * sx, sy])
    Ai = np.linalg.inv(A)
    off = np.array(R(*a_src)) - Ai @ np.array(R(*a_dst))
    im = Image.fromarray(part, "RGBA").convert("RGBa")
    out = im.transform(layer.size, Image.AFFINE, (Ai[0, 0], Ai[0, 1], off[0], Ai[1, 0], Ai[1, 1], off[1]),
                       resample=Image.BICUBIC).convert("RGBA")
    layer.alpha_composite(out)


def fire_grade(img, mask, center, radius, stops, pos=0.3):
    """Recolour masked pixels through colour stops by luminance mixed with closeness to center
    (reference px): far pixels (feather tips) take the dark end, pixels near the root the hot end."""
    a = np.array(img).astype(np.float64)
    lum = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]) / 255
    cx, cy = R(*center)
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    near = 1 - np.clip(np.hypot(xx - cx, yy - cy) / radius, 0, 1)
    t = np.clip((1 - pos) * lum + pos * near, 0, 1)[mask]
    cols = np.array([dk.hexrgba(c)[:3] for c in stops], np.float64)
    q = np.linspace(0, 1, len(cols))
    a[..., :3][mask] = np.stack([np.interp(t, q, cols[:, k]) for k in range(3)], -1)
    img.paste(Image.fromarray(a.clip(0, 255).astype(np.uint8), "RGBA"))


# ------------------------------------------------------------- compose ----
def stage2(img, ctx):
    src = img.copy()
    back = dk.blank(img)
    # longer tail plumes
    paste_part(back, src, TAIL_POLY, (470, 840), (470, 830), 1.25, -6)
    # far wing (right): mirrored copy opening behind the chest
    paste_part(back, src, WING_POLY, (520, 690), (735, 700), 1.0, 0, flip=True)
    # near wing: a bigger copy spreading behind the original one
    paste_part(back, src, WING_POLY, (520, 690), (520, 690), 1.35, 14)
    # taller crest
    paste_part(back, src, CREST_POLY, (640, 420), (640, 420), 1.32, 4)
    return dk.behind(img, back)


WING3 = ["#2a0612", "#6a0e1e", "#a8162a", "#d8282a", "#f2502a", "#fb8a2c", "#ffc244", "#ffe680"]
BEAK_POLY = [(680, 500), (772, 500), (778, 560), (752, 592), (698, 592), (676, 552)]
LEGS_POLY = [(470, 842), (600, 838), (615, 870), (745, 845), (775, 945), (455, 945)]
CREST3 = ["#3a0a14", "#b0202a", "#ee5a24", "#ff9a2c", "#ffcf48", "#ffee98", "#fffbe8"]
BODY3 = ["#4a0a18", "#9c1626", "#dc2c26", "#f4641e", "#fca22c", "#ffd858", "#fff4b8"]


def stage3(img, ctx):
    src = img.copy()
    # hotter body: shadows sink to crimson, highlights rise to white-gold
    feathers = dk.hsv_mask(src, h=(350, 55), s=(0.45, 1), v=(0.35, 1))
    feathers &= ~poly_mask(BEAK_POLY) & ~poly_mask(LEGS_POLY)
    dk.gradient_map(src, feathers, BODY3)

    back = dk.blank(img)
    paste_part(back, src, TAIL_POLY, (470, 840), (470, 834), (1.28, 1.08), -4)    # longer tail
    # massive raised wings in a crimson -> orange -> gold fire gradient
    for args in [((520, 690), (750, 714), 1.65, 34, True), ((520, 690), (520, 724), 1.7, -10, False)]:
        lay = dk.blank(img)
        paste_part(lay, src, WING_POLY, *args[:4], flip=args[4])
        fire_grade(lay, opaque(lay), (640, 640), 780, WING3, pos=0.42)
        back.alpha_composite(lay)
    # white-hot flame crest rising behind the head
    crest = dk.blank(img)
    paste_part(crest, src, CREST_POLY, (640, 430), (640, 430), 1.5, 2)
    fire_grade(crest, opaque(crest), (650, 440), 420, CREST3, pos=0.3)
    back.alpha_composite(crest)
    return dk.behind(src, back)


def compose(stage, img, ctx):
    if stage == 2:
        return stage2(img, ctx)
    if stage == 3:
        return stage3(img, ctx)
    return img


# ---------------------------------------------------- sprite-level helpers ----
N = gk.CANVAS


def SP(ctx, x, y):
    """Reference px -> sprite coordinates (float; a pixel's centre is at +0.5)."""
    ax, ay = R(x, y)
    return ax * ctx.scale + ctx.ox, ay * ctx.scale + ctx.oy


def m_ellipse(cx, cy, rx, ry):
    yy, xx = np.mgrid[0:N, 0:N]
    return ((xx + 0.5 - cx) / rx) ** 2 + ((yy + 0.5 - cy) / ry) ** 2 <= 1.0


def m_poly(pts):
    """Sprite mask of a polygon in sprite coordinates."""
    m = Image.new("L", (N, N), 0)
    ImageDraw.Draw(m).polygon([(x - 0.5, y - 0.5) for x, y in pts], fill=1)
    return np.array(m).astype(bool)


def m_diamond(cx, cy, hw, hh):
    yy, xx = np.mgrid[0:N, 0:N]
    return np.abs(xx + 0.5 - cx) / hw + np.abs(yy + 0.5 - cy) / hh <= 1.0


def erode4(m):
    p = np.pad(m, 1)
    return m & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]


def ring(m):
    return grow4(m) & ~m


def paint(art, m, col):
    art.idx[m] = art.color(P.get(col, col))


def pal_lab(art):
    return rgb2lab(np.array(art.palette, np.float64))


def is_blue(art):
    pal = np.array(art.palette)
    good = (pal[:, 2] > pal[:, 0] + 40) & (pal[:, 2] > 90)
    lut = np.r_[good, False]
    return lut[art.idx]           # idx -1 -> last entry (False)


def skin(art):
    """Palette index of the cream face colour."""
    return art.closest((247, 234, 217))


FACE_POLY = [(540, 400), (600, 368), (700, 378), (800, 398), (862, 450), (868, 560), (800, 612), (790, 860),
             (600, 870), (560, 760), (530, 520)]


def despeckle(art, ctx, max_area=30):
    """Fold small muted islands (grey / tan / peach / pale: anti-alias leftovers and scaled-up
    highlight strokes; the feathers themselves are all strongly saturated) outside the face and
    chest into their most common neighbour colour."""
    lab = pal_lab(art)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    odd = (chroma < 50) & (lab[:, 0] > 35)
    face = m_poly([SP(ctx, x, y) for x, y in FACE_POLY])
    k_ink = art.names["outline"]
    flat = art.idx.ravel()
    for c, region in list(components(art.idx)):
        if c == k_ink or not odd[c] or len(region) > max_area:
            continue
        ys, xs = np.divmod(region, N)
        if face[ys, xs].mean() > 0.5:
            continue
        nb = []
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            yy, xx = np.clip(ys + dy, 0, N - 1), np.clip(xs + dx, 0, N - 1)
            nb.append(flat[yy * N + xx])
        nb = np.concatenate(nb)
        nb = nb[(nb != c) & (nb != TRANSPARENT) & (nb != k_ink)]
        nb = nb[~odd[nb]] if (~odd[nb]).any() else nb
        if len(nb):
            vals, cnt = np.unique(nb, return_counts=True)
            flat[region] = vals[cnt.argmax()]


# ------------------------------------------------------------------ talons ----
# per foot (reference px): clearing box, hip (where the leg leaves the body), heel
FEET = [((468, 850, 602, 945), (532, 862), (534, 912)),
        ((606, 850, 762, 945), (676, 864), (680, 910))]


def draw_feet(art, ctx):
    """Redraw the brown talons: short scaly leg, three toes fanning onto the ground, dark claws."""
    for c in ("tal", "tal_l", "tal_d"):
        art.color(P[c])
    lab = pal_lab(art)
    pal = np.array(art.palette)
    brownish = (lab[:, 0] < 52) & (pal[:, 0] >= pal[:, 2])
    k_ink = art.names["outline"]
    s = ctx.scale / 0.1316
    yy, xx = np.mgrid[0:N, 0:N] + 0.5
    allm = np.zeros((N, N), bool)
    shade = []
    for box, hip, heel in FEET:
        x0, y0 = SP(ctx, box[0], box[1])
        x1, _ = SP(ctx, box[2], box[3])
        zone = (xx >= x0) & (xx < x1) & (yy >= y0)
        old = zone & (np.r_[brownish, False][art.idx] | (art.idx == k_ink))
        art.idx[old] = TRANSPARENT
        hx, hy = SP(ctx, *heel)
        px, py = SP(ctx, *hip)
        hy = min(hy, gk.GROUND - 2.0 * s)
        ground = gk.GROUND - 0.6
        leg = m_poly([(px - 1.5 * s, py), (px + 1.5 * s, py), (hx + 1.3 * s, hy), (hx - 1.3 * s, hy)])
        foot = leg.copy()
        for dx, w0 in ((-4.2, 1.5), (0.6, 1.6), (4.6, 1.4)):
            tx = hx + dx * s
            ang = np.arctan2(ground - hy, tx - hx)
            nx, ny = -np.sin(ang), np.cos(ang)
            toe = m_poly([(hx + nx * w0 * s, hy + ny * w0 * s), (hx - nx * w0 * s, hy - ny * w0 * s),
                          (tx - nx * 0.7, ground - ny * 0.7), (tx + nx * 0.7, ground + ny * 0.7)])
            foot |= toe
            shade.append((tx, ground))
        allm |= foot
        art.idx[foot] = art.color(P["tal"])
        art.idx[foot & (xx < hx - 0.6 * s) & (yy < hy + 0.5)] = art.color(P["tal_l"])
        art.idx[foot & (xx > hx + 0.8 * s) & (yy < hy + 1.5)] = art.color(P["tal_d"])
    for tx, gy in shade:   # claws
        art.put(int(np.floor(tx)), int(np.floor(gy)), P["ink"])
    rim = ring(allm) & (art.idx == TRANSPARENT)
    art.idx[rim] = k_ink
    # restore the outline ring around anything the clearing opened up
    M = art.idx != TRANSPARENT
    art.idx[ring(M) & (yy > gk.GROUND - 20) & (yy < gk.GROUND)] = k_ink


# ------------------------------------------------------------------- eyes ----
# reference-px eye geometry: centre, radii, which side the outer corner is on
EYES = [((634, 463), (55, 52), -1), ((805, 497), (35, 47), 1)]


def eye_geom(ctx, e):
    (x, y), (rx, ry), side = e
    cx, cy = SP(ctx, x, y)
    return cx, cy, max(2.6, rx * ctx.scale), max(3.2, ry * ctx.scale), side


def brow_cut(stage, cx, cy, rx, ry, side):
    """Pixels of the eye above the stern brow line (stages 2-3): line from the outer corner (higher)
    down to the inner corner."""
    if stage == 1:
        return np.zeros((N, N), bool), None
    drop = {2: 0.30, 3: 0.48}[stage] * ry
    top = cy - ry
    xo, xi = (cx - rx, cx + rx) if side < 0 else (cx + rx, cx - rx)
    yo, yi = top + 0.05 * ry, top + drop + 0.05 * ry
    yy, xx = np.mgrid[0:N, 0:N]
    t = (xx + 0.5 - xo) / (xi - xo)
    yline = yo + (yi - yo) * t
    return (yy + 0.5) < yline, (xo, yo, xi, yi)


def clean_eye_area(art, cx, cy, rx, ry):
    """Old pixelized eye (dark lashes, greyish anti-alias remnants) -> skin, so the hand-drawn eye
    starts from a clean face; saturated feather pixels around it are kept."""
    k = skin(art)
    area = m_ellipse(cx, cy + 0.8, rx + 1.8, ry + 2.6)
    lab = pal_lab(art)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    bad = (lab[:, 0] < 45) | ((chroma < 28) & (lab[:, 0] < 88))
    art.idx[area & np.r_[bad, False][art.idx]] = k
    # under the eye is always face: the old iris often reached lower than the new eye
    yy, xx = np.mgrid[0:N, 0:N] + 0.5
    art.idx[area & (yy > cy + 0.45 * ry) & (np.abs(xx - cx) < 0.85 * rx) & (art.idx != TRANSPARENT)] = k
    # stray remnants of the old iris: non-skin pixels mostly surrounded by skin
    for _ in range(2):
        sk = art.idx == k
        p = np.pad(sk, 1)
        cnt = sum(p[1 + dy:N + 1 + dy, 1 + dx:N + 1 + dx].astype(int)
                  for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
        art.idx[area & ~sk & (art.idx != TRANSPARENT) & (cnt >= 5)] = k


def draw_eye(art, ctx, stage, e, closed=False):
    cx, cy, rx, ry, side = eye_geom(ctx, e)
    k_skin = skin(art)
    E = m_ellipse(cx, cy, rx, ry)
    cut, line = brow_cut(stage, cx, cy, rx, ry, side)
    if closed:
        art.idx[E] = k_skin
        # lower-lid arc: ink pixels following the bottom third of the eye
        yy, xx = np.mgrid[0:N, 0:N]
        u = (xx + 0.5 - cx) / rx
        yarc = cy + 0.35 * ry - 0.25 * ry * u ** 2 + 0.25 * ry * (1 - u ** 2)
        arc = (np.abs(u) <= 0.95) & (np.abs(yy + 0.5 - yarc) <= (0.6 if stage < 3 else 0.9))
        art.idx[arc] = art.color(P["ink"])
        if stage == 1:   # lash flick at the outer corner
            xo = int(np.floor(cx + side * rx))
            yo = int(np.floor(cy + 0.25 * ry))
            art.put(xo, yo, P["ink"])
            art.put(xo + side, yo - 1, P["ink"])
    else:
        glow = stage == 3
        art.idx[E] = art.color(P["sclera"])
        icx, icy = cx + (0.16 if side < 0 else 0.04) * rx, cy + 0.1 * ry
        irx, iry = rx * 0.8, ry * 0.9
        I = m_ellipse(icx, icy, irx, iry) & E
        yy = np.mgrid[0:N, 0:N][0] + 0.5
        paint(art, I, "glow" if glow else "iris")
        paint(art, I & (yy < icy - 0.25 * iry), "glow_d" if glow else "iris_d")
        paint(art, I & (yy > icy + 0.45 * iry), "glow_l" if glow else "iris_l")
        if glow:   # slit pupil: the awakened divine-fire gaze
            pu = m_ellipse(icx + 0.04 * irx, icy + 0.02 * iry, max(1.0, irx * 0.3), iry * 0.66) & E
        else:
            pu = m_ellipse(icx + 0.05 * irx, icy - 0.02 * iry, irx * 0.5, iry * 0.56) & E
        paint(art, pu, "pupil")
        # outline: inner edge of the eye, top two rows heavier (lashes)
        edge = E & ~erode4(E)
        top = E & ~np.roll(E, 2, axis=0)
        paint(art, edge | top, "ink")
        # highlights
        hx = int(np.floor(icx - 0.32 * irx))
        hy = int(np.floor(icy - 0.42 * iry))
        hs = 3 if rx >= 9 else 2
        art.rect(hx, hy, hx + hs - 1, hy + hs - 1, P["white"])
        if hs == 3:
            art.put(hx, hy, P["sclera"] if not E[hy, hx] else P["white"])
        art.put(int(np.floor(icx + 0.38 * irx)), int(np.floor(icy + 0.42 * iry)), P["white"])
        if stage == 1:   # outer-corner lash flick (the stern brow replaces it later)
            xo = int(np.floor(cx + side * (rx - 0.5)))
            yt = int(np.floor(cy - 0.55 * ry))
            art.put(xo + side, yt, P["ink"])
            art.put(xo + 2 * side, yt - 1, P["ink"])
    if line is not None:
        # stern brow: cut the top of the eye, draw the lid line
        art.idx[E & cut] = k_skin
        xo, yo, xi, yi = line
        x0, x1 = sorted((xo, xi))
        for x in range(int(np.floor(x0)) - 1, int(np.floor(x1)) + 2):
            t = (x + 0.5 - xo) / (xi - xo)
            if not -0.12 <= t <= 1.05:
                continue
            y = yo + (yi - yo) * t
            art.put(x, int(np.floor(y)), P["ink"])
            if stage == 3 or t < 0.6:
                art.put(x, int(np.floor(y)) + 1, P["ink"])


# --------------------------------------------------------- gem & lettering ----
def draw_gem(art, ctx, c=(706, 722), hw=34, hh=46):
    """Faceted orange diamond (reference-px centre / half sizes): lit top-left facet, dark
    bottom-right facet, a white glint."""
    cx, cy = SP(ctx, *c)
    cx = np.floor(cx) + 0.5
    hw_s, hh_s = max(3.0, round(hw * ctx.scale)) + 0.5, max(4.0, round(hh * ctx.scale)) + 0.5
    D = m_diamond(cx, cy, hw_s, hh_s)
    yy, xx = np.mgrid[0:N, 0:N] + 0.5
    paint(art, ring(D), "ink")
    paint(art, D & (xx < cx) & (yy < cy), "gem_l")
    paint(art, D & (xx >= cx) & (yy < cy), "gem")
    paint(art, D & (xx < cx) & (yy >= cy), "gem_m")
    paint(art, D & (xx >= cx) & (yy >= cy), "gem_d")
    gx, gy = int(np.floor(cx - hw_s * 0.35)), int(np.floor(cy - hh_s * 0.35))
    art.put(gx, gy, P["gem_hi"])
    if hw_s > 5:
        art.put(gx + 1, gy - 1, P["gem_hi"])
        art.put(gx, gy + 1, P["gem_l"])


FONT = {
    "I": ["#", "#", "#", "#", "#"],
    "I3": ["###", ".#.", ".#.", ".#.", "###"],
    "Q": [".##.", "#..#", "#..#", "#.##", ".###"],
    "X": ["#.#", "#.#", ".#.", "#.#", "#.#"],
}


def draw_iqx(art, ctx, stage):
    """Hand-placed 'IQX' on the scarf, centred on the original lettering."""
    glyphs = ["I3" if stage == 3 else "I", "Q", "X"]
    w = sum(len(FONT[g][0]) for g in glyphs) + len(glyphs) - 1
    cx, cy = SP(ctx, 705, 650)
    x0 = int(round(cx - w / 2))
    blue = is_blue(art)
    # vertical centre of the scarf in the text columns
    cols = [np.where(blue[:, x])[0] for x in range(x0, x0 + w)]
    cols = [c[(c > cy - 12) & (c < cy + 12)] for c in cols]
    tops = [c.min() for c in cols if len(c)]
    bots = [c.max() for c in cols if len(c)]
    yc = (np.median(tops) + np.median(bots)) / 2 if tops else cy
    y0 = int(round(yc - 2.5))
    # clean blue field behind the letters
    box = np.zeros((N, N), bool)
    box[y0 - 1:y0 + 6, x0 - 2:x0 + w + 2] = True
    pal = np.array(art.palette)
    lab = pal_lab(art)
    pale = np.r_[(pal[:, 2] >= pal[:, 0] - 10) & (lab[:, 0] > 55), False][art.idx]
    paint(art, box & (blue | pale), "blue")
    x = x0
    for g in glyphs:
        art.blit(x, y0, FONT[g], {"#": P["blue_hi"]})
        x += len(FONT[g][0]) + 1


# ---------------------------------------------------------- stage 3 regalia ----
def draw_crown(art, ctx):
    """Golden crown on the crest base: three flame spikes, a royal-blue gem in the middle."""
    cx, cy = SP(ctx, 722, 352)
    cx = np.floor(cx) + 0.5
    s = ctx.scale / 0.1848
    W = 13 * s
    yy, xx = np.mgrid[0:N, 0:N] + 0.5
    # band follows the head's curve: ends sit lower
    u = (xx - cx) / W
    btop = cy - 1.2 + 1.6 * u ** 2
    band = (np.abs(u) <= 1) & (yy >= btop) & (yy < btop + 4.2)
    spikes = np.zeros((N, N), bool)
    for fx, h, hw in [(-0.7, 8.0, 3.2), (0.0, 12.5, 4.2), (0.7, 8.0, 3.2)]:
        sx = cx + fx * W
        base = cy - 1.2 + 1.6 * fx ** 2
        r = (base - yy) / h
        spikes |= (r >= 0) & (r <= 1) & (np.abs(xx - sx) <= hw * (1 - r) + 0.3)
    M = band | spikes
    paint(art, ring(M), "ink")
    paint(art, M, "au")
    paint(art, M & (xx < cx - 0.2) & spikes, "au_l")
    paint(art, band & (yy < btop + 1), "au_hi")
    paint(art, band & (yy >= btop + 3.0), "au_d")
    for fx in (-0.7, 0.7):
        sx = cx + fx * W
        paint(art, spikes & ~band & (xx > sx + 0.2), "au_d")
    # centre gem
    G = m_diamond(cx, cy - 4.6, 2.6, 3.8)
    paint(art, ring(G), "ink")
    paint(art, G, "blue")
    paint(art, G & (xx < cx) & (yy < cy - 4.6), "blue_l")
    paint(art, G & (xx > cx) & (yy > cy - 4.6), "blue_d")
    art.put(int(cx - 1), int(cy - 7), P["blue_hi"])


def draw_gorget(art, ctx):
    """Golden pectoral hugging the scarf's lower edge: thick in the middle, tapering into upturned
    horn tips past both ends of the scarf, carrying the gem."""
    blue = is_blue(art)
    xl, _ = SP(ctx, 556, 640)
    xr, _ = SP(ctx, 784, 640)
    xm, _ = SP(ctx, 700, 700)
    s = ctx.scale / 0.1848
    cols = {}
    for x in range(int(np.ceil(xl)), int(np.floor(xr)) + 1):
        ys = np.where(blue[:, x])[0]
        if len(ys):
            cols[x] = ys.max() + 2                  # leave the scarf's own outline row
    if not cols:
        return
    xs = sorted(cols)
    ext = int(round(3 * s))
    for k in range(1, ext + 1):                      # horn tips curl upwards past the scarf
        cols[xs[0] - k] = cols[xs[0]] - k
        cols[xs[-1] + k] = cols[xs[-1]] - k
    M = np.zeros((N, N), bool)
    x0, x1 = min(cols), max(cols)
    for x, top in cols.items():
        u = (x + 0.5 - xm) / ((x1 + 1 - xm) if x + 0.5 > xm else (xm - x0))
        th = max(1.0, (3.2 + 5.0 * (1 - u ** 2)) * s * min(1.0, 1.3 * (1 - abs(u)) + 0.35))
        M[top:int(round(top + th)), x] = True
    yy = np.mgrid[0:N, 0:N][0]
    tops = np.where(M.any(0), M.argmax(0), N)
    bots = np.where(M.any(0), N - 1 - M[::-1].argmax(0), 0)
    paint(art, ring(M) & ~blue, "ink")
    paint(art, M, "au")
    paint(art, M & (yy == tops[None, :]), "au_hi")
    paint(art, M & (yy >= bots[None, :] - 1) & (bots[None, :] - tops[None, :] >= 3), "au_d")
    draw_gem(art, ctx, (704, 742), 40, 54)


def fix_mouth(art, ctx):
    """The open beak's pink lining tends to fall into a brown cluster: repaint it."""
    m = m_poly([SP(ctx, x, y) for x, y in [(694, 548), (752, 548), (744, 580), (712, 584)]])
    lab = pal_lab(art)
    pal = np.array(art.palette)
    brownish = np.r_[(lab[:, 0] < 62) & (pal[:, 0] > pal[:, 2] + 20), False][art.idx]
    k_ink = art.names["outline"]
    tgt = m & brownish & (art.idx != k_ink)
    yy = np.mgrid[0:N, 0:N][0]
    if tgt.any():
        y0 = yy[tgt].min()
        paint(art, tgt & (yy <= y0), "mouth_d")
        paint(art, tgt & (yy > y0), "mouth")


SKY = 9   # the demo keeps the top 8 sprite rows as empty sky: nothing is drawn above this row


def draw_aura(art, top=150):
    """Divine-fire aura: a two-step crimson glow just outside the outline of the upper silhouette,
    with short flame tongues rising from the local peaks (wing tips, crest)."""
    M = art.idx != TRANSPARENT
    yy = np.mgrid[0:N, 0:N][0]
    a1 = ring(M) & (yy < top) & (yy >= SKY)
    a2 = ring(M | a1) & (yy < top - 12) & (yy >= SKY)
    # tongues: above each local peak of the silhouette top
    tops = np.array([np.argmax(M[:, x]) if M[:, x].any() else N for x in range(N)])
    tongues = np.zeros((N, N), bool)
    for x in range(3, N - 3):
        t = tops[x]
        if t < top - 20 and t == tops[x - 3:x + 4].min() and t <= tops[x - 1] and t < tops[x + 1] + 1:
            h = 4 if t > 20 else 3
            for k in range(1, h + 1):
                y = t - 1 - k
                hw = int((h - k) // 2)
                if y >= SKY:
                    tongues[y, x - hw:x + hw + 1] = True
    tongues &= ~M
    art.idx[(a2 | tongues) & ~M & (art.idx == TRANSPARENT)] = art.color("#5c0c1e")
    art.idx[a1 & (art.idx == TRANSPARENT)] = art.color("#a01828")


def sparkle(art, x, y, size=1):
    """Ember / energy sparkle on transparent background only."""
    pts = [(0, 0, "spark")] + [(dx, dy, "flame") for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
    if size > 1:
        pts += [(dx, dy, "red") for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2))]
        pts[0] = (0, 0, "white")
    for dx, dy, c in pts:
        if 0 <= x + dx < N and 0 <= y + dy < N and art.idx[y + dy, x + dx] == TRANSPARENT:
            art.put(x + dx, y + dy, P[c])


SPARKS = {
    1: [],
    2: [(40, 70, 1), (150, 64, 1), (176, 96, 1), (16, 118, 1)],
    3: [(8, 30, 2), (184, 24, 1), (100, 12, 1), (176, 152, 2), (6, 100, 1), (64, 14, 1)],
}


# ---------------------------------------------------------------- touchup ----
def touchup(stage, art, ctx):
    despeckle(art, ctx)
    draw_feet(art, ctx)
    for e in EYES:
        clean_eye_area(art, *eye_geom(ctx, e)[:4])
    for e in EYES:
        draw_eye(art, ctx, stage, e)
    draw_iqx(art, ctx, stage)
    fix_mouth(art, ctx)
    if stage == 3:
        draw_gorget(art, ctx)
        draw_crown(art, ctx)
        draw_aura(art)
    else:
        draw_gem(art, ctx)
    for x, y, size in SPARKS[stage]:
        sparkle(art, x, y, size)


def blink(stage, art, ctx):
    for e in EYES:
        draw_eye(art, ctx, stage, e, closed=True)
