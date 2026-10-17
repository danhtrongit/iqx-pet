"""Kim Quy: evolution line for the game sprites (see tools/gamekit.py).

Stage 1 is the original baby turtle. Stage 2 (Ngọc Quy) and stage 3 (Kim Quy Thần Kiếm) are rebuilt
at illustration level: a procedurally shaded dome shell with hexagon plates replaces the original
shell, the sprout grows into a sapling and then into a golden leaf crest with a bonsai rooted on the
shell, and stage 3 carries the Thuận Thiên sword. Small details (eyes, gem, IQX lettering, trims,
aura, sparks) are placed pixel by pixel in touchup().
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
    "name": "Kim Quy",
    "layer": "Định giá",
    "element": "value_jade",
    "accent": "#34d399",
    "tagline": "Chậm mà chắc, đo đúng giá trị thật của doanh nghiệp.",
}

INK = "#0a1430"
CFG = {"colors": 28, "ink_l": 24.0, "ink_frac": 0.45, "smooth": 1, "min_island": 4, "outline": INK}

STAGES = [
    {"rank": "Ấu thể", "name": "Tiểu Quy", "level": 1, "height": 104,
     "config": dict(CFG, colors=24),
     "desc": "", "skill": {"name": "", "desc": ""},
     "stats": {"hp": 46, "atk": 24, "def": 44, "spd": 18}},
    {"rank": "Trưởng thành", "name": "Ngọc Quy", "level": 10, "height": 130,
     "config": dict(CFG, colors=30),
     "desc": "", "skill": {"name": "", "desc": ""},
     "stats": {"hp": 72, "atk": 48, "def": 74, "spd": 30}},
    {"rank": "Thần thú", "name": "Kim Quy Thần Kiếm", "level": 25, "height": 150, "dx": -10,
     "config": dict(CFG, colors=34),
     "desc": "", "skill": {"name": "", "desc": ""},
     "stats": {"hp": 98, "atk": 80, "def": 96, "spd": 42}},
]

# Fixed detail colours (hand-placed pixels).
P = {
    "ink": INK, "white": "#ffffff", "sclera": "#f6fbff",
    "iris_d": "#0f5a2e", "iris": "#22a24a", "iris_l": "#6fd86a", "pupil": "#04140c",
    "jglow_d": "#0d7a5c", "jglow": "#34d399", "jglow_l": "#a8f7d6", "jglow_hi": "#eafff6",
    "gem_hi": "#ffffff", "gem_l": "#a6ecff", "gem": "#3cc4f4", "gem_m": "#1d8fe0", "gem_d": "#0b5cb4",
    "blue_dd": "#0a2a7a", "blue_d": "#0d3fa6", "blue": "#1456c8", "blue_l": "#3d80e8", "letter": "#f2f8ff",
    "au_dd": "#7a420c", "au_d": "#bd7a16", "au": "#eeb12c", "au_l": "#ffd95c", "au_hi": "#fff4c4",
    "jade_dd": "#06372c", "jade_d": "#0d6650", "jade": "#169a74", "jade_l": "#3fcf9a", "jade_hi": "#a6f2cf",
    "blush": "#f7a6a0", "tongue": "#f06a5c", "mouth": "#8e1f2e",
}

# ------------------------------------------------------- art-level helpers ----
def R(x, y):
    """Reference poster px -> art canvas px."""
    return x + gk.PAD, y + gk.PAD


def RP(pts):
    return [R(*p) for p in pts]


def poly_mask(pts):
    """Boolean art-canvas mask of a polygon given in reference px."""
    m = Image.new("L", (gk.ART, gk.ART), 0)
    ImageDraw.Draw(m).polygon([tuple(map(float, R(*q))) for q in pts], fill=255)
    return np.array(m) > 127


def opaque(layer):
    return np.array(layer)[..., 3] > 0


def grow(m, n):
    for _ in range(n):
        m = grow4(m)
    return m


_PARTS = {}


def parts():
    """Masks of the original's parts on the art canvas (computed once): flood fills of the areas
    enclosed by the ink lines, plus the ink lines that belong to them."""
    if _PARTS:
        return _PARTS
    a = np.array(gk.art_canvas(gk.PETS[3] if False else "kim-quy"))
    alpha = a[..., 3]
    op = alpha > 128
    L = np.zeros(op.shape)
    L[op] = rgb2lab(a[op][:, :3])[:, 0]
    rgbi = a[..., :3].astype(int)
    blue = (rgbi[..., 2] > rgbi[..., 0] + 70) & (rgbi[..., 2] > rgbi[..., 1] + 50)
    ink_bar = op & (L < 30)
    ink = ink_bar & ~blue
    bar = Image.fromarray(np.where(ink_bar | ~op, 255, 0).astype(np.uint8))

    def flood(*seeds):
        out = np.zeros(op.shape, bool)
        for sx, sy in seeds:
            b = bar.copy()
            ImageDraw.floodfill(b, R(sx, sy), 128)
            out |= np.array(b) == 128
        return out

    shell = flood((720, 560), (690, 420), (760, 640))
    leaves = flood((385, 262), (465, 215))
    head = flood((400, 420))
    interior = op & ~ink
    yy, xx = np.mgrid[0:gk.ART, 0:gk.ART]
    tail = flood((825, 725))
    leaf_r = flood((615, 280))
    known_fg = head | leaves | flood((270, 820), (450, 820), (610, 820), (750, 860))
    teal = dk.hsv_mask(Image.fromarray(a, "RGBA"), h=(138, 190), s=(0.35, 1))
    shell_zone = (xx > R(585, 0)[0]) & (yy < R(0, 814)[1]) & (yy > R(0, 330)[1])
    other = (shell | tail | leaf_r | (teal & shell_zone & ~known_fg)
             | (shell_zone & ~known_fg & ~blue & (xx > R(640, 0)[0]))) & interior
    fg_int = interior & ~other
    # split the ink lines between the foreground and the parts that get replaced (Voronoi growth)
    fg, ot = fg_int.copy(), other.copy()
    free = ink | ((alpha > 0) & ~op)
    for _ in range(24):
        gf = grow4(fg) & free & ~fg & ~ot
        go = grow4(ot) & free & ~fg & ~ot & ~gf
        fg |= gf
        ot |= go
    tail_all = (tail | (ink & grow(tail, 12))) & ~fg
    leaves_all = grow(leaves, 10) & (alpha > 0) & ~grow(head, 2)
    bump = head & (L > 75) & (a[..., 0].astype(int) > a[..., 2].astype(int) + 60) \
        & (yy < R(0, 352)[1]) & (xx > R(410, 0)[0]) & (xx < R(545, 0)[0])
    _PARTS.update(shell=shell, fg=fg, tail=tail_all, leaves=leaves_all, head=head, bump=bump,
                  leg_l=flood((270, 820)), leg_r=flood((610, 820)) & (yy > R(0, 760)[1]) & (xx > R(528, 0)[0]),
                  alpha=alpha > 0, ink=ink)
    return _PARTS


def erase(img, m):
    a = np.array(img)
    a[m, 3] = 0
    img.paste(Image.fromarray(a, "RGBA"))


def keep_only(img, m):
    a = np.array(img)
    a[~m, 3] = 0
    return Image.fromarray(a, "RGBA")


def affine_paste(layer, src, a_src, a_dst, A):
    """Composite src onto layer through the 2x2 matrix A (art px), mapping point a_src to a_dst
    (both reference px)."""
    Ai = np.linalg.inv(np.asarray(A, np.float64))
    off = np.array(R(*a_src)) - Ai @ np.array(R(*a_dst))
    im = src.convert("RGBa")
    out = im.transform(layer.size, Image.AFFINE, (Ai[0, 0], Ai[0, 1], off[0], Ai[1, 0], Ai[1, 1], off[1]),
                       resample=Image.BICUBIC).convert("RGBA")
    layer.alpha_composite(out)


def rot(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


# The original middle leaf: stem end and tip (reference px).
LEAF_BASE, LEAF_TIP = (503, 292), (414, 146)
LEAF_POLY = [(398, 150), (414, 132), (452, 150), (500, 190), (532, 240), (532, 300), (500, 302), (462, 278),
             (425, 238), (400, 196)]


def leaf_src(src):
    return keep_only(src, poly_mask(LEAF_POLY) & ~grow(parts()["fg"] & ~parts()["leaves"], 0))


def paste_leaf(layer, leaf, base, angle, scale=1.0, flip=False):
    """Paste the leaf with its stem at `base` (reference px), pointing `angle` degrees clockwise from
    straight up."""
    d0 = np.subtract(LEAF_TIP, LEAF_BASE)
    a0 = np.degrees(np.arctan2(d0[0], -d0[1]))           # clockwise from up
    F = np.eye(2)
    if flip:   # mirror across the leaf axis
        u = d0 / np.hypot(*d0)
        F = 2 * np.outer(u, u) - np.eye(2)
    A = rot(angle - a0) @ (np.eye(2) * scale) @ F
    affine_paste(layer, leaf, LEAF_BASE, base, A)


LEAF_COLS = ["#3a8a1c", "#68bb2a", "#a8e05a"]


def draw_leaf(img, base, angle, length, width=None, cols=LEAF_COLS, ow=9, flip=False):
    """Almond leaf (reference px) from its stem `base`, pointing `angle` degrees clockwise from up,
    ink outline, a lit half and a dark midrib."""
    width = width or length * 0.46
    t = np.radians(angle)
    d = np.array([np.sin(t), -np.cos(t)])
    n = np.array([-d[1], d[0]])
    b = np.array(R(*base), float)
    tip = b + d * length
    s_ = 1 if not flip else -1
    side1 = dk.bezier([tuple(b), tuple(b + d * length * 0.35 + n * width * 0.62),
                       tuple(b + d * length * 0.78 + n * width * 0.42), tuple(tip)], 24)
    side2 = dk.bezier([tuple(tip), tuple(b + d * length * 0.78 - n * width * 0.42),
                       tuple(b + d * length * 0.35 - n * width * 0.62), tuple(b)], 24)
    poly = side1 + side2[1:]
    dk.stroke(img, poly + [poly[0]], INK, ow * 2)
    dk.shape(img, poly, cols[1])
    lit = side1 if s_ > 0 else side2[::-1]
    mid = dk.bezier([tuple(b + d * 6), tuple(b + d * length * 0.5 + n * width * 0.04 * s_), tuple(tip - d * 6)], 16)
    dk.shape(img, lit + mid[::-1][1:-1], cols[2])
    dk.stroke(img, mid[:-3], cols[0], max(5, ow * 0.7))
    # short stem
    dk.stroke(img, [tuple(b - d * 18), tuple(b + d * 4)], INK, ow * 1.6)
    dk.stroke(img, [tuple(b - d * 16), tuple(b + d * 4)], cols[0], ow * 0.6)


def spike(img, base, tip, w, body, light, outline_w=10, tip_cols=None, tip_frac=0.38):
    """A cone spike (reference px) with an ink outline, lit from the left; optional coloured tip."""
    b = np.array(R(*base), float)
    t = np.array(R(*tip), float)
    d = t - b
    n = np.array([-d[1], d[0]]) / np.hypot(*d) * w / 2
    tri = [tuple(b + n), tuple(t), tuple(b - n)]
    if outline_w:
        dk.stroke(img, tri + [tri[0]], INK, outline_w * 2)
    dk.shape(img, tri, body)
    lit_side = n if n[0] < 0 else -n
    dk.shape(img, [tuple(b + lit_side), tuple(t), tuple(b + lit_side * 0.1 + d * 0.05)], light)
    if tip_cols:
        m = b + d * (1 - tip_frac)
        dk.shape(img, [tuple(m + n * tip_frac), tuple(t), tuple(m - n * tip_frac)], tip_cols[0])
        dk.shape(img, [tuple(m + lit_side * tip_frac), tuple(t), tuple(m + lit_side * 0.05)], tip_cols[1])


def dome_eval(xr, yr, sp):
    """Shell model at reference-px points: plate id, tone index, rim flag and shading terms.

    sp: c, rad (ellipsoid cap), hexr (central plate size in radians of the dome), apex (u, v of the
    central plate), ring2 (outer edge of the first plate ring, in hexr), rim_w, light, bevel, tones, rim."""
    cx, cy = sp["c"]
    rx, ry = sp["rad"]
    u = (xr - cx) / rx
    v = (yr - cy) / ry
    rr2 = np.clip(u * u + v * v, 0, 0.9999)
    z = np.sqrt(1 - rr2)
    Lv = np.array(sp.get("light", (-0.55, -0.7, 0.45)), float)
    Lv /= np.linalg.norm(Lv)
    lam = u * Lv[0] + v * Lv[1] + z * Lv[2]
    A = np.arcsin(np.clip(u, -0.9999, 0.9999))
    B = np.arcsin(np.clip(v, -0.9999, 0.9999))
    a0, b0 = np.arcsin(sp["apex"][0]), np.arcsin(sp["apex"][1])
    hexr, r2 = sp["hexr"], sp["ring2"]
    x, y = (A - a0) / hexr, (B - b0) / hexr
    dh = np.maximum(np.abs(y) * 2 / np.sqrt(3), np.abs(y) / np.sqrt(3) + np.abs(x))   # flat-top hex norm
    ang = np.degrees(np.arctan2(y, x)) % 360
    s6 = np.floor(ang / 60).astype(np.int64)
    s12 = np.floor(((ang + 15) % 360) / 30).astype(np.int64)
    cell = np.where(dh < 1, 0, np.where(dh < r2, 1 + s6, 7 + s12))
    cang = np.where(dh < r2, (s6 + 0.5) * 60, s12 * 30)
    crad = np.where(dh < 1, 0, np.where(dh < r2, (1 + r2) / 2, r2 + 0.75))
    half = np.where(dh < 1, 1.0, np.where(dh < r2, (r2 - 1) / 2 + 0.3, 0.85))
    ccx, ccy = crad * np.cos(np.radians(cang)), crad * np.sin(np.radians(cang))
    la, lb = (x - ccx) / half, (y - ccy) / half
    rr = np.sqrt(u * u + v * v)
    rim_w = sp.get("rim_w", 0.15)
    rim_m = (rr > 1 - rim_w) & (v > -0.3) & (u > -0.3) if sp.get("rim") else np.zeros(u.shape, bool)
    seg = np.floor((np.degrees(np.arctan2(v, u)) + 360) / 24).astype(np.int64)
    cell = np.where(rim_m, -1 - seg, cell)
    across = (rr - (1 - rim_w)) / rim_w - 0.5
    la = np.where(rim_m, 0, la)
    lb = np.where(rim_m, across * 1.6, lb)
    bev = -(la * Lv[0] + lb * Lv[1]) / np.hypot(Lv[0], Lv[1])
    val = 0.62 * lam + sp.get("bevel", 0.32) * bev
    nt = len(sp["tones"])
    th = sp.get("thresholds", [-0.08, 0.28, 0.66, 0.9][:nt - 1])
    k = np.digitize(val, th)
    nr = len(sp["rim"]) if sp.get("rim") else 1
    rv = 0.55 * lam - 0.45 * across
    kr = np.digitize(rv, [-0.45, -0.05, 0.35][:nr - 1])
    return dict(cell=cell, k=k, kr=kr, rim=rim_m, lam=lam, dh=dh, u=u, v=v, rr=rr)


def dome_layer(clip, sp, seam_w=4):
    """Art-canvas RGBA layer of the shell inside `clip` (gives the silhouette and the colours that the
    pixelizer clusters; the sprite's plates are re-rendered crisply in touchup)."""
    ys, xs = np.where(clip)
    d = dome_eval(xs - gk.PAD, ys - gk.PAD, sp)
    cols = np.array([dk.hexrgba(t)[:3] for t in sp["tones"]])
    rgb = cols[d["k"]]
    if sp.get("rim"):
        rc = np.array([dk.hexrgba(t)[:3] for t in sp["rim"]])
        rgb = np.where(d["rim"][:, None], rc[d["kr"]], rgb)
    out = np.zeros((gk.ART, gk.ART, 4), np.uint8)
    out[ys, xs, :3] = rgb
    out[ys, xs, 3] = 255
    idmap = np.full(clip.shape, -10 ** 9, np.int64)
    idmap[ys, xs] = d["cell"]
    sm = np.zeros(clip.shape, bool)
    for dy, dx in ((0, seam_w), (seam_w, 0)):
        sh = np.roll(idmap, (dy, dx), axis=(0, 1))
        sm |= (sh != idmap) & (sh > -10 ** 9)
    out[sm & clip, :3] = dk.hexrgba(sp["seam"][0])[:3]
    return Image.fromarray(out, "RGBA")


def on_ellipse(c, rad, t, inset=0.0, out=0.0):
    """Point on the ellipse at parameter t (degrees, 0 = right, 90 = down) moved `inset` px inwards,
    plus the point `out` px further along the outward normal (reference px)."""
    tr = np.radians(t)
    x, y = c[0] + rad[0] * np.cos(tr), c[1] + rad[1] * np.sin(tr)
    n = np.array([np.cos(tr) / rad[0], np.sin(tr) / rad[1]])
    n /= np.hypot(*n)
    return (x - n[0] * inset, y - n[1] * inset), (x + n[0] * out, y + n[1] * out)


def ellipse_mask(c, rad, ybot=None):
    cx, cy = R(*c)
    yy, xx = np.ogrid[0:gk.ART, 0:gk.ART]
    m = ((xx - cx) / rad[0]) ** 2 + ((yy - cy) / rad[1]) ** 2 <= 1
    if ybot is not None:
        m &= yy < R(0, ybot)[1]
    return m


def outline_of(m, w):
    """Ink ring `w` px wide just outside mask m."""
    return grow(m, w) & ~m


def rgba_fill(m, col):
    out = np.zeros((gk.ART, gk.ART, 4), np.uint8)
    out[m] = dk.hexrgba(col)
    return Image.fromarray(out, "RGBA")


# ------------------------------------------------------------- compose ----
JADE_TONES = ["#14634d", "#1b8264", "#26a47a", "#3fc794", "#8eebc4"]
GOLD_RIM = ["#8a5410", "#c98a1c", "#f0b834", "#ffe07a"]
GOLD_TONES = ["#a8650f", "#d6961f", "#f5c242", "#ffe596"]
JADE_RIM = ["#0e5c47", "#168467", "#27ad85", "#66dcb0"]
SHELL = {
    3: dict(c=(716, 562), rad=(170, 276), ybot=816, hexr=0.4, apex=(0.2, -0.26), ring2=2.15, rim_w=0.15,
            bevel=0.36, tones=GOLD_TONES, rim=JADE_RIM, seam=("#2fd096", "#14906a", "#b4fbe0"),
            thresholds=[-0.1, 0.24, 0.62]),
    2: dict(c=(706, 576), rad=(146, 252), ybot=812, hexr=0.42, apex=(0.2, -0.24), ring2=2.15, rim_w=0.15,
            bevel=0.34, tones=JADE_TONES[1:], rim=GOLD_RIM, seam=("#e8b030", "#a8701a", "#ffd95c")),
}
BARK = ["#5e3c1e", "#94653a", "#c4955c"]
SKIN = {2: ["#0b3a2a", "#147a52", "#2aa86e", "#5fd08e", "#b6efb4", "#effff0"],
        3: ["#0a3d34", "#127a62", "#22a585", "#4fd0a6", "#a8f0d0", "#ecfff6"]}
_FRONT = {}


def reskin(body, stage):
    pm = parts()
    m = dk.hsv_mask(body, h=(60, 170), s=(0.3, 1)) & pm["fg"] & ~pm["bump"] & ~pm["leaves"]
    dk.gradient_map(body, m, SKIN[stage])


def trunk(img, pts, w, cols=BARK, ow=10):
    """Woody stroke with outline (reference px), light stripe on the left, shade on the right."""
    P_ = [R(*p) for p in dk.bezier(pts, 40)]
    dk.stroke(img, P_, INK, w + 2 * ow)
    dk.stroke(img, P_, cols[1], w)
    dk.stroke(img, [(x - w * 0.2, y) for x, y in P_], cols[2], w * 0.4)
    dk.stroke(img, [(x + w * 0.3, y) for x, y in P_], cols[0], w * 0.3)


def shell_parts(img, back, stage, spikes=()):
    """Draw the stage's dome shell on `back`: spikes along its ridge are part of the silhouette (one ink
    outline around shell + spikes); returns the dome mask."""
    sp = SHELL[stage]
    m = ellipse_mask(sp["c"], sp["rad"], sp["ybot"])
    sl = dk.blank(img)
    for t, ln, w, body, light, tip in spikes:
        base, tipp = on_ellipse(sp["c"], sp["rad"], t, 60, ln)
        spike(sl, base, tipp, w, body, light, outline_w=0, tip_cols=tip)
    sil = m | opaque(sl)
    back.alpha_composite(rgba_fill(outline_of(sil, 11), INK))
    back.alpha_composite(sl)
    back.alpha_composite(dome_layer(m, sp))
    back.alpha_composite(rgba_fill(outline_of(parts()["fg"], 7) & sil, INK))
    return m


def stage2(img, ctx):
    pm = parts()
    src = img.copy()
    body = img.copy()
    erase(body, ~pm["fg"] | pm["leaves"])
    reskin(body, 2)
    back = dk.blank(img)
    # longer tail, swept up behind the shell
    dk.gradient_map(src, dk.hsv_mask(src, h=(60, 170), s=(0.3, 1)) & pm["tail"], SKIN[2])
    affine_paste(back, keep_only(src, pm["tail"]), (790, 760), (812, 752), rot(-14) * 1.35)
    # sapling: a forked trunk rising from the head, leafy branches spreading like a V crest
    trunk(back, [(478, 340), (484, 262), (474, 204)], 54)
    trunk(back, [(468, 214), (424, 162), (392, 104)], 38)
    trunk(back, [(480, 214), (526, 164), (556, 106)], 38)
    for base, ang, ln, fl in [((394, 110), -18, 164, False), ((554, 112), 20, 164, True),
                              ((418, 156), -92, 132, False), ((532, 158), 94, 132, True),
                              ((448, 306), -88, 120, False), ((510, 302), 76, 108, True)]:
        draw_leaf(back, base, ang, ln, flip=fl)
    jade_spike = ("#26a47a", "#5fdcaa", ("#d8961e", "#ffd95c"))
    m = shell_parts(img, back, 2, [(-104, 126, 100, *jade_spike), (-66, 146, 108, *jade_spike),
                                   (-28, 134, 100, *jade_spike)])
    _FRONT[2] = np.zeros(m.shape, bool)
    return dk.behind(body, back)


def pad(img, c, rx, ry, cols, bumps=5, ow=10):
    """Bonsai foliage pad (reference px): a cloud of overlapping discs with a flat-ish bottom, ink
    outline, dark underside, light bumps on top."""
    cx, cy = R(*c)
    m = Image.new("L", img.size, 0)
    d = ImageDraw.Draw(m)
    discs = []
    for i in range(bumps):
        t = i / (bumps - 1)
        x = cx - rx + 2 * rx * t
        r = ry * (0.62 + 0.38 * np.sin(np.pi * t))
        y = cy + (ry - r) * 0.7
        discs.append((x, y, r))
    for x, y, r in discs:
        d.ellipse([x - r, y - r, x + r, y + r], fill=255)
    d.ellipse([cx - rx * 0.94, cy - ry * 0.2, cx + rx * 0.94, cy + ry * 0.62], fill=255)
    M = np.array(m) > 127
    img.alpha_composite(rgba_fill(outline_of(M, ow), INK))
    img.alpha_composite(rgba_fill(M, cols[1]))
    under = M & ~np.roll(M, -int(ry * 0.32), axis=0)
    img.alpha_composite(rgba_fill(under, cols[0]))
    lm = Image.new("L", img.size, 0)
    dl = ImageDraw.Draw(lm)
    for x, y, r in discs:
        dl.ellipse([x - r * 0.7 - r * 0.22, y - r * 0.7 - r * 0.25, x + r * 0.7 - r * 0.22, y + r * 0.7 - r * 0.25],
                   fill=255)
    lit = (np.array(lm) > 127) & M & ~under
    img.alpha_composite(rgba_fill(lit, cols[2]))
    hm = Image.new("L", img.size, 0)
    dh_ = ImageDraw.Draw(hm)
    for x, y, r in discs[1:-1]:
        dh_.ellipse([x - r * 0.3 - r * 0.4, y - r * 0.25 - r * 0.5, x + r * 0.3 - r * 0.4, y + r * 0.25 - r * 0.5],
                    fill=255)
    img.alpha_composite(rgba_fill((np.array(hm) > 127) & lit, cols[3]))
    return M


def coin(img, c, r, ow=9):
    """Ancient gold coin with a square hole (reference px)."""
    x, y = R(*c)
    dk.ellipse(img, x, y, r, r, "#f2b833", outline=INK, width=ow)
    dk.ellipse(img, x - r * 0.12, y - r * 0.12, r * 0.78, r * 0.78, "#ffe07a")
    h = r * 0.3
    dk.shape(img, [(x - h, y - h), (x + h, y - h), (x + h, y + h), (x - h, y + h)], INK)


def sword(img, tip, guard, hilt_len=96, w=44, ow=10):
    """Thuận Thiên: a straight jade-glowing blade, wide gold guard, jade-wrapped hilt, gold pommel."""
    t = np.array(R(*tip), float)
    g = np.array(R(*guard), float)
    d = (g - t) / np.hypot(*(g - t))
    n = np.array([-d[1], d[0]])
    if n[1] > 0:
        n = -n                                    # n points up (lit edge)
    hw = w / 2
    blade = [tuple(t + d * 10), tuple(g + n * hw), tuple(g - n * hw)]
    blade = [tuple(t), tuple(t + d * 60 + n * hw), tuple(g + n * hw), tuple(g - n * hw), tuple(t + d * 60 - n * hw)]
    dk.stroke(img, blade + [blade[0]], INK, ow * 2)
    dk.shape(img, blade, "#9fcfc8")
    dk.shape(img, [tuple(t), tuple(t + d * 60 + n * hw), tuple(g + n * hw), tuple(g + n * hw * 0.1),
                   tuple(t + d * 60 + n * hw * 0.1)], "#e9fff8")
    dk.stroke(img, [tuple(t + d * 70 - n * hw * 0.35), tuple(g - d * 8 - n * hw * 0.35)], "#34d399", w * 0.22)
    # hilt + pommel
    h0, h1 = g, g + d * hilt_len
    dk.stroke(img, [tuple(h0), tuple(h1)], INK, w * 0.62 + ow * 2)
    dk.stroke(img, [tuple(h0), tuple(h1)], "#147a5e", w * 0.62)
    dk.stroke(img, [tuple(h0 + n * w * 0.12), tuple(h1 + n * w * 0.12)], "#2fb889", w * 0.22)
    for k in (0.33, 0.66):
        q = h0 + d * hilt_len * k
        dk.stroke(img, [tuple(q - n * w * 0.31), tuple(q + n * w * 0.31)], "#f2b833", 9)
    pm = h1 + d * 20
    dk.ellipse(img, pm[0], pm[1], 26, 26, "#f2b833", outline=INK, width=ow)
    dk.ellipse(img, pm[0] - 5, pm[1] - 6, 13, 13, "#34d399")
    # guard: a wide gold cross-bar with curled ends
    gl, gr = g + n * w * 1.55 + d * 4, g - n * w * 1.55 + d * 4
    dk.stroke(img, [tuple(gl), tuple(gr)], INK, 30 + ow * 2)
    dk.stroke(img, [tuple(gl), tuple(gr)], "#e3a52a", 30)
    dk.stroke(img, [tuple(gl - d * 6), tuple(gr - d * 6)], "#ffe07a", 11)
    dk.ellipse(img, g[0] + d[0] * 4, g[1] + d[1] * 4, 15, 15, "#34d399", outline=INK, width=6)


GOLD_LEAF = ["#b8740f", "#efb52e", "#fff0a6"]
PAD_COLS = ["#0f6b52", "#1f9e76", "#3fcf98", "#a6f2cf"]


def band(img, leg, yc, h, curve, cols, xr=None, ow=9):
    """Gold bracer across a leg (art mask): rows yc..yc+h (reference px), bent down by `curve` at
    the sides, ink lines along its edges, clipped to the leg."""
    yy, xx = np.mgrid[0:gk.ART, 0:gk.ART]
    xs = np.where(leg.any(0))[0]
    x0, x1 = (xs.min(), xs.max()) if xr is None else (R(xr[0], 0)[0], R(xr[1], 0)[0])
    t = (xx - (x0 + x1) / 2) / ((x1 - x0) / 2)
    top = R(0, yc)[1] + curve * t ** 2
    M = leg & (yy >= top) & (yy < top + h)
    edge = leg & (((yy >= top - ow) & (yy < top)) | ((yy >= top + h) & (yy < top + h + ow)))
    img.alpha_composite(rgba_fill(edge, INK))
    img.alpha_composite(rgba_fill(M, cols[1]))
    img.alpha_composite(rgba_fill(M & (yy < top + h * 0.38), cols[2]))
    img.alpha_composite(rgba_fill(M & (yy >= top + h * 0.75), cols[0]))
    return M


def stage3(img, ctx):
    pm = parts()
    src = img.copy()
    body = img.copy()
    erase(body, ~pm["fg"] | pm["leaves"])
    reskin(body, 3)
    # the sprout's node turns into a jade orb
    dk.gradient_map(body, pm["bump"], ["#0b6a50", "#22b585", "#7ff0c4", "#e8fff6"])
    back = dk.blank(img)
    sp = SHELL[3]
    # long tail swept up behind the shell
    tail_src = src.copy()
    dk.gradient_map(tail_src, dk.hsv_mask(tail_src, h=(60, 170), s=(0.3, 1)) & pm["tail"], SKIN[3])
    affine_paste(back, keep_only(tail_src, pm["tail"]), (790, 760), (830, 752), rot(-18) * 1.6)
    gold_spike = ("#e3a52a", "#ffe07a", ("#20b585", "#8ff2cc"))
    m = shell_parts(img, back, 3, [(-106, 132, 104, *gold_spike), (-60, 158, 118, *gold_spike),
                                   (-22, 158, 116, *gold_spike), (14, 124, 98, *gold_spike)])
    over = dk.blank(img)
    # bonsai rooted on the shell: twisted trunk, jade foliage pads, gold coins
    trunk(over, [(770, 336), (818, 250), (740, 190), (790, 60)], 64)
    trunk(over, [(756, 204), (700, 176), (650, 160)], 40)
    trunk(over, [(800, 132), (870, 132), (930, 144)], 36)
    for c, rx, ry in [((634, 150), 86, 58), ((946, 146), 80, 56), ((792, 20), 120, 74)]:
        pad(over, c, rx, ry, PAD_COLS)
    for c, r in [((604, 184), 27), ((720, 60), 26), ((864, 52), 27), ((976, 184), 25)]:
        coin(over, c, r)
    # Thuận Thiên, strapped along the shell
    sword(over, (584, 752), (868, 612), hilt_len=96, w=54)
    _FRONT[3] = opaque(over)
    back.alpha_composite(over)
    # golden leaf crest rising from behind the jade orb
    for base, ang, ln, fl in [((452, 304), -42, 200, False), ((506, 300), 24, 214, True),
                              ((478, 300), -8, 168, False), ((440, 308), -82, 140, False)]:
        draw_leaf(back, base, ang, ln, cols=GOLD_LEAF, flip=fl)
    out = dk.behind(body, back)
    # gold bracers on the front legs
    for leg, yc, xr in [(pm["leg_l"], 806, None), (pm["leg_r"], 812, (536, 700))]:
        band(out, leg, yc, 40, 22, ["#b8740f", "#efb52e", "#ffe48a"], xr=xr)
    return out


def compose(stage, img, ctx):
    if stage == 2:
        return stage2(img, ctx)
    if stage == 3:
        return stage3(img, ctx)
    return img


# ---------------------------------------------------- sprite-level helpers ----
N = gk.CANVAS
YY, XX = np.mgrid[0:N, 0:N]
YC, XC = YY + 0.5, XX + 0.5


def SP(ctx, x, y):
    """Reference px -> sprite coordinates (float; a pixel's centre is at +0.5)."""
    ax, ay = R(x, y)
    return ax * ctx.scale + ctx.ox, ay * ctx.scale + ctx.oy


def m_ellipse(cx, cy, rx, ry):
    return ((XC - cx) / rx) ** 2 + ((YC - cy) / ry) ** 2 <= 1.0


def m_poly(pts):
    """Sprite mask of a polygon in sprite coordinates."""
    m = Image.new("L", (N, N), 0)
    ImageDraw.Draw(m).polygon([(x - 0.5, y - 0.5) for x, y in pts], fill=1)
    return np.array(m).astype(bool)


def erode4(m):
    p = np.pad(m, 1)
    return m & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]


def ring(m):
    return grow4(m) & ~m


def paint(art, m, col):
    art.idx[m] = art.color(P.get(col, col))


def pal_lab(art):
    return rgb2lab(np.array(art.palette, np.float64))


def lut(art, good):
    return np.r_[np.asarray(good, bool), False][art.idx]      # idx -1 -> last entry (False)


def is_blue(art):
    """Royal-blue scarf tones (not the light cyan of the gem)."""
    pal = np.array(art.palette)
    lab = pal_lab(art)
    return lut(art, (pal[:, 2] > pal[:, 0] + 60) & (pal[:, 2] > pal[:, 1] + 40) & (pal[:, 2] > 110)
               & (lab[:, 0] < 58))


def ring_colour(art, m, exclude=()):
    """Most common colour just outside mask m (ignoring outline, transparent and `exclude`)."""
    r = ring(grow4(m)) | ring(m)
    v = art.idx[r]
    bad = {TRANSPARENT, art.names["outline"], *exclude}
    v = v[~np.isin(v, list(bad))]
    if not len(v):
        return art.closest((250, 244, 210))
    vals, cnt = np.unique(v, return_counts=True)
    return int(vals[cnt.argmax()])


# ------------------------------------------------------------------- eyes ----
# reference-px eye geometry: centre, radii, which side the outer corner (white crescent) is on
EYES = [((236, 484), (34, 56), -1), ((472, 490), (61, 58), 1)]


def eye_geom(ctx, e):
    (x, y), (rx, ry), side = e
    cx, cy = SP(ctx, x, y)
    return cx, cy, max(2.6, rx * ctx.scale), max(3.6, ry * ctx.scale), side


def eye_skin(art, ctx, e):
    cx, cy, rx, ry, _ = eye_geom(ctx, e)
    E = m_ellipse(cx, cy, rx + 1.2, ry + 1.2)
    lab = pal_lab(art)
    dark = np.where(lab[:, 0] < 50)[0].tolist()
    return ring_colour(art, E, exclude=dark + [art.color(P[k]) for k in ("sclera", "white")])


def clean_eye_area(art, ctx, e, k_skin):
    """Old pixelized eye -> skin, so the hand-drawn eye starts from a clean face."""
    cx, cy, rx, ry, _ = eye_geom(ctx, e)
    area = m_ellipse(cx, cy, rx + 1.4, ry + 1.4)
    lab = pal_lab(art)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    bad = (lab[:, 0] < 52) | ((chroma < 20) & (lab[:, 0] < 92)) | (lab[:, 0] > 96)
    green = (lab[:, 1] < -25) & (lab[:, 0] < 70)          # old iris greens
    art.idx[area & lut(art, bad | green)] = k_skin


def draw_eye(art, ctx, stage, e, k_skin, closed=False):
    cx, cy, rx, ry, side = eye_geom(ctx, e)
    E = m_ellipse(cx, cy, rx, ry)
    if closed:
        art.idx[m_ellipse(cx, cy, rx + 0.6, ry + 0.6) & (art.idx != TRANSPARENT)] = k_skin
        u = (XC - cx) / rx
        yarc = cy + 0.18 * ry + 0.32 * ry * (1 - u ** 2)
        th = 0.55 if stage == 1 else 0.75
        arc = (np.abs(u) <= 1.0) & (np.abs(YC - yarc) <= th)
        paint(art, arc, "ink")
        # lash flick at the outer corner
        xo = int(np.floor(cx + side * rx))
        yo = int(np.floor(cy + 0.15 * ry))
        art.put(xo + side, yo - 1, P["ink"])
        return
    glow = stage == 3
    paint(art, E, "sclera")
    icx, icy = cx - side * 0.2 * rx, cy + 0.06 * ry
    irx, iry = rx * 0.84, ry * 0.9
    I = m_ellipse(icx, icy, irx, iry) & E
    paint(art, I, "jglow" if glow else "iris")
    paint(art, I & (YC < icy - 0.2 * iry), "jglow_d" if glow else "iris_d")
    paint(art, I & (YC > icy + 0.45 * iry), "jglow_l" if glow else "iris_l")
    pr = 0.5 if stage < 3 else 0.42
    pu = m_ellipse(icx + side * 0.04 * irx, icy - 0.06 * iry, max(1.0, irx * pr), iry * 0.56) & E
    paint(art, pu, "pupil")
    # outline: inner edge of the eye, top rows heavier (lashes)
    edge = E & ~erode4(E)
    top = E & ~np.roll(E, 2 if ry > 6 else 1, axis=0)
    paint(art, edge | top, "ink")
    # highlights: a big one up-left on the pupil, a small one low on the other side
    hs = 3 if rx >= 9 else 2
    hx = int(np.floor(icx - 0.3 * irx - hs / 2 + 0.5))
    hy = int(np.floor(icy - 0.42 * iry))
    art.rect(hx, hy, hx + hs - 1, hy + hs - 1, P["white"])
    if hs == 3:
        art.put(hx + 2, hy + 2, P["pupil"])
    art.put(int(np.floor(icx + 0.36 * irx)), int(np.floor(icy + 0.38 * iry)), P["white"])
    # outer-corner lash flick
    xo = int(np.floor(cx + side * (rx - 0.5)))
    yt = int(np.floor(cy - 0.6 * ry))
    art.put(xo + side, yt, P["ink"])
    if rx > 5:
        art.put(xo + 2 * side, yt - 1, P["ink"])


# --------------------------------------------------------- gem & lettering ----
def draw_gem(art, ctx, c=(403, 798), hw=46, hh_top=44, hh_bot=62):
    """Faceted diamond: crown (top) shorter than the pavilion (bottom), girdle across."""
    cx, cy = SP(ctx, *c)
    cx = np.floor(cx) + 0.5
    cy = np.floor(cy) + 0.5
    a = max(3.0, round(hw * ctx.scale)) + 0.5
    bt = max(3.0, round(hh_top * ctx.scale)) + 0.5
    bb = max(4.0, round(hh_bot * ctx.scale)) + 0.5
    dx, dy = np.abs(XC - cx) / a, YC - cy
    D = np.where(dy < 0, dx + (-dy) / bt, dx + dy / bb) <= 1.0
    paint(art, ring(D), "ink")
    up, left = dy < 0, XC < cx
    paint(art, D & up & left, "gem_l")
    paint(art, D & up & ~left, "gem")
    paint(art, D & ~up & left, "gem_m")
    paint(art, D & ~up & ~left, "gem_d")
    # centre facet line down the pavilion, girdle highlight
    paint(art, D & ~up & (np.abs(XC - cx) < 0.6) & (dy > 1), "gem")
    paint(art, D & (np.abs(dy) < 0.6) & (np.abs(XC - cx) < a - 1), "gem_l")
    gx, gy = int(np.floor(cx - a * 0.4)), int(np.floor(cy - bt * 0.45))
    art.put(gx, gy, P["gem_hi"])
    if a > 5:
        art.put(gx + 1, gy, P["gem_hi"])
        art.put(gx, gy + 1, P["gem_hi"])
    art.put(int(np.floor(cx + a * 0.35)), int(np.floor(cy + bb * 0.25)), P["gem_l"])


FONT = {
    "I": ["#", "#", "#", "#", "#"],
    "I3": ["###", ".#.", ".#.", ".#.", "###"],
    "Q": [".##.", "#..#", "#..#", "#.##", ".###"],
    "X": ["#.#", "#.#", ".#.", "#.#", "#.#"],
}


def recolour_scarf(art):
    """Snap the scarf's blues onto three fixed royal-blue tones (by lightness)."""
    blue = is_blue(art)
    lab = pal_lab(art)
    L = lab[art.idx][..., 0]
    art.idx[blue & (L < 34)] = art.color(P["blue_d"])
    art.idx[blue & (L >= 34) & (L < 48)] = art.color(P["blue"])
    art.idx[blue & (L >= 48)] = art.color(P["blue_l"])
    return art.idx == art.color(P["blue"])


def scarf_band(art, x, cy):
    """Rows (top, bottom) of the scarf in column x around row cy: first ink above / below."""
    k = art.names["outline"]
    y0 = int(cy)
    if art.idx[y0, x] == k:
        y0 -= 1
    t = y0
    while t > 0 and art.idx[t - 1, x] not in (k, TRANSPARENT):
        t -= 1
    b = y0
    while b < N - 1 and art.idx[b + 1, x] not in (k, TRANSPARENT):
        b += 1
    return t, b


def draw_iqx(art, ctx, stage, cy_ref=704):
    """Hand-placed 'IQX' on the scarf, centred on the original lettering."""
    glyphs = ["I3" if stage == 3 else "I", "Q", "X"]
    w = sum(len(FONT[g][0]) for g in glyphs) + len(glyphs) - 1
    cx, cy = SP(ctx, 402, cy_ref)
    x0 = int(round(cx - w / 2))
    bands = [scarf_band(art, x, cy) for x in range(x0, x0 + w)]
    bands = [(t, b) for t, b in bands if b - t >= 6 and b - t <= 22]
    if bands:
        yc = (np.median([t for t, _ in bands]) + np.median([b for _, b in bands]) + 1) / 2
    else:
        yc = cy
    y0 = int(np.floor(yc - 2.5))
    box = np.zeros((N, N), bool)
    box[y0 - 1:y0 + 6, x0 - 1:x0 + w + 1] = True
    k = art.names["outline"]
    paint(art, box & (art.idx != k) & (art.idx != TRANSPARENT), "blue")
    # leftovers of the original (wider) lettering beside the new text
    wide = np.zeros((N, N), bool)
    wide[y0 - 2:y0 + 7, x0 - 6:x0 + w + 6] = True
    lab = pal_lab(art)
    pale = lut(art, (lab[:, 0] > 62) & (np.abs(lab[:, 1]) < 20) & (lab[:, 2] < 25))
    near_blue = grow(is_blue(art), 1)
    paint(art, wide & pale & near_blue, "blue")
    x = x0
    for g in glyphs:
        art.blit(x, y0, FONT[g], {"#": P["letter"]})
        x += len(FONT[g][0]) + 1


# ------------------------------------------------------------ shell plates ----
def frac(mask, ctx):
    """Share of each sprite pixel covered by an art-canvas mask."""
    return gk._place((mask.astype(np.uint8) * 255)[..., None], ctx, Image.BOX)[..., 0] / 255


def sprite_ref(ctx):
    """Reference-px coordinates of every sprite pixel centre."""
    ax = (XC - ctx.ox) / ctx.scale
    ay = (YC - ctx.oy) / ctx.scale
    return ax - gk.PAD, ay - gk.PAD


def paint_shell(art, ctx, stage):
    """Re-render the visible shell pixels from the dome model: cel-shaded plates, 1px seams, rim."""
    sp = SHELL[stage]
    xr, yr = sprite_ref(ctx)
    cx, cy = sp["c"]
    ell = ((xr - cx) / sp["rad"][0]) ** 2 + ((yr - cy) / sp["rad"][1]) ** 2 <= 1
    ell &= yr < sp["ybot"]
    cover = frac(parts()["fg"], ctx) + frac(_FRONT.get(stage, np.zeros((1, 1), bool)), ctx) \
        if stage in _FRONT and _FRONT[stage].shape[0] > 1 else frac(parts()["fg"], ctx)
    k_ink = art.names["outline"]
    vis = ell & (cover < 0.5) & (art.idx != k_ink) & (art.idx != TRANSPARENT)
    d = dome_eval(xr, yr, sp)
    tones = [art.color(t) for t in sp["tones"]]
    rim = [art.color(t) for t in sp["rim"]] if sp.get("rim") else tones
    col = np.where(d["rim"], np.array(rim)[d["kr"]], np.array(tones)[d["k"]])
    art.idx[vis] = col[vis]
    cell = d["cell"]
    right = np.roll(cell, -1, axis=1)
    below = np.roll(cell, -1, axis=0)
    ell_r = np.roll(ell, -1, axis=1)
    ell_b = np.roll(ell, -1, axis=0)
    seam = ((cell != right) & ell_r) | ((cell != below) & ell_b)
    seam &= vis
    lam = d["lam"]
    sc = np.where(lam > 0.42, art.color(sp["seam"][2]), np.where(lam < 0.0, art.color(sp["seam"][1]),
                                                                  art.color(sp["seam"][0])))
    in_rim = d["rim"] & np.roll(d["rim"], -1, axis=1) & np.roll(d["rim"], -1, axis=0)
    sc = np.where(in_rim, rim[0], sc)
    art.idx[seam] = sc[seam]
    return vis, d


# ---------------------------------------------------------------- touchup ----
def touchup(stage, art, ctx):
    if stage in SHELL:
        paint_shell(art, ctx, stage)
    recolour_scarf(art)
    skins = [eye_skin(art, ctx, e) for e in EYES]
    for e, k in zip(EYES, skins):
        clean_eye_area(art, ctx, e, k)
    for e, k in zip(EYES, skins):
        draw_eye(art, ctx, stage, e, k)
    art.names["skin_l"], art.names["skin_r"] = skins
    draw_iqx(art, ctx, stage)
    draw_gem(art, ctx)


def blink(stage, art, ctx):
    for e, k in zip(EYES, (art.names["skin_l"], art.names["skin_r"])):
        draw_eye(art, ctx, stage, e, k, closed=True)
