"""Kim Quy: evolution line for the game sprites (see tools/gamekit.py).

Stage 1 is the original baby turtle. Stages 2 (Ngọc Quy) and 3 (Kim Quy Thần Kiếm) are rebuilt at
illustration level: the original shell is cut away (flood-filled part masks) and replaced by a bigger
spiked dome whose scutes (central hexagon, two plate rings, marginal rim) come from a small 3D model.
The sprout grows into a sapling crowned with the original's own leaf (stamped, rotated, mirrored so the
light stays top-left); in stage 3 it becomes a golden leaf crest around a jade orb while a bonsai hung
with coins takes root on the shell and the Thuận Thiên sword is strapped diagonally across it. The
skin shifts towards jade.

touchup() re-renders the visible shell from the same model at sprite resolution (one flat tone per
plate with pillow shading, thinned 1px seams, a few glowing runes) and hand-places the small details:
eyes with highlights and a determined lid line, blush, mouth, IQX lettering, gem and its gold setting,
scarf trim, the sword's guard / hilt / pommel, the jade orb, hanging coins and sparks. No aura is baked
in: the demo draws its own flickering outline on the final form.
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
     "desc": "Chú rùa con đội mầm lá, chậm rãi lật từng trang báo cáo tài chính. Mầm chỉ nhú thêm "
             "khi chú tìm ra một doanh nghiệp đang bị thị trường định giá thấp hơn giá trị thật.",
     "skill": {"name": "Soi Giá Trị",
               "desc": "Đọc vị P/E, P/B của đối thủ: lộ giá trị thật, giảm 15% phòng thủ của mục tiêu "
                       "trong 2 lượt."},
     "stats": {"hp": 48, "atk": 26, "def": 46, "spd": 20}},
    {"rank": "Trưởng thành", "name": "Ngọc Quy", "level": 10, "height": 130,
     "config": dict(CFG, colors=30),
     "desc": "Mai hóa ngọc bích mọc gai, từng mảnh lục giác viền vàng khắc một mô hình định giá: "
             "P/E, P/B, DCF. Mầm non đã thành cây con, mỗi thương vụ có biên an toàn dày lại mọc "
             "thêm một lá.",
     "skill": {"name": "Khiên Biên An Toàn",
               "desc": "Dựng khiên ngọc lục giác hấp thụ sát thương bằng 30% máu tối đa; mua càng "
                       "dưới giá trị nội tại, khiên càng dày."},
     "stats": {"hp": 76, "atk": 50, "def": 78, "spd": 32}},
    {"rank": "Thần thú", "name": "Kim Quy Thần Kiếm", "level": 25, "height": 150, "dx": -10,
     "config": dict(CFG, colors=34),
     "desc": "Thần Quy Hồ Gươm nghìn tuổi: mai vàng khắc ngọc phù rực sáng, gươm Thuận Thiên đeo "
             "chéo sau lưng, cây thế cổ trên mai treo đầy tiền vàng. Ngài định giá mọi doanh nghiệp "
             "không sai một đồng.",
     "skill": {"name": "Thuận Thiên Định Giá",
               "desc": "Vung gươm Thuận Thiên kéo mọi mục tiêu về đúng giá trị thật: đòn đánh xuyên "
                       "phòng thủ, sát thương bằng khoảng chênh giữa giá và giá trị."},
     "stats": {"hp": 100, "atk": 84, "def": 100, "spd": 46}},
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
    "blush": "#f7a6a0", "tongue": "#f06a5c", "mouth": "#7a1a2a",
}

# ------------------------------------------------------- art-level helpers ----
def R(x, y):
    """Reference poster px -> art canvas px."""
    return x + gk.PAD, y + gk.PAD


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
    a = np.array(gk.art_canvas("kim-quy"))
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
    leaves_all = grow(leaves, 22) & (alpha > 0) & ~grow(head, 2)
    bump = head & (L > 75) & (a[..., 0].astype(int) > a[..., 2].astype(int) + 60) \
        & (yy < R(0, 352)[1]) & (xx > R(410, 0)[0]) & (xx < R(545, 0)[0])
    _PARTS.update(shell=shell, replace=other, fg=fg, tail=tail_all, leaves=leaves_all, head=head, bump=bump,
                  leg_l=flood((270, 820)), leg_r=flood((610, 820)) & (yy > R(0, 760)[1]) & (xx > R(528, 0)[0]),
                  alpha=alpha > 0, ink=ink, leaf_m=flood((465, 215)), leaf_l=flood((385, 262)))
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


LEAF_COLS = ["#2c7a1a", "#5aae28", "#b0e562"]


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


_STAMP = {}


def leaf_stamp():
    """The original's big middle leaf, cut out with its own ink outline: (image, base, tip) in ref px."""
    if not _STAMP:
        pm = parts()
        m = pm["leaf_m"]
        ys, xs = np.where(m)
        i_b, i_t = ys.argmax(), ys.argmin()
        base = (xs[i_b] - gk.PAD, ys[i_b] - gk.PAD)
        tip = (xs[i_t] - gk.PAD, ys[i_t] - gk.PAD)
        cut = grow(m, 16) & pm["alpha"] & ~pm["leaf_l"] & ~grow(pm["head"], 1)
        _STAMP.update(img=keep_only(gk.art_canvas("kim-quy"), cut), base=base, tip=tip)
    return _STAMP["img"], _STAMP["base"], _STAMP["tip"]


def stamp_leaf(layer, base, angle, scale, mirror=None, ink_w=3):
    """Paste the original leaf with its stem at `base` (ref px), pointing `angle` degrees clockwise
    from up. Leaves pointing left are mirrored so their lit half stays on top (light from upper left)."""
    img, b0, t0 = leaf_stamp()
    a0 = np.degrees(np.arctan2(t0[0] - b0[0], -(t0[1] - b0[1])))
    if mirror is None:
        mirror = angle < 0
    M = np.diag([-1.0, 1.0]) if mirror else np.eye(2)
    a_src = -a0 if mirror else a0
    one = dk.blank(layer)
    affine_paste(one, img, b0, base, rot(angle - a_src) @ M * scale)
    # its own ink rim, so it stays separated from the leaves and branch behind it
    layer.alpha_composite(rgba_fill(outline_of(np.array(one)[..., 3] > 100, ink_w), INK))
    layer.alpha_composite(one)


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
    x, y = (A - a0) / hexr, (B - b0) / (hexr * sp.get("vy", 1.0))
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
GOLD_TONES = ["#9c5414", "#cf8a1e", "#eeb02c", "#ffd04a", "#fff1a8"]
JADE_RIM = ["#0e5c47", "#168467", "#27ad85", "#66dcb0"]
SHELL = {
    3: dict(c=(716, 562), rad=(170, 276), ybot=816, hexr=0.4, vy=0.66, apex=(0.2, -0.26), ring2=2.15,
            rim_w=0.15, bevel=0.36, tones=GOLD_TONES, plate_th=[-0.02, 0.5], k0=1, rim=JADE_RIM,
            seam=("#22b585", "#0f7a5a", "#34d399"), thresholds=[-0.1, 0.24, 0.62]),
    2: dict(c=(706, 576), rad=(146, 252), ybot=812, hexr=0.42, vy=0.66, apex=(0.2, -0.24), ring2=2.15,
            rim_w=0.15, bevel=0.34, tones=JADE_TONES, plate_th=[-0.1, 0.32], k0=1, rim=GOLD_RIM,
            seam=("#e8b030", "#a8701a", "#ffd95c")),
}
BARK = ["#5e3c1e", "#94653a", "#c4955c"]
SKIN = {2: ["#0b3a2a", "#147a52", "#2aa86e", "#5fd08e", "#b6efb4", "#effff0"],
        3: ["#0a3d34", "#127a62", "#22a585", "#47c99e", "#86e2bc", "#c2f4de"]}
_FRONT = {}


def reskin(body, stage):
    pm = parts()
    m = dk.hsv_mask(body, h=(60, 170), s=(0.3, 1)) & pm["fg"] & ~pm["bump"] & ~pm["leaves"]
    dk.gradient_map(body, m, SKIN[stage])


def tree(img, limbs, cols=BARK, ow=10):
    """Woody limbs [(control points, width)] (reference px) drawn in passes, so forks join cleanly:
    all ink outlines, then the bark, then a light stripe on the left and a shade on the right."""
    paths = [([R(*p) for p in dk.bezier(pts, 40)], w) for pts, w in limbs]
    for P_, w in paths:
        dk.stroke(img, P_, INK, w + 2 * ow)
    for P_, w in paths:
        dk.stroke(img, P_, cols[1], w)
    for P_, w in paths:
        dk.stroke(img, [(x - w * 0.2, y) for x, y in P_], cols[2], w * 0.4)
    for P_, w in paths:
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


SAPLING_LIMBS = [([(478, 340), (482, 262), (477, 196)], 50),
                 ([(477, 214), (440, 168), (414, 116)], 34), ([(479, 214), (516, 170), (542, 118)], 34)]
SAPLING_LEAVES = [((452, 254), -66, 0.66), ((504, 256), 66, 0.66),           # behind the branches
                  ((436, 172), -84, 0.74), ((520, 174), 84, 0.74),
                  ((416, 124), -26, 0.98), ((540, 126), 26, 0.98), ((478, 200), 2, 0.84)]


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
    # sapling: a forked trunk rising from the head; the original's leaf, grown and multiplied, clusters
    # on the branch tips so the crown reads as a lush leafy V crest
    for base, ang, sc in SAPLING_LEAVES[:4]:          # lower and side leaves behind the branches
        stamp_leaf(back, base, ang, sc)
    tree(back, SAPLING_LIMBS)
    for base, ang, sc in SAPLING_LEAVES[4:]:          # crown leaves on the branch tips, over them
        stamp_leaf(back, base, ang, sc)
    jade_spike = ("#26a47a", "#5fdcaa", ("#d8961e", "#ffd95c"))
    shell_parts(img, back, 2, [(-104, 126, 100, *jade_spike), (-66, 146, 108, *jade_spike),
                                   (-28, 134, 100, *jade_spike)])
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
    # every bump gets a lit crescent along its upper-left edge, the middle ones a brighter rim on top
    yy, xx = np.ogrid[0:img.size[1], 0:img.size[0]]

    def disc(x, y, r):
        return (xx - x) ** 2 + (yy - y) ** 2 <= r * r

    lit = np.zeros(M.shape, bool)
    hi = np.zeros(M.shape, bool)
    for i, (x, y, r) in enumerate(discs):
        lit |= disc(x, y, r) & ~disc(x + r * 0.16, y + r * 0.36, r)
        if 0 < i < len(discs) - 1:
            hi |= disc(x, y, r) & ~disc(x + r * 0.06, y + r * 0.16, r) & (xx < x + r * 0.2)
    lit &= M & ~under
    img.alpha_composite(rgba_fill(lit, cols[2]))
    img.alpha_composite(rgba_fill(hi & lit, cols[3]))
    return M


def sword(img, tip, guard, hilt_len=96, w=44, ow=10):
    """Thuận Thiên: a straight jade-glowing blade, wide gold guard, jade-wrapped hilt, gold pommel."""
    t = np.array(R(*tip), float)
    g = np.array(R(*guard), float)
    d = (g - t) / np.hypot(*(g - t))
    n = np.array([-d[1], d[0]])
    if n[1] > 0:
        n = -n                                    # n points up (lit edge)
    hw = w / 2
    blade = [tuple(t), tuple(t + d * 60 + n * hw), tuple(g + n * hw), tuple(g - n * hw), tuple(t + d * 60 - n * hw)]
    dk.stroke(img, blade + [blade[0]], INK, ow * 2)
    dk.shape(img, blade, "#9fcfc8")
    dk.shape(img, [tuple(t), tuple(t + d * 60 + n * hw), tuple(g + n * hw), tuple(g + n * hw * 0.1),
                   tuple(t + d * 60 + n * hw * 0.1)], "#e9fff8")
    dk.stroke(img, [tuple(t + d * 70 - n * hw * 0.35), tuple(g - d * 8 - n * hw * 0.35)], "#34d399", w * 0.22)
    # guard, hilt and pommel are hand-drawn at sprite size in touchup (sword_hilt)


GOLD_LEAF = ["#a8640e", "#eaa82a", "#ffd95c"]
SWORD3 = ((596, 786), (852, 418))        # blade tip, guard (reference px): diagonal across the shell
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
    body = img.copy()
    erase(body, ~pm["fg"] | pm["leaves"])
    reskin(body, 3)
    # the sprout's node turns into a jade orb
    dk.gradient_map(body, pm["bump"], ["#0b6a50", "#22b585", "#7ff0c4", "#e8fff6"])
    back = dk.blank(img)
    # (the tail stays tucked away under the big shell: only a stray sliver of it would show)
    gold_spike = ("#e3a52a", "#ffe07a", ("#20b585", "#8ff2cc"))
    shell_parts(img, back, 3, [(-6, 176, 128, *gold_spike), (24, 120, 100, *gold_spike)])
    over = dk.blank(img)
    # bonsai rooted on the shell: twisted trunk, jade foliage pads, gold coins
    tree(over, [([(756, 210), (700, 190), (650, 176)], 40), ([(800, 140), (870, 146), (928, 160)], 36),
                ([(770, 336), (818, 250), (740, 190), (790, 84)], 64)])
    for c, rx, ry in [((634, 168), 86, 58), ((944, 164), 80, 56), ((792, 74), 118, 56)]:
        pad(over, c, rx, ry, PAD_COLS)
    # Thuận Thiên, strapped along the shell
    sword(over, *SWORD3, hilt_len=118, w=54)
    _FRONT[3] = opaque(over)
    back.alpha_composite(over)
    # golden leaf crest rising from behind the jade orb
    for base, ang, ln, fl in [((452, 304), -42, 200, False), ((506, 300), 24, 214, True),
                              ((478, 300), -8, 168, False), ((444, 310), -66, 136, False)]:
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
        clear = m_ellipse(cx, cy, rx + 0.6, ry + 0.6) | (m_ellipse(cx, cy, rx + 2.2, ry + 1.6)
                                                         & (art.idx == art.names["outline"]))
        art.idx[clear & (art.idx != TRANSPARENT)] = k_skin
        # the open eye's lash flick and lid line reach a little outside the eye: clear them too
        xo = int(np.floor(cx + side * (rx - 0.5)))
        yt = int(np.floor(cy - 0.6 * ry))
        old = [(xo + side, yt), (xo + 2 * side, yt - 1)]
        if stage > 1:
            old += brow_pixels(stage, cx, cy, rx, ry, side)[0]
        for x, y in old:
            if 0 <= x < N and 0 <= y < N and art.idx[y, x] == art.color(P["ink"]):
                art.idx[y, x] = k_skin
        # closed lid: a gentle smile-shaped arc, one pixel per column (two rows deep in the middle
        # for the bigger stages), plus a lash flick at the outer end
        xs = range(int(np.floor(cx - rx + 0.3)), int(np.floor(cx + rx - 0.3)) + 1)
        end = None
        for x in xs:
            u = (x + 0.5 - cx) / rx
            y = int(np.floor(cy + 0.15 * ry + 0.3 * ry * (1 - u * u)))
            art.put(x, y, P["ink"])
            if stage > 1 and abs(u) < 0.55:
                art.put(x, y + 1, P["ink"])
            if (side < 0 and end is None) or side > 0:
                end = (x, y)
        if end:
            art.put(end[0] + side, end[1] - 1, P["ink"])
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
    if stage > 1:
        brow(art, stage, cx, cy, rx, ry, side, k_skin)


def brow_pixels(stage, cx, cy, rx, ry, side):
    """Lid line of the determined look: from the outer corner down towards the nose."""
    drop = {2: 0.26, 3: 0.4}[stage] * ry
    top = cy - ry
    xo, xi = cx + side * rx, cx - side * rx
    yo, yi = top + 0.1 * ry, top + 0.1 * ry + drop
    pts = []
    x0, x1 = sorted((xo, xi))
    for x in range(int(np.floor(x0)) - 1, int(np.floor(x1)) + 2):
        tt = (x + 0.5 - xo) / (xi - xo)
        if not -0.1 <= tt <= 1.06:
            continue
        y = int(np.floor(yo + (yi - yo) * tt))
        pts.append((x, y))
        if stage == 3 or tt < 0.7:
            pts.append((x, y + 1))
    return pts, (xo, yo, xi, yi)


def brow(art, stage, cx, cy, rx, ry, side, k_skin):
    """Determined look: the top of the eye is cut by the lid line."""
    pts, (xo, yo, xi, yi) = brow_pixels(stage, cx, cy, rx, ry, side)
    t = (XC - xo) / (xi - xo)
    yline = yo + (yi - yo) * t
    E = m_ellipse(cx, cy, rx + 0.5, ry + 0.5)
    art.idx[E & (YC < yline - 0.5) & (art.idx != TRANSPARENT)] = k_skin
    for x, y in pts:
        art.put(x, y, P["ink"])


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
    pal = np.array(art.palette)
    bluish = lut(art, (pal[:, 2] > pal[:, 0] + 50) & (pal[:, 2] > pal[:, 1] + 20))
    paint(art, wide & bluish, "blue")
    x = x0
    for g in glyphs:
        art.blit(x, y0, FONT[g], {"#": P["letter"]})
        x += len(FONT[g][0]) + 1


# ------------------------------------------------------------ shell plates ----
def frac(mask, ctx):
    """Share of each sprite pixel covered by an art-canvas mask (same box downscale and placement as
    the pixelizer uses for the art canvas)."""
    n = max(1, round(gk.ART * ctx.scale))
    small = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((n, n), Image.BOX), np.float64) / 255
    out = np.zeros((N, N))
    ox, oy = round(ctx.ox), round(ctx.oy)
    sx0, sy0, dx0, dy0 = max(0, -ox), max(0, -oy), max(0, ox), max(0, oy)
    w, h = min(n - sx0, N - dx0), min(n - sy0, N - dy0)
    if w > 0 and h > 0:
        out[dy0:dy0 + h, dx0:dx0 + w] = small[sy0:sy0 + h, sx0:sx0 + w]
    return out


def sprite_ref(ctx):
    """Reference-px coordinates of every sprite pixel centre."""
    ax = (XC - ctx.ox) / ctx.scale
    ay = (YC - ctx.oy) / ctx.scale
    return ax - gk.PAD, ay - gk.PAD


def thin_lines(m):
    """Pixel-perfect 1px lines: drop the corner pixel of every L-step so diagonals are 8-connected."""
    m = m.copy()
    H, W = m.shape
    for y, x in zip(*np.where(m)):
        n = y > 0 and m[y - 1, x]
        s = y < H - 1 and m[y + 1, x]
        w = x > 0 and m[y, x - 1]
        e = x < W - 1 and m[y, x + 1]
        if (n + s + w + e) == 2 and (n or s) and (w or e):
            m[y, x] = False
    return m


def paint_shell(art, ctx, stage):
    """Re-render the visible shell pixels from the dome model at sprite size: every plate is one flat
    tone picked from the dome's lighting (so the dome reads by plate, not by noisy bands), with a 1px
    bevel (light along its upper-left seam, shade along its lower-right one); seams are thinned to
    clean 1px lines; the marginal rim keeps its own banded tones."""
    sp = SHELL[stage]
    xr, yr = sprite_ref(ctx)
    cx, cy = sp["c"]
    ell = ((xr - cx) / sp["rad"][0]) ** 2 + ((yr - cy) / sp["rad"][1]) ** 2 <= 1
    ell &= yr < sp["ybot"]
    cover = frac(parts()["fg"], ctx)
    if stage in _FRONT:                       # sword / bonsai drawn over the shell in compose()
        cover = cover + frac(_FRONT[stage], ctx)
    k_ink = art.names["outline"]
    vis = ell & (cover < 0.5) & (art.idx != k_ink) & (art.idx != TRANSPARENT)
    d = dome_eval(xr, yr, sp)
    cell = d["cell"]
    tones = [art.color(t) for t in sp["tones"]]
    rim = [art.color(t) for t in sp["rim"]] if sp.get("rim") else tones
    nt = len(tones)
    # flat tone per plate from its mean dome lighting
    k = np.zeros(cell.shape, np.int64)
    th = sp.get("plate_th", [-0.05, 0.3, 0.62][:nt - 2])
    for c in np.unique(cell[ell]):
        m = ell & (cell == c)
        k[m] = np.digitize(d["lam"][m].mean(), th) + sp.get("k0", 0)
    col = np.where(d["rim"], np.array(rim)[d["kr"]], np.array(tones)[np.clip(k, 0, nt - 1)])
    art.idx[vis] = col[vis]
    # seams: where the plate id changes towards the right / bottom neighbour, thinned to 1px
    seam = np.zeros(cell.shape, bool)
    for dy, dx in ((0, 1), (1, 0)):
        nb = np.roll(cell, (-dy, -dx), axis=(0, 1))
        seam |= (cell != nb) & np.roll(ell, (-dy, -dx), axis=(0, 1))
    seam = thin_lines(seam & vis)
    # pillow shading, like the original's bulgy scutes: a lit core pushed towards the upper left and a
    # shaded lower-right edge inside every plate
    kk = np.clip(k, 0, nt - 1)
    T = np.array(tones)
    for c in np.unique(cell[vis]):
        if c < 0:
            continue
        Pm = vis & ~seam & (cell == c)
        if Pm.sum() < 6:
            continue
        shade = Pm & ~np.roll(Pm, (-1, -1), axis=(0, 1))
        if Pm.sum() >= 14:
            shade |= Pm & ~np.roll(Pm, (-2, -2), axis=(0, 1)) & (kk < 2)
        core = np.roll(erode(Pm, 2 if Pm.sum() >= 40 else 1), (-1, -1), axis=(0, 1)) & Pm & ~shade
        art.idx[shade] = T[np.clip(kk - 1, 0, nt - 1)][shade]
        art.idx[core] = T[np.clip(kk + 1, 0, nt - 1)][core]
    lam = d["lam"]
    sc = np.where(lam > 0.42, art.color(sp["seam"][2]), np.where(lam < 0.0, art.color(sp["seam"][1]),
                                                                  art.color(sp["seam"][0])))
    in_rim = d["rim"] & np.roll(d["rim"], -1, axis=1) & np.roll(d["rim"], -1, axis=0)
    sc = np.where(in_rim, rim[0], sc)
    art.idx[seam] = sc[seam]
    return vis & ~seam, d


# --------------------------------------------------------------- face bits ----
MOUTH = [(302, 548), (320, 555), (340, 557), (360, 552), (372, 544), (366, 568), (354, 586), (336, 594),
         (318, 589), (306, 570)]


def draw_mouth(art, ctx, corners=True):
    """Open smile: dark red inside, pink tongue at the bottom, ink rim, little upturned corners."""
    M = m_poly([SP(ctx, x, y) for x, y in MOUTH])
    if M.sum() < 4:
        return
    paint(art, ring(M) & (art.idx != TRANSPARENT), "ink")
    paint(art, M, "mouth")
    tx, ty = SP(ctx, 338, 590)
    T = m_ellipse(tx, ty, 26 * ctx.scale + 0.6, 16 * ctx.scale + 0.6) & M
    paint(art, T, "tongue")
    paint(art, T & ~np.roll(T, 1, axis=0) & ~np.roll(T, -1, axis=1), "#ff9a8a")
    for (x, y), d in (((290, 534), -1), ((384, 528), 1)) if corners else ():
        px, py = SP(ctx, x, y)
        art.put(int(px), int(py), P["ink"])
        art.put(int(px) + d, int(py) - 1, P["ink"])


def face_box(ctx, x0=168, y0=280, x1=600, y1=660):
    """Sprite mask of the head/face area (reference-px box)."""
    a, b = SP(ctx, x0, y0)
    c, d = SP(ctx, x1, y1)
    return (XX >= int(a)) & (XX <= int(c)) & (YY >= int(b)) & (YY <= int(d))


def despeckle(art, region, max_size=3):
    """Leftover specks of the original (old eye rims, grey antialiasing, pale glints) inside `region`:
    tiny same-colour blobs, and greyish blobs of any size, take the colour of their surroundings."""
    lab = pal_lab(art)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    greyish = (chroma < 16) & (lab[:, 0] > 30) & (lab[:, 0] < 92)
    keep = {art.names["outline"]} | {art.color(P[k]) for k in
                                     ("ink", "white", "sclera", "pupil", "iris", "iris_d", "iris_l", "jglow",
                                      "jglow_d", "jglow_l", "jglow_hi", "mouth", "tongue", "blush",
                                      "letter", "blue_d", "blue", "blue_l", "jade", "jade_d", "jade_l")}
    keep |= {art.color("#ff9a8a"), art.color("#fbc8b8")}
    H, W = art.idx.shape
    for _ in range(2):
        sub = np.where(region, art.idx, TRANSPARENT)
        for c, reg in components(sub):
            if c in keep or (len(reg) > max_size and not greyish[c]):
                continue
            ys, xs = np.divmod(reg, W)
            nb = []
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                yy, xx = ys + dy, xs + dx
                ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
                nb.append(art.idx[yy[ok], xx[ok]])
            nb = np.concatenate(nb)
            nb = nb[(nb != c) & (nb != TRANSPARENT) & ~np.isin(nb, list(keep))]
            nb = nb[~greyish[nb]]
            if len(nb) < 3:
                continue
            vals, cnt = np.unique(nb, return_counts=True)
            art.idx[ys, xs] = vals[cnt.argmax()]


BLUSH = [(206, 566), (494, 568)]           # cheek centres (reference px)


def draw_blush(art, ctx):
    """Pink blush on both cheeks, as in the original: a soft oval, deeper in the middle, only on the
    cream skin (never over the eyes, mouth or outline)."""
    lab = pal_lab(art)
    cream = lut(art, (lab[:, 0] > 74) & (lab[:, 2] > 8) & (np.abs(lab[:, 1]) < 16))
    rx, ry = max(3.0, 36 * ctx.scale), max(1.6, 17 * ctx.scale)
    for x, y in BLUSH:
        cx, cy = SP(ctx, x, y)
        outer = m_ellipse(cx, cy, rx, ry) & (cream | (art.idx == art.color("#fbc8b8")))
        paint(art, outer, "#fbc8b8")
        paint(art, m_ellipse(cx, cy, rx * 0.62, ry * 0.6) & outer, "blush")


def draw_nostrils(art, ctx):
    for x, y in ((298, 497), (338, 495)):
        px, py = SP(ctx, x, y)
        art.put(int(px), int(py), P["ink"])


def scarf_shadow(art):
    """Darker top row of the scarf: the shadow under the chin."""
    blue = np.isin(art.idx, [art.color(P[k]) for k in ("blue", "blue_l")])
    above = np.roll(art.idx, 1, axis=0)
    top = blue & ~np.isin(above, [art.color(P[k]) for k in ("blue", "blue_l", "blue_d", "letter")])
    paint(art, top, "blue_d")


def scarf_trim(art):
    """Gold trim along the lower edge of the scarf (stages 2-3)."""
    blue = is_blue(art) | np.isin(art.idx, [art.color(P[k]) for k in ("blue", "blue_d", "blue_l", "letter")])
    below = np.roll(blue, -1, axis=0)
    under = np.roll(art.idx, -1, axis=0)
    edge = blue & ~below & (under != TRANSPARENT) & (art.idx != art.color(P["letter"]))
    paint(art, edge, "au")


def gem_frame(art, ctx, stage, c=(403, 798), hw=46, hh_top=44, hh_bot=62):
    """Gold setting behind the gem: a bail on top (stage 2) or a full frame with prongs (stage 3)."""
    cx, cy = SP(ctx, *c)
    cx = np.floor(cx) + 0.5
    cy = np.floor(cy) + 0.5
    a = max(3.0, round(hw * ctx.scale)) + 0.5
    bt = max(3.0, round(hh_top * ctx.scale)) + 0.5
    bb = max(4.0, round(hh_bot * ctx.scale)) + 0.5
    dx, dy = np.abs(XC - cx), YC - cy
    if stage == 3:
        g = 4.0
        F = np.where(dy < 0, dx / (a + g) + (-dy) / (bt + g), dx / (a + g) + dy / (bb + g)) <= 1.0
        paint(art, ring(F), "ink")
        paint(art, F, "au")
        paint(art, F & (dy < 0) & (XC < cx), "au_l")
        paint(art, F & (dy > 0) & (XC > cx), "au_d")
        # side prongs
        for sx in (-1, 1):
            x = int(np.floor(cx + sx * (a + g)))
            y = int(np.floor(cy))
            art.put(x + sx, y, P["au"])
            art.put(x + 2 * sx, y, P["ink"])
            art.put(x + sx, y - 1, P["ink"])
            art.put(x + sx, y + 1, P["ink"])
    else:
        bx, by = int(np.floor(cx)), int(np.floor(cy - bt - 1))
        art.blit(bx - 2, by - 1, ["#####", "#ooo#", ".#o#."], {"#": P["ink"], "o": P["au"]})
        art.put(bx - 1, by, P["au_l"])


def erode(m, n):
    for _ in range(n):
        m = erode4(m)
    return m


def sparkle(art, x, y, kind="plus"):
    """Energy sparkle (jade plus) or a tiny floating gold coin, on transparent background only."""
    if kind == "plus":
        pts = [(0, 0, "jglow_hi"), (1, 0, "jglow"), (-1, 0, "jglow"), (0, 1, "jglow"), (0, -1, "jglow")]
    elif kind == "big":
        pts = [(0, 0, "white"), (1, 0, "jglow_l"), (-1, 0, "jglow_l"), (0, 1, "jglow_l"), (0, -1, "jglow_l"),
               (2, 0, "jglow"), (-2, 0, "jglow"), (0, 2, "jglow"), (0, -2, "jglow")]
    elif kind == "coin":
        rows = [".kkk.", "kaaak", "kadak", "kaaak", ".kkk."]
        pts = [(i - 2, j - 2, {"k": "ink", "a": "au", "d": "au_dd"}[ch])
               for j, row in enumerate(rows) for i, ch in enumerate(row) if ch != "."]
        pts += [(-1, -1, "au_hi"), (1, 1, "au_d"), (0, -1, "au_l"), (-1, 0, "au_l")]
        pts = list({(dx, dy): (dx, dy, c) for dx, dy, c in pts}.values())
    else:
        pts = [(0, 0, "jglow_l")]
    for dx, dy, c in pts:
        if 0 <= x + dx < N and 0 <= y + dy < N and art.idx[y + dy, x + dx] == TRANSPARENT:
            art.put(x + dx, y + dy, P[c])


def shell_runes(art, vis, d, n=3):
    """Stage 3: a glowing jade rune (diamond ring, bright core) on the few biggest visible plates."""
    cell = d["cell"]
    plates = []
    for c in np.unique(cell[vis]):
        if c < 0:
            continue
        m = vis & (cell == c)
        plates.append((int(erode(m, 2).sum()), int(c), m))
    for _, c, m in sorted(plates, key=lambda t: -t[0])[:n]:
        core = erode(m, 2)
        if not core.any():
            continue
        ys, xs = np.where(core)
        x, y = int(round(xs.mean())), int(round(ys.mean()))
        pts = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3) if abs(dx) + abs(dy) == 2]
        if all(0 <= y + dy < N and 0 <= x + dx < N and m[y + dy, x + dx] for dx, dy in pts):
            for dx, dy in pts:
                art.put(x + dx, y + dy, P["jglow"])
            art.put(x, y, P["jglow_hi"])


def sword_glow(art, ctx):
    """Stage 3: the Thuận Thiên blade's fuller glows jade (1px line on the visible light blade pixels)."""
    (tx, ty), (gx, gy) = SWORD3
    d = np.array([gx - tx, gy - ty], float)
    L = np.hypot(*d)
    d /= L
    n = np.array([d[1], -d[0]])                # points up
    lab = pal_lab(art)
    light = lut(art, lab[:, 0] > 72)
    pts = []
    for t in np.linspace(70, L - 14, 200):
        x, y = SP(ctx, tx + d[0] * t - n[0] * 4, ty + d[1] * t - n[1] * 4)
        pts.append((int(x), int(y)))
    for x, y in dict.fromkeys(pts):
        if 0 <= x < N and 0 <= y < N and light[y, x]:
            art.put(x, y, P["jglow"])
    # brightest spark near the guard
    x, y = SP(ctx, gx - d[0] * 40 - n[0] * 4, gy - d[1] * 40 - n[1] * 4)
    if light[int(y), int(x)] or art.idx[int(y), int(x)] == art.color(P["jglow"]):
        art.put(int(x), int(y), P["jglow_hi"])


def sword_hilt(art, ctx, hilt=19.0, hw=2.8, guard=12.5, gt=1.9, pr=3.8):
    """Stage 3: Thuận Thiên's gold cross-guard, jade-wrapped hilt and gold pommel, drawn crisply at
    sprite size (sprite px) on the blade's axis, beyond the guard point."""
    (tx, ty), (gx, gy) = SWORD3
    d = np.array([gx - tx, gy - ty], float)
    d /= np.hypot(*d)
    n = np.array([d[1], -d[0]])
    if n[1] > 0:
        n = -n                                             # n points up-left (lit side)
    G = np.array(SP(ctx, gx, gy))
    rel_x, rel_y = XC - G[0], YC - G[1]
    a = rel_x * d[0] + rel_y * d[1]                        # along the blade axis (towards the hilt)
    b = rel_x * n[0] + rel_y * n[1]                        # across it (towards the light)
    H = (a > 0.5) & (a < hilt) & (np.abs(b) < hw)
    pc = G + d * (hilt + pr - 0.8)
    Pm = m_ellipse(pc[0], pc[1], pr, pr)
    Gd = (np.abs(a - 0.4) < gt) & (np.abs(b) < guard)
    k = art.names["outline"]
    for part in (H, Pm, Gd):
        paint(art, ring(part) & ~H & ~Pm & ~Gd, "ink")
    # hilt: jade wrap with diagonal binding, lit stripe on the upper-left side
    paint(art, H, "jade")
    paint(art, H & (b > hw - 1.2), "jade_l")
    paint(art, H & (b < -hw + 1.2), "jade_d")
    paint(art, H & (np.mod(a + b * 0.8, 3.2) < 0.9), "jade_dd")
    paint(art, H & (np.abs(a - hilt * 0.55) < 0.9), "au")
    # pommel
    paint(art, Pm, "au")
    paint(art, Pm & ~np.roll(Pm, (-1, -1), axis=(0, 1)), "au_d")
    paint(art, m_ellipse(pc[0] - 0.9, pc[1] - 0.9, pr * 0.55, pr * 0.55) & Pm, "au_l")
    art.put(int(pc[0] - 1.4), int(pc[1] - 1.4), P["au_hi"])
    # guard: gold bar, lit towards the hilt, shaded towards the blade, darker curled tips, jade stone
    paint(art, Gd, "au")
    paint(art, Gd & (a - 0.4 > gt - 1.0), "au_l")
    paint(art, Gd & (a - 0.4 < -gt + 1.0), "au_d")
    paint(art, Gd & (np.abs(b) > guard - 1.5), "au_d")
    paint(art, m_ellipse(G[0] + d[0] * 0.4, G[1] + d[1] * 0.4, 1.6, 1.6), "jglow")
    art.put(int(G[0] + d[0] * 0.4 - 0.6), int(G[1] + d[1] * 0.4 - 0.6), P["jglow_hi"])
    del k


def jade_orb(art, ctx):
    """Stage 3: the sprout's seed bump has become a round jade orb set in the head, crest behind it."""
    x, y = SP(ctx, 478, 312)
    cx, cy = np.floor(x) + 0.5, np.floor(y) - 0.5
    O = m_ellipse(cx, cy, 5.2, 4.8)
    paint(art, ring(O), "ink")
    paint(art, O, "jade")
    paint(art, O & ~np.roll(O, -1, axis=0), "jade_d")
    paint(art, O & ~np.roll(O, (-2, -1), axis=(0, 1)), "jade_d")
    paint(art, m_ellipse(cx - 1.2, cy - 1.0, 3.2, 2.6) & O, "jade_l")
    paint(art, m_ellipse(cx - 1.8, cy - 1.6, 1.5, 1.2) & O, "jade_hi")
    art.put(int(cx) - 2, int(cy) - 2, P["white"])


# coins hung from the bonsai pads on red cords: (attach x, attach y, cord length) in sprite px
COINS3 = [(104, 56, 6), (176, 54, 4)]
COIN = ["..kkkk..",
        ".khhaak.",
        "khhaaadk",
        "khakkadk",
        "kaakkadk",
        "kaaaaddk",
        ".kadddk.",
        "..kkkk.."]


def hanging_coins(art):
    """Ancient square-holed gold coins on short red cords under the bonsai's foliage pads."""
    for x, y, ln in COINS3:
        for j in range(ln):
            art.put(x, y + j, "#d8324a" if j % 3 != 2 else "#9c1830")
        art.blit(x - 3, y + ln, COIN, {"k": P["ink"], "h": P["au_l"], "a": P["au"], "d": P["au_d"]})
        art.put(x - 2, y + ln + 2, P["au_hi"])


SPARKS = {
    3: [(14, 62, "big"), (24, 112, "plus"), (64, 18, "plus"), (184, 150, "plus")],
}


def despeckle_shell(art, ctx):
    """Stage 1: the original shell's white specular streaks -> the surrounding plate colour."""
    zone = frac(parts()["replace"], ctx) > 0.3
    lab = pal_lab(art)
    pale = lut(art, (lab[:, 0] > 76) & (np.hypot(lab[:, 1], lab[:, 2]) < 42))
    bad = zone & pale
    bad_cols = np.unique(art.idx[bad])
    for _ in range(2):
        for y, x in zip(*np.where(bad & np.isin(art.idx, bad_cols))):
            nb = art.idx[max(0, y - 1):y + 2, max(0, x - 1):x + 2].ravel()
            nb = nb[(nb != TRANSPARENT) & (nb != art.names["outline"]) & ~np.isin(nb, bad_cols)]
            if len(nb):
                v, c = np.unique(nb, return_counts=True)
                art.idx[y, x] = v[c.argmax()]


# ---------------------------------------------------------------- touchup ----
def touchup(stage, art, ctx):
    vis = d = None
    if stage == 1:
        despeckle_shell(art, ctx)
    # grey antialiasing left between the original's ink and its colours -> the neighbouring colour
    # (the light sword blade is the one greyish part that belongs)
    blade = frac(_FRONT[stage], ctx) > 0.2 if stage in _FRONT else np.zeros((N, N), bool)
    despeckle(art, ~blade, max_size=0)
    if stage in SHELL:
        vis, d = paint_shell(art, ctx, stage)
    recolour_scarf(art)
    skins = [eye_skin(art, ctx, e) for e in EYES]
    for e, k in zip(EYES, skins):
        clean_eye_area(art, ctx, e, k)
    for e, k in zip(EYES, skins):
        draw_eye(art, ctx, stage, e, k)
    art.names["skin_l"], art.names["skin_r"] = skins
    draw_mouth(art, ctx, corners=stage > 1)
    draw_nostrils(art, ctx)
    despeckle(art, face_box(ctx, y1=630))
    draw_blush(art, ctx)
    draw_iqx(art, ctx, stage)
    scarf_shadow(art)
    if stage > 1:
        scarf_trim(art)
        gem_frame(art, ctx, stage)
    draw_gem(art, ctx)
    if stage == 3:
        shell_runes(art, vis, d)
        sword_glow(art, ctx)
        sword_hilt(art, ctx)
        jade_orb(art, ctx)
        hanging_coins(art)
    for x, y, kind in SPARKS.get(stage, []):
        sparkle(art, x, y, kind)


def blink(stage, art, ctx):
    for e, k in zip(EYES, (art.names["skin_l"], art.names["skin_r"])):
        draw_eye(art, ctx, stage, e, k, closed=True)
