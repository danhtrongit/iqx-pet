"""Lộc Hươu: evolution line for the game sprites (see tools/gamekit.py)."""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

_TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

import drawkit as dk  # noqa: E402
import gamekit as gk  # noqa: E402
from pixelize import TRANSPARENT, components, hex2rgb, nearest, rgb2lab  # noqa: E402

PET = {
    "name": "Lộc Hươu",
    "layer": "Nội bộ",
    "element": "observer_gold",
    "accent": "#f5b942",
    "tagline": "Quan sát lặng lẽ giao dịch của người trong cuộc.",
}

INK = "#2a160e"
CFG = {"colors": 28, "ink_l": 26.0, "ink_frac": 0.4, "smooth": 0, "min_island": 1, "outline": INK}

STAGES = [
    {"rank": "Ấu thể", "name": "Tiểu Lộc", "level": 1, "height": 104,
     "config": {**CFG, "colors": 24},
     "desc": "Chú hươu con tò mò với đôi tai thính, lặng lẽ dỏng tai nghe ngóng mỗi lần ban lãnh đạo "
             "mua hay bán cổ phiếu của chính công ty mình.",
     "skill": {"name": "Dỏng Tai Nghe Ngóng",
               "desc": "Bắt tín hiệu giao dịch của ban lãnh đạo và người có liên quan, báo ngay cho chủ nhân."},
     "stats": {"hp": 38, "atk": 28, "def": 30, "spd": 52}},
    {"rank": "Trưởng thành", "name": "Kim Lộc", "level": 10, "height": 128,
     "config": {**CFG, "colors": 34},
     "desc": "Gạc vàng đâm chồi lộc non mỗi khi người trong cuộc âm thầm gom hàng; những đốm sáng trên lưng "
             "khắc ghi từng giao dịch của cổ đông lớn.",
     "skill": {"name": "Gạc Vàng Đâm Chồi",
               "desc": "Khi nội bộ mua vào, gạc nảy lộc: cả đội tăng 20% độ chính xác tín hiệu trong 3 lượt."},
     "stats": {"hp": 62, "atk": 55, "def": 50, "spd": 74}},
    {"rank": "Thần thú", "name": "Thần Lộc Thiên Nhãn", "level": 25, "height": 146,
     "config": {**CFG, "colors": 30},
     "desc": "Thần thú mang Thiên Nhãn giữa trán, nhìn thấu mọi giao dịch nội bộ trước khi thị trường kịp hay "
             "biết. Con mắt thứ ba ấy không bao giờ khép lại.",
     "skill": {"name": "Thiên Nhãn Khai Mở",
               "desc": "Soi thấu toàn bộ giao dịch của người trong cuộc, lật tẩy đợt xả hàng ngầm: "
                       "đối thủ bị lộ diện mất 30% phòng thủ."},
     "stats": {"hp": 86, "atk": 82, "def": 72, "spd": 95}},
]

# --------------------------------------------------------------- palette ----
BASE = {
    "fur_dd": "#9a5418", "fur_d": "#df8a2c", "fur": "#f6a944", "fur_l": "#f8c070",
    "cream_d": "#e8c6a8", "cream": "#f9e4c2", "white": "#fbf6ee", "pink": "#f6b8a0",
    "ant_d": "#9a601b", "ant": "#e09a26", "ant_l": "#f0c45a", "ant_hi": "#f4e4ac",
    "blue_dd": "#0b2a74", "blue_d": "#063f9e", "blue": "#0a56c5", "blue_l": "#4a82d6", "blue_hi": "#b4d4f4",
    "gem_d": "#cd8d26", "gem": "#f5bd22", "gem_l": "#fde27a",
    "hoof_d": "#3c2215", "hoof": "#62381b", "hoof_l": "#8a5e3a",
    "iris_d": "#7a4512", "iris": "#e0a010", "iris_l": "#f6cf52",
    "nose": "#cc5a46", "mouth": "#e0605a",
}
EXTRA = {
    "gold_dd": "#8a4a0c", "gold_d": "#c87c14", "gold": "#f2b230", "gold_l": "#ffdc60", "gold_hi": "#fff6c8",
    "leaf_d": "#2f7a4a", "leaf": "#4cae62", "leaf_l": "#9ee07e",
    "petal": "#ffc4d6", "petal_d": "#f08aac",
    "glow": "#fff1a8", "star": "#ffffff",
    "aqua": "#7fe3ff",
}
NAMES = list(BASE) + list(EXTRA)
K_INK = len(NAMES)
STAGE_COLS = {
    1: {},
    2: {"fur_dd": "#a3561a", "fur_d": "#e8902a", "fur": "#f9ae40", "fur_l": "#fdc86e",
        "ant_d": "#a8640e", "ant": "#eda826", "ant_l": "#ffd451", "ant_hi": "#fff2b0"},
    3: {"fur_dd": "#a3501a", "fur_d": "#e8902c", "fur": "#fbb442", "fur_l": "#ffd07a",
        "ant_d": "#b0650c", "ant": "#f5b72a", "ant_l": "#ffe066", "ant_hi": "#fffbe0"},
}


def palette(stage):
    return {n: STAGE_COLS[stage].get(n, BASE.get(n) or EXTRA.get(n)) for n in NAMES}


def L(name):
    return K_INK if name == "ink" else NAMES.index(name)


_STATE: dict = {}

# --------------------------------------------------------------- geometry ----
# reference-poster regions (1024px poster coordinates)
SCARF_POLY = [(375, 610), (470, 590), (560, 600), (650, 588), (662, 640), (650, 685), (560, 680), (470, 668),
              (440, 700), (395, 690), (370, 640)]
HOOF_BOXES = [(290, 880, 375, 950), (420, 880, 470, 935), (470, 880, 565, 950), (575, 880, 665, 945)]
EAR_POLYS = [[(236, 312), (330, 330), (440, 360), (440, 440), (380, 460), (300, 420)],
             [(690, 330), (795, 285), (790, 360), (740, 445), (700, 430)]]
STRETCH = {1: (815, 875, 1.0), 2: (815, 875, 1.5), 3: (815, 875, 1.85)}   # ref y band, factor


def lift(stage):
    ya, yb, f = STRETCH[stage]
    return (yb - ya) * (f - 1)


def P(stage, x, y):
    """Reference poster px -> art canvas for this stage (after the leg stretch)."""
    ya, yb, f = STRETCH[stage]
    if y <= ya:
        y2 = y - lift(stage)
    elif y < yb:
        y2 = yb - (yb - y) * f
    else:
        y2 = y
    return x + gk.PAD, y2 + gk.PAD


def stretch_legs(img, stage):
    ya, yb, f = STRETCH[stage]
    if f == 1.0:
        return img
    a = np.array(img)
    Ya, Yb = ya + gk.PAD, yb + gk.PAD
    seg = Image.fromarray(a[Ya:Yb]).convert("RGBa")
    nh = int(round((Yb - Ya) * f))
    seg = np.array(seg.resize((a.shape[1], nh), Image.BICUBIC).convert("RGBA"))
    d = nh - (Yb - Ya)
    out = np.zeros_like(a)
    out[:Ya - d] = a[d:Ya]
    out[Yb - nh:Yb] = seg
    out[Yb:] = a[Yb:]
    return Image.fromarray(out, "RGBA")


def poly_mask(pts):
    m = Image.new("L", (gk.ART, gk.ART), 0)
    ImageDraw.Draw(m).polygon([tuple(map(float, q)) for q in pts], fill=255)
    return np.array(m) > 127


def box_mask(stage, box):
    m = np.zeros((gk.ART, gk.ART), bool)
    x0, y0 = P(stage, box[0], box[1])
    x1, y1 = P(stage, box[2], box[3])
    m[int(y0):int(y1), int(x0):int(x1)] = True
    return m


def ref_poly(stage, pts):
    return poly_mask([P(stage, *q) for q in pts])


# ------------------------------------------------------------ flattening ----
def flatten(img, thr, mode=0, regions=None):
    """Snap opaque art pixels to the BASE palette; pixels darker than `thr` (L*, scalar or map) are ink.
    regions: {name: bool mask} limits where a colour may be used (elsewhere the nearest free colour wins).
    Returns a label image (-1 transparent, K_INK = ink) indexed like NAMES."""
    a = np.array(img.convert("RGBA"))
    names = list(BASE)
    regions = regions or {}
    cols = np.array([hex2rgb(BASE[n]) for n in names], np.float64)
    op = a[..., 3] >= 128
    ys, xs = np.where(op)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = a[y0:y1, x0:x1]
    sop = op[y0:y1, x0:x1]
    lab = rgb2lab(sub[sop][:, :3].astype(np.float64))
    t = thr[y0:y1, x0:x1][sop] if isinstance(thr, np.ndarray) else thr
    plab = rgb2lab(cols)
    v = nearest(lab, plab)
    free = [i for i, n in enumerate(names) if n not in regions]
    vf = np.array(free)[nearest(lab, plab[free])]
    bad = np.zeros(len(v), bool)
    for n, m in regions.items():
        bad |= (v == names.index(n)) & ~m[y0:y1, x0:x1][sop]
    v[bad] = vf[bad]
    v = np.where(lab[:, 0] < t, K_INK, v)
    lbl = np.full(sop.shape, 255, np.uint8)
    lbl[sop] = v
    if mode:
        f = np.array(Image.fromarray(lbl, "L").filter(ImageFilter.ModeFilter(mode)))
        keep = (lbl == 255) | (lbl == K_INK) | (f == 255) | (f == K_INK)
        lbl = np.where(keep, lbl, f)
    full = np.full(op.shape, -1, np.int16)
    full[y0:y1, x0:x1] = np.where(lbl == 255, -1, lbl.astype(np.int16))
    return full


def render(lbl, pal):
    """Label image -> flat RGBA image."""
    cols = np.array([hex2rgb(pal[n]) for n in NAMES] + [hex2rgb(INK)], np.uint8)
    out = np.zeros(lbl.shape + (4,), np.uint8)
    m = lbl >= 0
    out[..., :3][m] = cols[lbl[m]]
    out[..., 3][m] = 255
    return Image.fromarray(out, "RGBA")


def thicken_ink(lbl, size):
    if size <= 1:
        return lbl
    ink = Image.fromarray(((lbl == K_INK) * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(size))
    out = lbl.copy()
    out[(np.array(ink) > 0) & (lbl >= 0)] = K_INK
    return out


def relabel(lbl, src, dst, mask=None):
    """Change label(s) `src` (name or list of names) to `dst` inside mask."""
    src = [src] if isinstance(src, str) else src
    m = np.isin(lbl, [L(s) for s in src])
    if mask is not None:
        m &= mask
    lbl[m] = -1 if dst is None else L(dst)


# ---------------------------------------------------------- label canvas ----
class Layer:
    """Draw additions straight into palette labels (no antialias blends)."""

    def __init__(self):
        self.im = Image.new("L", (gk.ART, gk.ART), 0)
        self.d = ImageDraw.Draw(self.im)

    @staticmethod
    def code(name):
        return L(name) + 1

    def poly(self, pts, name):
        if len(pts) < 3:
            return
        self.d.polygon([tuple(map(float, q)) for q in pts], fill=self.code(name))

    def line(self, pts, w, name, closed=False):
        pts = [tuple(map(float, q)) for q in pts]
        if closed:
            pts = pts + [pts[0], pts[1]]
        c = self.code(name)
        self.d.line(pts, fill=c, width=max(1, int(round(w))), joint="curve")
        r = w / 2
        if not closed:
            for x, y in (pts[0], pts[-1]):
                self.d.ellipse([x - r, y - r, x + r, y + r], fill=c)

    def ellipse(self, cx, cy, rx, ry, name):
        self.d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=self.code(name))

    def parts(self, parts, ow):
        """Outline of every part first, then the fills by depth, so joined parts merge without seams."""
        for layers in parts:
            self.line(layers[0][0], 2 * ow, "ink", closed=True)
            self.poly(layers[0][0], "ink")
        for depth in range(max(len(p) for p in parts)):
            for layers in parts:
                if depth < len(layers):
                    self.poly(layers[depth][0], layers[depth][1])

    def labels(self):
        return np.array(self.im).astype(np.int16) - 1


def over(lbl, layer):
    m = layer >= 0
    lbl[m] = layer[m]


def under(lbl, layer):
    m = (lbl < 0) & (layer >= 0)
    lbl[m] = layer[m]


# ------------------------------------------------------------ shapes ----
LIGHT = np.array([-0.55, -0.83])


def strip(pts, wfun, a=-1.0, b=1.0, t0=0.0, t1=1.0, cap=False):
    """Polygon of the band [a, b] (-1..1 across the width) of a variable-width stroke, for t in [t0, t1].
    cap=True rounds the far end (full-width bands only)."""
    pts = np.asarray(pts, np.float64)
    n = len(pts)
    left, right = [], []
    nrm = np.array([1.0, 0.0])
    for i, q in enumerate(pts):
        t = i / (n - 1)
        if t < t0 - 1e-9 or t > t1 + 1e-9:
            continue
        tan = pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]
        nrm = np.array([-tan[1], tan[0]]) / (np.hypot(*tan) or 1)
        hw = wfun(t) / 2
        left.append(tuple(q + nrm * hw * a))
        right.append(tuple(q + nrm * hw * b))
    arc = []
    if cap:
        q = pts[-1]
        hw = wfun(1.0) / 2
        fwd = np.array([nrm[1], -nrm[0]])
        for ang in np.linspace(0, np.pi, 9)[1:-1]:
            arc.append(tuple(q + nrm * hw * np.cos(ang) * b + fwd * hw * np.sin(ang)))
    return left + arc + right[::-1]


def light_side(pts):
    pts = np.asarray(pts, np.float64)
    tan = pts[-1] - pts[0]
    nrm = np.array([-tan[1], tan[0]]) / (np.hypot(*tan) or 1)
    return 1.0 if nrm @ LIGHT > 0 else -1.0


def taper(w0, w1, p=1.0):
    return lambda t: w0 + (w1 - w0) * (t ** p)


def flat_tube(pts, wfun, base, tip=None, tip_from=0.62):
    """Flat-filled tube (shaded later at sprite level); `tip` colours the far end."""
    lay = [(strip(pts, wfun, cap=True), base)]
    if tip:
        lay.append((strip(pts, wfun, t0=tip_from, cap=True), tip))
    return lay


def curve(pts, n=64):
    return dk.bezier(pts, n)


def at(pts, t):
    pts = np.asarray(pts)
    i = min(len(pts) - 1, int(round(t * (len(pts) - 1))))
    return tuple(pts[i])


# -------------------------------------------------------------- antlers ----
ANTLER = {
    # beam: local control points (out, up); tines: (t on beam, local control points, w0, w1)
    2: {"beam": [(0, -22), (16, 80), (80, 150), (88, 265)], "w": (64, 24),
        "tines": [(0.22, [(0, 0), (-22, 20), (-40, 45), (-42, 80)], 34, 16),
                  (0.50, [(0, 0), (42, 10), (80, 36), (98, 84)], 38, 17),
                  (0.74, [(0, 0), (-10, 34), (-16, 70), (-8, 102)], 32, 15)]},
    3: {"beam": [(0, -22), (20, 90), (100, 170), (110, 330)], "w": (74, 26),
        "tines": [(0.20, [(0, 0), (-26, 22), (-48, 52), (-50, 94)], 40, 18),
                  (0.42, [(0, 0), (50, 10), (95, 40), (120, 98)], 44, 18),
                  (0.62, [(0, 0), (-12, 40), (-20, 82), (-10, 120)], 38, 16),
                  (0.80, [(0, 0), (36, 12), (60, 40), (70, 82)], 32, 15)]},
}


def antler(base, side, s, spec, cols, tip=None):
    """Parts of one antler. side=-1 grows to the viewer's left, +1 to the right; s scales it.
    Returns (parts, centre lines [beam, tine...])."""
    bx, by = base

    def to(o, u, ox=bx, oy=by):
        return ox + side * o * s, oy - u * s

    beam = curve([to(o, u) for o, u in spec["beam"]])
    w0, w1 = spec["w"]
    parts = [flat_tube(beam, taper(w0 * s, w1 * s, 0.8), cols, tip=tip, tip_from=0.8)]
    lines = [beam]
    for t, ctrl, a, b in spec["tines"]:
        ox, oy = at(beam, t)
        pts = curve([to(o, u, ox, oy) for o, u in ctrl])
        parts.append(flat_tube(pts, taper(a * s, b * s, 0.8), cols, tip=tip))
        lines.append(pts)
    return parts, lines


ERASE_L = [(240, 100), (535, 100), (535, 290), (518, 300), (516, 352), (462, 352), (455, 338), (430, 322),
           (400, 310), (360, 300), (300, 296), (240, 280)]
ERASE_R = [(590, 100), (830, 100), (830, 250), (790, 268), (740, 288), (702, 308), (688, 332), (672, 350),
           (650, 346), (630, 322), (600, 302), (590, 290)]


def erase_old_antlers(stage, lbl):
    """Remove the original antlers above the head (everything inside the antler regions)."""
    lbl[ref_poly(stage, ERASE_L) | ref_poly(stage, ERASE_R)] = -1


# --------------------------------------------------------------- compose ----
GROW = {1: 3, 2: 3, 3: 1}
OW = {1: 6, 2: 6, 3: 5}      # outline width of additions (art px)


GEM_BOX = (552, 662, 640, 772)
FACE_BOX = (575, 505, 675, 585)
EYE_BOXES = [(455, 435, 572, 552), (645, 432, 712, 540)]
ANT_BASE = [(455, 300), (530, 290), (530, 368), (455, 368)]


def base_labels(stage, img):
    """Flatten the original illustration to the base palette with per-region ink thresholds."""
    thr = np.full((gk.ART, gk.ART), 26.0)
    hooves = np.zeros((gk.ART, gk.ART), bool)
    for b in HOOF_BOXES:
        hooves |= box_mask(stage, b)
    thr[hooves] = 12.0
    scarf = ref_poly(stage, SCARF_POLY)
    thr[scarf] = 12.0
    ears = ref_poly(stage, EAR_POLYS[0]) | ref_poly(stage, EAR_POLYS[1])
    ants = ref_poly(stage, ERASE_L) | ref_poly(stage, ERASE_R) | ref_poly(stage, ANT_BASE)
    eyes = box_mask(stage, EYE_BOXES[0]) | box_mask(stage, EYE_BOXES[1])
    regions = {"pink": ears, "gem_d": box_mask(stage, GEM_BOX), "nose": box_mask(stage, FACE_BOX)}
    regions["gem"] = regions["gem_l"] = regions["gem_d"]
    regions["mouth"] = regions["nose"]
    for n in ("ant_d", "ant", "ant_l", "ant_hi"):
        regions[n] = ants
    for n in ("blue_dd", "blue_d", "blue", "blue_l", "blue_hi"):
        regions[n] = scarf
    for n in ("iris_d", "iris", "iris_l"):
        regions[n] = eyes
    for n in ("hoof_d", "hoof", "hoof_l"):
        regions[n] = hooves
    lbl = flatten(img, thr, mode=5, regions=regions)
    lbl = thicken_ink(lbl, GROW[stage])
    relabel(lbl, ["ant_d", "ant_l", "ant_hi"], "ant", ants)
    return lbl


# ------------------------------------------------------------ additions ----
def ribbon(pts, w, cols, trim=None, notch=0.9, twist=0.5):
    """Scarf ribbon with a twist and a swallow-tail end: [(polygon, colour), ...]."""
    wf = lambda t: w * (1 - 0.5 * np.exp(-((t - twist) / 0.08) ** 2)) * (0.9 + 0.25 * t)
    s = light_side(pts)
    poly = strip(pts, wf)
    half = len(poly) // 2
    end = np.asarray(pts[-1])
    back = np.asarray(pts[-6])
    v = (back - end) / (np.hypot(*(back - end)) or 1)
    notch_pt = tuple(end + v * w * notch)
    poly = poly[:half] + [notch_pt] + poly[half:]
    base, dark, light = cols
    lay = [(poly, base)]
    for sg, t0, t1 in ((s, 0.0, twist), (-s, twist, 0.9)):
        lay.append((strip(pts, wf, *sorted((-sg * 1.0, -sg * 0.3)), t0=t0, t1=t1), dark))
        lay.append((strip(pts, wf, *sorted((sg * 0.2, sg * 0.55)), t0=t0 + 0.06, t1=t1 - 0.06), light))
        if trim:
            lay.append((strip(pts, wf, *sorted((-sg * 1.0, -sg * 0.6)), t0=t0, t1=t1), trim[0]))
            lay.append((strip(pts, wf, *sorted((-sg * 0.82, -sg * 0.68)), t0=t0 + 0.04, t1=t1 - 0.04), trim[1]))
    return lay


def leaf(base, ang, length, width):
    """Leaf: [(polygon, colour)...], pointing at `ang` degrees (0 = right, 90 = up)."""
    a = np.radians(ang)
    d = np.array([np.cos(a), -np.sin(a)])
    nrm = np.array([-d[1], d[0]])
    b = np.asarray(base, np.float64)
    pts = [tuple(b + d * length * t + nrm * length * 0.1 * np.sin(np.pi * t)) for t in np.linspace(0, 1, 24)]
    wf = lambda t: max(3.0, width * np.sin(np.pi * min(1.0, 0.08 + t)) ** 0.6)
    return [(strip(pts, wf), "leaf"), (strip(pts, wf, -1.0, -0.05), "leaf_d"),
            (strip(pts, wf, 0.2, 0.6, t0=0.2, t1=0.7), "leaf_l")]


def lock(base, tip, bend, w):
    """Flame-like lock of mane/tail from base to tip, bowed sideways by `bend` (fraction of length)."""
    bx, by = base
    tx, ty = tip
    dx, dy = tx - bx, ty - by
    nx, ny = -dy * bend, dx * bend
    pts = curve([base, (bx + dx * 0.4 + nx, by + dy * 0.4 + ny), (bx + dx * 0.75 + nx * 0.6, by + dy * 0.75 + ny * 0.6),
                 tip])
    wf = lambda t: max(6.0, w * np.sin(np.pi * (0.3 + 0.7 * t)) ** 0.7)
    return pts, wf


def flame_parts(pts, wf, cols, light=(0.15, 0.8), dark=0.45, tip=None):
    """Cel-shaded lock; tip=(colour, colour2) paints the far end (glowing tips)."""
    s = light_side(pts)
    d, m, l, h = cols
    lay = [(strip(pts, wf, cap=True), m),
           (strip(pts, wf, *sorted((-s * 1.0, -s * (1 - dark)))), d),
           (strip(pts, wf, *sorted((s * light[0], s * light[1])), t0=0.12, t1=0.92), l)]
    if h:
        lay.append((strip(pts, wf, *sorted((s * 0.35, s * 0.65)), t0=0.3, t1=0.85), h))
    if tip:
        lay.append((strip(pts, wf, t0=0.62, cap=True), tip[0]))
        lay.append((strip(pts, wf, *sorted((-s * 1.0, -s * 0.2)), t0=0.66), tip[1]))
    return lay


# buds on the stage 2 antlers: (antler side, centre line index, t, angle for the left antler, length)
LEAVES = [(-1, 0, 0.40, 150, 80), (-1, 0, 0.45, 60, 66), (-1, 2, 0.55, 105, 64),
          (1, 0, 0.42, 150, 74), (1, 0, 0.47, 62, 58), (1, 2, 0.55, 105, 60)]
BLOSSOMS = [(-1, 3, 0.78), (1, 1, 0.85)]
# aura ruff tufts (stage 3): base, tip, bend, width (reference px)
RUFF = [((560, 684), (556, 792), 0.06, 54), ((596, 686), (612, 810), -0.05, 58), ((628, 682), (668, 798), -0.12, 56),
        ((652, 668), (720, 764), -0.18, 52), ((662, 650), (744, 706), -0.2, 44)]
SCARF_NAMES = ("blue_dd", "blue_d", "blue", "blue_l", "blue_hi")
ANT_POS = {2: {"l": ((490, 350), 1.12), "r": ((652, 340), 1.02)},
           3: {"l": ((490, 350), 1.0), "r": ((652, 340), 0.92)}}


def head_patch(stage, lbl, ow):
    """Rebuild the head top where the original near antler sat: fur below a new ink contour."""
    top = curve([P(stage, *q) for q in [(444, 348), (466, 322), (492, 305), (522, 298)]])
    lay = Layer()
    lay.poly(top + [P(stage, 524, 372), P(stage, 444, 372)], "fur")
    lay.line(top, 2 * ow + 2, "ink")
    m = lay.labels()
    hole = ref_poly(stage, ERASE_L)
    sel = (m >= 0) & (hole | (lbl < 0) | (m == K_INK))
    sel &= ref_poly(stage, [(436, 290), (530, 290), (530, 375), (436, 375)])
    lbl[sel] = m[sel]


# stage 3 cape (reference px): attached at the scarf knot, streaming back and up, three trailing points
CAPE_TOP = [(455, 598), (380, 528), (280, 498), (180, 468), (92, 392)]
CAPE_TRAIL = [(92, 392), (152, 486), (86, 530), (160, 590), (110, 650), (240, 676)]
CAPE_BOTTOM = [(240, 676), (350, 694), (472, 668)]
CAPE_FOLDS = [((152, 486), (112, 440)), ((160, 590), (118, 560)), ((240, 676), (170, 662))]
CAPE_STARS = [(215, 515), (160, 560), (290, 545), (240, 610), (360, 585), (180, 640), (130, 470)]


def cape(stage, layer, ow):
    """Royal-blue starry cape with gold trim; drawn into `layer` (outline ring + clipped interior)."""
    top = np.array(curve([P(stage, *q) for q in CAPE_TOP], 48))
    tan = np.gradient(top, axis=0)
    nrm = np.stack([-tan[:, 1], tan[:, 0]], 1) / np.maximum(1e-6, np.hypot(tan[:, 0], tan[:, 1]))[:, None]
    tt = np.linspace(0, 1, len(top))
    top = [tuple(q) for q in top + nrm * (16 * np.sin(np.pi * 2.2 * tt) * np.sin(np.pi * tt))[:, None]]
    trail = [P(stage, *q) for q in CAPE_TRAIL]
    bottom = curve([P(stage, *q) for q in CAPE_BOTTOM], 24)
    poly = top + trail[1:] + bottom[1:]
    t = Layer()
    t.poly(poly, "blue")
    root = P(stage, 470, 640)
    for a, b in CAPE_FOLDS:
        t.poly([root, P(stage, *a), P(stage, *b)], "blue_d")
    for a, b in CAPE_FOLDS[:2]:
        pa, pb = np.array(P(stage, *a)), np.array(P(stage, *b))
        t.poly([root, tuple(pb), tuple(pb + (pa - pb) * 0.25)], "blue_l")
    edge = top + trail[1:]
    t.line(edge, 2 * ow + 22, "gold")
    t.line(edge, 2 * ow + 8, "gold_l")
    ring = Layer()
    ring.line(poly + [poly[0]], 2 * ow, "ink", closed=True)
    rl = ring.labels()
    inside = poly_mask(poly) & (rl < 0)
    tl = t.labels()
    out = rl.copy()
    out[inside & (tl >= 0)] = tl[inside & (tl >= 0)]
    m = out >= 0
    layer_arr = np.array(layer.im)
    layer_arr[m] = out[m] + 1
    layer.im = Image.fromarray(layer_arr, "L")
    layer.d = ImageDraw.Draw(layer.im)


def compose(stage, img, ctx):
    img = stretch_legs(img, stage)
    lbl = base_labels(stage, img)
    ow = OW[stage]
    st = {"lbl": lbl}
    _STATE[stage] = st
    if stage == 1:
        return render(lbl, palette(stage))

    erase_old_antlers(stage, lbl)
    spec = ANTLER[stage]
    cols = "ant"
    tip = "ant_hi" if stage == 3 else None
    (bl, sl), (br, sr) = ANT_POS[stage]["l"], ANT_POS[stage]["r"]
    pl, ll = antler(P(stage, *bl), -1, sl, spec, cols, tip=tip)
    pr, lr = antler(P(stage, *br), +1, sr, spec, cols, tip=tip)
    st["tips"] = [ln[-1] for ln in ll + lr]
    st["lines"] = {-1: ll, 1: lr}
    back, front = Layer(), Layer()

    # --- behind the body: scarf ribbons, tail plume, far antler
    blue = ("blue", "blue_d", "blue_l")
    if stage == 2:
        rib = [ribbon(curve([P(stage, *q) for q in [(436, 652), (350, 650), (300, 720), (205, 680)]]), 46, blue,
                      trim=("gold", "gold_l"), twist=0.55),
               ribbon(curve([P(stage, *q) for q in [(440, 630), (350, 548), (290, 620), (190, 545)]]), 54, blue,
                      trim=("gold", "gold_l"))]
        back.parts(rib, ow)
    else:
        cape(stage, back, ow)
    if stage == 3:
        plume = [flame_parts(*lock(P(stage, *b), P(stage, *t), bd, w), ("glow", "white", "white", None))
                 for b, t, bd, w in [((340, 700), (262, 548), -0.2, 76), ((335, 720), (215, 628), -0.14, 70),
                                     ((335, 735), (226, 712), 0.05, 56)]]
        back.parts(plume, ow)
    back.parts(pr, ow)
    back.parts(pl, ow)
    head_patch(stage, lbl, ow)

    # --- stage 2 buds (lộc non) on both antlers
    if stage == 2:
        for side, li, t, ang, ln in LEAVES:
            q = at(st["lines"][side][li], t)
            a0 = ang if side < 0 else 180 - ang
            (front if side < 0 else back).parts([leaf(q, a0, ln, 34)], ow)
        st["blossoms"] = [at(st["lines"][side][li], t) for side, li, t in BLOSSOMS]

    under(lbl, back.labels())
    over(lbl, front.labels())

    # --- scarf: gold trim along the lower edge of the band
    trim = Layer()
    trim.line(curve([P(stage, *q) for q in [(468, 646), (530, 668), (605, 680), (652, 650)]]),
              18 if stage == 2 else 22, "gold")
    trim.line(curve([P(stage, *q) for q in [(470, 642), (530, 663), (605, 674), (650, 645)]]), 7, "gold_l")
    tl_ = trim.labels()
    scarf = np.isin(lbl, [L(n) for n in SCARF_NAMES])
    lbl[scarf & (tl_ >= 0)] = tl_[scarf & (tl_ >= 0)]

    if stage == 3:
        # aura ruff of light on the chest, tucked under the scarf
        ruff = Layer()
        ruff.parts([flame_parts(*lock(P(stage, *b), P(stage, *t), bd, w), ("cream", "white", "white", None),
                                dark=0.4, tip=("glow", "gold_l")) for b, t, bd, w in RUFF], ow)
        rl = ruff.labels()
        keep = np.isin(lbl, [L(n) for n in SCARF_NAMES + ("gold", "gold_l")])
        m = (rl >= 0) & ~keep
        lbl[m] = rl[m]
        # golden hooves
        hooves = np.zeros(lbl.shape, bool)
        for b in HOOF_BOXES:
            hooves |= box_mask(stage, b)
        relabel(lbl, "hoof_d", "gold_d", hooves)
        relabel(lbl, "hoof", "gold", hooves)
        relabel(lbl, "hoof_l", "gold_l", hooves)
    return render(lbl, palette(stage))


# --------------------------------------------------------------- repaint ----
def repaint(stage, art, ctx):
    """Rebuild the sprite from the curated label image: majority label per sprite pixel, ink -> outline."""
    pal = palette(stage)
    lbl = _STATE[stage]["lbl"]
    cfg = STAGES[stage - 1]["config"]
    covs = np.zeros((K_INK + 1, gk.CANVAS, gk.CANVAS))
    for k in np.unique(lbl):
        if k < 0:
            continue
        m = lbl == k
        covs[k] = gk._place((m * 255).astype(np.uint8)[..., None], ctx, Image.BOX)[..., 0] / 255
    opaque = covs.sum(0) >= 0.5
    ink = covs[K_INK] >= cfg["ink_frac"]
    col = covs[:K_INK].argmax(0)
    idx = np.full((gk.CANVAS, gk.CANVAS), TRANSPARENT, int)
    idx[opaque] = col[opaque]
    idx[opaque & ink] = K_INK
    art.idx[...] = idx
    art.palette[:] = [hex2rgb(pal[n]) for n in NAMES] + [hex2rgb(INK)]
    art.names.clear()
    art.names.update({n: i for i, n in enumerate(NAMES)})
    art.names["outline"] = K_INK


def _n4(m, fill=False):
    p = np.pad(m, 1, constant_values=fill)
    return [p[0:-2, 1:-1], p[2:, 1:-1], p[1:-1, 0:-2], p[1:-1, 2:]]


def adj4(m):
    return np.logical_or.reduce(_n4(m))


def outline_pass(art):
    """Outer 1px outline ring where colour meets transparency, then peel doubled outer outline."""
    k = art.names["outline"]
    idx = art.idx
    tr = idx == TRANSPARENT
    fill = ~tr & (idx != k)
    idx[tr & adj4(fill)] = k
    for _ in range(2):
        tr = idx == TRANSPARENT
        ol = idx == k
        fill = ~tr & ~ol
        drop = ol & adj4(tr) & ~adj4(fill)
        idx[drop] = TRANSPARENT


def despeckle(art, min_area=3, protect=()):
    """Fold small colour islands into their most common colour neighbour (outline only as a last resort)."""
    k = art.names["outline"]
    idx = art.idx
    h, w = idx.shape
    flat = idx.ravel()
    keep = {k} | {art.color(p) for p in protect}
    for c, region in list(components(idx.copy())):
        if len(region) >= min_area or c in keep:
            continue
        ys, xs = np.divmod(region, w)
        nb = []
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            yy, xx = ys + dy, xs + dx
            ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
            nb.append(flat[yy[ok] * w + xx[ok]])
        nb = np.concatenate(nb)
        nb = nb[(nb != c) & (nb != TRANSPARENT)]
        fill_nb = nb[nb != k]
        use = fill_nb if len(fill_nb) else nb
        if len(use):
            vals, cnt = np.unique(use, return_counts=True)
            flat[region] = vals[cnt.argmax()]


def peel(art, passes=2):
    k = art.names["outline"]
    idx = art.idx
    for _ in range(passes):
        tr = idx == TRANSPARENT
        ol = idx == k
        fill = ~tr & ~ol
        drop = ol & adj4(tr) & ~adj4(fill)
        if not drop.any():
            break
        idx[drop] = TRANSPARENT


# ------------------------------------------------------------ pixel art ----
EYES = {"l": (511, 494, 49, 51), "r": (680, 487, 25, 47)}       # ref centre, radii (outline included)
LID = {1: 0.0, 2: 0.22, 3: 0.34}                              # top-lid slant towards the nose per stage


def sp(stage, ctx, x, y):
    """Reference poster px -> sprite px (float) for this stage."""
    ax, ay = P(stage, x, y)
    return ax * ctx.scale + ctx.ox, ay * ctx.scale + ctx.oy


def eye_mask(stage, ctx, key):
    """Pixels of one eye (inside its outline) and its normalised coordinates."""
    x, y, rx, ry = EYES[key]
    cx, cy = sp(stage, ctx, x, y)
    rx, ry = rx * ctx.scale, ry * ctx.scale
    yy, xx = np.mgrid[0:gk.CANVAS, 0:gk.CANVAS]
    u = (xx + 0.5 - cx) / rx
    v = (yy + 0.5 - cy) / ry
    inner = u if key == "l" else -u                     # +1 towards the nose
    lid = -1.0 + LID[stage] * (1 + inner) / 2
    m = (u ** 2 + v ** 2 <= 1.0)
    return m, m & (v >= lid), u, v, inner, (cx, cy, rx, ry)


EYE_JUNK = ("iris_d", "iris", "iris_l", "white", "gold", "gold_l")


def clean_eye(stage, art, ctx, key):
    """Turn leftover eye pixels (ink, iris, highlights) around an eye into skin before (re)painting it."""
    full, m, u, v, inner, (cx, cy, rx, ry) = eye_mask(stage, ctx, key)
    ex = 1.16 if key == "l" else 1.0
    big = (u / ex) ** 2 + (v / 1.14) ** 2 <= 1.0
    junk = np.isin(art.idx, [art.names["outline"]] + [art.color(n) for n in EYE_JUNK])
    art.idx[big & junk & ~full] = art.color("cream")
    art.idx[full] = art.color("cream")
    return cx, cy, rx, ry


def lash_pixels(stage, cx, cy, rx, ry):
    x0 = int(np.floor(cx - rx))
    y0 = int(np.floor(cy - 0.35 * ry))
    px = [(x0 - 1, y0), (x0 - 1, y0 - 1)]
    if stage >= 2:
        px.append((x0 - 2, y0 - 1))
    return px


def paint_eye(stage, art, ctx, key):
    clean_eye(stage, art, ctx, key)
    full, m, u, v, inner, (cx, cy, rx, ry) = eye_mask(stage, ctx, key)
    idx = art.idx
    k = art.names["outline"]
    # lid area cut away by the slant becomes skin
    idx[full & ~m] = art.color("cream")
    er = m.copy()
    er[1:-1, 1:-1] = m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
    ring = m & ~er
    # thicker upper lid (lash line)
    lidline = er & ~np.roll(m, 2, axis=0) & (v < 0)
    inside = er & ~lidline
    outer = -inner
    white = inside & (outer > 0.48 - 0.25 * v ** 2) & (v > -0.7)
    iris = inside & ~white
    cols = {"d": "iris_d", "a": "iris", "l": "iris_l"}
    if stage == 3:
        cols = {"d": "iris_d", "a": "gold", "l": "gold_l"}
    idx[ring | lidline] = k
    idx[white] = art.color("white")
    idx[iris] = art.color(cols["d"])
    # iris centre (a bit towards the nose)
    ic = 0.12 * (1 if key == "l" else -1)
    ui = u - ic
    amber = iris & (v > 0.05)
    idx[amber] = art.color(cols["a"])
    idx[iris & (v > 0.45) & (np.abs(ui) < 0.45)] = art.color(cols["l"])
    pupil = iris & ((ui / 0.5) ** 2 + ((v + 0.12) / 0.62) ** 2 <= 1.0)
    idx[pupil] = k
    # highlights: a block at the top of the pupil (towards the light), a dot low on the other side
    pcx, pcy = cx + ic * rx, cy - 0.12 * ry
    hs = (2, 2) if stage < 3 else (3, 2)
    if key == "r":
        hs = (1, 2) if stage < 3 else (2, 2)
    hx = int(np.floor(pcx + 0.2 * rx)) - (hs[0] - 1) // 2
    hy = int(np.floor(pcy - 0.42 * ry))
    for dy in range(hs[1]):
        for dx in range(hs[0]):
            if inside[hy + dy, hx + dx]:
                idx[hy + dy, hx + dx] = art.color("white")
    lx = int(np.floor(pcx - 0.3 * rx))
    ly = int(np.floor(cy + 0.36 * ry))
    if iris[ly, lx] and stage >= 2 and key == "l":
        idx[ly, lx] = art.color("white" if stage == 3 else cols["l"])
    # lash flick at the outer top corner of the big eye
    if key == "l":
        for x, y in lash_pixels(stage, cx, cy, rx, ry):
            art.put(x, y, k)


FONT = {
    "I": ["#", "#", "#", "#", "#"],
    "Q": [".#.", "#.#", "#.#", "#.#", ".##"],
    "X": ["#.#", "#.#", ".#.", "#.#", "#.#"],
}
FONT_BIG = {
    "I": ["###", ".#.", ".#.", ".#.", ".#.", "###"],
    "Q": [".##.", "#..#", "#..#", "#..#", ".##.", "...#"],
    "X": ["#..#", "#..#", ".##.", ".##.", "#..#", "#..#"],
}
TEXT_AT = {1: (101, 147), 2: (102, 133), 3: (103, 121)}       # top-left of "IQX" (sprite px)


def paint_text(stage, art):
    font = FONT if stage < 3 else FONT_BIG
    x, y = TEXT_AT[stage]
    w = sum(len(font[c][0]) for c in "IQX") + 2
    h = len(font["I"])
    # clear the original lettering to plain scarf blue first
    for n in ("blue_hi", "blue_l", "white", "cream", "cream_d"):
        art.replace(n, "blue", (x - 1, y - 1, x + w, y + h - 1))
    for ch in "IQX":
        rows = font[ch]
        art.blit(x, y, rows, {"#": "white"})
        x += len(rows[0]) + 1


def paint_gem(stage, art, cx, cy, a, b, frame=False):
    """Faceted diamond pendant centred on (cx, cy), half sizes a x b (sprite px)."""
    k = art.names["outline"]
    for y in range(int(cy - b) - 1, int(cy + b) + 2):
        for x in range(int(cx - a) - 1, int(cx + a) + 2):
            dx, dy = x + 0.5 - cx, y + 0.5 - cy
            d = abs(dx) / (a + 0.5) + abs(dy) / (b + 0.5)
            if d > 1.0:
                continue
            edge = d > 1.0 - 1.15 / min(a, b) * 1.0
            if edge:
                art.put(x, y, k)
                continue
            if frame and d > 1.0 - 2.3 / min(a, b):
                art.put(x, y, "gold_l" if dy < 0 else "gold_d")
                continue
            if dy < 0:
                c = "gem_l" if dx < 0 else "gem"
            else:
                c = "gem" if dx < 0 else "gem_d"
            art.put(x, y, c)
    # sparkle on the upper-left facet
    hx, hy = int(np.floor(cx - a * 0.35)), int(np.floor(cy - b * 0.35))
    art.put(hx, hy, "white")
    if frame:
        art.put(hx, hy + 1, "white")
        art.put(hx + 1, hy, "gem_l")


def star(art, x, y, big=False):
    """Four-point star sparkle: white core, glow arms."""
    art.put(x, y, "white")
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        art.put(x + dx, y + dy, "glow")
    if big:
        for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            art.put(x + dx, y + dy, "gold_l")


def paint_orb(art, x, y):
    """Glowing orb at an antler tip: outlined 7x7 disc with a white core and sparkle rays."""
    k = art.names["outline"]
    rows = ["..###..",
            ".#lll#.",
            "#lgggl#",
            "#lgwgl#",
            "#lgggl#",
            ".#lll#.",
            "..###.."]
    art.blit(x - 3, y - 3, rows, {"#": k, "l": "gold_l", "g": "glow", "w": "white"})
    art.put(x - 1, y - 1, "white")


def paint_third_eye(art, cx, cy):
    """Thiên Nhãn: a vertical gem-eye on the forehead, gold lids, IQX-blue iris, slit pupil."""
    rows = ["...#...",
            "..#l#..",
            ".#lgl#.",
            "#gbBbg#",
            "#bBkwb#",
            "#bBkBb#",
            "#bbkBb#",
            "#gbbbg#",
            ".#ggg#.",
            "..#g#..",
            "...#..."]
    k = art.names["outline"]
    key = {"#": k, "g": "gold", "l": "gold_l", "b": "blue", "B": "blue_l", "k": k, "w": "white"}
    art.blit(cx - 3, cy - 5, rows, key)


def paint_blossom(art, x, y):
    """Five-petal blossom (hoa đào) for the stage 2 antlers."""
    rows = [".##.##.",
            "#pp#pp#",
            "#pdcdp#",
            ".#ccc#.",
            "#pdcdp#",
            "#pp#pp#",
            ".##.##."]
    art.blit(x - 3, y - 3, rows, {"#": art.names["outline"], "p": "petal", "d": "petal_d", "c": "gold_l"})


SPARKLES = [(300, 90), (868, 300), (150, 330)]
SPOTS = [(395, 693), (412, 711), (372, 717), (333, 738), (392, 730), (352, 700), (430, 735), (447, 705), (318, 715)]


def paint_spots(stage, art, ctx):
    fur = {art.color(n) for n in ("fur_dd", "fur_d", "fur", "fur_l", "white", "cream")}
    for i, (x, y) in enumerate(SPOTS):
        fx, fy = sp(stage, ctx, x, y)
        px, py = int(fx), int(fy)
        if art.idx[py, px] not in fur:
            continue
        if stage == 1:
            if i in (0, 1, 2, 3, 6):
                art.put(px, py, "white")
        elif stage == 2:
            for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
                art.put(px + dx, py + dy, "glow")
            art.put(px, py, "white")
        else:
            star(art, px, py, big=(i % 3 == 0))


def shade_material(art, mat, dark, light, spec=None, solid=()):
    """Rim-shade a flat material: shadow on lower-right edges, light on upper-left, specular corners.
    `solid` names colours that count as the same object (no rim against them)."""
    idx = art.idx
    m = idx == art.color(mat)
    same = m.copy()
    for n in solid:
        same |= idx == art.color(n)
    p = np.pad(same, 2, constant_values=False)
    H, W = idx.shape

    def nb(dx, dy):
        return p[2 + dy:2 + dy + H, 2 + dx:2 + dx + W]

    shadow = m & (~nb(1, 0) | ~nb(0, 1) | ~nb(1, 1))
    lit = m & (~nb(-1, 0) | ~nb(0, -1) | ~nb(-1, -1)) & ~shadow
    wide_shadow = m & ~shadow & ~lit & (~nb(2, 1) | ~nb(1, 2) | ~nb(2, 2)) & nb(-2, 0) & nb(0, -2)
    idx[shadow | wide_shadow] = art.color(dark)
    idx[lit] = art.color(light)
    if spec:
        corner = m & ~nb(-1, 0) & ~nb(0, -1) & nb(1, 1)
        idx[corner] = art.color(spec)


MOUTH = {
    1: ["..nnn..",
        "...#...",
        "#.###.#",
        ".#ttt#.",
        "..###.."],
    2: ["...nnn...",
        "....#....",
        "....#....",
        "#.#####.#",
        ".##ttt##.",
        "..#ttt#..",
        "...###..."],
    3: ["....nnn....",
        ".....#.....",
        ".....#.....",
        "#..#####..#",
        ".##rrrrr##.",
        "..#rtttr#..",
        "...#ttt#...",
        "....###...."],
}
NOSE = (634, 524)      # reference px, nose centre


def paint_mouth(stage, art, ctx):
    rows = MOUTH[stage]
    nx, ny = sp(stage, ctx, *NOSE)
    col = rows[0].index("n") + 1
    x0, y0 = int(np.floor(nx)) - col, int(np.floor(ny)) - 1
    w, h = len(rows[0]), len(rows)
    k = art.names["outline"]
    for n in (k, "nose", "mouth", "pink", "cream_d"):
        art.replace(n, "white", (x0 - 2, y0, x0 + w + 1, y0 + h + 3))
    art.blit(x0, y0, rows, {"n": "nose", "#": k, "t": "mouth", "r": "nose"})


def to_px(ctx, q):
    x, y = ctx.to_sprite(*q)
    return x, y


# --------------------------------------------------------------- touchup ----
# --------------------------------------------------------------- touchup ----
def touchup(stage, art, ctx):
    repaint(stage, art, ctx)
    outline_pass(art)
    despeckle(art, 3)
    peel(art)
    st = _STATE[stage]
    if stage >= 2:
        if stage == 3:
            shade_material(art, "ant_hi", "ant_l", "white", solid=("ant",))
            shade_material(art, "ant", "ant_d", "ant_l", spec="ant_hi", solid=("ant_hi", "ant_l", "white"))
        else:
            shade_material(art, "ant", "ant_d", "ant_l", spec="ant_hi")
    else:
        shade_material(art, "ant", "ant_d", "ant_l", spec="ant_hi")
    for key in ("l", "r"):
        paint_eye(stage, art, ctx, key)
    paint_mouth(stage, art, ctx)
    paint_text(stage, art)
    gx, gy = sp(stage, ctx, 595, 716)
    if stage == 1:
        paint_gem(stage, art, gx, gy, 4.0, 5.6)
    elif stage == 2:
        paint_gem(stage, art, gx, gy + 1.5, 4.8, 6.8)
    else:
        paint_gem(stage, art, gx, gy + 2.5, 6.2, 8.6, frame=True)
    art.replace("blue_hi", "blue_l")
    art.replace("hoof_l", "hoof")
    paint_spots(stage, art, ctx)
    if stage == 2:
        for q in st.get("blossoms", []):
            paint_blossom(art, *to_px(ctx, q))
    if stage == 3:
        for i, q in enumerate(st["tips"]):
            x, y = to_px(ctx, q)
            paint_orb(art, x, y)
        for x, y in CAPE_STARS:
            fx, fy = sp(stage, ctx, x, y)
            px, py = int(fx), int(fy)
            if art.idx[py, px] in (art.color("blue"), art.color("blue_d"), art.color("blue_l")):
                star(art, px, py, big=False)
        fx, fy = sp(stage, ctx, 582, 352)
        paint_third_eye(art, int(fx), int(fy))
        # floating motes of light around the antlers and cape
        for x, y in SPARKLES:
            fx, fy = sp(stage, ctx, x, y)
            if (art.idx[int(fy) - 2:int(fy) + 3, int(fx) - 2:int(fx) + 3] == TRANSPARENT).all():
                star(art, int(fx), int(fy), big=True)


def blink(stage, art, ctx):
    """Close both eyes: skin over the eye, a soft lower-lid arc with a lash flick (the third eye stays open)."""
    k = art.names["outline"]
    for key in ("l", "r"):
        cx, cy, rx, ry = clean_eye(stage, art, ctx, key)
        if key == "l":
            for x, y in lash_pixels(stage, cx, cy, rx, ry):
                art.put(x, y, "cream")
        x0, x1 = int(np.floor(cx - rx * 0.92)), int(np.floor(cx + rx * 0.92))
        pts = []
        for x in range(x0, x1 + 1):
            uu = (x + 0.5 - cx) / rx
            y = cy + ry * (0.12 + 0.28 * (1 - min(1.0, uu * uu)))
            pts.append((x, int(np.floor(y))))
        for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
            art.line(xa, ya, xb, yb, k)
        if stage == 3:
            # heavier lash line for the big eye
            for x, y in pts[1:-1]:
                if key == "l":
                    art.put(x, y - 1, k) if abs(x + 0.5 - cx) < rx * 0.55 else None
        if key == "l":
            xe, ye = pts[0]
            art.put(xe - 1, ye - 1, k)
            if stage >= 2:
                art.put(xe - 2, ye - 2, k)
        else:
            xe, ye = pts[-1]
            art.put(xe + 1, ye - 1, k)
