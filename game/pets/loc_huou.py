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
     "desc": "Thần thú đội vương miện gạc vàng đầu pha lê, khoác áo choàng sao trời. Thiên Nhãn giữa trán nhìn "
             "thấu mọi giao dịch nội bộ trước khi thị trường kịp hay biết, và con mắt ấy không bao giờ khép lại.",
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
    "glow": "#fff1a8", "star": "#ffffff",
    "aqua": "#8ee8ff", "aqua_d": "#3d9ee6",
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


def spline(pts, n=64):
    """Catmull-Rom curve through every point (beams that must bend where they are told to)."""
    p = np.asarray(pts, np.float64)
    p = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    seg = len(p) - 3
    out = []
    for i in range(seg):
        p0, p1, p2, p3 = p[i:i + 4]
        for t in np.linspace(0, 1, max(2, n // seg), endpoint=(i == seg - 1)):
            t2, t3 = t * t, t * t * t
            q = 0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (3 * p1 - p0 - 3 * p2 + p3) * t3)
            out.append((float(q[0]), float(q[1])))
    return out


def at(pts, t):
    pts = np.asarray(pts)
    i = min(len(pts) - 1, int(round(t * (len(pts) - 1))))
    return tuple(pts[i])


# -------------------------------------------------------------- antlers ----
ANTLER = {
    # beam: points (out, up) the beam passes through; w: (base, tip) width;
    # tines: (t on beam, bezier control points, w0, w1)
    # Stage 2: a young stag's lyre: the beam bows out and turns back up, a brow tine, an outer and an inner tine.
    2: {"spline": 1, "beam": [(0, -22), (18, 62), (72, 122), (134, 170), (156, 240), (154, 322)], "w": (70, 18),
        "tines": [(0.15, [(0, 0), (-30, 18), (-54, 46), (-58, 86)], 42, 12),
                  (0.47, [(0, 0), (44, 10), (86, 38), (112, 92)], 44, 12),
                  (0.77, [(0, 0), (-30, 16), (-56, 42), (-66, 80)], 36, 11)]},
    # Stage 3: a wide crown; the beam sweeps far out and curls up, tines fan from inward to outward along it.
    3: {"spline": 1, "beam": [(0, -22), (24, 60), (92, 128), (178, 176), (238, 246), (252, 330)], "w": (84, 20),
        "tines": [(0.12, [(0, 0), (-32, 22), (-56, 54), (-58, 98)], 48, 13),
                  (0.36, [(0, 0), (-8, 44), (-20, 90), (-34, 136)], 46, 13),
                  (0.56, [(0, 0), (0, 44), (0, 90), (-10, 134)], 42, 12),
                  (0.76, [(0, 0), (40, 22), (72, 54), (88, 100)], 38, 11)]},
}


def antler(base, side, s, spec, cols, tip=None, tip_from=0.7):
    """Parts of one antler. side=-1 grows to the viewer's left, +1 to the right; s scales it.
    Returns (parts, centre lines [beam, tine...])."""
    bx, by = base

    def to(o, u, ox=bx, oy=by):
        return ox + side * o * s, oy - u * s

    beam = (spline if spec.get("spline") else curve)([to(o, u) for o, u in spec["beam"]])
    w0, w1 = spec["w"]
    parts = [flat_tube(beam, taper(w0 * s, w1 * s, 1.15), cols, tip=tip, tip_from=0.82)]
    lines = [beam]
    for t, ctrl, a, b in spec["tines"]:
        ox, oy = at(beam, t)
        pts = curve([to(o, u, ox, oy) for o, u in ctrl])
        parts.append(flat_tube(pts, taper(a * s, b * s, 1.2), cols, tip=tip, tip_from=tip_from))
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
def streamer(pts, w0, w1, trim=True, notch=0.85):
    """Clean wind-blown scarf tail: light band on the lit side, shadow + gold edge on the other, swallow-tail end."""
    wf = taper(w0, w1, 1.0)
    s = light_side(pts)
    poly = strip(pts, wf)
    half = len(poly) // 2
    end = np.asarray(pts[-1])
    back = np.asarray(pts[-8])
    v = (back - end) / (np.hypot(*(back - end)) or 1)
    poly = poly[:half] + [tuple(end + v * w1 * notch)] + poly[half:]
    lay = [(poly, "blue"),
           (strip(pts, wf, *sorted((-s * 1.0, -s * 0.35)), t0=0.0, t1=0.97), "blue_d"),
           (strip(pts, wf, *sorted((s * 0.15, s * 0.6)), t0=0.08, t1=0.9), "blue_l")]
    if trim:
        lay.append((strip(pts, wf, *sorted((-s * 1.0, -s * 0.62)), t0=0.0, t1=0.96), "gold"))
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


def flame_parts(pts, wf, cols, light=(0.15, 0.8), dark=0.45, tip=None, tip_from=0.62):
    """Cel-shaded lock; tip=(colour, colour2) paints the far end (glowing tips)."""
    s = light_side(pts)
    d, m, l, h = cols
    lay = [(strip(pts, wf, cap=True), m),
           (strip(pts, wf, *sorted((-s * 1.0, -s * (1 - dark)))), d),
           (strip(pts, wf, *sorted((s * light[0], s * light[1])), t0=0.12, t1=0.92), l)]
    if h:
        lay.append((strip(pts, wf, *sorted((s * 0.35, s * 0.65)), t0=0.3, t1=0.85), h))
    if tip:
        lay.append((strip(pts, wf, t0=tip_from, cap=True), tip[0]))
        lay.append((strip(pts, wf, *sorted((-s * 1.0, -s * 0.2)), t0=tip_from + 0.04), tip[1]))
    return lay


# buds on the stage 2 antlers (lộc non): (centre line index, t, angle for the left antler, length)
LEAVES = [(0, 0.55, 26, 92), (0, 0.61, 66, 80)]
# chest mane tufts (stage 3): base, tip, bend, width (reference px)
RUFF = [((556, 684), (548, 772), 0.05, 44), ((580, 688), (578, 794), 0.03, 48), ((606, 690), (612, 808), 0.0, 50),
        ((632, 688), (648, 800), -0.04, 48), ((656, 680), (684, 782), -0.08, 44), ((664, 664), (712, 750), -0.12, 44),
        ((650, 648), (696, 690), -0.22, 40)]
SCARF_NAMES = ("blue_dd", "blue_d", "blue", "blue_l", "blue_hi")
ANT_POS = {2: {"l": ((508, 344), 1.2), "r": ((640, 338), 1.08)},     # bases kept clear of the ear valleys
           3: {"l": ((508, 344), 1.0), "r": ((640, 336), 0.92)}}


HEAD_TOP = [(446, 352), (468, 326), (494, 308), (526, 300)]      # new head contour where the near antler sat
EAR_TOP = [(402, 314), (426, 330), (446, 352)]                   # top edge of the left ear, up to that contour


def head_patch(stage, lbl, ow):
    """Rebuild the head top where the original near antler sat: fur below a clean 1px-ish contour that
    continues the left ear's top edge (the original's ear/antler gap otherwise turns into an ink blob)."""
    top = spline([P(stage, *q) for q in HEAD_TOP], 32)
    ear = spline([P(stage, *q) for q in EAR_TOP], 16)
    edge = ear + top[1:]
    lay = Layer()
    lay.poly(edge + [P(stage, 530, 384), P(stage, 402, 384)], "fur")
    lay.line(edge, 2 * ow - 1, "ink")
    m = lay.labels()
    hole = ref_poly(stage, ERASE_L)
    # inside the erased antler: always; elsewhere only the band right under the new contour (ear shading stays)
    band = Layer()
    band.line(edge, 2 * ow + 34, "fur")
    near = (band.labels() >= 0) & ((lbl < 0) | (lbl == K_INK) | (m == K_INK))
    box = ref_poly(stage, [(398, 290), (532, 290), (532, 384), (398, 384)])
    sel = (m >= 0) & (hole | near) & box
    lbl[sel] = m[sel]
    # old ink just above the new contour (the original ear/antler gap) opens up so the antler behind shows
    lbl[(band.labels() >= 0) & (m < 0) & (lbl == K_INK) & box] = -1


# stage 2 scarf tails (reference px, spline points) and the top of the deer's own tail (ribbons pass in front)
RIBBONS = [[(450, 612), (395, 552), (330, 508), (262, 528), (198, 552), (136, 500)],
           [(450, 636), (390, 616), (325, 596), (262, 622), (206, 656), (148, 632)]]
TAIL_TOP = [(236, 590), (300, 566), (362, 572), (394, 600), (394, 654), (236, 654)]

# the original bow's flap (stage 3: the cape covers it)
BOW_AREA = [(362, 606), (398, 588), (472, 588), (484, 640), (470, 658), (400, 650), (362, 624)]
CLASP = (468, 632)

# stage 3 cape (reference px): attached at the scarf knot, streaming back and up, three trailing points
CAPE_TOP = [(470, 604), (404, 556), (322, 518), (236, 494), (156, 460), (102, 428)]
CAPE_TRAIL = [(102, 428), (98, 458), (116, 482), (112, 512), (92, 536), (126, 566), (156, 588), (134, 618),
              (122, 646), (164, 662), (204, 672), (230, 690)]
CAPE_BOTTOM = [(230, 690), (340, 702), (474, 672)]
CAPE_FOLDS = [(116, 482), (156, 588), (204, 672)]           # notches: dark folds run from the clasp to them
CAPE_STARS = [(210, 520, 1), (150, 500, 0), (300, 552, 0), (236, 600, 1), (176, 632, 0), (380, 600, 0)]


def cape(stage, layer, ow):
    """Royal-blue starry cape with gold trim, billowing back from the clasp; drawn into `layer`."""
    top = spline([P(stage, *q) for q in CAPE_TOP], 48)
    trail = spline([P(stage, *q) for q in CAPE_TRAIL], 64)
    bottom = curve([P(stage, *q) for q in CAPE_BOTTOM], 24)
    poly = top + trail[1:] + bottom[1:]
    t = Layer()
    t.poly(poly, "blue")
    root = np.array(P(stage, *CLASP))
    for r in CAPE_FOLDS:
        # a fold: a long dark wedge from the clasp, widest where it meets the notch in the trailing edge
        b = np.array(P(stage, *r))
        d = b - root
        n = np.array([-d[1], d[0]]) / (np.hypot(*d) or 1)
        t.poly([tuple(root), tuple(b + n * 26 - d * 0.06), tuple(b - n * 18)], "blue_d")
    edge = top + trail[1:]
    t.line(top, 2 * ow + 24 + 34, "blue_l")
    t.line(edge, 2 * ow + 24, "gold")
    t.line(edge, 2 * ow + 10, "gold_l")
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


# far hind leg (seen between the near legs): its dark-brown shading turns muddy once the legs stretch
FAR_LEG = [(412, 772), (486, 772), (486, 892), (412, 892)]
# scarf knot on the left of the neck (reference px): upper flap, hanging end, knot
KNOT_FLAP = [(478, 626), (450, 612), (420, 604), (392, 604), (370, 612)]
KNOT_END = [(472, 640), (472, 660), (468, 676), (462, 690)]
KNOT_CLEAR = [(400, 646), (468, 646), (480, 700), (400, 700)]   # where the original hanging end was


def far_leg(stage, lbl):
    relabel(lbl, "fur_dd", "fur_d", ref_poly(stage, FAR_LEG))


def scarf_knot(stage, lbl, ow):
    """Redraw the knot as clean cel-shaded shapes over the original (its many ink folds turn to mud when small)."""
    blue = ("blue_d", "blue", "blue_l", None)
    old = ref_poly(stage, KNOT_CLEAR)
    relabel(lbl, list(SCARF_NAMES) + ["ink"], "fur", old)
    relabel(lbl, "fur_dd", "fur_d", old)          # the old knot's cast shadow
    flap = curve([P(stage, *q) for q in KNOT_FLAP], 32)
    end = curve([P(stage, *q) for q in KNOT_END], 24)
    parts = [flame_parts(end, lambda t: 40 - 12 * t, blue, light=(0.1, 0.7), dark=0.4),
             flame_parts(flap, lambda t: 66 - 58 * t ** 1.4, blue, light=(0.25, 0.85), dark=0.45),
             [(dk_ellipse(P(stage, 470, 630), 24, 30), "blue"),
              (dk_ellipse(P(stage, 462, 640), 14, 18), "blue_d")]]
    lay = Layer()
    lay.parts(parts, ow)
    over(lbl, lay.labels())


def dk_ellipse(c, rx, ry, n=28):
    cx, cy = c
    return [(cx + rx * np.cos(a), cy + ry * np.sin(a)) for a in np.linspace(0, 2 * np.pi, n, endpoint=False)]


def clean_stage1_antlers(lbl):
    """Inside the original antlers every non-ink label becomes plain antler gold (shaded later per pixel)."""
    m = ref_poly(1, ERASE_L) | ref_poly(1, ERASE_R)
    m &= ~ref_poly(1, [(505, 290), (560, 290), (560, 360), (512, 360)])   # forehead right of the near antler
    m &= (lbl >= 0) & (lbl != K_INK)
    lbl[m] = L("ant")


def compose(stage, img, ctx):
    img = stretch_legs(img, stage)
    lbl = base_labels(stage, img)
    ow = OW[stage]
    st = {"lbl": lbl}
    _STATE[stage] = st
    far_leg(stage, lbl)
    if stage == 1:
        clean_stage1_antlers(lbl)
        scarf_knot(stage, lbl, ow)
        return render(lbl, palette(stage))

    erase_old_antlers(stage, lbl)
    spec = ANTLER[stage]
    cols = "ant"
    tip = "aqua" if stage == 3 else None
    (bl, sl), (br, sr) = ANT_POS[stage]["l"], ANT_POS[stage]["r"]
    pl, ll = antler(P(stage, *bl), -1, sl, spec, cols, tip=tip)
    pr, lr = antler(P(stage, *br), +1, sr, spec, cols, tip=tip)
    st["tips"] = [ln[-1] for ln in ll + lr]
    st["lines"] = {-1: ll, 1: lr}
    back, front = Layer(), Layer()

    # --- behind the body: scarf tails (stage 2) or cape (stage 3), then both antlers
    blue = ("blue", "blue_d", "blue_l")
    if stage == 2:
        # two long scarf tails stream back from under the knot, above the back and in front of the deer's tail
        rib = [streamer(spline([P(stage, *q) for q in RIBBONS[0]], 72), 58, 50),
               streamer(spline([P(stage, *q) for q in RIBBONS[1]], 72), 54, 46)]
        rl = Layer()
        rl.parts(rib[::-1], ow)
        rl = rl.labels()
        front_of_tail = ref_poly(stage, TAIL_TOP) & (rl >= 0)
        lbl[front_of_tail] = rl[front_of_tail]
        under(lbl, rl)
    else:
        capel = Layer()
        cape(stage, capel, ow)
        cl = capel.labels()
        # the cape is fastened by a clasp (painted per pixel): it covers the old bow, and sits behind the body
        bow = ref_poly(stage, BOW_AREA) & (cl >= 0)
        lbl[bow] = cl[bow]
        under(lbl, cl)
    back.parts(pr, ow)
    back.parts(pl, ow)
    head_patch(stage, lbl, ow)

    # --- stage 2 buds (lộc non) on both antlers
    if stage == 2:
        for side in (-1, 1):
            sc = ANT_POS[stage]["l" if side < 0 else "r"][1]
            for li, t, ang, ln in LEAVES:
                q = at(st["lines"][side][li], t)
                back.parts([leaf(q, ang if side < 0 else 180 - ang, ln * sc, 30 * sc)], ow)

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
    if stage == 2:
        scarf_knot(stage, lbl, ow)
    else:
        relabel(lbl, list(SCARF_NAMES) + ["ink", "fur_dd"], "fur", ref_poly(stage, KNOT_CLEAR))

    if stage == 3:
        # a fluffy chest mane with glowing tips, tucked under the scarf: fur fills only, no ink, so it grows out of
        # the chest fluff instead of sitting on it (touchup still outlines it where it meets the background)
        ruff = Layer()
        tufts = [flame_parts(*lock(P(stage, *b), P(stage, *t), bd, w), ("cream", "white", "white", None),
                             dark=0.38, tip=("glow", "gold_l"), tip_from=0.8) for b, t, bd, w in RUFF]
        for depth in range(max(len(p) for p in tufts)):
            for p in tufts:
                if depth < len(p):
                    ruff.poly(*p[depth])
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
    # the original's dark under-eye shading leaves brown specks on the cheek
    cheek = ((u / 1.2) ** 2 + (v / 1.35) ** 2 <= 1.0) & (v > 0.35) & (np.abs(u) < 1.05) & ~full
    art.idx[cheek & np.isin(art.idx, [art.color(n) for n in ("fur_dd", "fur_d")])] = art.color("cream")
    art.idx[cheek & (inner > -0.5) & (art.idx == art.color("fur"))] = art.color("cream")
    art.idx[full] = art.color("cream")
    return cx, cy, rx, ry


def separate(art, family, p0, p1):
    """Inside the box p0..p1 (sprite px), fur pixels touching a `family` colour (e.g. the cape blues) become outline,
    so a cloth laid over the body keeps a 1px edge."""
    x0, y0 = int(p0[0]), int(p0[1])
    x1, y1 = int(p1[0]) + 1, int(p1[1]) + 1
    idx = art.idx
    cloth = np.zeros(idx.shape, bool)
    for n in [m for m in NAMES if m.startswith(family)] + ["gold", "gold_l"]:
        cloth |= idx == art.color(n)
    fur = np.isin(idx, [art.color(n) for n in ("fur_dd", "fur_d", "fur", "fur_l", "cream_d", "cream", "white")])
    hit = fur & adj4(cloth)
    box = np.zeros(idx.shape, bool)
    box[y0:y1, x0:x1] = True
    idx[hit & box] = art.names["outline"]


def lone_pixels(art, box, names=("fur", "fur_d", "fur_dd", "fur_l", "cream_d", "white", "cream")):
    """A pixel whose four neighbours all share another colour takes that colour (inside box, listed colours)."""
    x0, y0, x1, y1 = box
    idx = art.idx
    cand = {art.color(n) for n in names}
    for y in range(max(1, y0), min(idx.shape[0] - 1, y1)):
        for x in range(max(1, x0), min(idx.shape[1] - 1, x1)):
            c = idx[y, x]
            if c not in cand:
                continue
            nb = {idx[y - 1, x], idx[y + 1, x], idx[y, x - 1], idx[y, x + 1]}
            if len(nb) == 1:
                n = nb.pop()
                if n != c and n != TRANSPARENT and n != art.names["outline"]:
                    idx[y, x] = n


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
    "Q": [".##.", "#..#", "#..#", "#.#.", ".#.#"],
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


def paint_third_eye(art, cx, cy):
    """Thiên Nhãn: a vertical eye on the forehead in a gold diamond frame, IQX-blue iris, slit pupil."""
    rows = [".....#.....",
            "....#l#....",
            "...#lgd#...",
            "..#g###d#..",
            ".#g#bBb#d#.",
            "#g#bBwBb#d#",
            "#l#bBkBb#d#",
            "#g#bBkBb#d#",
            ".#g#bkb#d#.",
            "..#g###d#..",
            "...#gdd#...",
            "....#d#....",
            ".....#....."]
    k = art.names["outline"]
    key = {"#": k, "g": "gold", "l": "gold_l", "d": "gold_d", "b": "blue", "B": "blue_l", "k": k, "w": "white"}
    art.blit(cx - 5, cy - 6, rows, key)


def paint_clasp(art, cx, cy):
    """Stage 3 cape clasp: a round gold brooch with an IQX-blue cabochon."""
    rows = ["..#####..",
            ".#lllgg#.",
            "#llbbbgd#",
            "#lbwBBbd#",
            "#lbBBBbd#",
            "#gbBBBbd#",
            "#ggbbbdd#",
            ".#gdddd#.",
            "..#####.."]
    k = art.names["outline"]
    art.blit(cx - 4, cy - 4, rows, {"#": k, "l": "gold_l", "g": "gold", "d": "gold_d", "b": "blue",
                                     "B": "blue_l", "w": "white"})


SPARKLES = [(4, -7, -6), (9, 7, -8)]      # (antler tip index, dx, dy) in sprite px
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
        elif i in (0, 3, 6):
            star(art, px, py)
        elif i in (1, 4, 8):
            art.put(px, py, "white")


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
    3: ["...nnn...",          # calm, closed 'ω' smile: the divine form no longer grins
        "....#....",
        "....#....",
        "#..#.#..#",
        ".##...##."],
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
        art.replace(n, "white", (x0 - 2, y0, x0 + max(w, 11) + 1, y0 + max(h, 8) + 3))
    art.blit(x0, y0, rows, {"n": "nose", "#": k, "t": "mouth", "r": "nose"})


def to_px(ctx, q):
    x, y = ctx.to_sprite(*q)
    return x, y


# --------------------------------------------------------------- touchup ----
def touchup(stage, art, ctx):
    repaint(stage, art, ctx)
    outline_pass(art)
    despeckle(art, 3)
    peel(art)
    lone_pixels(art, (0, 0, gk.CANVAS, gk.CANVAS))
    st = _STATE[stage]
    if stage == 3:
        shade_material(art, "aqua", "aqua_d", "white", solid=("ant",))
        shade_material(art, "ant", "ant_d", "ant_l", spec="ant_hi", solid=("aqua", "aqua_d", "white"))
    else:
        shade_material(art, "ant", "ant_d", "ant_l", spec="ant_hi")
    for key in ("l", "r"):
        paint_eye(stage, art, ctx, key)
    paint_mouth(stage, art, ctx)
    fx0, fy0 = sp(stage, ctx, 430, 420)
    fx1, fy1 = sp(stage, ctx, 740, 600)
    lone_pixels(art, (int(fx0), int(fy0), int(fx1) + 1, int(fy1) + 1))
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
        pass
    if stage == 3:
        for x, y, big in CAPE_STARS:
            fx, fy = sp(stage, ctx, x, y)
            px, py = int(fx), int(fy)
            near = art.idx[py - 1:py + 2, px - 1:px + 2]
            if np.isin(near, [art.color("blue"), art.color("blue_d"), art.color("blue_l")]).all():
                if big:
                    star(art, px, py)
                else:
                    art.put(px, py, "white")
        fx, fy = sp(stage, ctx, 582, 352)
        # the original's white forehead blaze would crowd the third eye: soften it to light fur
        art.replace("white", "fur_l", (int(fx) - 8, int(fy) - 6, int(fx) + 8, int(fy) + 10))
        paint_third_eye(art, int(fx), int(fy) + 2)
        separate(art, "blue", sp(stage, ctx, 360, 580), sp(stage, ctx, 490, 680))
        fx, fy = sp(stage, ctx, *CLASP)
        paint_clasp(art, int(fx), int(fy))
        # twinkles of light off the two outermost crystal tips
        for i, dx, dy in SPARKLES:
            x, y = to_px(ctx, st["tips"][i])
            x, y = x + dx, max(2, y + dy)
            if (art.idx[y - 2:y + 3, x - 2:x + 3] == TRANSPARENT).all():
                star(art, x, y, big=True)


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
    fx0, fy0 = sp(stage, ctx, 430, 420)
    fx1, fy1 = sp(stage, ctx, 740, 600)
    lone_pixels(art, (int(fx0), int(fy0), int(fx1) + 1, int(fy1) + 1))
