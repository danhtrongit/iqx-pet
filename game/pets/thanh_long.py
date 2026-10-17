"""Thanh Long: evolution line for the game sprites (see tools/gamekit.py)."""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageFilter

_TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

import drawkit as dk  # noqa: E402
import gamekit as gk  # noqa: E402
from pixelize import TRANSPARENT, hex2rgb, merge_islands, nearest, rgb2lab  # noqa: E402

PET = {
    "name": "Thanh Long",
    "layer": "Dòng tiền",
    "element": "flow_cyan",
    "accent": "#22d3ee",
    "tagline": "Theo dấu dòng tiền lớn chảy vào thị trường.",
}

CFG = {"colors": 30, "ink_l": 24.0, "ink_frac": 0.4, "smooth": 0, "min_island": 1, "outline": "#0b1b52"}

STAGES = [
    {"rank": "Ấu thể", "name": "Tiểu Thanh Long", "level": 1, "height": 104,
     "config": dict(CFG),
     "desc": "Rồng con nở ra từ dòng vốn nhỏ đầu tiên chảy vào thị trường. Chiếc khăn IQX giúp nó "
             "đánh hơi xem tiền đang chảy về đâu.",
     "skill": {"name": "Đánh Hơi Dòng Vốn",
               "desc": "Dò ra mã có dòng tiền vào bất thường trong phiên, tăng 10% tốc độ cho cả đội."},
     "stats": {"hp": 42, "atk": 36, "def": 30, "spd": 55}},
    {"rank": "Trưởng thành", "name": "Thanh Long Lưu Thủy", "level": 10, "height": 130,
     "config": dict(CFG),
     "desc": "Đôi sừng vàng nhú lên khi nó đọc được dòng tiền khối ngoại và tự doanh. Vây nước mọc dọc "
             "đuôi, cuộn sóng dẫn thanh khoản chảy theo ý mình.",
     "skill": {"name": "Xoáy Thủy Lưu",
               "desc": "Cuộn dòng tiền thành xoáy nước, hút thanh khoản của đối thủ và gây sát thương theo "
                       "khối lượng giao dịch."},
     "stats": {"hp": 64, "atk": 62, "def": 50, "spd": 76}},
    {"rank": "Thần thú", "name": "Thanh Long Vương", "level": 25, "height": 150, "dx": -4,
     "config": dict(CFG),
     "desc": "Vua của mọi dòng chảy: sừng vàng uy nghi, bờm rực lửa xanh. Long châu soi thấu dòng vốn lớn trước khi "
             "thị trường kịp phản ứng; rồng đến đâu, mây lành và thanh khoản hội tụ đến đó.",
     "skill": {"name": "Long Châu Hội Lưu",
               "desc": "Long châu gom mọi dòng vốn lớn thành cột nước thần, gây sát thương diện rộng và đánh "
                       "dấu điểm gom hàng của cá mập."},
     "stats": {"hp": 86, "atk": 90, "def": 72, "spd": 95}},
]

# --------------------------------------------------------------- palette ----
INK = "#0b1b52"
INK_HI = 32.0
SCARF_INK = 15.0      # the dark royal-blue scarf and its knot are colour, not line art
GROW = {1: 5, 2: 3, 3: 3}      # line-art thickening (MaxFilter size, art px) per stage
BASE = {
    "black": "#070a16",
    "teal_dd": "#0b7ea6", "teal_d": "#16a8c0", "teal": "#2ecdc9", "teal_l": "#5cdfc8",
    "cream": "#faf4e2", "cream_y": "#f0e1b6", "cream_s": "#d1c2a1",
    "blue_dd": "#0a3aa6", "blue_d": "#1258db", "blue": "#1587ec", "blue_l": "#25aaf3",
    "sky": "#88d7f5", "ice": "#d8f1f7",
    "cyan": "#22c6e6", "green": "#78dca6", "sea": "#4fa89a",
    "scarf_l": "#3d7ff0",
    "m_dark": "#3a0c14", "m_red": "#b9474a", "tongue": "#ec8a80",
    "white": "#ffffff",
}
EXTRA = {1: {}, 2: {}, 3: {}}


def palette(stage):
    return {**BASE, **EXTRA[stage]}


_STATE: dict = {}


def R(x, y):
    """reference poster px -> art canvas."""
    return x + gk.PAD, y + gk.PAD


# ------------------------------------------------------------ flattening ----
def flatten(img, pal, ink_l=24.0, mode=13, ink_hi=None, grow=0):
    """Snap every opaque art pixel to the curated palette (ink kept), mode-filter the labels.

    ink_hi: a higher ink threshold used everywhere except on the scarf (keeps thin line art).
    Returns (flat RGBA image, label image) where label -1 = transparent, len(pal) = ink.
    """
    a = np.array(img.convert("RGBA"))
    names = list(pal)
    cols = np.array([hex2rgb(pal[n]) for n in names], np.float64)
    op = a[..., 3] >= 128
    ys, xs = np.where(a[..., 3] > 0)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = a[y0:y1, x0:x1]
    sop = op[y0:y1, x0:x1]
    lab = rgb2lab(sub[sop][:, :3].astype(np.float64))
    lbl = np.full(sop.shape, 255, np.uint8)
    k_ink = len(names)
    thr = np.full(sop.shape, ink_l)
    if ink_hi is not None:
        thr[:] = ink_hi
        low = poly_mask(op.shape, [R(*q) for q in SCARF_POLY])[y0:y1, x0:x1]
        thr[low] = min(ink_l, SCARF_INK)
    v = np.where(lab[:, 0] < thr[sop], k_ink, nearest(lab, rgb2lab(cols)))
    lbl[sop] = v
    if grow:
        # thicken the line art so thin inner lines survive the downscale
        ink = Image.fromarray(((lbl == k_ink) * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(grow))
        lbl[(np.array(ink) > 0) & (lbl != 255)] = k_ink
    if mode:
        f = np.array(Image.fromarray(lbl, "L").filter(ImageFilter.ModeFilter(mode)))
        keep = (lbl == 255) | (lbl == k_ink) | (f == 255) | (f == k_ink)
        f = np.where(keep, lbl, f)
        lbl = f
    full = np.full(op.shape, -1, np.int16)
    full[y0:y1, x0:x1] = np.where(lbl == 255, -1, lbl.astype(np.int16))
    out = a.copy()
    m = (full >= 0) & (full < k_ink)
    out[..., :3][m] = cols[full[m]].astype(np.uint8)
    out[..., 3][full >= 0] = 255
    ink_rgb = hex2rgb(INK)
    out[..., :3][full == k_ink] = ink_rgb
    return Image.fromarray(out, "RGBA"), full


def mode_down(lbl, ctx, n):
    """Majority label of every sprite pixel's footprint (colour labels only, ink ignored)."""
    best = np.full((gk.CANVAS, gk.CANVAS), -1, int)
    bestv = np.zeros((gk.CANVAS, gk.CANVAS))
    for k in range(n):
        m = lbl == k
        if not m.any():
            continue
        f = gk._place((m * 255).astype(np.uint8)[..., None], ctx, Image.BOX)[..., 0]
        upd = f > bestv
        best[upd] = k
        bestv[upd] = f[upd]
    return best


def repaint(stage, art, ctx):
    """Replace the k-means colours of the sprite by the curated palette, picked by block majority."""
    pal = _STATE[stage]["pal"]
    names = list(pal)
    small = mode_down(_STATE[stage]["lbl"], ctx, len(names))
    k_out_old = art.names["outline"]
    outline_rgb = art.palette[k_out_old]
    new = np.full(art.idx.shape, TRANSPARENT, int)
    op = art.idx != TRANSPARENT
    ol = art.idx == k_out_old
    new[op & ~ol] = small[op & ~ol]
    k_out = len(names)
    new[op & (ol | (new < 0))] = k_out
    art.idx[...] = new
    art.palette[:] = [hex2rgb(pal[n]) for n in names] + [outline_rgb]
    art.names.clear()
    art.names.update({n: i for i, n in enumerate(names)})
    art.names["outline"] = k_out


def _n4(m, fill=False):
    p = np.pad(m, 1, constant_values=fill)
    return [p[0:-2, 1:-1], p[2:, 1:-1], p[1:-1, 0:-2], p[1:-1, 2:]]


def thin_outline(art, passes=4):
    """Peel outer outline pixels that only back other outline (2-3px rims and tip clumps -> 1px)."""
    k = art.names["outline"]
    for _ in range(passes):
        idx = art.idx
        ol = idx == k
        tr = idx == TRANSPARENT
        fill = ~ol & ~tr
        touch_t = np.logical_or.reduce(_n4(tr, True))
        touch_f = np.logical_or.reduce(_n4(fill))
        # an outline pixel is "deep" if it does not touch the outside, or it carries a colour edge
        deep = ol & (~touch_t | touch_f)
        backed = np.logical_or.reduce(_n4(deep))
        drop = ol & touch_t & ~touch_f & backed
        if not drop.any():
            break
        idx[drop] = TRANSPARENT
    prune_spurs(art)


def prune_spurs(art, n=4):
    """Remove dangling outline-only spurs (tips of tapered shapes) up to n px long."""
    k = art.names["outline"]
    for _ in range(n):
        idx = art.idx
        ol = idx == k
        op = idx != TRANSPARENT
        fill = op & ~ol
        touch_f = np.logical_or.reduce(_n4(fill))
        p = np.pad(op, 1)
        n8 = sum(p[1 + dy:p.shape[0] - 1 + dy, 1 + dx:p.shape[1] - 1 + dx].astype(int)
                 for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
        drop = ol & ~touch_f & (n8 <= 1)
        if not drop.any():
            break
        idx[drop] = TRANSPARENT


def smooth_fill(art, passes=1):
    """3x3 majority over colour pixels only: outline and transparency never change or vote."""
    k = art.names["outline"]
    n = len(art.palette)
    for _ in range(passes):
        idx = art.idx
        h, w = idx.shape
        p = np.pad(idx, 1, constant_values=TRANSPARENT)
        counts = np.zeros((n, h, w), np.int16)
        for dy in range(3):
            for dx in range(3):
                win = p[dy:dy + h, dx:dx + w]
                for c in range(n):
                    if c != k:
                        counts[c] += win == c
        best = counts.argmax(0)
        strong = counts.max(0) >= 5
        fill = (idx != TRANSPARENT) & (idx != k)
        idx[fill & strong] = best[fill & strong]


def seal_edges(art, box=None):
    """Colour pixels that touch the outside become outline, so the silhouette keeps its 1px line."""
    k = art.names["outline"]
    tr = art.idx == TRANSPARENT
    p = np.pad(tr, 1, constant_values=True)
    edge = ~tr & (p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]) & (art.idx != k)
    if box is not None:
        m = np.zeros_like(edge)
        x0, y0, x1, y1 = [int(v) for v in box]
        m[max(0, y0):y1 + 1, max(0, x0):x1 + 1] = True
        edge &= m
    art.idx[edge] = k


def drop_ink_specks(art, max_size=2):
    """Interior outline specks (8-connected outline blobs of <= max_size px that do not touch the
    outside) become the most common neighbouring fill colour."""
    k = art.names["outline"]
    idx = art.idx
    h, w = idx.shape
    ol = idx == k
    tr = idx == TRANSPARENT
    seen = np.zeros_like(ol)
    for y0, x0 in zip(*np.where(ol)):
        if seen[y0, x0]:
            continue
        stack, comp, outside = [(y0, x0)], [], False
        seen[y0, x0] = True
        while stack:
            y, x = stack.pop()
            comp.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if not (0 <= yy < h and 0 <= xx < w):
                        outside = True
                        continue
                    if tr[yy, xx] and abs(dy) + abs(dx) == 1:
                        outside = True
                    if ol[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
        if outside or len(comp) > max_size:
            continue
        for y, x in comp:
            nb = [idx[y + dy, x + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                  if (dy or dx) and 0 <= y + dy < h and 0 <= x + dx < w]
            nb = [v for v in nb if v != k and v != TRANSPARENT]
            if nb:
                vals, cnt = np.unique(nb, return_counts=True)
                idx[y, x] = vals[cnt.argmax()]


def despeckle(art, min_area=3, box=None):
    k = art.names["outline"]
    if box is None:
        art.idx[...] = merge_islands(art.idx, min_area, keep=(k,))
    else:
        x0, y0, x1, y1 = box
        art.idx[y0:y1 + 1, x0:x1 + 1] = merge_islands(art.idx[y0:y1 + 1, x0:x1 + 1], min_area, keep=(k,))


# ------------------------------------------------------------ drawing ----
OW = 5.0          # outline half-width in art px (a ~10px ink line -> 1 sprite px)


def wlin(w0, w1, bulge=0.0, p=1.0):
    return lambda t: w0 + (w1 - w0) * (t ** p) + bulge * np.sin(np.pi * t)


def strip(pts, wfun, a=-1.0, b=1.0, t0=0.0, t1=1.0):
    """Polygon of the band [a, b] (-1 = left edge, +1 = right edge) of a variable-width stroke,
    limited to the part t0..t1 of its length."""
    pts = np.array(pts, np.float64)
    n = len(pts)
    left, right = [], []
    for i, q in enumerate(pts):
        t = i / (n - 1)
        if t < t0 - 1e-9 or t > t1 + 1e-9:
            continue
        tan = pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]
        nrm = np.array([-tan[1], tan[0]]) / (np.hypot(*tan) or 1)
        hw = wfun(t) / 2
        left.append(tuple(q + nrm * hw * a))
        right.append(tuple(q + nrm * hw * b))
    return left + right[::-1]


def curve(*pts, n=40):
    return dk.bezier([R(*q) for q in pts], n)


def part(pts, wfun, cols, side=1, bands=None):
    """A cel-shaded stroke: list of (polygon, colour) layers, first one is the full silhouette."""
    if bands is None:
        bands = ((-0.3, 1.0), (0.35, 0.8)) if side > 0 else ((-1.0, 0.3), (-0.8, -0.35))
    layers = [(strip(pts, wfun), cols[0])]
    for (a, b), c in zip(bands, cols[1:]):
        layers.append((strip(pts, wfun, a, b), c))
    return layers


def draw_parts(img, parts, ow=OW, outline=INK):
    """Outline of every part first, then the fills, so joined parts merge without seams."""
    for layers in parts:
        poly = layers[0][0]
        dk.stroke(img, poly + [poly[0]], outline, ow * 2)
    for depth in range(max(len(p) for p in parts)):
        for layers in parts:
            if depth < len(layers):
                dk.shape(img, layers[depth][0], layers[depth][1])


def poly_mask(shape_hw, pts):
    from PIL import ImageDraw
    m = Image.new("L", (shape_hw[1], shape_hw[0]), 0)
    ImageDraw.Draw(m).polygon([tuple(q) for q in pts], fill=255)
    return np.array(m) > 127


def snap(img, pal):
    """Label image of an already-flat image (no mode filter): nearest palette colour, ink kept."""
    return flatten(img, pal, mode=0)


# --------------------------------------------------------------- compose ----
GOLD = {"gold_dd": "#7c3c0e", "gold_d": "#c4711a", "gold": "#f2aa26", "gold_l": "#ffd95a", "gold_hi": "#fff6c4"}
FLAME = {"flame": "#22d3ee", "aqua": "#8ef0fb", "foam": "#c6f8fd", "frost": "#ecfdff", "cloud": "#7fd0ee"}
EXTRA[2] = {k: GOLD[k] for k in ("gold_d", "gold", "gold_l")} | {k: FLAME[k] for k in ("aqua", "frost")}
EXTRA[3] = dict(GOLD) | dict(FLAME)

# reference-poster boxes to keep out of mane recolours
SCARF_POLY = [(470, 640), (520, 615), (600, 628), (700, 632), (770, 618), (790, 640), (775, 720), (700, 735),
              (600, 725), (540, 735), (470, 720)]
EYE_BOX = (560, 410, 840, 560)
GEM_BOX = (640, 690, 750, 820)


def mane_mask(img):
    """Blue hair (mane + tail tuft), not the scarf, eyes or gem."""
    m = dk.hsv_mask(img, h=(195, 232), s=(0.35, 1), v=(0.3, 1))
    a = np.array(img.convert("RGB"))
    ys, xs = np.where(m)
    dark = rgb2lab(a[ys, xs].astype(np.float64))[:, 0] < INK_HI
    m[ys[dark], xs[dark]] = False
    sh = m.shape
    m &= ~poly_mask(sh, [R(*q) for q in SCARF_POLY])
    for b in (EYE_BOX, GEM_BOX):
        x0, y0 = R(b[0], b[1])
        x1, y1 = R(b[2], b[3])
        m[int(y0):int(y1), int(x0):int(x1)] = False
    return m


TAIL_BOX = (60, 470, 390, 860)      # ref px box around the tail tuft


def flame_mask(img):
    """Mane + tail tuft including their pale highlight streaks (for the stage-3 recolour)."""
    m = mane_mask(img)
    region = np.array(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(31))) > 0
    pale = dk.hsv_mask(img, h=(170, 250), s=(0.0, 0.5), v=(0.55, 1)) & region
    # pure-white streaks of the tail tuft (no hue)
    tx0, ty0 = R(*TAIL_BOX[:2])
    tx1, ty1 = R(*TAIL_BOX[2:])
    white = dk.hsv_mask(img, s=(0.0, 0.05), v=(0.85, 1), box=(int(tx0), int(ty0), int(tx1), int(ty1))) & region
    pale |= white
    a = np.array(img.convert("RGB")).astype(np.float64)
    sh = pale.shape
    pale &= ~poly_mask(sh, [R(*q) for q in SCARF_POLY])
    for b in (EYE_BOX, GEM_BOX):
        x0, y0 = R(b[0], b[1])
        x1, y1 = R(b[2], b[3])
        pale[int(y0):int(y1), int(x0):int(x1)] = False
    return m | pale


def flame_recolour(img, mask, centre, r0, r1, steps, w_lum=0.45, tail_cap=0.9):
    """Cel-shade the mane like a blue flame: deep at the roots, bright at the tips.
    Mixes distance from `centre` (ref px, r0..r1) with the original luminance, then steps it."""
    a = np.array(img.convert("RGBA"))
    lum = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]) / 255
    lum = np.array(Image.fromarray((lum * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(6))) / 255
    cx, cy = R(*centre)
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    r = np.clip((np.hypot(xx - cx, yy - cy) - r0) / (r1 - r0), 0, 1)
    v = (1 - w_lum) * r + w_lum * np.clip((lum - 0.25) / 0.45, 0, 1)
    # the tail tuft tops out at aqua: no frost-white patch on it
    tx0, ty0 = R(*TAIL_BOX[:2])
    tx1, ty1 = R(*TAIL_BOX[2:])
    tail = np.zeros(mask.shape, bool)
    tail[int(ty0):int(ty1), int(tx0):int(tx1)] = True
    v = np.where(tail, np.minimum(v, tail_cap), v)
    out = a.copy()
    done = np.zeros(mask.shape, bool)
    for hi, col in steps:
        m = mask & ~done & (v < hi)
        out[..., :3][m] = hex2rgb(col)
        done |= m
    img.paste(Image.fromarray(out, "RGBA"))


# the tall mane lock of the original, used as a template for new locks (ref px polygon, base point)
LOCK_POLY = [(500, 125), (530, 110), (528, 180), (560, 228), (605, 262), (622, 322), (560, 334), (500, 302),
             (470, 282), (440, 230), (436, 175), (470, 135)]
LOCK_BASE = (560, 312)
TUFT_POLY = [(130, 600), (150, 540), (200, 515), (270, 512), (330, 545), (350, 610), (300, 640), (240, 600),
             (200, 600), (160, 625)]
TUFT_BASE = (300, 625)


def template(img, poly, base):
    m = poly_mask((img.height, img.width), [R(*q) for q in poly])
    ys, xs = np.where(m)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    t = img.crop((x0, y0, x1, y1))
    a = np.minimum(np.array(t)[..., 3], (m[y0:y1, x0:x1] * 255).astype(np.uint8))
    t.putalpha(Image.fromarray(a))
    bx, by = R(*base)
    return t, (bx - x0, by - y0)


def paste(layer, tpl, at, angle=0.0, scale=1.0, flip=False):
    """Composite a template rotated (degrees, counter-clockwise) and scaled about its base point onto layer."""
    t, (bx, by) = tpl
    if flip:
        t = t.transpose(Image.FLIP_LEFT_RIGHT)
        bx = t.width - bx
    r = int(np.ceil(max(np.hypot(bx, by), np.hypot(t.width - bx, by), np.hypot(bx, t.height - by),
                        np.hypot(t.width - bx, t.height - by)))) + 2
    sq = Image.new("RGBA", (2 * r, 2 * r), (0, 0, 0, 0))
    sq.alpha_composite(t, (int(r - bx), int(r - by)))
    sq = sq.rotate(angle, resample=Image.BICUBIC, center=(r, r))
    n = max(1, round(2 * r * scale))
    sq = sq.resize((n, n), Image.LANCZOS)
    ax, ay = R(*at)
    layer.alpha_composite(sq, (int(round(ax - n / 2)), int(round(ay - n / 2))))


TOES = [(324, 928), (354, 931), (380, 931), (542, 928), (580, 931), (620, 926), (760, 928), (797, 928)]


def whisker(img, pts, w0=18, w1=6):
    """Long flowing whisker continuing the original right whisker."""
    c = curve(*pts, n=40)
    draw_parts(img, [part(c, wlin(w0, w1), ("#4fa89a", "#78dca6"), side=-1, bands=((-1.0, 0.3),))], ow=4.5)


# water fin along the outer curve of the tail (ref px): ray roots inside the tail, ray tips outside
FIN_ROOTS = [(300, 800), (240, 780), (215, 715)]
FIN_TIPS = [(214, 900), (112, 822), (96, 700)]
FIN_BASE = [(330, 790), (300, 760), (240, 740), (215, 690)]   # inner edge (hidden under the tail)


def _tip(root, tip, grow):
    r = np.array(root, float)
    return tuple(r + (np.array(tip, float) - r) * grow)


def tail_fin(layer, cols, grow=1.0, sag=0.55):
    """Webbed water fin: membrane with a scalloped edge between rays sweeping back.
    cols = (membrane edge, membrane core, ray, ray highlight). Drawn on a layer that goes behind the body."""
    tips = [_tip(r, t, grow) for r, t in zip(FIN_ROOTS, FIN_TIPS)]
    root = (np.array(FIN_ROOTS[0]) + np.array(FIN_ROOTS[-1])) / 2

    def outline_poly(scale):
        tp = [tuple(root + (np.array(t) - root) * scale) for t in tips]
        outer = [R(*FIN_BASE[0])]
        outer += dk.bezier([R(*FIN_BASE[0]), R(*((np.array(FIN_BASE[0]) + np.array(tp[0])) / 2 + (20, -10))),
                            R(*tp[0])], 16)
        for a, b in zip(tp, tp[1:]):
            a, b = np.array(a), np.array(b)
            mid = (a + b) / 2
            ctrl = mid + (root - mid) * sag          # pull the scallop toward the tail
            outer += dk.bezier([R(*a), R(*ctrl), R(*b)], 20)
        outer += dk.bezier([R(*tp[-1]), R(*((np.array(tp[-1]) + np.array(FIN_BASE[-1])) / 2 + (0, -24))),
                            R(*FIN_BASE[-1])], 16)
        return outer + [R(*q) for q in FIN_BASE[::-1]]

    poly = outline_poly(1.0)
    dk.stroke(layer, poly + [poly[0]], INK, OW * 2)
    dk.shape(layer, poly, cols[0])
    dk.shape(layer, outline_poly(0.62), cols[1])
    for r, tip in zip(FIN_ROOTS, tips):
        c = curve(r, ((r[0] + tip[0]) / 2 + 6, (r[1] + tip[1]) / 2 + 10), tip, n=30)
        for poly_, col in part(c, wlin(36 * (0.7 + 0.3 * grow), 7, p=0.75), cols[2:], side=1,
                               bands=((0.25, 1.0),)):
            dk.shape(layer, poly_, col)


def stage2(img, ctx):
    pal = palette(2)
    lock = template(img, LOCK_POLY, LOCK_BASE)
    tuft = template(img, TUFT_POLY, TUFT_BASE)
    back = dk.blank(img)
    # longer, flowing mane: extra locks fanning out behind
    for at, ang, sc in (((452, 268), 22, 1.08), ((398, 352), 50, 1.12), ((372, 452), 78, 1.0)):
        paste(back, lock, at, ang, sc)
    # finned tail: a second wave crest rising behind the tuft
    paste(back, tuft, (262, 560), -28, 0.95)
    # "Luu Thuy": water fins sweeping back along the tail
    tail_fin(back, ("#88d7f5", "#25aaf3", "#1258db", "#1587ec"))
    img = dk.behind(img, back)
    flat, _ = flatten(img, pal, ink_hi=INK_HI, grow=GROW[2])
    g = (pal["gold_d"], pal["gold"], pal["gold_l"])
    # golden horns breaking through the mane, swept back
    far = curve((800, 352), (806, 300), (796, 250), (788, 200), n=36)
    draw_parts(flat, [part(far, wlin(44, 13), g)])
    horn = curve((702, 372), (680, 300), (630, 250), (600, 205), (598, 160), n=48)
    draw_parts(flat, [part(horn, wlin(62, 14), g)])
    whisker(flat, ((862, 574), (900, 596), (936, 592), (958, 570)), 19, 10)
    return flat


def stage3(img, ctx):
    pal = palette(3)
    lock = template(img, LOCK_POLY, LOCK_BASE)
    tuft = template(img, TUFT_POLY, TUFT_BASE)
    back = dk.blank(img)
    for at, ang, sc in (((470, 250), 14, 1.25), ((418, 318), 40, 1.32), ((385, 405), 64, 1.3), ((372, 500), 88, 1.1)):
        paste(back, lock, at, ang, sc)
    paste(back, tuft, (250, 560), -30, 1.15)
    paste(back, tuft, (215, 600), -62, 0.9)
    img = dk.behind(img, back)
    # mane and tail tuft burn like a blue flame: royal-blue roots, cyan body, frost-white tips
    flame_recolour(img, flame_mask(img), (600, 470), 150, 480,
                   [(0.36, "#1258db"), (0.60, "#1587ec"), (0.80, "#22d3ee"), (0.93, "#8ef0fb"), (1.01, "#ecfdff")])
    back = dk.blank(img)
    tail_fin(back, (FLAME["aqua"], FLAME["flame"], "#1258db", "#1587ec"), grow=1.3)
    img = dk.behind(img, back)
    flat, _ = flatten(img, pal, ink_hi=INK_HI, grow=GROW[3])

    g = (pal["gold_d"], pal["gold"], pal["gold_l"])
    back = dk.blank(flat)
    # far antler
    far = curve((792, 362), (803, 290), (792, 222), (765, 172), (758, 104), n=48)
    far_t = curve((799, 272), (830, 225), (858, 172), n=24)
    draw_parts(back, [part(far_t, wlin(54, 16), g), part(far, wlin(78, 20), g)])
    # flaming dragon pearl (the pearl itself is drawn as pixels in touchup)
    pearl_flames(back)
    flat = dk.behind(flat, back)

    # near antler: a stout beam sweeping back over the mane, two tines curling forward
    beam = curve((702, 375), (680, 296), (625, 238), (545, 205), (472, 172), (425, 125), (418, 66), n=72)
    t1 = curve((642, 252), (650, 196), (684, 142), n=26)
    t2 = curve((540, 206), (538, 150), (566, 92), n=26)
    draw_parts(flat, [part(t1, wlin(62, 18), g), part(t2, wlin(60, 18), g), part(beam, wlin(98, 24), g)])

    whisker(flat, ((862, 574), (905, 600), (950, 602), (976, 578), (984, 546)), 20, 10)
    # gold trim along the bottom of the scarf (the gold pendant and gem are drawn as pixels)
    dk.stroke(flat, curve((540, 712), (620, 716), (700, 722), (775, 700), n=40), INK, 22)
    dk.stroke(flat, curve((540, 712), (620, 716), (700, 722), (775, 700), n=40), pal["gold"], 11)
    return flat


PEARL = ((916, 752), 40)        # ref px centre, radius


def pearl_flames(layer):
    """A cyan fireball behind the pearl: round base, three flame tongues licking up."""
    (px, py), pr = PEARL
    cols = (FLAME["flame"], FLAME["aqua"], FLAME["frost"])
    tongues = [
        (((px - 34, py + 6), (px - 58, py - 48), (px - 46, py - 96), (px - 22, py - 122)), 58, 5),
        (((px + 34, py + 6), (px + 56, py - 44), (px + 50, py - 86), (px + 30, py - 106)), 54, 5),
        (((px, py - 10), (px - 14, py - 70), (px + 2, py - 120), (px + 22, py - 156)), 64, 5),
    ]
    disc = [R(px + (pr + 20) * np.cos(t), py + 6 + (pr + 16) * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 48)[:-1]]
    parts = [[(disc, cols[0])]]
    parts += [part(curve(*pts, n=36), wlin(w0, w1, p=0.8), cols, side=-1, bands=((-0.55, 0.45), (-0.25, 0.15)))
              for pts, w0, w1 in tongues]
    draw_parts(layer, parts)


def compose(stage, img, ctx):
    pal = palette(stage)
    if stage in (2, 3):
        img = (stage2 if stage == 2 else stage3)(img, ctx)
        _, lbl = snap(img, pal)
        flat = img
    else:
        flat, lbl = flatten(img, pal, ink_hi=INK_HI, grow=GROW[1])
    _STATE[stage] = {"pal": pal, "lbl": lbl}
    if os.environ.get("TL_DEBUG"):
        flat.save(os.environ["TL_DEBUG"])
    return flat


# --------------------------------------------------------------- touchup ----
class Geo:
    """reference poster px -> continuous sprite coordinates for one stage."""

    def __init__(self, ctx):
        self.s = ctx.scale
        self.ox, self.oy = ctx.ox, ctx.oy

    def __call__(self, x, y):
        return (x + gk.PAD) * self.s + self.ox, (y + gk.PAD) * self.s + self.oy

    def d(self, v):
        return v * self.s


def cover(fn, box=None, ss=6, thr=0.5):
    """Pixels whose area is at least `thr` inside fn(x, y) (continuous sprite coords, vectorised)."""
    H = W = gk.CANVAS
    x0, y0, x1, y1 = box or (0, 0, W, H)
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(W, int(np.ceil(x1))), min(H, int(np.ceil(y1)))
    out = np.zeros((H, W), bool)
    if x1 <= x0 or y1 <= y0:
        return out
    yy, xx = np.mgrid[0:(y1 - y0) * ss, 0:(x1 - x0) * ss].astype(np.float64)
    xx = x0 + (xx + 0.5) / ss
    yy = y0 + (yy + 0.5) / ss
    m = fn(xx, yy).astype(np.float64)
    c = m.reshape(y1 - y0, ss, x1 - x0, ss).mean((1, 3))
    out[y0:y1, x0:x1] = c >= thr
    return out


def ell(cx, cy, rx, ry, rot=0.0):
    ca, sa = np.cos(rot), np.sin(rot)

    def fn(x, y):
        u = (x - cx) * ca + (y - cy) * sa
        v = -(x - cx) * sa + (y - cy) * ca
        return (u / rx) ** 2 + (v / ry) ** 2 <= 1
    return fn


def ebox(cx, cy, rx, ry, pad=2):
    r = max(rx, ry) + pad
    return (cx - r, cy - r, cx + r + 1, cy + r + 1)


def inpaint(art, mask, avoid=()):
    """Fill mask with the colour of the nearest pixel outside it (skin under a closed eye)."""
    bad = set([TRANSPARENT, art.names["outline"]] + [art.names[a] for a in avoid if a in art.names])
    ys, xs = np.where(mask)
    src = ~mask & ~np.isin(art.idx, list(bad))
    sy, sx = np.where(src)
    if not len(sy):
        return
    for y, x in zip(ys, xs):
        d = (sy - y) ** 2 + (sx - x) ** 2
        j = d.argmin()
        art.idx[y, x] = art.idx[sy[j], sx[j]]


EYE_N = dict(c=(638, 484), r=(61, 62), iris=(658, 494), ri=(40, 47), pupil=(657, 489), rp=(23, 27),
             hl=(671, 471), hl2=(637, 517))
EYE_F = dict(c=(805, 487), r=(25, 52), hl=(797, 470))


def eyes(stage, art, G, closed=False):
    k = art.names["outline"]
    # ---- near (big) eye
    ex, ey = G(*EYE_N["c"])
    rx, ry = G.d(EYE_N["r"][0]), G.d(EYE_N["r"][1])
    box = ebox(ex, ey, rx, ry, 4)
    outer = cover(ell(ex, ey, rx, ry), box)
    if closed:
        inpaint(art, outer)
        return outer
    art.idx[outer] = k
    inner = cover(ell(ex + 0.15, ey + 0.35, rx - 1.15, ry - 1.3), box)
    art.idx[inner] = art.color("white")
    ix, iy = G(*EYE_N["iris"])
    irx, iry = G.d(EYE_N["ri"][0]), G.d(EYE_N["ri"][1])
    iris = cover(ell(ix, iy, irx, iry), box) & inner
    glow = stage == 3
    c_top, c_mid, c_low, c_rim = (("blue_dd", "blue_d", "blue", "blue_l") if not glow else
                                  ("blue", "flame", "aqua", "frost"))
    art.idx[iris] = art.color(c_top)
    art.idx[iris & cover(ell(ix, iy + iry * 0.35, irx * 0.95, iry * 0.75), box)] = art.color(c_mid)
    art.idx[iris & cover(ell(ix, iy + iry * 0.75, irx * 0.75, iry * 0.4), box)] = art.color(c_low)
    art.idx[iris & cover(ell(ix + 0.2, iy + iry * 0.95, irx * 0.45, iry * 0.18), box)] = art.color(c_rim)
    px, py = G(*EYE_N["pupil"])
    prx, pry = G.d(EYE_N["rp"][0]), G.d(EYE_N["rp"][1])
    if glow:
        prx *= 0.42
    art.idx[cover(ell(px, py, prx, pry), box) & inner] = art.color("black")
    # sclera shadow under the upper lid
    sh = inner & ~iris & cover(ell(ex, ey - ry * 0.55, rx * 1.1, ry * 0.5), box)
    art.idx[sh & (art.idx == art.color("white"))] = art.color("ice")
    # highlights
    hx, hy = G(*EYE_N["hl"])
    hr = max(0.75, G.d(8.5))
    art.idx[cover(ell(hx, hy, hr, hr), ebox(hx, hy, hr, hr), thr=0.35)] = art.color("white")
    h2x, h2y = G(*EYE_N["hl2"])
    art.put(int(h2x), int(h2y), "white")
    # lash flick at the outer corner
    lx, ly = G(568, 452)
    art.put(int(lx), int(ly), k)
    art.put(int(lx) - 1, int(ly) - 1, k)

    # ---- far (small) eye
    fx, fy = G(*EYE_F["c"])
    frx, fry = G.d(EYE_F["r"][0]), G.d(EYE_F["r"][1])
    fbox = ebox(fx, fy, frx, fry, 4)
    fo = cover(ell(fx, fy, frx, fry), fbox)
    art.idx[fo] = k
    fi = cover(ell(fx - 0.1, fy + 0.4, frx - 1.05, fry - 1.3), fbox)
    art.idx[fi] = art.color(c_top)
    art.idx[fi & cover(ell(fx, fy + fry * 0.55, frx, fry * 0.5), fbox)] = art.color(c_mid if not glow else c_low)
    fhx, fhy = G(*EYE_F["hl"])
    art.put(int(fhx), int(fhy), "white")
    if stage >= 2:
        art.put(int(fhx), int(fhy) + 1, "white")
    return outer | fo


def halfplane(p0, p1):
    """Points left of the directed line p0 -> p1 (above it when p1 is to the right of p0)."""
    (x0, y0), (x1, y1) = p0, p1
    return lambda x, y: (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0) < 0


def near_line(p0, p1, half):
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy

    def fn(x, y):
        t = np.clip(((x - x0) * dx + (y - y0) * dy) / L2, 0, 1)
        return (x - x0 - t * dx) ** 2 + (y - y0 - t * dy) ** 2 <= half * half
    return fn


def convex(pts):
    """Inside test for a convex polygon given in either winding."""
    pts = list(pts)

    def fn(x, y):
        pos = np.ones_like(x, bool)
        neg = np.ones_like(x, bool)
        for (ax, ay), (bx, by) in zip(pts, pts[1:] + pts[:1]):
            c = (bx - ax) * (y - ay) - (by - ay) * (x - ax)
            pos &= c >= 0
            neg &= c <= 0
        return pos | neg
    return fn


# fierce upper lids: (near eye left point, right point), (far eye left point, right point) in eye-radius units
LIDS = {2: ((-1.15, -0.62), (1.1, -0.42), (-1.2, -0.38), (1.2, -0.6)),
        3: ((-1.15, -0.55), (1.1, -0.12), (-1.2, -0.12), (1.2, -0.55))}


def lids(stage, art, G):
    if stage not in LIDS:
        return
    k = art.names["outline"]
    nl, nr, fl, fr = LIDS[stage]
    for (cx, cy), (rx, ry), a, b in (((*EYE_N["c"],), EYE_N["r"], nl, nr), ((*EYE_F["c"],), EYE_F["r"], fl, fr)):
        ex, ey = G(cx, cy)
        rx, ry = G.d(rx), G.d(ry)
        p0 = (ex + a[0] * rx, ey + a[1] * ry)
        p1 = (ex + b[0] * rx, ey + b[1] * ry)
        box = ebox(ex, ey, rx, ry, 3)
        region = cover(ell(ex, ey, rx + 0.9, ry + 0.9), box)
        cut = region & cover(halfplane(p0, p1), box)
        inpaint(art, cut, avoid=("white", "ice", "black", "blue_dd", "blue_d", "blue", "flame", "aqua"))
        line = cover(near_line(p0, p1, 0.75 if stage == 2 else 0.95), box, thr=0.4) & \
            cover(ell(ex, ey, rx + 0.6, ry + 0.6), box)
        art.idx[line] = k


GLYPHS = {
    5: {"I": ["#", "#", "#", "#", "#"],
        "Q": [".##.", "#..#", "#..#", "#.#.", ".#.#"],
        "X": ["#...#", ".#.#.", "..#..", ".#.#.", "#...#"]},
    6: {"I": ["##", "##", "##", "##", "##", "##"],
        "Q": [".###.", "#...#", "#...#", "#.#.#", "#..#.", ".##.#"],
        "X": ["#...#", ".#.#.", "..#..", "..#..", ".#.#.", "#...#"]},
    7: {"I": ["###", ".#.", ".#.", ".#.", ".#.", ".#.", "###"],
        "Q": [".###.", "#...#", "#...#", "#...#", "#.#.#", "#..#.", ".##.#"],
        "X": ["#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"]},
}
SCARF_COLS = ("blue_dd", "blue_d", "blue", "blue_l", "scarf_l", "white", "ice", "sky", "cyan")


def lettering(stage, art, G):
    h = {1: 5, 2: 6, 3: 7}[stage]
    g = GLYPHS[h]
    word = [g[c] for c in "IQX"]
    w = sum(len(r[0]) for r in word) + len(word) - 1
    cx, cy = G(677, 676)
    x0 = int(round(cx - w / 2))
    y0 = int(round(cy - h / 2))
    # flat band behind the letters
    bx0, by0, bx1, by1 = x0 - 2, y0 - 1, x0 + w + 1, y0 + h
    sub = art.idx[by0:by1 + 1, bx0:bx1 + 1]
    sub[np.isin(sub, [art.color(c) for c in SCARF_COLS])] = art.color("blue_d")
    x = x0
    for rows in word:
        art.blit(x, y0, rows, {"#": "white"})
        x += len(rows[0]) + 1


def gem(art, cx, cy, a, b, big=False):
    """Faceted diamond (sprite coords, half width a, half height b)."""
    k = art.names["outline"]
    gy = cy - b * 0.28                                   # girdle
    T, Rt, B, L = (cx, cy - b), (cx + a, gy), (cx, cy + b), (cx - a, gy)
    box = (cx - a - 2, cy - b - 2, cx + a + 3, cy + b + 3)
    full = cover(convex([T, Rt, B, L]), box, thr=0.45)
    art.idx[full] = k
    inner = cover(convex([(cx, cy - b + 1.3), (cx - a + 1.2, gy), (cx, cy + b - 1.4), (cx + a - 1.2, gy)]), box, thr=0.5)
    up = inner & cover(lambda x, y: y < gy, box)
    lo = inner & ~up
    left = cover(lambda x, y: x < cx, box)
    art.idx[up & left] = art.color("sky")
    art.idx[up & ~left] = art.color("ice")
    art.idx[lo & left] = art.color("blue_l")
    art.idx[lo & ~left] = art.color("blue")
    # centre ridge of the pavilion and the girdle line
    mid = lo & cover(lambda x, y: np.abs(x - cx) < 0.55 + (cy + b - y) * 0.06, box)
    art.idx[mid] = art.color("cyan")
    girdle = inner & cover(lambda x, y: np.abs(y - gy) < 0.5, box)
    art.idx[girdle] = art.color("white" if big else "ice")
    # sparkle on the upper-left facet
    sx, sy = int(cx - a * 0.4), int(gy - b * 0.25)
    art.put(sx, sy, "white")
    if big:
        art.put(sx, sy - 1, "white")


GEMS = {1: ((697, 758), (42, 51)), 2: ((697, 772), (56, 66)), 3: ((687, 792), (40, 54))}


def gems(stage, art, G):
    (x, y), (a, b) = GEMS[stage]
    cx, cy = G(x, y)
    if stage == 3:
        pendant(art, cx, cy, G.d(a), G.d(b))
    gem(art, cx, cy, G.d(a), G.d(b), big=stage >= 2)


MOUTH_COLS = ("m_dark", "m_red", "tongue")


def fangs(stage, art, G):
    """White fangs hanging from the upper lip, against the dark of the open mouth."""
    if stage < 2:
        return
    k = art.names["outline"]
    teal = [art.names[c] for c in ("teal_dd", "teal_d", "teal", "teal_l") if c in art.names]
    _, y0 = G(700, 535)
    _, y1 = G(700, 600)
    spec = ((722, 2, 2), (748, 2, 1)) if stage == 3 else ((724, 2, 1),)
    for xr, tall, wide in spec:
        x = int(G(xr, 560)[0])
        yl = None
        for y in range(int(y0), int(y1)):
            if art.idx[y, x] in teal and art.idx[y + 1, x] == k:
                yl = y + 1
                break
        if yl is None:
            continue
        for j in range(tall):
            art.put(x, yl + 1 + j, "white")
        if wide > 1:
            art.put(x + 1, yl + 1, "white")


def orb(art, cx, cy, r):
    """Crisp pixel dragon pearl (its cyan flames are drawn in compose)."""
    k = art.names["outline"]
    box = ebox(cx, cy, r, r, 3)
    art.idx[cover(ell(cx, cy, r + 0.9, r + 0.9), box)] = k
    body = cover(ell(cx, cy, r, r), box)
    art.idx[body] = art.color("aqua")
    art.idx[body & cover(ell(cx + r * 0.22, cy - r * 0.22, r * 0.8, r * 0.8), box)] = art.color("frost")
    art.idx[body & ~cover(ell(cx + r * 0.16, cy - r * 0.16, r * 1.02, r * 1.02), box)] = art.color("flame")
    hx, hy = int(cx + r * 0.25), int(cy - r * 0.5)
    art.rect(hx, hy, hx + 1, hy + 1, "white")
    art.put(int(cx - r * 0.4), int(cy + r * 0.3), "white")


SPARKLE = {"s": ["..a..", "..f..", "afWfa", "..f..", "..a.."], "m": [".a.", "aWa", ".a."]}


def sparkle(art, x, y, size="s", force=False):
    rows = SPARKLE[size]
    key = {"a": "aqua", "f": "frost", "W": "white"}
    h = len(rows)
    for j, row in enumerate(rows):
        for i, ch in enumerate(row):
            if ch in key:
                X, Y = x + i - h // 2, y + j - h // 2
                if force or art.idx[Y, X] == TRANSPARENT:
                    art.put(X, Y, key[ch])


def glow(stage, art, G):
    if stage != 3:
        return
    (px, py), pr = PEARL
    cx, cy = G(px, py)
    orb(art, cx, cy, G.d(pr))
    for (x, y), sz in ((G(985, 640), "s"), (G(330, 40), "s"), (G(860, 120), "m"), (G(120, 470), "m"),
                       (G(1000, 860), "m")):
        sparkle(art, int(x), int(y), sz)


# auspicious clouds at the feet, sprite px relative to the cloud's left end on the ground:
# (dx, dy_above_ground, rx, ry) puffs; the first puff carries the curl
CLOUDS = [
    (13, [(8.0, 6.6, 8.0, 6.4), (18.5, 4.2, 5.2, 4.2), (3.0, 2.6, 3.4, 2.6)], False),
    (152, [(8.0, 6.6, 8.0, 6.4), (18.5, 4.2, 5.2, 4.2), (3.0, 2.6, 3.4, 2.6)], True),
    # low bank behind the legs joining the two curls
    (38, [(0, 3.4, 5.4, 3.4), (8, 4.6, 6.0, 4.4), (16, 3.4, 5.4, 3.4), (24, 4.6, 6.0, 4.4), (32, 3.4, 5.4, 3.4), (40, 4.6, 6.0, 4.4), (48, 3.4, 5.4, 3.4), (56, 4.6, 6.0, 4.4), (64, 3.4, 5.4, 3.4), (72, 4.6, 6.0, 4.4), (80, 3.4, 5.4, 3.4), (88, 4.6, 6.0, 4.4), (96, 3.4, 5.4, 3.4), (104, 4.6, 6.0, 4.4), (112, 3.4, 5.4, 3.4)], None),
]


def pixel_cloud(art, x0, puffs, flip=False):
    """Xiangyun cloud in pixels, painted only on empty pixels so it stays behind the body:
    frost tops, foam body, cyan underside, a curl inside the big puff, 1px outline."""
    k = art.names["outline"]
    yb = gk.GROUND - 1.0
    w = max(p[0] + p[2] for p in puffs)
    pts = [((x0 + w - dx) if flip else (x0 + dx), yb - dy, rx, ry) for dx, dy, rx, ry in puffs]
    shape = np.zeros(art.idx.shape, bool)
    for cx, cy, rx, ry in pts:
        shape |= cover(ell(cx, cy, rx, ry), ebox(cx, cy, rx, ry))
    shape[int(yb):, :] = False          # flat bottom, one row above the ground for the outline
    free = art.idx == TRANSPARENT
    p = np.pad(shape, 1)
    ring = (p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]) & ~shape
    art.idx[ring & free] = k
    body = shape & free
    art.idx[body] = art.color("foam")
    yy, xx = np.mgrid[0:gk.CANVAS, 0:gk.CANVAS]
    art.idx[body & (yy >= yb - 2.5)] = art.color("cloud")
    for cx, cy, rx, ry in pts:
        top = cover(ell(cx - rx * 0.25, cy - ry * 0.45, rx * 0.5, ry * 0.4), ebox(cx, cy, rx, ry))
        art.idx[body & top] = art.color("frost")
    if flip is None:
        return
    # curl: a 3/4 ring of cloud blue inside the big puff, opening toward the rest of the cloud
    cx, cy, rx, ry = pts[0]
    ang = np.degrees(np.arctan2(yy + 0.5 - cy, (xx + 0.5 - cx) * (-1 if flip else 1)))
    rr = np.hypot((xx + 0.5 - cx) / (rx * 0.62), (yy + 0.5 - cy) / (ry * 0.62))
    curl = body & (np.abs(rr - 1.0) < 0.28) & ~((ang > -10) & (ang < 80))
    art.idx[curl] = art.color("cloud")
    art.put(int(round(cx)), int(round(cy)), "cloud")


def clouds(stage, art, G):
    if stage != 3:
        return
    for x0, puffs, flip in CLOUDS:
        pixel_cloud(art, x0, puffs, flip)


FEET = [(0, 3, 7, 9), (3, 6, 6, 7), (6, 8, 7, 9)]   # TOES index range, x margin, rows above the toes
CLAW_GOLD = {"cream_s": "gold_d", "cream_y": "gold", "cream": "gold_l", "white": "gold_l"}


def gold_claws(stage, art, G):
    """Stage 3: the original cream claws turn to gold."""
    if stage != 3:
        return
    for i0, i1, mx, my in FEET:
        xs = [G(*TOES[i])[0] for i in range(i0, i1)]
        y = G(*TOES[i0])[1]
        x0, x1 = int(min(xs)) - mx, int(max(xs)) + mx
        y0 = int(y) - my
        for src, dst in CLAW_GOLD.items():
            if src in art.names:
                art.replace(src, dst, (x0, y0, x1, gk.GROUND - 1))


def pendant(art, cx, cy, a, b):
    """Gold setting for the stage-3 gem: diamond frame with side flares (the gem goes on top)."""
    k = art.names["outline"]
    gy = cy - b * 0.28
    A, B = a + 2.3, b + 2.7
    box = (cx - A - 6, cy - B - 3, cx + A + 7, cy + B + 4)
    frame = cover(convex([(cx, cy - B), (cx + A, gy), (cx, cy + B), (cx - A, gy)]), box, thr=0.45)
    fl = cover(convex([(cx - A - 3.6, gy - 0.4), (cx - A + 1.5, gy - 2.6), (cx - A + 1.5, gy + 2.2)]), box, thr=0.45)
    fr = cover(convex([(cx + A + 3.6, gy - 0.4), (cx + A - 1.5, gy - 2.6), (cx + A - 1.5, gy + 2.2)]), box, thr=0.45)
    shape = frame | fl | fr
    p = np.pad(shape, 1)
    ring = (p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]) & ~shape
    art.idx[ring] = k
    art.idx[shape] = art.color("gold")
    yy, xx = np.mgrid[0:gk.CANVAS, 0:gk.CANVAS]
    lit = (xx + 0.5 - cx) * 0.8 + (yy + 0.5 - gy) < -1.0
    dark = (xx + 0.5 - cx) * 0.8 + (yy + 0.5 - gy) > 3.0
    art.idx[shape & lit] = art.color("gold_l")
    art.idx[shape & dark] = art.color("gold_d")
    # bright glint on the left flare tip and top of the frame
    art.put(int(cx - A - 2), int(gy - 1), "gold_hi")
    art.put(int(cx), int(cy - B + 0.6), "gold_hi")


# near whisker of the original: from the snout down across the cheek, looping up on the left (ref px)
WHISKER_N = [(590, 538), (548, 546), (524, 566), (510, 592), (492, 612), (466, 622), (440, 614), (421, 596),
             (416, 572), (424, 551), (438, 541)]
MINT = "#a6ecc4"
WHISKER_R = {1: (1.35, 0.8), 2: (1.55, 0.9), 3: (1.75, 1.0)}       # core radius at root, at tip (sprite px)


def whisker_px(stage, art, G):
    """Redraw the curly near whisker as a clean pixel line: mint core, sea-green underside, 1px outline."""
    k = art.names["outline"]
    pts = np.array([G(*q) for q in WHISKER_N])
    # dense polyline through the points (Catmull-Rom)
    dense = []
    P = np.vstack([pts[:1], pts, pts[-1:]])
    for i in range(1, len(P) - 2):
        for t in np.linspace(0, 1, 12, endpoint=False):
            p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
            dense.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t +
                                (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    dense.append(pts[-1])
    dense = np.array(dense)
    r0, r1 = WHISKER_R[stage]
    rad = np.linspace(r0, r1, len(dense))
    x0, y0 = dense.min(0) - 4
    x1, y1 = dense.max(0) + 4
    box = (x0, y0, x1, y1)

    def within(extra, below=False):
        def fn(x, y):
            out = np.zeros(x.shape, bool)
            for (cx, cy), r in zip(dense, rad):
                d2 = (x - cx) ** 2 + (y - cy - (0.55 if below else 0)) ** 2
                out |= d2 <= (r + extra) ** 2
            return out
        return fn
    ring = cover(within(0.95), box, ss=4, thr=0.4)
    core = cover(within(0.0), box, ss=4, thr=0.45)
    art.idx[ring & ~core] = k
    art.idx[core] = art.color(MINT)
    # shade the lower edge of the line
    sh = core & np.roll(~core, -1, axis=0)
    art.idx[sh] = art.color("green")


def touchup(stage, art, ctx):
    repaint(stage, art, ctx)
    if "gold_hi" in art.names:
        # cream fur that snapped to the near-identical pale gold stays cream
        art.replace("gold_hi", "cream")
    smooth_fill(art, 1)
    despeckle(art, 4)
    thin_outline(art)
    drop_ink_specks(art)
    G = Geo(ctx)
    whisker_px(stage, art, G)
    eyes(stage, art, G)
    lids(stage, art, G)
    lettering(stage, art, G)
    gems(stage, art, G)
    fangs(stage, art, G)
    gold_claws(stage, art, G)
    clouds(stage, art, G)
    seal_edges(art)
    glow(stage, art, G)


def smooth_region(art, mask, passes=2, r=2):
    """Majority filter over a (2r+1)^2 window, applied only to `mask` pixels (outline and empty never vote),
    so inpainted skin gets organic edges instead of straight nearest-pixel wedges."""
    k = art.names["outline"]
    ys, xs = np.where(mask)
    for _ in range(passes):
        idx = art.idx
        new = idx.copy()
        for y, x in zip(ys, xs):
            win = idx[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].ravel()
            win = win[(win != TRANSPARENT) & (win != k)]
            if len(win):
                vals, cnt = np.unique(win, return_counts=True)
                new[y, x] = vals[cnt.argmax()]
        art.idx[...] = new


EYE_COLS = ("white", "ice", "black", "blue_dd", "blue_d", "blue", "blue_l", "flame", "aqua", "frost")


def blink(stage, art, ctx):
    G = Geo(ctx)
    k = art.names["outline"]
    ex, ey = G(*EYE_N["c"])
    rx, ry = G.d(EYE_N["r"][0]), G.d(EYE_N["r"][1])
    box = ebox(ex, ey, rx, ry, 4)
    m = cover(ell(ex, ey, rx + 0.8, ry + 0.8), box)
    lash = cover(ell(ex - rx * 0.85, ey - ry * 0.45, rx * 0.55, ry * 0.5), box) & (art.idx == k)
    m |= lash
    m &= ~np.isin(art.idx, [art.color(MINT), art.names["green"]])     # keep the whisker
    # closed upper lid: a bold 2px lash line, soft smile at stage 1, flatter and angled as it grows fierce
    sag = {1: 0.36, 2: 0.24, 3: 0.12}[stage]
    tilt = {1: 0.0, 2: 0.06, 3: 0.2}[stage]
    yline = ey + ry * 0.12

    def yc(x):
        u = (x - ex) / (rx * 0.95)
        return yline + sag * ry * (1 - u * u) - (x - ex) / rx * ry * tilt

    def arc(x, y):
        u = (x - ex) / (rx * 0.95)
        return (np.abs(u) <= 1) & (np.abs(y - yc(x)) <= 1.0)
    # the lid above the line is face skin (teal), the socket below it takes the surrounding cheek colours
    lid = m & cover(lambda x, y: y < yc(x), box, thr=0.5) & cover(ell(ex, ey, rx + 0.2, ry + 0.2), box)
    inpaint(art, m, avoid=EYE_COLS)
    art.idx[lid] = art.color("teal_l")
    smooth_region(art, m & ~lid, passes=2)
    art.idx[cover(arc, box, thr=0.3)] = k
    # lash flick at the outer corner
    xl = ex - rx * 0.95
    yl = yc(xl)
    art.put(int(np.floor(xl)) - 1, int(np.floor(yl)) - 1, k)
    if stage >= 2:
        art.put(int(np.floor(xl)) - 2, int(np.floor(yl)) - 2, k)
    # far eye: short closed line
    fx, fy = G(*EYE_F["c"])
    frx, fry = G.d(EYE_F["r"][0]), G.d(EYE_F["r"][1])
    fbox = ebox(fx, fy, frx, fry, 3)
    fm = cover(ell(fx, fy, frx + 0.3, fry + 0.3), fbox)
    inpaint(art, fm, avoid=EYE_COLS)
    smooth_region(art, fm, passes=1, r=1)
    fl = cover(near_line((fx - frx * 0.95, fy + fry * 0.15),
                         (fx + frx * 0.75, fy + fry * 0.3 - (0.0 if stage < 3 else 0.9)), 0.85), fbox, thr=0.4)
    art.idx[fl] = k
    seal_edges(art, fbox)
