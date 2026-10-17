"""Core pixel-art conversion for the IQX mascots.

reference/<slug>.webp (1024px illustration)
  -> box downscale to the sprite size
  -> hard alpha (no anti-aliasing) + 1px outline ring
  -> dark ink lines from the original kept as a single outline colour
  -> palette reduced with k-means in Lab
  -> majority filter so colour areas are flat and edges are clean

The result is an `Art`: an index image plus a palette, with small drawing
helpers that the per-pet touch-up scripts in pets/ use for hand fixes.
"""
from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRANSPARENT = -1

DEFAULTS = {
    "size": 512,        # sprite is size x size pixels
    "margin": 6,        # transparent pixels kept below/around the body
    "colors": 48,       # palette size, outline excluded
    "ink_l": 34.0,      # Lab L* under which an original pixel counts as ink
    "ink_frac": 0.5,    # share of ink inside an output pixel to draw it as outline
    "smooth": 2,        # majority-filter passes
    "min_island": 12,   # colour regions smaller than this (px) merge into a neighbour
    "seed": 0,
}


def rgb2lab(rgb: np.ndarray) -> np.ndarray:
    c = np.asarray(rgb, np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def nearest(x: np.ndarray, centers: np.ndarray, chunk: int = 65536) -> np.ndarray:
    out = np.empty(len(x), int)
    for i in range(0, len(x), chunk):
        d = ((x[i:i + chunk, None] - centers[None]) ** 2).sum(-1)
        out[i:i + chunk] = d.argmin(1)
    return out


def kmeans(x: np.ndarray, k: int, seed: int = 0, iters: int = 30, sample: int = 30000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    s = x[rng.choice(len(x), min(len(x), sample), replace=False)]
    centers = [s[rng.integers(len(s))]]
    dmin = ((s - centers[0]) ** 2).sum(-1)
    for _ in range(k - 1):
        centers.append(s[rng.choice(len(s), p=dmin / dmin.sum())])
        dmin = np.minimum(dmin, ((s - centers[-1]) ** 2).sum(-1))
    c = np.array(centers)
    for _ in range(iters):
        lab = nearest(s, c)
        for j in range(k):
            if (lab == j).any():
                c[j] = s[lab == j].mean(0)
    return c


def grow4(m: np.ndarray) -> np.ndarray:
    p = np.pad(m, 1)
    return p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]


def majority(idx: np.ndarray, n_colors: int, passes: int) -> np.ndarray:
    """3x3 majority filter on an index image; transparent pixels stay put."""
    h, w = idx.shape
    for _ in range(passes):
        counts = np.zeros((n_colors, h, w), np.int16)
        p = np.pad(idx, 1, constant_values=TRANSPARENT)
        for dy in range(3):
            for dx in range(3):
                win = p[dy:dy + h, dx:dx + w]
                for c in range(n_colors):
                    counts[c] += win == c
        best = counts.argmax(0)
        strong = counts.max(0) >= 5
        idx = np.where((idx != TRANSPARENT) & strong, best, idx)
    return idx


def components(idx: np.ndarray):
    """4-connected same-colour regions: yields (colour, flat pixel indices)."""
    h, w = idx.shape
    flat = idx.ravel()
    seen = np.zeros(h * w, bool)
    seen[flat == TRANSPARENT] = True
    for start in range(h * w):
        if seen[start]:
            continue
        c = flat[start]
        stack, region = [start], []
        seen[start] = True
        while stack:
            p = stack.pop()
            region.append(p)
            y, x = divmod(p, w)
            for q, ok in ((p - w, y > 0), (p + w, y < h - 1), (p - 1, x > 0), (p + 1, x < w - 1)):
                if ok and not seen[q] and flat[q] == c:
                    seen[q] = True
                    stack.append(q)
        yield c, np.array(region)


def merge_islands(idx: np.ndarray, min_area: int, keep=()) -> np.ndarray:
    """Fold colour regions smaller than min_area into their most common neighbour."""
    if min_area <= 1:
        return idx
    h, w = idx.shape
    out = idx.copy()
    flat = out.ravel()
    for c, region in list(components(idx)):
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
        if len(nb):
            vals, cnt = np.unique(nb, return_counts=True)
            flat[region] = vals[cnt.argmax()]
    return out


@dataclass
class Art:
    idx: np.ndarray                      # (H, W) palette index, -1 = transparent
    palette: list[tuple[int, int, int]]
    names: dict[str, int]                # semantic names -> palette index ("outline", ...)

    # ---- palette -------------------------------------------------------
    def color(self, c) -> int:
        """Palette index for a name, '#rrggbb', (r, g, b) or an index."""
        if isinstance(c, (int, np.integer)):
            return int(c)
        if isinstance(c, str) and c in self.names:
            return self.names[c]
        rgb = hex2rgb(c) if isinstance(c, str) else tuple(int(v) for v in c)
        if rgb in self.palette:
            return self.palette.index(rgb)
        self.palette.append(rgb)
        return len(self.palette) - 1

    def name(self, label: str, c) -> int:
        self.names[label] = self.color(c)
        return self.names[label]

    def closest(self, rgb) -> int:
        pal = rgb2lab(np.array(self.palette))
        return int(nearest(rgb2lab(np.array([rgb], np.float64)), pal)[0])

    # ---- drawing -------------------------------------------------------
    def put(self, x: int, y: int, c) -> None:
        h, w = self.idx.shape
        if 0 <= x < w and 0 <= y < h:
            self.idx[y, x] = TRANSPARENT if c is None else self.color(c)

    def rect(self, x0: int, y0: int, x1: int, y1: int, c) -> None:
        """Filled rectangle, inclusive corners."""
        self.idx[y0:y1 + 1, x0:x1 + 1] = TRANSPARENT if c is None else self.color(c)

    def ellipse(self, cx: float, cy: float, rx: float, ry: float, c, only=None) -> None:
        """Filled ellipse; `only` limits painting to pixels currently in that colour set."""
        h, w = self.idx.shape
        yy, xx = np.mgrid[0:h, 0:w]
        m = ((xx + 0.5 - cx) / rx) ** 2 + ((yy + 0.5 - cy) / ry) ** 2 <= 1.0
        if only is not None:
            m &= np.isin(self.idx, [self.color(o) for o in only])
        self.idx[m] = TRANSPARENT if c is None else self.color(c)

    def line(self, x0: int, y0: int, x1: int, y1: int, c, width: int = 1) -> None:
        n = max(abs(x1 - x0), abs(y1 - y0), 1)
        for t in range(n + 1):
            x = round(x0 + (x1 - x0) * t / n)
            y = round(y0 + (y1 - y0) * t / n)
            r = width // 2
            for dy in range(-r, width - r):
                for dx in range(-r, width - r):
                    self.put(x + dx, y + dy, c)

    def replace(self, src, dst, box=None) -> None:
        """Recolour src -> dst, optionally inside box=(x0, y0, x1, y1) inclusive."""
        sl = np.s_[:, :] if box is None else np.s_[box[1]:box[3] + 1, box[0]:box[2] + 1]
        region = self.idx[sl]
        region[region == self.color(src)] = TRANSPARENT if dst is None else self.color(dst)

    def blit(self, x: int, y: int, rows: list[str], key: dict[str, object]) -> None:
        """Paint a small hand-drawn pattern; chars not in key are left untouched."""
        for j, row in enumerate(rows):
            for i, ch in enumerate(row):
                if ch in key:
                    self.put(x + i, y + j, key[ch])

    # ---- output --------------------------------------------------------
    def rgba(self) -> np.ndarray:
        pal = np.array(self.palette + [(0, 0, 0)], np.uint8)
        out = np.zeros(self.idx.shape + (4,), np.uint8)
        out[..., :3] = pal[self.idx]
        out[..., 3] = np.where(self.idx == TRANSPARENT, 0, 255)
        return out

    def image(self) -> Image.Image:
        return Image.fromarray(self.rgba(), "RGBA")

    def save(self, path: str) -> None:
        """Indexed PNG, palette entry 0 is transparent."""
        used = sorted(set(np.unique(self.idx)) - {TRANSPARENT})
        if len(used) > 255:
            raise ValueError(f"{len(used)} colours, indexed PNG holds 255")
        remap = np.zeros(len(self.palette) + 1, np.uint8)
        flat = [0, 0, 0]
        for n, i in enumerate(used, start=1):
            remap[i] = n
            flat.extend(self.palette[i])
        img = Image.fromarray(remap[self.idx], "P")
        img.putpalette(flat)
        img.info["transparency"] = 0
        img.save(path, optimize=True, transparency=0)


def hex2rgb(s: str) -> tuple[int, int, int]:
    s = s.lstrip("#")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def reference(slug: str) -> Image.Image:
    return Image.open(os.path.join(ROOT, "reference", f"{slug}.webp")).convert("RGBA")


def placement(slug: str, size: int, margin: int):
    """Crop box in the reference and where it lands in the sprite: (box, (ox, oy, nw, nh))."""
    a = np.array(reference(slug))
    ys, xs = np.where(a[..., 3] > 16)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    bw, bh = box[2] - box[0], box[3] - box[1]
    s = (size - 2 * margin) / max(bw, bh)
    nw, nh = round(bw * s), round(bh * s)
    return box, ((size - nw) // 2, size - margin - nh, nw, nh)


def to_sprite(slug: str, x: float, y: float, size: int = 512, margin: int = 6) -> tuple[int, int]:
    """Map a pixel position in reference/<slug>.webp to sprite coordinates."""
    box, (ox, oy, nw, nh) = placement(slug, size, margin)
    return (round(ox + (x - box[0]) * nw / (box[2] - box[0])),
            round(oy + (y - box[1]) * nh / (box[3] - box[1])))


def pixelize(slug: str, **opts) -> Art:
    o = {**DEFAULTS, **opts}
    size = o["size"]
    im = reference(slug)
    box, (ox, oy, nw, nh) = placement(slug, size, o["margin"])
    crop = im.crop(box)
    small = np.array(crop.convert("RGBa").resize((nw, nh), Image.BOX).convert("RGBA")).astype(np.float64)

    hi = np.array(crop).astype(np.float64)
    hi_lab = rgb2lab(hi[..., :3])
    ink = ((hi_lab[..., 0] < o["ink_l"]) & (hi[..., 3] > 128)).astype(np.uint8) * 255
    ink_frac = np.array(Image.fromarray(ink).resize((nw, nh), Image.BOX)).astype(np.float64) / 255

    canvas = np.zeros((size, size, 4))
    frac = np.zeros((size, size))
    canvas[oy:oy + nh, ox:ox + nw] = small
    frac[oy:oy + nh, ox:ox + nw] = ink_frac
    mask = canvas[..., 3] >= 128

    px = canvas[mask][:, :3]
    lab = rgb2lab(px)
    centers = kmeans(lab, o["colors"], o["seed"])
    labels = nearest(lab, centers)
    pal = [tuple(int(round(v)) for v in px[labels == j].mean(0)) if (labels == j).any() else (255, 0, 255)
           for j in range(o["colors"])]

    op = hi[..., 3] > 200
    dark = hi[op][hi_lab[op][:, 0] < min(30.0, o["ink_l"])][:, :3]
    outline = tuple(int(round(v)) for v in dark.mean(0)) if len(dark) else (20, 30, 60)
    pal.append(outline)
    k_out = len(pal) - 1

    idx = np.full((size, size), TRANSPARENT, int)
    idx[mask] = labels
    idx[mask & (frac >= o["ink_frac"])] = k_out
    idx = majority(idx, len(pal), o["smooth"])
    idx = merge_islands(idx, o["min_island"])
    idx[grow4(mask) & ~mask] = k_out

    return Art(idx, pal, {"outline": k_out})
