"""Illustration-level drawing helpers for evolution compose() functions.

Everything works on RGBA PIL images (the 2048px art canvas). Shapes are
supersampled so edges stay smooth before pixelization; give shapes a dark
outline (L* < ~30, e.g. the pet's ink colour) and the pixelizer turns it into
the sprite's outline, matching the original line art.
"""
from __future__ import annotations

import colorsys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

Point = tuple[float, float]


def hexrgba(c, alpha: int = 255) -> tuple[int, int, int, int]:
    if isinstance(c, str):
        s = c.lstrip("#")
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), alpha
    return tuple(c) if len(c) == 4 else (*c, alpha)


def bezier(points: list[Point], n: int = 48) -> list[Point]:
    """Bezier curve of any degree through control points."""
    pts = np.array(points, np.float64)
    out = []
    for t in np.linspace(0, 1, n):
        p = pts.copy()
        while len(p) > 1:
            p = (1 - t) * p[:-1] + t * p[1:]
        out.append((float(p[0, 0]), float(p[0, 1])))
    return out


def _layer(size, draw_fn, ss: int = 4) -> Image.Image:
    w, h = size
    big = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(big), ss)
    return big.resize((w, h), Image.LANCZOS)


def _bbox(points, pad):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return int(min(xs) - pad), int(min(ys) - pad), int(max(xs) + pad) + 1, int(max(ys) + pad) + 1


def shape(img: Image.Image, points: list[Point], fill, outline=None, width: float = 0, ss: int = 4) -> None:
    """Filled polygon, composited onto img in place; `outline` adds a border `width` px wide outside it."""
    if outline and width:
        stroke(img, list(points) + [points[0]], outline, width * 2, ss)
    x0, y0, x1, y1 = _bbox(points, 2)

    def fn(d, s):
        d.polygon([((x - x0) * s, (y - y0) * s) for x, y in points], fill=hexrgba(fill))

    img.alpha_composite(_layer((x1 - x0, y1 - y0), fn, ss), (x0, y0))


def stroke(img: Image.Image, points: list[Point], color, width: float, ss: int = 4) -> None:
    """Polyline with round caps and joins."""
    x0, y0, x1, y1 = _bbox(points, width + 2)

    def fn(d, s):
        pts = [((x - x0) * s, (y - y0) * s) for x, y in points]
        w = max(1, int(width * s))
        d.line(pts, fill=hexrgba(color), width=w, joint="curve")
        r = w / 2
        for px, py in (pts[0], pts[-1]):
            d.ellipse([px - r, py - r, px + r, py + r], fill=hexrgba(color))

    img.alpha_composite(_layer((x1 - x0, y1 - y0), fn, ss), (x0, y0))


def outlined_stroke(img, points, color, width, outline, outline_width, ss: int = 4) -> None:
    """A tapered-free 'tube': outline stroke underneath, colour stroke on top."""
    stroke(img, points, outline, width + 2 * outline_width, ss)
    stroke(img, points, color, width, ss)


def ellipse(img: Image.Image, cx, cy, rx, ry, fill, outline=None, width: float = 0, ss: int = 4) -> None:
    x0, y0 = int(cx - rx - width - 2), int(cy - ry - width - 2)
    x1, y1 = int(cx + rx + width + 2) + 1, int(cy + ry + width + 2) + 1

    def fn(d, s):
        box = [(cx - rx - x0) * s, (cy - ry - y0) * s, (cx + rx - x0) * s, (cy + ry - y0) * s]
        if outline and width:
            d.ellipse([box[0] - width * s, box[1] - width * s, box[2] + width * s, box[3] + width * s],
                      fill=hexrgba(outline))
        d.ellipse(box, fill=hexrgba(fill))

    img.alpha_composite(_layer((x1 - x0, y1 - y0), fn, ss), (x0, y0))


def tapered(img, points: list[Point], w0: float, w1: float, fill, outline=None, outline_width: float = 0) -> None:
    """A horn/feather/flame shape: centre line `points`, width from w0 at the start to w1 at the end."""
    pts = np.array(points, np.float64)
    left, right = [], []
    for i, p in enumerate(pts):
        a = pts[max(i - 1, 0)]
        b = pts[min(i + 1, len(pts) - 1)]
        t = b - a
        nrm = np.array([-t[1], t[0]]) / (np.hypot(*t) or 1)
        w = (w0 + (w1 - w0) * i / max(1, len(pts) - 1)) / 2
        left.append(tuple(p + nrm * w))
        right.append(tuple(p - nrm * w))
    poly = left + right[::-1]
    if outline and outline_width:
        # Outline straddles the edge: a wide outline-coloured stroke, then the fill on top.
        stroke(img, poly + [poly[0]], outline, outline_width * 2)
    shape(img, poly, fill)


def hsv_mask(img: Image.Image, h: tuple[float, float] | None = None, s: tuple[float, float] = (0, 1),
             v: tuple[float, float] = (0, 1), box=None, alpha_min: int = 128) -> np.ndarray:
    """Boolean mask of pixels inside hue/sat/value ranges (h in degrees, wraps if h0 > h1)."""
    a = np.array(img.convert("RGBA")).astype(np.float64) / 255
    rgb = a[..., :3]
    mx, mn = rgb.max(-1), rgb.min(-1)
    d = mx - mn
    hue = np.zeros_like(mx)
    nz = d > 1e-6
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    hr = nz & (mx == r)
    hg = nz & (mx == g) & ~hr
    hb = nz & ~hr & ~hg
    hue[hr] = ((g - b)[hr] / d[hr]) % 6
    hue[hg] = (b - r)[hg] / d[hg] + 2
    hue[hb] = (r - g)[hb] / d[hb] + 4
    hue *= 60
    sat = np.where(mx > 0, d / np.maximum(mx, 1e-6), 0)
    m = (a[..., 3] * 255 >= alpha_min) & (sat >= s[0]) & (sat <= s[1]) & (mx >= v[0]) & (mx <= v[1])
    if h is not None:
        m &= ((hue >= h[0]) & (hue <= h[1])) if h[0] <= h[1] else ((hue >= h[0]) | (hue <= h[1]))
    if box is not None:
        bm = np.zeros_like(m)
        bm[box[1]:box[3], box[0]:box[2]] = True
        m &= bm
    return m


def gradient_map(img: Image.Image, mask: np.ndarray, stops: list) -> None:
    """Recolour masked pixels by luminance through colour stops (dark -> light), in place.

    stops: ['#2a1200', '#c98a1a', '#ffe9a0'] maps black..white across those colours.
    Good for turning fur gold, a shell into jade, feathers into white-hot fire.
    """
    a = np.array(img.convert("RGBA")).astype(np.float64)
    lum = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]) / 255
    cols = np.array([hexrgba(c)[:3] for c in stops], np.float64)
    pos = np.linspace(0, 1, len(cols))
    t = lum[mask]
    out = np.stack([np.interp(t, pos, cols[:, k]) for k in range(3)], -1)
    a[..., :3][mask] = out
    img.paste(Image.fromarray(a.clip(0, 255).astype(np.uint8), "RGBA"))


def shift_hsv(img: Image.Image, mask: np.ndarray, dh: float = 0, ds: float = 1, dv: float = 1) -> None:
    """Rotate hue by dh degrees, scale saturation/value, for masked pixels, in place."""
    a = np.array(img.convert("RGBA"))
    ys, xs = np.where(mask)
    for y, x in zip(ys, xs):
        r, g, b = a[y, x, :3] / 255
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        r, g, b = colorsys.hsv_to_rgb((h + dh / 360) % 1, min(1, s * ds), min(1, v * dv))
        a[y, x, :3] = (r * 255, g * 255, b * 255)
    img.paste(Image.fromarray(a, "RGBA"))


def copy_part(img: Image.Image, box, dest: Point, scale: float = 1.0, angle: float = 0.0,
              flip: bool = False, mask: np.ndarray | None = None) -> None:
    """Copy a region (optionally masked) of img, transform it and composite it at dest (top-left), in place."""
    part = img.crop(box)
    if mask is not None:
        m = Image.fromarray((mask[box[1]:box[3], box[0]:box[2]] * 255).astype(np.uint8))
        alpha = Image.fromarray(np.minimum(np.array(part)[..., 3], np.array(m)))
        part.putalpha(alpha)
    if flip:
        part = part.transpose(Image.FLIP_LEFT_RIGHT)
    if scale != 1.0:
        part = part.resize((max(1, round(part.width * scale)), max(1, round(part.height * scale))), Image.LANCZOS)
    if angle:
        part = part.rotate(angle, resample=Image.BICUBIC, expand=True)
    img.alpha_composite(part, (int(dest[0]), int(dest[1])))


def behind(img: Image.Image, layer: Image.Image) -> Image.Image:
    """Return layer composited UNDER img (for wings, auras, tails that go behind the body)."""
    out = layer.copy()
    out.alpha_composite(img)
    return out


def blank(img: Image.Image) -> Image.Image:
    return Image.new("RGBA", img.size, (0, 0, 0, 0))


def soft_glow(layer: Image.Image, color, radius: float) -> Image.Image:
    """A blurred coloured copy of the layer's silhouette (keep it mostly opaque or it vanishes when pixelized)."""
    a = np.array(layer)[..., 3]
    g = Image.new("RGBA", layer.size, hexrgba(color))
    g.putalpha(Image.fromarray(a).filter(ImageFilter.GaussianBlur(radius)))
    return g
