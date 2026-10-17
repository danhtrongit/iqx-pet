"""Game sprites: three evolution stages per pet on one fixed canvas.

Every stage starts from the pet's original illustration placed on a padded
2048x2048 canvas (the "art canvas"). A pet module may draw on that canvas
(compose), then the result is pixelized onto a CANVAS x CANVAS sprite where
the original body is `height` pixels tall, centred, feet on GROUND. Because
all stages share the canvas, ground line and pixel size, the pet visibly grows
as it evolves.

Pet modules live in game/pets/<slug_with_underscores>.py:

    PET = {"name": ..., "layer": ..., "element": ..., "accent": "#rrggbb", ...}
    STAGES = [ {"name": ..., "height": 104, "config": {...}, ...}, {...}, {...} ]
    def compose(stage, img, ctx): ...  # optional, draw on the 2048 art canvas, return it
    def touchup(stage, art, ctx): ...  # optional, pixel fixes on the final sprite
    def blink(stage, art, ctx): ...    # optional, close the eyes (called on a copy)

`stage` is 1, 2 or 3. `ctx` is a StageContext with the coordinate helpers.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import importlib.util
import os

import numpy as np
from PIL import Image

from pixelize import (TRANSPARENT, Art, grow4, kmeans, majority, merge_islands,
                      nearest, reference, rgb2lab)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAME = os.path.join(ROOT, "game")
CANVAS = 192          # sprite is CANVAS x CANVAS
GROUND = 188          # first row below the feet
PAD = 512             # reference poster sits at (PAD, PAD) on the art canvas
ART = 1024 + 2 * PAD  # art canvas size

PETS = ["thanh-long", "loc-huou", "phung-hoang", "kim-quy"]

PIXEL_DEFAULTS = {
    "colors": 28,       # palette size, outline excluded
    "ink_l": 34.0,      # Lab L* under which an art pixel counts as ink
    "ink_frac": 0.45,   # share of ink inside a sprite pixel to draw it as outline
    "smooth": 1,        # majority-filter passes
    "min_island": 3,    # colour regions smaller than this merge into a neighbour
    "seed": 0,
    "outline": None,    # '#rrggbb' to force the outline colour
}


def body_box(slug: str) -> tuple[int, int, int, int]:
    """Bounding box of the original body on the art canvas."""
    a = np.array(reference(slug))[..., 3]
    ys, xs = np.where(a > 16)
    return int(xs.min()) + PAD, int(ys.min()) + PAD, int(xs.max()) + 1 + PAD, int(ys.max()) + 1 + PAD


@dataclass
class StageContext:
    slug: str
    stage: int
    height: int
    scale: float          # sprite px per art px
    ox: float             # sprite = art * scale + (ox, oy)
    oy: float
    box: tuple[int, int, int, int]

    def to_sprite(self, x: float, y: float) -> tuple[int, int]:
        """Art-canvas coordinates -> sprite pixel."""
        return int(np.floor(x * self.scale + self.ox)), int(np.floor(y * self.scale + self.oy))

    def to_art(self, x: float, y: float) -> tuple[float, float]:
        """Sprite pixel (its centre) -> art-canvas coordinates."""
        return (x + 0.5 - self.ox) / self.scale, (y + 0.5 - self.oy) / self.scale

    @staticmethod
    def ref(x: float, y: float) -> tuple[float, float]:
        """reference/<slug>.webp pixel -> art-canvas coordinates."""
        return x + PAD, y + PAD


def stage_context(slug: str, stage: int, height: int, dx: float = 0.0) -> StageContext:
    box = body_box(slug)
    scale = height / (box[3] - box[1])
    cx = (box[0] + box[2]) / 2
    ox = CANVAS / 2 + dx - cx * scale
    oy = GROUND - box[3] * scale
    return StageContext(slug, stage, height, scale, ox, oy, box)


def art_canvas(slug: str) -> Image.Image:
    img = Image.new("RGBA", (ART, ART), (0, 0, 0, 0))
    img.alpha_composite(reference(slug), (PAD, PAD))
    return img


def _place(arr: np.ndarray, ctx: StageContext, resample) -> np.ndarray:
    """Resize an art-canvas array (H, W, C) into sprite space with clipping."""
    n = max(1, round(ART * ctx.scale))
    mode = {4: "RGBA", 1: "L"}[arr.shape[2]]
    im = Image.fromarray(arr if arr.shape[2] == 4 else arr[..., 0], mode)
    if mode == "RGBA":
        small = np.array(im.convert("RGBa").resize((n, n), resample).convert("RGBA")).astype(np.float64)
    else:
        small = np.array(im.resize((n, n), resample)).astype(np.float64)[..., None]
    out = np.zeros((CANVAS, CANVAS, small.shape[2]))
    ox, oy = round(ctx.ox), round(ctx.oy)
    sx0, sy0 = max(0, -ox), max(0, -oy)
    dx0, dy0 = max(0, ox), max(0, oy)
    w = min(n - sx0, CANVAS - dx0)
    h = min(n - sy0, CANVAS - dy0)
    if w > 0 and h > 0:
        out[dy0:dy0 + h, dx0:dx0 + w] = small[sy0:sy0 + h, sx0:sx0 + w]
    return out


def pixelize_art(img: Image.Image, ctx: StageContext, **opts) -> Art:
    o = {**PIXEL_DEFAULTS, **opts}
    hi = np.array(img.convert("RGBA"))
    # Only the area that can land on the sprite matters; skip the rest for speed.
    lab_l = np.zeros(hi.shape[:2])
    op = hi[..., 3] > 128
    lab_l[op] = rgb2lab(hi[op][:, :3])[:, 0]
    ink = ((lab_l < o["ink_l"]) & op).astype(np.uint8)[..., None] * 255

    canvas = _place(hi, ctx, Image.BOX)
    frac = _place(ink, ctx, Image.BOX)[..., 0] / 255
    mask = canvas[..., 3] >= 128

    px = canvas[mask][:, :3]
    lab = rgb2lab(px)
    centers = kmeans(lab, min(o["colors"], len(px)), o["seed"])
    labels = nearest(lab, centers)
    pal = [tuple(int(round(v)) for v in px[labels == j].mean(0)) if (labels == j).any() else (255, 0, 255)
           for j in range(len(centers))]

    if o["outline"]:
        s = o["outline"].lstrip("#")
        outline = (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    else:
        dark = hi[op & (lab_l < min(30.0, o["ink_l"]))][:, :3]
        outline = tuple(int(round(v)) for v in dark.mean(0)) if len(dark) else (20, 30, 60)
    pal.append(outline)
    k_out = len(pal) - 1

    idx = np.full((CANVAS, CANVAS), TRANSPARENT, int)
    idx[mask] = labels
    idx[mask & (frac >= o["ink_frac"])] = k_out
    idx = majority(idx, len(pal), o["smooth"])
    idx = merge_islands(idx, o["min_island"], keep=(k_out,))
    idx[grow4(idx != TRANSPARENT) & (idx == TRANSPARENT)] = k_out
    return Art(idx, pal, {"outline": k_out})


def load_pet(slug: str):
    path = os.path.join(GAME, "pets", f"{slug.replace('-', '_')}.py")
    spec = importlib.util.spec_from_file_location(f"game_pet_{slug.replace('-', '_')}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_stage(slug: str, stage: int, mod=None) -> tuple[Art, Art, StageContext]:
    """Returns (sprite, blink sprite, context) for stage 1..3."""
    mod = mod or load_pet(slug)
    st = mod.STAGES[stage - 1]
    ctx = stage_context(slug, stage, st["height"], st.get("dx", 0.0))
    img = art_canvas(slug)
    if hasattr(mod, "compose"):
        img = mod.compose(stage, img, ctx) or img
    art = pixelize_art(img, ctx, **st.get("config", {}))
    if hasattr(mod, "touchup"):
        mod.touchup(stage, art, ctx)
    closed = Art(art.idx.copy(), list(art.palette), dict(art.names))
    if hasattr(mod, "blink"):
        mod.blink(stage, closed, ctx)
    return art, closed, ctx


def clipped_edges(art: Art) -> list[str]:
    """Which canvas edges the sprite touches (a sign it was cut off)."""
    m = art.idx != TRANSPARENT
    return [name for name, edge in (("top", m[0]), ("bottom", m[-1]), ("left", m[:, 0]), ("right", m[:, -1]))
            if edge.any()]


def deepcopy_art(art: Art) -> Art:
    return copy.deepcopy(art)
