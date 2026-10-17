"""Zoomed side-by-side of the same region in the original and in the sprite.

    python tools/inspect_region.py kim-quy 180 120 330 240 --zoom 4 --out /tmp/face.png
    python tools/inspect_region.py kim-quy --full --out /tmp/full.png

Coordinates are sprite pixels (0..511), inclusive-exclusive. A light grid every
8 sprite pixels, with labels every 16, helps place touch-ups exactly.
"""
from __future__ import annotations

import argparse
import os
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build import build  # noqa: E402
from pixelize import DEFAULTS, ROOT, placement, reference  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("box", nargs="*", type=int, help="x0 y0 x1 y1 in sprite pixels")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--zoom", type=int, default=4)
    ap.add_argument("--sprite", help="read this PNG instead of rebuilding")
    ap.add_argument("--no-grid", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.sprite:
        spr = Image.open(args.sprite).convert("RGBA")
    else:
        spr = build(args.slug).image()
    size = spr.width
    x0, y0, x1, y1 = (0, 0, size, size) if args.full or not args.box else args.box
    z = args.zoom

    # Same region in the reference, resampled onto the sprite grid.
    (bx0, by0, bx1, by1), (ox, oy, nw, nh) = placement(args.slug, size, DEFAULTS["margin"])
    sx, sy = (bx1 - bx0) / nw, (by1 - by0) / nh
    ref_box = (bx0 + (x0 - ox) * sx, by0 + (y0 - oy) * sy, bx0 + (x1 - ox) * sx, by0 + (y1 - oy) * sy)
    w, h = (x1 - x0) * z, (y1 - y0) * z
    ref = reference(args.slug).resize((w, h), Image.LANCZOS, box=ref_box)
    crop = spr.crop((x0, y0, x1, y1)).resize((w, h), Image.NEAREST)

    pad = 30
    out = Image.new("RGBA", (w * 2 + pad * 3, h + pad * 2), (255, 255, 255, 255))
    out.alpha_composite(ref, (pad, pad))
    out.alpha_composite(crop, (w + pad * 2, pad))
    d = ImageDraw.Draw(out)
    if not args.no_grid:
        for ox_ in (pad, w + pad * 2):
            for gx in range(x0 - x0 % 8, x1 + 1, 8):
                if gx < x0:
                    continue
                X = ox_ + (gx - x0) * z
                d.line([(X, pad), (X, pad + h)], fill=(255, 0, 90, 60 if gx % 16 else 130))
                if gx % 16 == 0:
                    d.text((X + 1, pad - 12), str(gx), fill=(200, 0, 60, 255))
            for gy in range(y0 - y0 % 8, y1 + 1, 8):
                if gy < y0:
                    continue
                Y = pad + (gy - y0) * z
                d.line([(ox_, Y), (ox_ + w, Y)], fill=(255, 0, 90, 60 if gy % 16 else 130))
                if gy % 16 == 0:
                    d.text((ox_ - 26, Y + 1), str(gy), fill=(200, 0, 60, 255))
    out.save(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
