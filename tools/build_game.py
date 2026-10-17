"""Build the game sprites (3 evolution stages per pet) and the demo's data file.

    python tools/build_game.py                    # every pet
    python tools/build_game.py kim-quy            # one pet
    python tools/build_game.py kim-quy --lineup /tmp/kq.png   # original + 3 stages at 3x
    python tools/build_game.py kim-quy --zoom 2 40 20 150 120 --out /tmp/z.png
                                                  # stage 2, sprite box x0 y0 x1 y1, gridded, 6x

Writes game/sprites/<slug>-<n>.png and <slug>-<n>-blink.png (192x192, indexed),
game/sprites/lineup.png (all pets x stages) and game/data/pets.js.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gamekit import CANVAS, GAME, GROUND, PETS, build_stage, clipped_edges, load_pet  # noqa: E402
from pixelize import reference  # noqa: E402

BG = (24, 22, 38, 255)


def checker(w: int, h: int, cell: int = 8) -> Image.Image:
    im = Image.new("RGBA", (w, h), BG)
    d = ImageDraw.Draw(im)
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            if (x // cell + y // cell) % 2:
                d.rectangle([x, y, x + cell - 1, y + cell - 1], fill=(32, 30, 50, 255))
    return im


def lineup(slug: str, sprites: list[Image.Image], z: int = 3) -> Image.Image:
    side = CANVAS * z
    out = checker(side * (len(sprites) + 1), side)
    ref = reference(slug)
    out.alpha_composite(ref.resize((side, side), Image.LANCZOS), (0, 0))
    d = ImageDraw.Draw(out)
    for i, sp in enumerate(sprites, start=1):
        out.alpha_composite(sp.resize((side, side), Image.NEAREST), (i * side, 0))
        d.line([(i * side, GROUND * z), ((i + 1) * side, GROUND * z)], fill=(90, 80, 140, 255))
        d.text((i * side + 6, 6), f"stage {i}", fill=(200, 200, 230, 255))
    return out


def zoom(img: Image.Image, box, z: int = 6) -> Image.Image:
    x0, y0, x1, y1 = box
    crop = img.crop(box).resize(((x1 - x0) * z, (y1 - y0) * z), Image.NEAREST)
    pad = 28
    out = checker(crop.width + pad, crop.height + pad, z)
    out.alpha_composite(crop, (pad, pad))
    d = ImageDraw.Draw(out)
    for gx in range(x0, x1 + 1):
        if gx % 4 == 0:
            X = pad + (gx - x0) * z
            d.line([(X, pad), (X, pad + crop.height)], fill=(255, 0, 90, 140 if gx % 16 == 0 else 50))
            if gx % 16 == 0:
                d.text((X + 1, 4), str(gx), fill=(255, 120, 160, 255))
    for gy in range(y0, y1 + 1):
        if gy % 4 == 0:
            Y = pad + (gy - y0) * z
            d.line([(pad, Y), (pad + crop.width, Y)], fill=(255, 0, 90, 140 if gy % 16 == 0 else 50))
            if gy % 16 == 0:
                d.text((2, Y + 1), str(gy), fill=(255, 120, 160, 255))
    return out


def pet_record(slug: str, mod) -> dict:
    stages = []
    for n, st in enumerate(mod.STAGES, start=1):
        rec = {k: v for k, v in st.items() if k not in ("config", "height", "dx")}
        rec["sprite"] = f"sprites/{slug}-{n}.png"
        rec["blink"] = f"sprites/{slug}-{n}-blink.png"
        stages.append(rec)
    return {"slug": slug, **mod.PET, "stages": stages}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("slugs", nargs="*", default=PETS)
    ap.add_argument("--lineup")
    ap.add_argument("--zoom", nargs=5, type=int, metavar=("STAGE", "X0", "Y0", "X1", "Y1"))
    ap.add_argument("--blink", action="store_true", help="with --zoom: show the blink frame")
    ap.add_argument("--out")
    args = ap.parse_args()

    sprites_dir = os.path.join(GAME, "sprites")
    os.makedirs(sprites_dir, exist_ok=True)
    os.makedirs(os.path.join(GAME, "data"), exist_ok=True)

    for slug in args.slugs:
        mod = load_pet(slug)
        imgs = []
        for n in (1, 2, 3):
            if args.zoom and args.zoom[0] != n:
                continue
            art, closed, _ = build_stage(slug, n, mod)
            art.save(os.path.join(sprites_dir, f"{slug}-{n}.png"))
            closed.save(os.path.join(sprites_dir, f"{slug}-{n}-blink.png"))
            imgs.append(art.image())
            edges = clipped_edges(art)
            used = len(set(art.idx.ravel().tolist()) - {-1})
            print(f"{slug} stage {n}: {used} colours" + (f"  WARNING touches {edges}" if edges else ""))
            if args.zoom:
                img = closed.image() if args.blink else art.image()
                zoom(img, args.zoom[1:]).save(args.out or f"/tmp/{slug}-{n}-zoom.png")
        if args.lineup and not args.zoom:
            lineup(slug, imgs).save(args.lineup if len(args.slugs) == 1 else
                                    args.lineup.replace(".png", f"-{slug}.png"))

    if args.zoom:
        return

    # Data file for the demo + an overview image, from every pet module.
    records = [pet_record(s, load_pet(s)) for s in PETS]
    with open(os.path.join(GAME, "data", "pets.js"), "w", encoding="utf-8") as f:
        f.write("// Generated by tools/build_game.py from game/pets/*.py. Do not edit.\n")
        f.write("window.IQX_PETS = " + json.dumps(records, ensure_ascii=False, indent=2) + ";\n")

    rows = []
    for s in PETS:
        row = [Image.open(os.path.join(sprites_dir, f"{s}-{n}.png")).convert("RGBA")
               for n in (1, 2, 3) if os.path.exists(os.path.join(sprites_dir, f"{s}-{n}.png"))]
        rows.append(row)
    z = 2
    sheet = checker(CANVAS * z * 3, CANVAS * z * len(rows))
    for j, row in enumerate(rows):
        for i, im in enumerate(row):
            sheet.alpha_composite(im.resize((CANVAS * z, CANVAS * z), Image.NEAREST), (i * CANVAS * z, j * CANVAS * z))
    sheet.save(os.path.join(sprites_dir, "lineup.png"))


if __name__ == "__main__":
    main()
