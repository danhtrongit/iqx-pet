"""Build the pixel sprites.

    python tools/build.py                 # all pets
    python tools/build.py kim-quy         # one pet
    python tools/build.py --size 128      # quick low-res preview (touch-ups skipped)

For every pet: pixelize(reference) with pets/<slug>.py CONFIG, then that
file's touchup(art), then write
    sprites/<slug>.png          native sprite, indexed PNG
    dist/<slug>@2x.png          nearest-neighbour upscale
    dist/compare/<slug>.png     original | pixel, side by side
    dist/sheet.png              all pets in one row
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pixelize import DEFAULTS, ROOT, pixelize, reference  # noqa: E402

PETS = ["thanh-long", "loc-huou", "phung-hoang", "kim-quy"]


def load_pet(slug: str):
    path = os.path.join(ROOT, "pets", f"{slug.replace('-', '_')}.py")
    if not os.path.exists(path):
        return {}, None
    spec = importlib.util.spec_from_file_location(f"pet_{slug.replace('-', '_')}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "CONFIG", {}), getattr(mod, "touchup", None)


def build(slug: str, size: int | None = None):
    config, touchup = load_pet(slug)
    if size is not None and size != config.get("size", DEFAULTS["size"]):
        # Touch-ups are drawn in native sprite coordinates, so skip them at other sizes.
        return pixelize(slug, **{**config, "size": size})
    art = pixelize(slug, **config)
    if touchup:
        touchup(art)
    return art


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("slugs", nargs="*", default=PETS)
    ap.add_argument("--size", type=int)
    ap.add_argument("--out", default=ROOT)
    args = ap.parse_args()

    sprites = os.path.join(args.out, "sprites")
    dist = os.path.join(args.out, "dist")
    os.makedirs(sprites, exist_ok=True)
    os.makedirs(os.path.join(dist, "compare"), exist_ok=True)

    for slug in args.slugs:
        art = build(slug, args.size)
        art.save(os.path.join(sprites, f"{slug}.png"))
        img = art.image()
        side = img.width
        img.resize((side * 2, side * 2), Image.NEAREST).save(os.path.join(dist, f"{slug}@2x.png"), optimize=True)

        shown = max(side, 512)
        cmp_ = Image.new("RGBA", (shown * 2, shown), (255, 255, 255, 255))
        cmp_.alpha_composite(reference(slug).resize((shown, shown), Image.LANCZOS), (0, 0))
        cmp_.alpha_composite(img.resize((shown, shown), Image.NEAREST), (shown, 0))
        cmp_.save(os.path.join(dist, "compare", f"{slug}.png"))
        used = len(set(art.idx.ravel().tolist()) - {-1})
        print(f"{slug}: {side}x{side}, {used} colours")

    # Sprite sheet from whatever is in sprites/ for the canonical pets.
    built = [os.path.join(sprites, f"{s}.png") for s in PETS if os.path.exists(os.path.join(sprites, f"{s}.png"))]
    if built:
        imgs = [Image.open(p).convert("RGBA") for p in built]
        w, h = imgs[0].size
        sheet = Image.new("RGBA", (w * len(imgs), h), (0, 0, 0, 0))
        for i, im in enumerate(imgs):
            sheet.alpha_composite(im.resize((w, h), Image.NEAREST), (i * w, 0))
        sheet.save(os.path.join(dist, "sheet.png"), optimize=True)


if __name__ == "__main__":
    main()
