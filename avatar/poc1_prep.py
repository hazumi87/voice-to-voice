#!/usr/bin/env python3
"""PoC1 prep - mouth mask + crop tile for viseme sprite generation.

GENERATOR-AGNOSTIC. Produces the inputs BOTH paths need (local Flux inpaint OR
Gemini): the base face, a mouth-region mask, the cropped mouth tile, and a feather
mask for clean recompositing. No GPU, no API, no model - just PIL. This is the
unglamorous shared prep so that once a generator is chosen, sprite gen is fast.

The mouth box was pinned visually against working/reference_chat.png (2528x1696):
the smiling-with-teeth mouth sits at ~0.40-0.60 W, ~0.53-0.68 H. We expand it a bit
so the generator has lip + chin + cheek context (inpaint needs margin to blend).

Outputs (avatar/poc1/):
  base.png            - full face, unchanged (the canvas)
  mouth_tile.png      - the cropped mouth region (the thing a generator edits)
  mouth_mask.png      - white = editable mouth area, black = keep (for inpaint)
  mouth_mask_feather.png - soft-edged version for recompositing
  box.json            - the exact pixel box + feather, for the assembler to reverse
"""
import os, json
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "working", "reference_chat_neutral.png")
OUT = os.path.join(HERE, "poc1")

# Mouth box as fractions of the full image (pinned visually). The "edit" box is the
# tight mouth; the "tile" box is expanded for inpaint context.
# NEUTRAL base (closed neutral mouth) - re-pinned for the smaller, closed mouth which
# sits slightly lower than the old smiling source. This is the correct canvas: opening
# a mouth FROM neutral is far more reliable than restructuring an existing smile.
EDIT = (0.430, 0.555, 0.570, 0.650)   # the actual closed mouth - where the mask is white
TILE = (0.395, 0.510, 0.605, 0.700)   # the crop fed to the generator (with margin)
FEATHER_PX = 24                        # soft edge for recompositing


def frac_box(W, H, fr):
    return (int(W * fr[0]), int(H * fr[1]), int(W * fr[2]), int(H * fr[3]))


def main():
    os.makedirs(OUT, exist_ok=True)
    im = Image.open(SRC).convert("RGB")
    W, H = im.size
    edit = frac_box(W, H, EDIT)
    tile = frac_box(W, H, TILE)

    im.save(os.path.join(OUT, "base.png"))

    # crop tile (generator input)
    mouth_tile = im.crop(tile)
    mouth_tile.save(os.path.join(OUT, "mouth_tile.png"))

    # hard mask in TILE-local coords: white ellipse over the edit region
    tw, th = tile[2] - tile[0], tile[3] - tile[1]
    mask = Image.new("L", (tw, th), 0)
    d = ImageDraw.Draw(mask)
    # edit box relative to the tile origin
    ex0, ey0 = edit[0] - tile[0], edit[1] - tile[1]
    ex1, ey1 = edit[2] - tile[0], edit[3] - tile[1]
    d.ellipse((ex0, ey0, ex1, ey1), fill=255)
    mask.save(os.path.join(OUT, "mouth_mask.png"))

    feather = mask.filter(ImageFilter.GaussianBlur(FEATHER_PX))
    feather.save(os.path.join(OUT, "mouth_mask_feather.png"))

    with open(os.path.join(OUT, "box.json"), "w", encoding="utf-8") as f:
        json.dump({"image_size": [W, H], "edit_box": edit, "tile_box": tile,
                   "tile_size": [tw, th], "feather_px": FEATHER_PX,
                   "edit_in_tile": [ex0, ey0, ex1, ey1]}, f, indent=2)

    print(f"[poc1-prep] base {W}x{H}")
    print(f"[poc1-prep] edit box {edit}")
    print(f"[poc1-prep] tile box {tile}  ({tw}x{th})")
    print(f"[poc1-prep] wrote base.png, mouth_tile.png, mouth_mask.png, "
          f"mouth_mask_feather.png, box.json -> {OUT}")


if __name__ == "__main__":
    main()
