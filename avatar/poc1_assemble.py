#!/usr/bin/env python3
"""PoC1 assembler - composite generated mouth tiles back into full-face sprites.

GENERATOR-AGNOSTIC. Whatever produced the edited mouth tiles (local Flux inpaint
or Gemini), drop them in avatar/poc1/tiles/ named <ID>.png (01.png, 02.png, ... 10.png,
each the same size as mouth_tile.png / box.json tile_size). This script feathers each
back onto base.png at the exact mask location and writes mouth_<ID>.png sprites that
match the viseme_map contract - the exact files PoC3 PixiJS will swap.

If a tile is the FULL face (some generators return the whole edited image rather than
just the tile), pass --full and it crops the tile region out before compositing - so
the rest of the face stays byte-identical to base across all 9 sprites (critical: only
the mouth should change between frames, nothing else).

Run:
  .venv/Scripts/python.exe avatar/poc1_assemble.py            # tiles are mouth-sized
  .venv/Scripts/python.exe avatar/poc1_assemble.py --full     # tiles are full-face
"""
import os, json, argparse, sys
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
POC1 = os.path.join(HERE, "poc1")
TILES = os.path.join(POC1, "tiles")
SPRITES = os.path.join(POC1, "sprites")

# Numbered scheme "blair-10-numbered" rendered_set (id 7 aliases 6, not rendered).
IDS = ["01", "02", "03", "04", "05", "06", "08", "09", "10"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="tiles are full-face images; crop the tile region first")
    args = ap.parse_args()

    box = json.load(open(os.path.join(POC1, "box.json")))
    tile_box = tuple(box["tile_box"])
    tw, th = box["tile_size"]
    base = Image.open(os.path.join(POC1, "base.png")).convert("RGBA")
    feather = Image.open(os.path.join(POC1, "mouth_mask_feather.png")).convert("L")

    os.makedirs(SPRITES, exist_ok=True)
    if not os.path.isdir(TILES):
        sys.exit(f"[poc1-assemble] no tiles dir yet: {TILES}\n"
                 f"   drop generated mouth tiles there as A.png..X.png, then re-run.")

    made = []
    for vid in IDS:
        tpath = os.path.join(TILES, f"{vid}.png")
        if not os.path.exists(tpath):
            print(f"[poc1-assemble] SKIP {vid} (no tile {tpath})")
            continue
        tile = Image.open(tpath).convert("RGBA")
        if args.full:
            tile = tile.crop(tile_box)
        if tile.size != (tw, th):
            tile = tile.resize((tw, th), Image.LANCZOS)
        sprite = base.copy()
        sprite.paste(tile, (tile_box[0], tile_box[1]), feather)
        outp = os.path.join(SPRITES, f"mouth_{vid}.png")
        sprite.convert("RGB").save(outp)
        made.append(vid)
        print(f"[poc1-assemble] mouth_{vid}.png")

    print(f"\n[poc1-assemble] built {len(made)}/9 sprites -> {SPRITES}")
    missing = [v for v in IDS if v not in made]
    if missing:
        print(f"[poc1-assemble] still missing: {', '.join(missing)}")


if __name__ == "__main__":
    main()
