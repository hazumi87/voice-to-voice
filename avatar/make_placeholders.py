"""
make_placeholders.py
Generates 9 placeholder mouth sprites (512x512 RGBA) for lip-sync prototype.
Reads viseme_map.json and writes one PNG per viseme into avatar/web/public/sprites/.
Re-runnable: overwrites existing files.
Uses PIL only (no cv2, no rembg, no external TTF required).
"""

import json
import math
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
VISEME_MAP = SCRIPT_DIR / "viseme_map.json"
OUT_DIR    = SCRIPT_DIR / "web" / "public" / "sprites"

# ── Canvas constants ───────────────────────────────────────────────────────────
SIZE        = 512
CX, CY      = SIZE // 2, SIZE // 2          # canvas center
MOUTH_W     = 320                           # max horizontal mouth width
MOUTH_H_MAX = 220                           # max vertical opening at openness=1.0

# ── Colour palette ─────────────────────────────────────────────────────────────
COL_OUTLINE  = (30,  20,  20,  255)
COL_LIP      = (210, 100, 110, 255)
COL_LIP_DK   = (170,  60,  70, 255)
COL_INTERIOR = (60,   20,  20, 255)
COL_TEETH    = (240, 235, 225, 255)
COL_TONGUE   = (220, 100, 100, 255)
COL_LABEL_BG = (0,    0,   0, 140)
COL_LABEL_FG = (255, 255, 255, 255)
COL_ID_FG    = (255, 220,  80, 255)


def load_font(size: int) -> ImageFont.ImageFont:
    """Load a font at the requested size; fall back to PIL default."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except (IOError, OSError):
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except (IOError, OSError):
            return ImageFont.load_default()


def bbox_center(draw, text, font):
    """Return (w, h) of text bounding box."""
    bb = draw.textbbox((0, 0), text, font=font)
    return bb[2] - bb[0], bb[3] - bb[1]


def draw_label_overlay(draw: ImageDraw.ImageDraw, vid: str, label: str):
    """Draw viseme ID (large, top-right) and label (small, bottom-left)."""
    font_big   = load_font(72)
    font_small = load_font(26)

    # Viseme ID — top-right corner
    iw, ih = bbox_center(draw, vid, font_big)
    pad = 14
    ix = SIZE - iw - pad - 8
    iy = pad
    draw.rectangle([ix - 6, iy - 4, ix + iw + 6, iy + ih + 4], fill=COL_LABEL_BG)
    draw.text((ix, iy), vid, font=font_big, fill=COL_ID_FG)

    # Label — bottom-left corner
    lw, lh = bbox_center(draw, label, font_small)
    lx, ly = pad, SIZE - lh - pad - 8
    draw.rectangle([lx - 4, ly - 3, lx + lw + 4, ly + lh + 3], fill=COL_LABEL_BG)
    draw.text((lx, ly), label, font=font_small, fill=COL_LABEL_FG)


# ── Per-viseme drawing functions ───────────────────────────────────────────────

def draw_closed(img: Image.Image, lip_color=COL_LIP, pressed=False):
    """Closed/rest mouth: a thin flat ellipse with lip outline."""
    draw = ImageDraw.Draw(img)
    lip_h = 18 if not pressed else 22
    x0 = CX - MOUTH_W // 2
    x1 = CX + MOUTH_W // 2
    # Upper lip arc
    draw.ellipse([x0, CY - lip_h, x1, CY + lip_h], fill=lip_color, outline=COL_OUTLINE, width=3)
    # Cupid's bow suggestion — dark center line
    draw.arc([x0, CY - lip_h, x1, CY + lip_h], start=0, end=180, fill=COL_OUTLINE, width=3)


def draw_open(img: Image.Image, openness: float,
              show_teeth: bool = False, show_tongue: bool = False,
              teeth_top: bool = True, puckered: bool = False,
              teeth_bottom: bool = False):
    """
    Generic open-mouth shape driven by openness (0..1).
    puckered: narrows the width and makes the opening round.
    """
    draw = ImageDraw.Draw(img)

    h = max(10, int(MOUTH_H_MAX * openness))

    if puckered:
        w = int(MOUTH_W * 0.38)
    else:
        w = MOUTH_W

    x0 = CX - w // 2
    x1 = CX + w // 2
    y0 = CY - h // 2
    y1 = CY + h // 2

    # ── Mouth interior ──────────────────────────────────────────────────────
    draw.ellipse([x0, y0, x1, y1], fill=COL_INTERIOR)

    # ── Teeth ───────────────────────────────────────────────────────────────
    teeth_h = min(32, h // 3)
    if show_teeth and teeth_top and h > 20:
        draw.rectangle([x0 + 4, y0 + 2, x1 - 4, y0 + teeth_h], fill=COL_TEETH)
        # tooth dividers
        tw = (w - 8) // 5
        for i in range(1, 5):
            tx = x0 + 4 + i * tw
            draw.line([(tx, y0 + 4), (tx, y0 + teeth_h - 2)], fill=COL_OUTLINE, width=1)

    if show_teeth and teeth_bottom and h > 20:
        draw.rectangle([x0 + 4, y1 - teeth_h, x1 - 4, y1 - 2], fill=COL_TEETH)

    # ── Tongue ──────────────────────────────────────────────────────────────
    if show_tongue and h > 30:
        ty0 = CY - h // 5
        ty1 = CY + h // 2 - 4
        tx0 = CX - int(w * 0.28)
        tx1 = CX + int(w * 0.28)
        draw.ellipse([tx0, ty0, tx1, ty1], fill=COL_TONGUE, outline=COL_LIP_DK, width=2)

    # ── Outer lip ellipse ────────────────────────────────────────────────────
    lip_pad = 14 if not puckered else 10
    lx0 = x0 - lip_pad
    lx1 = x1 + lip_pad
    ly0 = y0 - lip_pad
    ly1 = y1 + lip_pad
    draw.ellipse([lx0, ly0, lx1, ly1], outline=COL_LIP, width=lip_pad)
    draw.ellipse([lx0, ly0, lx1, ly1], outline=COL_OUTLINE, width=3)


def draw_teeth_on_lip(img: Image.Image):
    """G — upper teeth resting on lower lip (labiodental f/v)."""
    draw = ImageDraw.Draw(img)
    # Lower lip — wide arc
    lip_y = CY + 30
    lx0 = CX - MOUTH_W // 2
    lx1 = CX + MOUTH_W // 2
    draw.ellipse([lx0, lip_y - 20, lx1, lip_y + 40], fill=COL_LIP, outline=COL_OUTLINE, width=3)

    # Upper teeth bar sitting on top of lower lip
    teeth_h = 28
    tx0 = CX - int(MOUTH_W * 0.38)
    tx1 = CX + int(MOUTH_W * 0.38)
    ty0 = lip_y - teeth_h + 4
    ty1 = lip_y + 6
    draw.rectangle([tx0, ty0, tx1, ty1], fill=COL_TEETH, outline=COL_OUTLINE, width=2)
    # dividers
    tw = (tx1 - tx0) // 5
    for i in range(1, 5):
        tx = tx0 + i * tw
        draw.line([(tx, ty0 + 3), (tx, ty1 - 3)], fill=COL_OUTLINE, width=1)

    # Upper lip arc above teeth
    draw.arc([lx0, CY - 60, lx1, lip_y + 10], start=200, end=340, fill=COL_LIP_DK, width=14)
    draw.arc([lx0, CY - 60, lx1, lip_y + 10], start=200, end=340, fill=COL_OUTLINE, width=3)


# ── Main generator ─────────────────────────────────────────────────────────────

# Numbered scheme "blair-10-numbered". Keys are viseme id strings 1..10.
# id 7 (W/Q) aliases id 6 and is not drawn (it shares mouth_06.png).
VISEME_DRAW = {
    # 1 — closed M/B/P: pressed lips
    "1": lambda img, v: draw_closed(img, pressed=True),

    # 2 — consonants: slightly open, teeth close
    "2": lambda img, v: draw_open(img, v["openness"], show_teeth=True, teeth_top=True, teeth_bottom=True),

    # 3 — open E/eh
    "3": lambda img, v: draw_open(img, v["openness"], show_teeth=True, teeth_top=True),

    # 4 — wide open A/ah
    "4": lambda img, v: draw_open(img, v["openness"], show_teeth=True, teeth_top=True, show_tongue=False),

    # 5 — rounded O
    "5": lambda img, v: draw_open(img, v["openness"], show_teeth=False),

    # 6 — pucker U
    "6": lambda img, v: draw_open(img, v["openness"], puckered=True),

    # 8 — teeth on lip F/V
    "8": lambda img, v: draw_teeth_on_lip(img),

    # 9 — tongue visible L
    "9": lambda img, v: draw_open(img, v["openness"], show_teeth=True, show_tongue=True, teeth_top=True),

    # 10 — rest: thin closed, neutral lip line
    "10": lambda img, v: draw_closed(img, pressed=False),
}


def make_sprite(vid: str, vdata: dict, out_path: Path):
    img  = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw_fn = VISEME_DRAW.get(vid)
    if draw_fn:
        draw_fn(img, vdata)
    draw = ImageDraw.Draw(img)
    draw_label_overlay(draw, vid, vdata["label"])
    img.save(str(out_path), "PNG")
    return out_path.stat().st_size


def main():
    with open(VISEME_MAP, "r", encoding="utf-8") as f:
        vmap = json.load(f)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    visemes = vmap["visemes"]
    # Skip alias ids (e.g. 7 -> 6): they reuse another id's sprite, no need to draw.
    drawable = {k: v for k, v in visemes.items() if "alias_of" not in v}
    print(f"Generating {len(drawable)} placeholder sprites -> {OUT_DIR}\n")

    total = 0
    for vid, vdata in drawable.items():
        sprite_name = vdata["sprite"]
        out_path    = OUT_DIR / sprite_name
        size_bytes  = make_sprite(vid, vdata, out_path)
        total += 1
        print(f"  [{vid}] {sprite_name:20s}  openness={vdata['openness']:.2f}  {size_bytes:>7,} bytes")

    print(f"\nDone. {total} sprites written to {OUT_DIR}")


if __name__ == "__main__":
    main()
