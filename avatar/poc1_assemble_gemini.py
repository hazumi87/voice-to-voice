#!/usr/bin/env python3
"""PoC1 assembler (Gemini whole-frame swap) - build numbered viseme sprites from
the hand-generated Gemini pose set in working/manual_gemini/.

WHY WHOLE-FRAME (not mouth-composite): the Gemini poses are INDEPENDENT
regenerations of the same character - different head size/position and even
different canvas dimensions (2400x1792 vs 2528x1696). They do NOT pixel-align, so
cropping one mouth box and pasting onto a single neutral base (the original PoC1
plan) produces a mis-scaled, mis-placed mouth. Instead each viseme id gets the
WHOLE pose as its sprite, normalized to one common canvas. The head shifts a
little between poses, but you get REAL art lip-syncing to real audio with zero
alignment guesswork. Cleaner long-term option: slice the isolated-mouth chart
working/manual_gemini/Mouth_Poses.png over one static face (Eric's call).

Pose -> numbered viseme id (scheme blair-10-numbered):
  1  closed M/B/P  <- pose_neutral
  2  consonants    <- pose_C
  3  open E/eh     <- pose_smile
  4  wide open A   <- pose_A
  5  rounded O     <- pose_O
  6  pucker U      <- pose_pucker   (id 7 W/Q aliases this; not rendered)
  8  teeth-on-lip  <- pose_F
  9  tongue L      <- pose_L
  10 rest/idle     <- pose_rest

Run:
  .venv/Scripts/python.exe avatar/poc1_assemble_gemini.py
"""
import os, glob
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "working", "manual_gemini")
OUT_LOCAL = os.path.join(HERE, "poc1", "sprites")
OUT_WEB = os.path.join(HERE, "web", "public", "sprites")

# Common output canvas (4:3, matches the dominant 2400x1792 pose aspect; the
# Pixi stage is 640x480 = 4:3, so no distortion when it fills the stage).
TARGET_W, TARGET_H = 800, 600

# viseme id -> source pose filename (without extension)
POSE_FOR_ID = {
    "01": "pose_neutral",
    "02": "pose_C",
    "03": "pose_smile",
    "04": "pose_A",
    "05": "pose_O",
    "06": "pose_pucker",
    "08": "pose_F",
    "09": "pose_L",
    "10": "pose_rest",
}


def cover_resize(im, tw, th):
    """Scale to COVER (tw,th) then center-crop - no distortion, no letterbox."""
    w, h = im.size
    scale = max(tw / w, th / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    im = im.resize((nw, nh), Image.LANCZOS)
    left = (nw - tw) // 2
    top = (nh - th) // 2
    return im.crop((left, top, left + tw, top + th))


def main():
    os.makedirs(OUT_LOCAL, exist_ok=True)
    os.makedirs(OUT_WEB, exist_ok=True)

    # Clear stale letter-scheme sprites (mouth_A.png .. mouth_X.png) from web.
    for old in glob.glob(os.path.join(OUT_WEB, "mouth_*.png")):
        name = os.path.basename(old)
        # keep numbered (mouth_NN.png); drop single-letter ids
        stem = name[len("mouth_"):-len(".png")]
        if not stem.isdigit():
            os.remove(old)
            print(f"[gemini] removed stale {name}")

    made = []
    for vid, pose in POSE_FOR_ID.items():
        spath = os.path.join(SRC, f"{pose}.png")
        if not os.path.exists(spath):
            print(f"[gemini] SKIP id {vid}: no source {spath}")
            continue
        im = Image.open(spath).convert("RGB")
        out = cover_resize(im, TARGET_W, TARGET_H)
        fname = f"mouth_{vid}.png"
        out.save(os.path.join(OUT_LOCAL, fname))
        out.save(os.path.join(OUT_WEB, fname))
        made.append(vid)
        print(f"[gemini] id {vid:2s} <- {pose:14s} -> {fname}  ({TARGET_W}x{TARGET_H})")

    print(f"\n[gemini] built {len(made)}/{len(POSE_FOR_ID)} sprites")
    print(f"[gemini]   local: {OUT_LOCAL}")
    print(f"[gemini]   web:   {OUT_WEB}")
    missing = [v for v in POSE_FOR_ID if v not in made]
    if missing:
        print(f"[gemini] missing: {', '.join(missing)}")


if __name__ == "__main__":
    main()
