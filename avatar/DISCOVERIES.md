# 2D Avatar Prototype — Discoveries

Session 2026-06-13. Three PoCs stood up. What was learned, so a fresh session doesn't repeat it.

## PoC2 — Timing (DONE)
- **OmniVoice exposes NO per-token timing.** Its only "duration" machinery is `RuleDurationEstimator`
  (`.venv/.../omnivoice/utils/duration.py`) — a *whole-clip length* predictor (char phonetic weights ÷
  reference speed). No phonemes, no alignment, no timestamps. `generate()` returns only `generated_audios`.
  So we forced-align AFTER synthesis. Don't go looking for hidden timing in OmniVoice — it isn't there.
- **We already own a word-level aligner: faster-whisper 1.2.1**, `transcribe(word_timestamps=True)` gives
  `Word{start,end,word,probability}`. Same CPU/int8 config server.py uses. Zero new dependency. No MFA needed.
- **Word-level alignment is good enough for a 9-shape 2D mouth.** On the Stella phonetic pangram (20.4s,
  69 words), boundaries were tight, gaps landed on real pauses, 80% mouth-moving. What sells 2D lip-sync is
  *openness tracking the envelope + resting on silence*, not phoneme-perfect shapes. MFA stays "benchmark only."
- Mapping is **grapheme→viseme** (whisper gives orthographic words, not phones). Crude but reads right.
  `avatar/viseme_map.json` also has the ARPABET `phoneme_to_viseme` table ready IF a phone aligner is ever wired in.
- Files: `avatar/poc2_align.py`, output `avatar/full_stella_timeline.json`. Anti-strobe floor = 60ms (`MIN_VISEME_S`).

## PoC3 — Integration (DONE)
- Vite + React 19 + **pixi.js 8.19 + @pixi/react 8.0.5** in `avatar/web/`. `npm run dev` → :5173.
- **@pixi/react v8 API gotcha:** NO `<Sprite>`/`<Graphics>` components. Must `extend({ Sprite, Container, Texture })`
  (classes from 'pixi.js') then use lowercase JSX `<sprite>`. v8 also REQUIRES React 19 (peer dep).
- **Clock master = `<audio>` element's `currentTime`**, NOT Date.now(), NOT the ticker's own time. Audio is the
  single source of truth so mouth never drifts. Virtual-clock fallback engages automatically when no audio file
  is present (animates for testing now; becomes audio-driven the instant a real clip is dropped in `public/audio/`
  — zero rework). Timeline lookup is a pure module (`src/lib/visemeTimeline.js`), smoke-tested.
- Data served from `avatar/web/public/data/`; sprites from `public/sprites/`.

## PoC1 — Poses (PIPELINE WORKS, ART IS WEAK)
- **Path B (Gemini) is BLOCKED by billing**, not a bug. Every image model → `429 free_tier limit=0`; image-OUT
  isn't enabled on this `GEMINI_API_KEY` (text + vision-in work). No side-by-side possible. Script staged at
  `avatar/poc1/generate_pathB.py` if image billing is ever enabled. Given cost-weaning, leave it.
- **Path A (local SD-inpaint) works, free, on the 4080.** `runwayml/stable-diffusion-inpainting`, fp16, ~2.1GB
  VRAM, seed 12345, 30 steps. Inpaints the 607×356 mouth tile (resized→768×448 mult-of-8→back), feather-composites
  onto the 2528×1696 base. Script: `avatar/poc1/generate_pathA.py`. Sprites: `avatar/poc1/sprites/mouth_*.png`.
- **BUG (fixed, keep in mind): CLIP 77-token truncation.** First run = 9 byte-identical tiles because the verbose
  `style_anchor + global_constraints` preamble (167 tokens) ate the whole window and truncated the per-viseme
  instruction off the tail. Fix: front-load the short viseme-specific mouth text, append a compact style tag.
- **KNOWN-WEAK OUTPUT — visemes don't differentiate.** Verified by eye: "A" (should be closed) and "D" (should be
  wide-open) come out nearly identical — both smiling-with-teeth, both over-saturated red lips. Root cause: text-only
  SD-inpaint has NO spatial control over mouth openness; the surrounding face context ("smiling person") dominates and
  it just repaints a smile regardless of prompt. More prompt tuning won't fix this — it needs STRUCTURAL guidance.
  Open options for the next pass: (a) accept as concept-proof, (b) ControlNet/scribble shape-hint per viseme, or
  (c) human-supplied per-viseme reference mouths. This is a creative decision left for Eric.

## ID SCHEME REFACTOR — DONE (2026-06-15)
**Decision: killed the A-X letter viseme ids. Now NUMBERS 1-10 (the Blair pose grid).**
Reason: letter ids (A,B,C..) collide with phoneme letters (pose_A = the SOUND "ah"), which
confuses both humans and the image agent. Numbers can never collide with a sound.

Mapping old letter -> new number: A→1(MBP) B→2(consonants) C→3(eh) D→4(ah-wide) E→5(O)
F→6(U-pucker) G→8(F/V) H→9(L) X→10(rest). 7 = W/Q, folds into 6 (shares sprite).
Sprites are mouth_01.png .. mouth_10.png (zero-padded). rendered_set = 9 distinct (7 aliases 6).

**All consumers migrated to numbers:**
  - `avatar/viseme_map.json` — numbered scheme "blair-10-numbered" (single source of truth).
  - `avatar/poc2_align.py` — already read the map; re-ran it -> `full_stella_timeline.json` now numbered (127 frames, 80% mouth-moving, scheme=blair-10-numbered).
  - `avatar/make_placeholders.py` — VISEME_DRAW rekeyed to "1".."10", skips alias ids.
  - `avatar/poc1_assemble.py` — IDS = ["01".."10"] (rendered_set), tiles named NN.png.
  - `avatar/web/src/lib/visemeTimeline.js` — REST_VISEME = 10 (number).
  - `avatar/web/src/lib/visemeTimeline.smoke.js` — mid-clip expectation now self-derived (no stale fixture).
  - `avatar/web/src/components/VisemePlayer.jsx` — VISEME_IDS = numbered rendered_set, REST_ID='10'.
  - `avatar/web/src/components/MouthStage.jsx` — init texture textures['10']; sprite now FILLS the stage (full-face poses, see below).
  - `avatar/web/public/data/` — viseme_map.json + full_stella_timeline.json re-copied; `full_stella.wav` copied to public/audio.

## GEMINI POSES — ASSEMBLED via WHOLE-FRAME SWAP (2026-06-15)
The hand-generated Gemini poses (`working/manual_gemini/pose_*.png`) are INDEPENDENT
regenerations of the same character — different head size/position AND different canvas
dims (2400x1792 vs the 2528x1696 base). They do NOT pixel-align, so the original
"crop one mouth box, composite onto one neutral base" plan CANNOT work (mis-scaled,
mis-placed mouth). **Decision: whole-frame swap** — each viseme id gets the WHOLE pose
as its sprite, normalized to one 800x600 canvas (cover-crop, no distortion). Head shifts
a little between poses but you get real art lip-syncing to real audio, zero alignment guesswork.
  - Script: `avatar/poc1_assemble_gemini.py` (pose->id map inside). Built mouth_01..10.png
    into both `avatar/poc1/sprites/` and `avatar/web/public/sprites/` (cleared stale letter sprites).
  - Pose->id: 1<-neutral 2<-C 3<-smile 4<-A 5<-O 6<-pucker 8<-F 9<-L 10<-rest.
**CLEANER ALTERNATIVE (Eric's call, not yet done):** slice the isolated-mouth chart
`working/manual_gemini/Mouth_Poses.png` (clean per-sound mouth shapes) into mouth sprites
over ONE static face — true locked-camera lip-sync, but mouth art style differs from the rendered face.

## SERVED + TABBED (2026-06-15)
- Avatar SPA built with Vite `base:'/avatar/'` -> `avatar/web/dist/`, mounted in server.py at
  `/avatar` (StaticFiles html=True, before the catch-all "/" mount). Same-origin so it works over Tailscale.
- App now has TWO prototype views (App.jsx sub-nav): **Avatar · Lip-Sync** (PixiJS player) and
  **Phonetics · Alignment** (`PhoneticsView.jsx` — visual timeline strip of the word->viseme mapping).
- Nav tab **Lip-Sync ▸** added to index.html / prototype.html / talkback.html -> `/avatar/`.
- After any avatar code change: `cd avatar/web && npm run build`, then restart voice-to-voice (harbor MCP) to reserve the new dist. `npm run dev` (:5173) still works for live dev.

## GEMINI MANUAL POSES — COMPLETE SET IN HAND (working/manual_gemini/)
Eric generated poses in Gemini web. Filenames = the SOUND spoken (pose_A = "ah", NOT viseme A).
Coverage verified — ALL 10 grid shapes covered, NOTHING more to generate:
  1 MBP closed = pose_rest/pose_neutral | 2 consonants = pose_C (spares C2,G2) | 3 eh = pose_smile
  4 ah-wide = pose_A | 5 O = pose_O | 6/7 U,W/Q = pose_pucker | 8 F/V = pose_F (spare pose_V) | 9 L = pose_L | 10 rest = pose_rest
Duplicates to dedupe at assembly: pose_F & pose_V both =F/V; pose_C/C2/G2 all =consonants.
NEXT: re-pin mouth box per pose (faces may shift slightly), crop+mask each, assemble onto base -> mouth_01..10.png.
Base canvas = working/reference_chat_neutral.png (closed neutral mouth — the good base). Old smiling run archived in avatar/poc1_smiling_archive/.

## How to run the whole thing right now
1. `cd avatar/web && npm run dev` → http://localhost:5173
2. Mouth animates through the real Stella timeline. Swap sprite source between placeholder set
   (`public/sprites/`, clear+labeled, proves timing) and the Path A face set to judge differentiation.
