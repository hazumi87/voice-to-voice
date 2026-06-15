#!/usr/bin/env python3
"""PoC2 - timing layer for the 2D avatar.

OmniVoice exposes NO per-token timing (confirmed: its only "duration" machinery is
RuleDurationEstimator, a whole-clip length predictor - no phonemes, no alignment).
So we forced-align AFTER synthesis. We already own faster-whisper (CPU/int8, same
config server.py uses), which emits WORD-level {start,end} timestamps for free - no
new dependency, no MFA. This proves whether word-level alignment is tight enough to
drive a 9-shape 2D mouth before we pay for phone-level accuracy.

Pipeline:
  WAV (an OmniVoice clip)
    -> faster-whisper transcribe(word_timestamps=True)  -> [{word,start,end,prob}]
    -> per word, spread visemes across [start,end] via grapheme->viseme map
    -> insert rest (X) in the gaps between words
    -> viseme timeline: [{t_start, t_end, viseme}]

Run:
  .venv/Scripts/python.exe avatar/poc2_align.py full_stella.wav
  .venv/Scripts/python.exe avatar/poc2_align.py full_stella.wav --json out.json
"""
import sys, os, json, argparse, wave, contextlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MAP_PATH = os.path.join(HERE, "viseme_map.json")

# Minimum on-screen time for a viseme so the mouth doesn't strobe. Below this we
# merge adjacent identical-or-tiny visemes. 60ms ~= a brisk but readable mouth flap.
MIN_VISEME_S = 0.06


def load_map():
    with open(MAP_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def wav_duration(path):
    with contextlib.closing(wave.open(path, "rb")) as w:
        return w.getnframes() / float(w.getframerate())


def transcribe_words(path):
    """CPU/int8 faster-whisper, word timestamps on. Mirrors server.py's STT config."""
    from faster_whisper import WhisperModel
    model = WhisperModel("base.en", device="cpu", compute_type="int8")
    segments, info = model.transcribe(path, word_timestamps=True, language="en")
    words = []
    for seg in segments:
        if not seg.words:
            continue
        for w in seg.words:
            tok = w.word.strip()
            if tok:
                words.append({"word": tok, "start": float(w.start),
                              "end": float(w.end), "prob": float(w.probability)})
    return words


def word_to_visemes(word, vmap):
    """Map a whole word to an ordered list of viseme ids via the grapheme heuristic.
    Collapses consecutive duplicate visemes (so 'mall' -> A,D,H not A,D,H,H)."""
    g2v = vmap["grapheme_to_viseme"]
    default = g2v["_default"]
    out = []
    for ch in word.lower():
        if not ch.isalpha():
            continue
        v = g2v.get(ch, default)
        if not out or out[-1] != v:
            out.append(v)
    if not out:
        out = [vmap["_meta"]["rest_viseme"]]
    return out


def build_timeline(words, total_dur, vmap):
    rest = vmap["_meta"]["rest_viseme"]
    tl = []
    cursor = 0.0
    for wd in words:
        # gap before this word -> rest
        if wd["start"] > cursor + 1e-3:
            tl.append({"t_start": round(cursor, 3), "t_end": round(wd["start"], 3),
                       "viseme": rest, "src": "gap"})
        cursor = wd["start"]
        vis = word_to_visemes(wd["word"], vmap)
        span = max(wd["end"] - wd["start"], 1e-3)
        step = span / len(vis)
        for i, v in enumerate(vis):
            ts = wd["start"] + i * step
            te = wd["start"] + (i + 1) * step
            tl.append({"t_start": round(ts, 3), "t_end": round(te, 3),
                       "viseme": v, "src": wd["word"]})
        cursor = wd["end"]
    # trailing rest to end of clip
    if total_dur > cursor + 1e-3:
        tl.append({"t_start": round(cursor, 3), "t_end": round(total_dur, 3),
                   "viseme": rest, "src": "gap"})
    return merge_short(tl)


def merge_short(tl):
    """Merge sub-MIN_VISEME_S frames into their neighbor of the same viseme, and
    collapse runs of the same viseme. Keeps the timeline from strobing."""
    if not tl:
        return tl
    merged = [dict(tl[0])]
    for f in tl[1:]:
        last = merged[-1]
        if f["viseme"] == last["viseme"]:
            last["t_end"] = f["t_end"]   # extend run
        else:
            merged.append(dict(f))
    # second pass: absorb any still-too-short frame into the previous one
    out = []
    for f in merged:
        if out and (f["t_end"] - f["t_start"]) < MIN_VISEME_S:
            out[-1]["t_end"] = f["t_end"]
        else:
            out.append(f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", help="path to an OmniVoice WAV (relative to repo root ok)")
    ap.add_argument("--json", help="write timeline json here")
    args = ap.parse_args()

    wav = args.wav if os.path.isabs(args.wav) else os.path.join(ROOT, args.wav)
    if not os.path.exists(wav):
        sys.exit(f"no such wav: {wav}")

    vmap = load_map()
    dur = wav_duration(wav)
    print(f"[poc2] clip = {os.path.basename(wav)}  duration = {dur:.2f}s")
    print(f"[poc2] transcribing with faster-whisper (CPU/int8, word timestamps)...")
    words = transcribe_words(wav)
    text = " ".join(w["word"] for w in words)
    print(f"[poc2] ground-truth text ({len(words)} words):\n    {text}\n")

    print(f"[poc2] word timings:")
    for w in words:
        print(f"    {w['start']:6.2f} - {w['end']:6.2f}  ({w['end']-w['start']:.2f}s)  "
              f"p={w['prob']:.2f}  {w['word']}")

    tl = build_timeline(words, dur, vmap)
    print(f"\n[poc2] viseme timeline ({len(tl)} frames, scheme={vmap['_meta']['scheme']}):")
    for f in tl:
        bar = "#" * max(1, int((f["t_end"] - f["t_start"]) / 0.04))
        print(f"    {f['t_start']:6.2f} - {f['t_end']:6.2f}  {f['viseme']}  "
              f"{bar:20s} {f['src']}")

    # quick health stats
    frame_durs = [f["t_end"] - f["t_start"] for f in tl]
    speaking = sum(d for f, d in zip(tl, frame_durs) if f["viseme"] != vmap["_meta"]["rest_viseme"])
    print(f"\n[poc2] frames={len(tl)}  avg_frame={sum(frame_durs)/len(frame_durs)*1000:.0f}ms  "
          f"min={min(frame_durs)*1000:.0f}ms  speaking={speaking:.2f}s/{dur:.2f}s "
          f"({speaking/dur*100:.0f}% mouth-moving)")

    if args.json:
        outp = args.json if os.path.isabs(args.json) else os.path.join(HERE, args.json)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump({"clip": os.path.basename(wav), "duration": dur,
                       "text": text, "scheme": vmap["_meta"]["scheme"],
                       "timeline": tl}, f, indent=2)
        print(f"[poc2] wrote {outp}")


if __name__ == "__main__":
    main()
