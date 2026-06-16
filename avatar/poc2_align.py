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
    -> per word, run g2p_en to get ARPABET phones, map phones->visemes
    -> insert rest in the gaps between words
    -> viseme timeline: [{t_start, t_end, viseme}]

g2p method: g2p_en (handles OOV/names via seq2seq; ARPABET with stress digits stripped).
Fallback: phoneme_to_viseme[""] = 10 (rest) for any unmapped phone.

Run:
  .venv/Scripts/python.exe avatar/poc2_align.py full_stella.wav
  .venv/Scripts/python.exe avatar/poc2_align.py full_stella.wav --json out.json
"""
import sys, os, json, argparse, wave, contextlib, re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MAP_PATH = os.path.join(HERE, "viseme_map.json")

# Minimum on-screen time for a viseme so the mouth doesn't strobe. Below this we
# merge adjacent identical-or-tiny visemes. 60ms ~= a brisk but readable mouth flap.
MIN_VISEME_S = 0.06
# Epsilon for float comparison: rounded timestamps can produce 0.0599999... instead of
# 0.06 exactly, so use a slightly-below threshold to avoid absorbing on-target frames.
_MIN_EPS = MIN_VISEME_S - 1e-9


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


_g2p_instance = None
_unmapped_phones = set()

def _get_g2p():
    global _g2p_instance
    if _g2p_instance is None:
        from g2p_en import G2p
        _g2p_instance = G2p()
    return _g2p_instance


def word_to_visemes(word, vmap):
    """Map a whole word to an ordered list of viseme ids via phoneme->viseme mapping.

    Uses g2p_en to convert the word to ARPABET phonemes, strips stress digits,
    discards non-phone tokens (spaces, punctuation), then looks up each phone in
    phoneme_to_viseme. Consecutive duplicate visemes are collapsed to avoid
    visually identical back-to-back frames.

    Falls back to rest (10) for any phone not in the table, and logs them.
    """
    p2v = vmap["phoneme_to_viseme"]
    rest = vmap["_meta"]["rest_viseme"]

    g2p = _get_g2p()
    raw_phones = g2p(word)

    # Strip stress digits, keep only uppercase ARPABET tokens (skip spaces/punct)
    phones = []
    for tok in raw_phones:
        tok_clean = re.sub(r'[0-9]$', '', tok)  # strip trailing stress digit
        if tok_clean and re.match(r'^[A-Z]+$', tok_clean):
            phones.append(tok_clean)

    out = []
    for ph in phones:
        v = p2v.get(ph, None)
        if v is None:
            _unmapped_phones.add(ph)
            v = rest
        if not out or out[-1] != v:
            out.append(v)

    if not out:
        out = [rest]
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
        word_frames = []
        for i, v in enumerate(vis):
            ts = wd["start"] + i * step
            te = wd["start"] + (i + 1) * step
            word_frames.append({"t_start": round(ts, 3), "t_end": round(te, 3),
                                "viseme": v, "src": wd["word"]})
        # Merge short frames within this word, but protect the last phoneme so
        # word-final vowels (e.g. the AH in "stella") are never swallowed.
        tl.extend(merge_short_protect_last(word_frames))
        cursor = wd["end"]
    # trailing rest to end of clip
    if total_dur > cursor + 1e-3:
        tl.append({"t_start": round(cursor, 3), "t_end": round(total_dur, 3),
                   "viseme": rest, "src": "gap"})
    return merge_short(tl)


def merge_short_protect_last(tl):
    """Like merge_short but NEVER absorbs the final frame of a word sequence.

    This prevents word-final phonemes (e.g. the AH in 'stella', the R in 'store')
    from being swallowed by the anti-strobe pass — fixing the 'missing ah' class of bug.
    The last frame gets its t_start shifted forward to enforce MIN_VISEME_S minimum
    rather than being dropped entirely.
    """
    if not tl:
        return tl
    if len(tl) == 1:
        return [dict(tl[0])]

    # First collapse consecutive identical visemes
    merged = [dict(tl[0])]
    for f in tl[1:]:
        last = merged[-1]
        if f["viseme"] == last["viseme"]:
            last["t_end"] = f["t_end"]
        else:
            merged.append(dict(f))

    if len(merged) == 1:
        return merged

    # Second pass: absorb short frames into previous, but protect the last entry
    out = []
    last_idx = len(merged) - 1
    for idx, f in enumerate(merged):
        dur = f["t_end"] - f["t_start"]
        if idx == last_idx:
            # Protected: ensure minimum duration by shifting t_start back if needed
            if out and dur < _MIN_EPS:
                borrow = MIN_VISEME_S - dur
                f = dict(f)
                f["t_start"] = max(out[-1]["t_start"] + MIN_VISEME_S,
                                   f["t_start"] - borrow)
            out.append(f)
        elif out and dur < _MIN_EPS:
            out[-1]["t_end"] = f["t_end"]
        else:
            out.append(f)
    return out


def merge_short(tl):
    """Merge sub-MIN_VISEME_S frames into their neighbor and collapse same-viseme runs.
    Used for the full timeline (gap/rest frames between words). Word-internal frames
    are already handled by merge_short_protect_last before reaching this pass."""
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
        if out and (f["t_end"] - f["t_start"]) < _MIN_EPS:
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

    if _unmapped_phones:
        print(f"[poc2] WARNING: unmapped phonemes (fell back to rest): {sorted(_unmapped_phones)}")

    if args.json:
        outp = args.json if os.path.isabs(args.json) else os.path.join(HERE, args.json)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump({"clip": os.path.basename(wav), "duration": dur,
                       "text": text, "scheme": vmap["_meta"]["scheme"],
                       "timeline": tl}, f, indent=2)
        print(f"[poc2] wrote {outp}")


if __name__ == "__main__":
    main()
