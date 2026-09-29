"""Hold clips: pre-rendered lines played while OmniVoice isn't loaded (Eric, 2026-09-29).

After a GPU release (or a restart), the reply voice needs ~10 s to reload, but Whisper and
speaker ID stay loaded, so v2v still knows who spoke. It plays one of these clips in that
speaker's voice at once, loads TTS, then pushes the real reply to the device.

Logic tier only: file layout, the line texts, lookup and status. server.py renders the clips
(it owns synth) and plays them.

Layout: working/hold_clips/<voice_id>/<kind>.wav, one WAV per (voice, kind). The clips are
static, so they play with no GPU at all.
"""
import os
import re
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
CLIP_DIR = os.path.join(HERE, "working", "hold_clips")

# kind -> the line. No dynamic parts: a clip is rendered once and can't name who holds the GPU.
LINES = {
    "warming": "One moment, I'm just warming up my voice.",
    "busy": "The graphics card is busy with another job right now. I'll answer as soon as it frees up.",
    "still": "Still working on it. One more moment.",
    "giveup": "Sorry, I couldn't get the graphics card in time. Please ask me again in a minute.",
}
KINDS = tuple(LINES)

_lock = threading.Lock()


def _safe(voice_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", voice_id or "")


def path(voice_id: str, kind: str) -> str:
    return os.path.join(CLIP_DIR, _safe(voice_id), f"{kind}.wav")


def has(voice_id: str, kind: str) -> bool:
    return os.path.isfile(path(voice_id, kind))


def missing(voice_id: str) -> list:
    return [k for k in KINDS if not has(voice_id, k)]


def save(voice_id: str, kind: str, wav: bytes) -> str:
    p = path(voice_id, kind)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with _lock:
        with open(tmp, "wb") as f:
            f.write(wav)
        os.replace(tmp, p)
    return p


def load(voice_id: str, kind: str, fallback_voice: str = None):
    """(wav_bytes, voice_used) for the speaker's voice, else the fallback voice, else (None, None)."""
    for v in (voice_id, fallback_voice):
        if v and has(v, kind):
            with open(path(v, kind), "rb") as f:
                return f.read(), v
    return None, None


def remove(voice_id: str) -> None:
    """Drop a deleted voice's clips."""
    d = os.path.join(CLIP_DIR, _safe(voice_id))
    if os.path.isdir(d):
        for fn in os.listdir(d):
            try:
                os.remove(os.path.join(d, fn))
            except OSError:
                pass
        try:
            os.rmdir(d)
        except OSError:
            pass


def status(voice_ids) -> dict:
    """{voice_id: [kinds present]} for the dev panel / debugging."""
    return {v: [k for k in KINDS if has(v, k)] for v in voice_ids}
