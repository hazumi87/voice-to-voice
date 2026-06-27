"""
Speaker identification for the voice satellite — closed-set, personalization.

ECAPA-TDNN voiceprints (SpeechBrain). Enroll a person from browser-recorded
speech, then identify the speaker of each device utterance in /api/converse and
hand the name to the agent so it can personalize the reply. IDENTIFICATION ONLY,
NOT access control: forgiving threshold + mandatory 'unknown' fallback, never
gate anything on a match.

Runs in-process in v2v on the VRPC GPU, parallel to faster-whisper STT (the audio
is already in /api/converse — no LAN hop). Blueprint: KB 5f60bd007777d0bb.

Storage mirrors custom_voices/: speakers/<id>.json (metadata) + speakers/<id>.npy
(stacked per-clip embeddings, averaged into the voiceprint) + speakers/<id>__<k>.wav
(each enrollment clip, kept for re-derivation/backup/audit). All gitignored.
"""
import io
import json
import os
import threading
import time
import uuid

import av
import numpy as np
import soundfile as sf
import torch

SR = 16000                  # ECAPA expects 16 kHz mono
EMBED_DIM = 192

# Matching params — tune empirically per family during validation (blueprint).
# Personalization use case → lean permissive, but require a margin so two similar
# voices don't flip-flop.
THRESHOLD = 0.55            # cosine floor below which it's 'unknown'
MARGIN = 0.10              # top match must beat 2nd-best by this, else 'unknown'
MIN_SPEECH_S = 1.2         # utterances shorter than this abstain (STT still runs)
MIN_ENROLL_S = 6.0         # reject enrollment clips shorter than this

HERE = os.path.dirname(os.path.abspath(__file__))
SPEAKERS_DIR = os.path.join(HERE, "speakers")

_model = None
_lock = threading.Lock()    # ECAPA encode serialized (model not guaranteed reentrant)
_registry = {}              # id -> metadata dict
_vectors = {}               # id -> np.ndarray(192,) averaged unit voiceprint


def _device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load():
    """Load ECAPA once + read the registry from disk. Safe to call repeatedly."""
    global _model
    os.makedirs(SPEAKERS_DIR, exist_ok=True)
    _load_registry()
    if _model is not None:
        return
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except Exception:  # noqa: BLE001 — older speechbrain layout
        from speechbrain.pretrained import EncoderClassifier
    # SpeechBrain symlinks the fetched model into savedir by default, which needs
    # elevated privileges on Windows (WinError 1314). Force COPY instead.
    kw = dict(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.join(SPEAKERS_DIR, "_ecapa_model"),
        run_opts={"device": _device()},
    )
    try:
        from speechbrain.utils.fetching import LocalStrategy
        kw["local_strategy"] = LocalStrategy.COPY
    except Exception:  # noqa: BLE001 — older speechbrain without LocalStrategy
        pass
    _model = EncoderClassifier.from_hparams(**kw)
    print(f"[spkid] ECAPA loaded on {_device()}; {len(_registry)} speaker(s) enrolled",
          flush=True)


def is_ready():
    return _model is not None


# ---------------------------------------------------------------------------
# Registry persistence
# ---------------------------------------------------------------------------
def _meta_path(sid):
    return os.path.join(SPEAKERS_DIR, sid + ".json")


def _vecs_path(sid):
    return os.path.join(SPEAKERS_DIR, sid + ".npy")


def _load_registry():
    _registry.clear()
    _vectors.clear()
    if not os.path.isdir(SPEAKERS_DIR):
        return
    for fn in sorted(os.listdir(SPEAKERS_DIR)):
        if not fn.endswith(".json"):
            continue
        try:
            meta = json.load(open(os.path.join(SPEAKERS_DIR, fn), encoding="utf-8"))
            sid = meta["id"]
            stack = np.load(_vecs_path(sid))          # [n_clips, 192]
        except Exception as e:  # noqa: BLE001
            print(f"[spkid] failed to load speaker {fn}: {e}", flush=True)
            continue
        _registry[sid] = meta
        _vectors[sid] = _unit(stack.mean(axis=0))


def _save(sid, meta, stack):
    json.dump(meta, open(_meta_path(sid), "w", encoding="utf-8"), indent=2)
    np.save(_vecs_path(sid), stack)
    _registry[sid] = meta
    _vectors[sid] = _unit(stack.mean(axis=0))


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def _unit(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def decode_16k_mono(raw: bytes):
    """Decode any uploaded/recorded audio (mp4/AAC/webm/wav) -> mono float32 @16k."""
    container = av.open(io.BytesIO(raw))
    stream = container.streams.audio[0]
    resampler = av.audio.resampler.AudioResampler(format="flt", layout="mono", rate=SR)
    chunks = []
    for frame in container.decode(stream):
        for rf in resampler.resample(frame):
            chunks.append(rf.to_ndarray().reshape(-1))
    container.close()
    if not chunks:
        return None
    return np.concatenate(chunks).astype(np.float32)


def _trim_silence(wav, frame_ms=20, rel_thr=0.06, abs_thr=0.004, pad_ms=80):
    """Trim leading/trailing low-energy audio so silence/quiet padding doesn't
    dilute the voiceprint. Keeps interior gaps between words; only cuts the ends
    (first..last frame whose RMS clears max(abs_thr, rel_thr*peak)). Never returns
    empty — falls back to the original clip if nothing clears the threshold."""
    if wav is None or len(wav) == 0:
        return wav
    fl = max(1, int(SR * frame_ms / 1000))
    n = len(wav) // fl
    if n < 2:
        return wav
    frames = wav[:n * fl].reshape(n, fl).astype(np.float32)
    energy = np.sqrt((frames ** 2).mean(axis=1))
    peak = float(energy.max())
    if peak <= 0:
        return wav
    thr = max(abs_thr, rel_thr * peak)
    voiced = np.where(energy >= thr)[0]
    if len(voiced) == 0:
        return wav
    pad = int(SR * pad_ms / 1000)
    start = max(0, voiced[0] * fl - pad)
    end = min(len(wav), (voiced[-1] + 1) * fl + pad)
    return wav[start:end]


def _embed_wave(wav):
    """wav: float32 mono @16k -> unit-norm 192-d embedding (np.ndarray).
    End-silence is trimmed first so the print reflects speech, not dead air."""
    wav = _trim_silence(wav)
    t = torch.from_numpy(np.ascontiguousarray(wav)).float().unsqueeze(0)
    with _lock, torch.no_grad():
        emb = _model.encode_batch(t).squeeze().detach().cpu().numpy()
    return _unit(emb)


# ---------------------------------------------------------------------------
# Enrollment + CRUD
# ---------------------------------------------------------------------------
def enroll(name: str, raw: bytes):
    """Create a new speaker from one clip. Returns (entry, None) or (None, error)."""
    if _model is None:
        return None, "speaker-id model not loaded"
    wav = decode_16k_mono(raw)
    if wav is None:
        return None, "could not decode audio"
    secs = len(wav) / SR
    if secs < MIN_ENROLL_S:
        return None, f"clip too short ({secs:.1f}s); need >= {MIN_ENROLL_S:.0f}s"
    sid = uuid.uuid4().hex[:12]
    emb = _embed_wave(wav)
    clip_file = f"{sid}__0.wav"
    sf.write(os.path.join(SPEAKERS_DIR, clip_file), wav, SR)
    now = int(time.time())
    meta = {
        "id": sid,
        "name": (name or "").strip() or sid,
        "enrolled_at": now,
        "updated_at": now,
        "clips": [{"file": clip_file, "seconds": round(secs, 2), "added_at": now}],
    }
    _save(sid, meta, emb.reshape(1, -1))
    print(f"[spkid] enrolled '{meta['name']}' ({sid}) from {secs:.1f}s", flush=True)
    return _public(sid), None


def add_clip(sid: str, raw: bytes):
    """Add another enrollment clip to an existing speaker (averaged into the print)."""
    if _model is None:
        return None, "speaker-id model not loaded"
    if sid not in _registry:
        return None, "unknown speaker id"
    wav = decode_16k_mono(raw)
    if wav is None:
        return None, "could not decode audio"
    secs = len(wav) / SR
    if secs < MIN_ENROLL_S:
        return None, f"clip too short ({secs:.1f}s); need >= {MIN_ENROLL_S:.0f}s"
    meta = _registry[sid]
    k = len(meta["clips"])
    clip_file = f"{sid}__{k}.wav"
    sf.write(os.path.join(SPEAKERS_DIR, clip_file), wav, SR)
    emb = _embed_wave(wav)
    stack = np.load(_vecs_path(sid))
    stack = np.vstack([stack, emb.reshape(1, -1)])
    now = int(time.time())
    meta["clips"].append({"file": clip_file, "seconds": round(secs, 2), "added_at": now})
    meta["updated_at"] = now
    _save(sid, meta, stack)
    print(f"[spkid] added clip {k} to '{meta['name']}' ({sid}); {len(stack)} total", flush=True)
    return _public(sid), None


def rename(sid: str, name: str):
    if sid not in _registry:
        return None, "unknown speaker id"
    name = (name or "").strip()
    if not name:
        return None, "name required"
    _registry[sid]["name"] = name
    _registry[sid]["updated_at"] = int(time.time())
    json.dump(_registry[sid], open(_meta_path(sid), "w", encoding="utf-8"), indent=2)
    return _public(sid), None


def delete(sid: str):
    if sid not in _registry:
        return False
    for clip in _registry[sid].get("clips", []):
        try:
            os.remove(os.path.join(SPEAKERS_DIR, clip["file"]))
        except OSError:
            pass
    for p in (_meta_path(sid), _vecs_path(sid)):
        try:
            os.remove(p)
        except OSError:
            pass
    _registry.pop(sid, None)
    _vectors.pop(sid, None)
    print(f"[spkid] deleted speaker {sid}", flush=True)
    return True


def set_voice(sid: str, voice: str):
    """Associate a TTS voice with a speaker (used to reply in that person's voice)."""
    if sid not in _registry:
        return None, "unknown speaker id"
    _registry[sid]["voice"] = (voice or "").strip()
    _registry[sid]["updated_at"] = int(time.time())
    json.dump(_registry[sid], open(_meta_path(sid), "w", encoding="utf-8"), indent=2)
    return _public(sid), None


def get_voice(sid: str):
    """Saved voice for a speaker id, or '' if none/unknown."""
    return (_registry.get(sid, {}) or {}).get("voice", "")


def _public(sid):
    m = _registry[sid]
    return {
        "id": sid,
        "name": m["name"],
        "voice": m.get("voice", ""),
        "clips": len(m.get("clips", [])),
        "seconds": round(sum(c["seconds"] for c in m.get("clips", [])), 1),
        "enrolled_at": m.get("enrolled_at"),
        "updated_at": m.get("updated_at"),
    }


def list_speakers():
    return [_public(sid) for sid in sorted(_registry, key=lambda s: _registry[s]["name"].lower())]


# ---------------------------------------------------------------------------
# Identification
# ---------------------------------------------------------------------------
def identify(raw: bytes):
    """Identify the speaker of an utterance.
    Returns (name|'unknown', confidence_float, detail_str, sid|None). Never raises.
    The sid lets callers key per-speaker state (history, voice) stably by id."""
    try:
        if _model is None or not _vectors:
            return "unknown", 0.0, ("no_model" if _model is None else "no_enrollments"), None
        wav = decode_16k_mono(raw)
        if wav is None:
            return "unknown", 0.0, "decode_failed", None
        secs = len(wav) / SR
        if secs < MIN_SPEECH_S:
            return "unknown", 0.0, f"too_short({secs:.1f}s)", None
        v = _embed_wave(wav)
        scored = sorted(
            ((float(np.dot(v, vec)), sid) for sid, vec in _vectors.items()),
            reverse=True,
        )
        top_sim, top_sid = scored[0]
        second_sim = scored[1][0] if len(scored) > 1 else -1.0
        if top_sim < THRESHOLD:
            return "unknown", top_sim, f"below_threshold({top_sim:.2f})", None
        if (top_sim - second_sim) < MARGIN:
            return "unknown", top_sim, f"ambiguous(d={top_sim - second_sim:.2f})", None
        return _registry[top_sid]["name"], top_sim, f"match({top_sim:.2f})", top_sid
    except Exception as e:  # noqa: BLE001 — ID must never break the converse path
        print(f"[spkid] identify failed: {e!r}", flush=True)
        return "unknown", 0.0, "error", None
