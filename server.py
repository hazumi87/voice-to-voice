"""Voice-to-voice prototype server (VRPC, throwaway).

One FastAPI/uvicorn process serves:
  - the static front-end (index.html + main.js)
  - GET  /api/voices    -> list of voice-design presets for the radio buttons
  - POST /api/converse  -> audio in (mp4/webm/wav) -> STT -> ollama -> TTS -> audio out
  - POST /api/reset     -> clear conversation history
  - GET  /api/health    -> component status

All three legs run locally on the VRPC:
  STT  = faster-whisper (base.en, cuda/float16, decodes iPad mp4/AAC via PyAV)
  CHAT = ollama (llama3.1) at 127.0.0.1:11434, called SERVER-SIDE only
  TTS  = OmniVoice (k2-fsa/OmniVoice, cuda/float16), voice-design mode (no ref WAV)

Production note: this is a prove-it prototype. Nothing here is harbor-supervised
and it touches no production service (neutts :8220, agent-speech-relay).
"""
import io
import os
import re
import sys
import time
import threading
import urllib.parse
import urllib.request
import json

import av  # decode arbitrary uploaded/recorded audio (mp4/AAC/webm) to a waveform
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, UploadFile, File, Form, Body, Request
from fastapi.responses import Response, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Windows HF symlink footgun: copy instead of symlink.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "llama3.2:3b"  # smaller/faster for snappy turns

# Appended to every personality so all replies stay short and TTS-friendly.
VOICE_STYLE = (
    " Keep every reply to one or two short spoken sentences. "
    "Do NOT use markdown, bullet points, lists, code blocks, asterisks, or emojis - "
    "your words are read aloud by a text-to-speech engine, so write only what should be spoken."
)

# Distinct agent personalities. Each is a system prompt; VOICE_STYLE is appended.
# "group" controls how the front-end dropdown is grouped.
PERSONALITIES = [
    # --- Characters ---
    {"id": "friendly", "label": "Friendly Companion", "group": "Characters",
     "system": "You are a warm, upbeat voice companion. You chat naturally, show "
               "genuine curiosity about the person, and keep things light and kind."},
    {"id": "sardonic", "label": "Sardonic Wit", "group": "Characters",
     "system": "You are a dry, razor-sharp companion with a sarcastic streak. You're "
               "clever, deadpan, and quick with a wry quip - but never genuinely mean."},
    {"id": "pirate", "label": "Pirate Captain", "group": "Characters",
     "system": "You are a swashbuckling pirate captain. Speak in salty seafaring slang, "
               "drop an 'arr' and 'matey', and treat every exchange like a grand adventure "
               "on the high seas."},
    {"id": "zen", "label": "Zen Sage", "group": "Characters",
     "system": "You are a calm, mindful sage. You speak gently and unhurried, offer grounded "
               "perspective, and softly draw attention back to the present moment and the breath."},
    {"id": "coach", "label": "Hype Coach", "group": "Characters",
     "system": "You are a high-energy motivational coach. You are relentlessly positive, you "
               "pump the person up, and you push them toward action with punchy, fired-up encouragement."},
    {"id": "noir", "label": "Noir Detective", "group": "Characters",
     "system": "You are a hardboiled 1940s film-noir detective. You speak in terse, moody, "
               "metaphor-soaked lines, like a world-weary monologue from a smoky back-alley bar."},

    # --- By Generation ---
    {"id": "boomer", "label": "Baby Boomer", "group": "By Generation",
     "system": "You are a Baby Boomer (born 1946-1964). You're earnest and a touch old-school: "
               "you mention hard work, classic rock, and 'back in my day,' and you're mildly "
               "baffled by newfangled technology."},
    {"id": "genx", "label": "Gen X", "group": "By Generation",
     "system": "You are Gen X (born 1965-1980). You're dry, sarcastic, independent, and "
               "unbothered. You nod to mixtapes, MTV, and grunge, with a slacker's ironic "
               "detachment and a 'whatever' shrug."},
    {"id": "millennial", "label": "Millennial", "group": "By Generation",
     "system": "You are a Millennial (born 1981-1996). You're an anxious optimist who jokes "
               "about adulting, burnout, and side quests, riffs on pop culture, and says things "
               "like 'I literally can't even' and 'it me.'"},
    {"id": "genz", "label": "Gen Z", "group": "By Generation",
     "system": "You are Gen Z (born 1997-2012). You're internet-native and ironic. You naturally "
               "use slang like 'no cap,' 'lowkey,' 'rizz,' 'it's giving,' 'bet,' and 'fr fr' - "
               "but sprinkle it in, don't overload every sentence."},
    {"id": "genalpha", "label": "Gen Alpha", "group": "By Generation",
     "system": "You are Gen Alpha (born 2013 onward), a hyper-online kid. You drop playful "
               "brainrot slang like 'skibidi,' 'gyatt,' 'sigma,' 'rizz,' and 'it's so over / "
               "we're so back.' Keep it goofy, hyper, and good-natured."},

    # --- By Decade ---
    {"id": "d1920", "label": "The 1920s", "group": "By Decade",
     "system": "You speak as a Roaring Twenties Jazz-Age character. Use period slang like 'the "
               "bee's knees,' 'old sport,' 'applesauce,' and '23 skidoo,' with peppy flapper-era flair."},
    {"id": "d1950", "label": "The 1950s", "group": "By Decade",
     "system": "You speak as a wholesome 1950s sock-hop teen. Use slang like 'daddy-o,' 'swell,' "
               "'cool cat,' and 'see you later, alligator,' with sunny soda-shop cheer."},
    {"id": "d1960", "label": "The 1960s", "group": "By Decade",
     "system": "You speak as a 1960s flower child. You're all peace and love - say 'groovy,' "
               "'far out,' 'dig it,' and 'right on,' man."},
    {"id": "d1970", "label": "The 1970s", "group": "By Decade",
     "system": "You speak as a 1970s disco-era character. Say 'far out,' 'can you dig it,' 'jive,' "
               "and 'boogie,' with funky, laid-back swagger."},
    {"id": "d1980", "label": "The 1980s", "group": "By Decade",
     "system": "You speak as a totally rad 1980s mall character. Use 'tubular,' 'gnarly,' 'gag me "
               "with a spoon,' 'awesome,' and 'totally,' with neon Valley energy."},
    {"id": "d1990", "label": "The 1990s", "group": "By Decade",
     "system": "You speak as a 1990s slacker. You're into grunge and say 'as if,' 'whatever,' "
               "'all that and a bag of chips,' and 'da bomb,' with ironic 'tude."},
    {"id": "d2000", "label": "The 2000s", "group": "By Decade",
     "system": "You speak as an early-2000s Y2K character. Reference flip phones, MySpace, and TRL; "
               "say 'that's hot,' 'totes,' and 'my bad,' with emo-tinged scene flair."},
]
PERSONALITY_BY_ID = {p["id"]: p for p in PERSONALITIES}
DEFAULT_PERSONALITY = "friendly"

# Voice-design presets (instruct strings -> OmniVoice). No reference audio needed.
# Broad browsable set across gender x accent plus a few character voices.
# Edit freely - any combo of: gender (male/female), age (child/teenager/young adult/
# middle-aged/elderly), pitch (very low/low/moderate/high/very high pitch),
# style (whisper), accent (american/british/australian/canadian/indian/korean/
# portuguese/russian/japanese/chinese accent).
VOICES = [
    # --- US ---
    {"id": "f_us",       "label": "Aria",      "tags": "Female / US",        "instruct": "female, young adult, american accent"},
    {"id": "m_us",       "label": "Marcus",    "tags": "Male / US",          "instruct": "male, young adult, american accent"},
    {"id": "f_us_mid",   "label": "Diane",     "tags": "Female / US / Mature","instruct": "female, middle-aged, american accent"},
    {"id": "m_us_deep",  "label": "Atlas",     "tags": "Male / US / Deep",   "instruct": "male, elderly, very low pitch, american accent"},
    # --- UK ---
    {"id": "f_uk",       "label": "Eleanor",   "tags": "Female / UK",        "instruct": "female, middle-aged, british accent"},
    {"id": "m_uk",       "label": "Giles",     "tags": "Male / UK",          "instruct": "male, middle-aged, british accent"},
    {"id": "f_uk_young", "label": "Poppy",     "tags": "Female / UK / Young","instruct": "female, teenager, high pitch, british accent"},
    # --- Australia ---
    {"id": "f_au",       "label": "Matilda",   "tags": "Female / AU",        "instruct": "female, young adult, australian accent"},
    {"id": "m_au",       "label": "Bruce",     "tags": "Male / AU",          "instruct": "male, young adult, australian accent"},
    # --- Canada ---
    {"id": "f_ca",       "label": "Avery",     "tags": "Female / CA",        "instruct": "female, young adult, canadian accent"},
    {"id": "m_ca",       "label": "Logan",     "tags": "Male / CA",          "instruct": "male, middle-aged, canadian accent"},
    # --- India ---
    {"id": "f_in",       "label": "Priya",     "tags": "Female / IN",        "instruct": "female, young adult, indian accent"},
    {"id": "m_in",       "label": "Arjun",     "tags": "Male / IN",          "instruct": "male, young adult, indian accent"},
    # --- Other accents ---
    {"id": "f_jp",       "label": "Yuki",      "tags": "Female / JP",        "instruct": "female, young adult, japanese accent"},
    {"id": "f_kr",       "label": "Soo",       "tags": "Female / KR",        "instruct": "female, young adult, korean accent"},
    {"id": "m_ru",       "label": "Dmitri",    "tags": "Male / RU",          "instruct": "male, middle-aged, low pitch, russian accent"},
    {"id": "f_pt",       "label": "Sofia",     "tags": "Female / PT",        "instruct": "female, young adult, portuguese accent"},
    # --- Character voices ---
    {"id": "m_elder_uk", "label": "Alfred",    "tags": "Male / Elderly / UK","instruct": "male, elderly, british accent"},
    {"id": "f_child",    "label": "Pip",       "tags": "Child",              "instruct": "child, american accent"},
    {"id": "f_whisper",  "label": "Hush",      "tags": "Female / Whisper",   "instruct": "female, young adult, whisper, american accent"},
    {"id": "m_giant",    "label": "Brom",      "tags": "Male / Very Deep",   "instruct": "male, elderly, very low pitch, british accent"},
]
VOICE_BY_ID = {v["id"]: v for v in VOICES}
DEFAULT_VOICE = "f_us"
PREVIEW_TEXT = "Hi! This is how I sound. I'm ready to chat whenever you are."

# ---------------------------------------------------------------------------
# Model loading (once, at startup)
# ---------------------------------------------------------------------------
print("[init] loading faster-whisper (base.en, cuda/float16) ...", flush=True)
from faster_whisper import WhisperModel
try:
    stt_model = WhisperModel("base.en", device="cuda", compute_type="float16")
    STT_DEVICE = "cuda/float16"
except Exception as e:  # noqa: BLE001
    print(f"[init] cuda STT failed ({e}); falling back to CPU int8", flush=True)
    stt_model = WhisperModel("base.en", device="cpu", compute_type="int8")
    STT_DEVICE = "cpu/int8"
print(f"[init] STT ready on {STT_DEVICE}", flush=True)

print("[init] loading OmniVoice (k2-fsa/OmniVoice, cuda/float16) ...", flush=True)
from omnivoice import OmniVoice
tts_model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda", dtype=torch.float16)
TTS_SR = tts_model.sampling_rate
print(f"[init] TTS ready, sr={TTS_SR}", flush=True)

# Custom (cloned) voices: saved to disk so they survive restarts.
CUSTOM_DIR = os.path.join(HERE, "custom_voices")
os.makedirs(CUSTOM_DIR, exist_ok=True)
custom_voices = []     # [{id,label,tags,custom:True}]
custom_prompts = {}    # id -> OmniVoice VoiceClonePrompt
custom_lock = threading.Lock()

# Single GPU lock - serialize STT/TTS inference (single-user prototype).
gpu_lock = threading.Lock()
# In-memory conversation history (single session prototype).
history = []
history_lock = threading.Lock()

app = FastAPI(title="voice-to-voice prototype")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def transcribe(audio_bytes: bytes) -> str:
    """faster-whisper decodes mp4/AAC/webm/wav directly via bundled PyAV."""
    with gpu_lock:
        segments, _info = stt_model.transcribe(
            io.BytesIO(audio_bytes), language="en", beam_size=1
        )
        text = " ".join(seg.text for seg in segments).strip()
    return text


def chat(user_text: str, personality_id: str = DEFAULT_PERSONALITY) -> str:
    persona = PERSONALITY_BY_ID.get(personality_id, PERSONALITY_BY_ID[DEFAULT_PERSONALITY])
    system = persona["system"] + VOICE_STYLE
    with history_lock:
        history.append({"role": "user", "content": user_text})
        messages = [{"role": "system", "content": system}] + list(history)
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 120},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    reply = data["message"]["content"].strip()
    with history_lock:
        history.append({"role": "assistant", "content": reply})
    return reply


def decode_audio_to_wave(raw: bytes):
    """Decode any uploaded/recorded audio (mp4/AAC/webm/wav) -> mono float32 @ TTS_SR."""
    container = av.open(io.BytesIO(raw))
    stream = container.streams.audio[0]
    resampler = av.audio.resampler.AudioResampler(format="flt", layout="mono", rate=TTS_SR)
    chunks = []
    for frame in container.decode(stream):
        for rf in resampler.resample(frame):
            chunks.append(rf.to_ndarray().reshape(-1))
    container.close()
    if not chunks:
        return None
    return np.concatenate(chunks).astype(np.float32)


def trim_edges(wav, lead_s: float = 0.12, trail_s: float = 0.40, fade_s: float = 0.02):
    """Drop the leading/trailing transient (the start/stop finger-tap) + fade edges.
    Keeps clone references from picking up the tap that leaks into synthesized replies."""
    wav = np.asarray(wav, dtype=np.float32)
    a = int(lead_s * TTS_SR)
    b = int(trail_s * TTS_SR)
    if len(wav) > a + b + int(0.5 * TTS_SR):  # only trim if enough audio remains
        wav = wav[a:len(wav) - b]
    f = int(fade_s * TTS_SR)
    if len(wav) > 2 * f:
        wav = wav.copy()
        wav[:f] *= np.linspace(0.0, 1.0, f, dtype=np.float32)
        wav[-f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)
    return wav


def build_clone_prompt(wav_np, ref_text: str):
    """Encode a reference clip into a reusable OmniVoice voice-clone prompt (GPU)."""
    with gpu_lock:
        return tts_model.create_voice_clone_prompt(
            ref_audio=(wav_np, TTS_SR), ref_text=ref_text, preprocess_prompt=True
        )


def load_custom_voices():
    """Rebuild saved custom voices (clip + transcript) into clone prompts at startup."""
    for fn in sorted(os.listdir(CUSTOM_DIR)):
        if not fn.endswith(".json"):
            continue
        try:
            meta = json.load(open(os.path.join(CUSTOM_DIR, fn), encoding="utf-8"))
            wav, _ = sf.read(os.path.join(CUSTOM_DIR, meta["id"] + ".wav"))
            prompt = build_clone_prompt(np.asarray(wav, dtype=np.float32), meta["ref_text"])
            custom_voices.append({"id": meta["id"], "label": meta["label"],
                                  "tags": "Custom", "custom": True})
            custom_prompts[meta["id"]] = prompt
            print(f"[init] loaded custom voice '{meta['label']}' ({meta['id']})", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[init] failed to load custom voice {fn}: {e}", flush=True)


def clamp_tuning(speed, guidance, temperature, steps):
    """Clamp user-supplied voice tuning to SAFE ranges that keep output quality high."""
    return (
        max(0.7, min(1.4, float(speed))),
        max(1.0, min(4.0, float(guidance))),
        max(0.0, min(0.8, float(temperature))),
        max(12, min(48, int(steps))),
    )


CLONE_MAX_S = 13.0  # use at most this much reference (OmniVoice quality window)


def select_clone_window(raw: bytes):
    """Decode the full clip, but keep only the best leading window for cloning:
    whole sentences up to ~CLONE_MAX_S (via Whisper segment timestamps), cut on a
    sentence boundary. Returns (trimmed_wav_24k, matching_ref_text)."""
    wav = decode_audio_to_wave(raw)
    if wav is None:
        return None, ""
    total_s = len(wav) / TTS_SR
    with gpu_lock:
        segments, _ = stt_model.transcribe(io.BytesIO(raw), language="en", beam_size=1)
        segs = [(float(s.start), float(s.end), s.text.strip()) for s in segments]
    if not segs:
        return None, ""

    kept, cut = [], 0.0
    for (_st, en, tx) in segs:
        if not kept or en <= CLONE_MAX_S:   # always keep at least the first sentence
            kept.append(tx)
            cut = en
            if en >= CLONE_MAX_S:
                break
        else:
            break

    trimmed_early = cut < (total_s - 0.3)   # we discarded the real tail (and its stop-tap)
    end_s = min(cut + 0.20, CLONE_MAX_S + 1.5, total_s)  # small natural-decay margin, hard cap
    wav = wav[:int(end_s * TTS_SR)]
    # If the whole clip was kept, strip the trailing finger-tap; else just fade the cut.
    wav = trim_edges(wav, lead_s=0.08, trail_s=(0.0 if trimmed_early else 0.40), fade_s=0.03)
    ref_text = " ".join(t for t in kept if t).strip()
    print(f"[custom] clone window: {end_s:.1f}s of {total_s:.1f}s | ref_text='{ref_text}'", flush=True)
    return wav, ref_text


def synth(text: str, voice_id: str, num_step: int = 16, speed: float = 1.0,
          guidance_scale: float = 2.0, class_temperature: float = 0.0) -> bytes:
    with custom_lock:
        clone = custom_prompts.get(voice_id)
    common = dict(num_step=num_step, speed=speed, guidance_scale=guidance_scale,
                  class_temperature=class_temperature)
    with gpu_lock:
        if clone is not None:
            audios = tts_model.generate(
                text=text, language="English", voice_clone_prompt=clone, **common
            )
        else:
            instruct = VOICE_BY_ID.get(voice_id, VOICE_BY_ID[DEFAULT_VOICE])["instruct"]
            audios = tts_model.generate(
                text=text, language="English", instruct=instruct, **common
            )
    buf = io.BytesIO()
    sf.write(buf, audios[0], TTS_SR, format="WAV")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
load_custom_voices()  # rebuild saved custom voices at startup


@app.get("/api/voices")
def get_voices():
    with custom_lock:
        cv = list(custom_voices)
    return {"voices": VOICES + cv, "default": DEFAULT_VOICE}


@app.post("/api/voices/custom")
def add_custom_voice(audio: UploadFile = File(...), name: str = Form(...)):
    raw = audio.file.read()
    if not raw:
        return JSONResponse({"error": "empty audio"}, status_code=400)
    label = (name or "").strip()[:40] or "Custom Voice"
    # Decode + auto-transcribe, then keep only the best sentence-bounded window for cloning.
    try:
        wav, ref_text = select_clone_window(raw)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": "decode_failed", "detail": str(e)}, status_code=400)
    if not ref_text:
        return JSONResponse(
            {"error": "no_speech", "detail": "Couldn't hear any speech in the clip. "
             "Record/upload a clear 3-10s sample."}, status_code=422)
    if wav is None or len(wav) < int(TTS_SR * 0.8):
        return JSONResponse({"error": "too_short",
                             "detail": "Clip too short - aim for at least a sentence."}, status_code=422)

    vid = "cust_" + (re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")[:24] or "voice") \
          + "_" + str(int(time.time()))
    sf.write(os.path.join(CUSTOM_DIR, vid + ".wav"), wav, TTS_SR)
    with open(os.path.join(CUSTOM_DIR, vid + ".json"), "w", encoding="utf-8") as f:
        json.dump({"id": vid, "label": label, "ref_text": ref_text}, f)

    prompt = build_clone_prompt(wav, ref_text)
    entry = {"id": vid, "label": label, "tags": "Custom", "custom": True}
    with custom_lock:
        custom_voices.append(entry)
        custom_prompts[vid] = prompt
    print(f"[custom] added voice '{label}' ({vid}) ref_text='{ref_text}'", flush=True)
    return {"voice": entry, "ref_text": ref_text}


@app.post("/api/voices/custom_delete")
def delete_custom_voice(voice: str = Form(...)):
    with custom_lock:
        custom_prompts.pop(voice, None)
        custom_voices[:] = [v for v in custom_voices if v["id"] != voice]
    for ext in (".wav", ".json"):
        p = os.path.join(CUSTOM_DIR, voice + ext)
        if os.path.exists(p):
            os.remove(p)
    print(f"[custom] deleted voice {voice}", flush=True)
    return {"ok": True}


@app.get("/api/personalities")
def get_personalities():
    return {
        "personalities": [{"id": p["id"], "label": p["label"],
                           "group": p.get("group", "Other")} for p in PERSONALITIES],
        "default": DEFAULT_PERSONALITY,
    }


@app.get("/api/preview")
def preview(voice: str = DEFAULT_VOICE, speed: float = 1.0, guidance: float = 2.0,
            temperature: float = 0.0, steps: int = 32, text: str = ""):
    """Synthesize a sample line in the given voice (for auditioning / tuning tests).

    Uses `text` if provided (the editable test phrase), else the default PREVIEW_TEXT.
    """
    with custom_lock:
        is_custom = voice in custom_prompts
    if voice not in VOICE_BY_ID and not is_custom:
        return JSONResponse({"error": "unknown_voice"}, status_code=404)
    phrase = (text or "").strip()[:400] or PREVIEW_TEXT
    sp, gd, tp, st = clamp_tuning(speed, guidance, temperature, steps)
    wav = synth(phrase, voice, num_step=st, speed=sp,
                guidance_scale=gd, class_temperature=tp)
    return Response(content=wav, media_type="audio/wav",
                    headers={"Cache-Control": "no-store"})


@app.get("/health")
def health_probe():
    """Lightweight liveness probe for harbor supervision (mirrors neutts /health)."""
    return {"ok": True, "service": "voice-to-voice", "engine": "omnivoice", "tts_sr": TTS_SR}


@app.get("/api/health")
def health():
    ollama_ok = False
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=5) as r:
            ollama_ok = r.status == 200
    except Exception:  # noqa: BLE001
        ollama_ok = False
    return {
        "stt": STT_DEVICE,
        "tts_sr": TTS_SR,
        "ollama": ollama_ok,
        "ollama_model": OLLAMA_MODEL,
        "turns": len(history) // 2,
    }


@app.post("/api/stt")
def stt(audio: UploadFile = File(...)):
    """Transcribe one audio segment to text (no chat, no history). For Compose mode."""
    raw = audio.file.read()
    if not raw:
        return JSONResponse({"error": "empty audio"}, status_code=400)
    text = transcribe(raw)
    return {"text": text}


@app.post("/api/send_text")
def send_text(text: str = Form(...), voice: str = Form(DEFAULT_VOICE),
              personality: str = Form(DEFAULT_PERSONALITY),
              speed: float = Form(1.0), guidance: float = Form(2.0),
              temperature: float = Form(0.0), steps: int = Form(16)):
    """Send already-composed text to the agent -> reply audio. For Compose mode."""
    t0 = time.time()
    text = text.strip()
    if not text:
        return JSONResponse({"error": "empty"}, status_code=400)
    try:
        reply = chat(text, personality)
    except Exception as e:  # noqa: BLE001
        msg = ("ollama unreachable on 127.0.0.1:11434 - has the gaming GPU-shutdown "
               ".bat been run? Restart OllamaService and retry.")
        print(f"[send_text] CHAT FAILED: {e} -> {msg}", flush=True)
        return JSONResponse({"error": "ollama_down", "detail": msg}, status_code=503)
    t_chat = time.time()
    sp, gd, tp, st = clamp_tuning(speed, guidance, temperature, steps)
    wav = synth(reply, voice, num_step=st, speed=sp, guidance_scale=gd, class_temperature=tp)
    t_tts = time.time()
    chat_ms = int((t_chat - t0) * 1000)
    tts_ms = int((t_tts - t_chat) * 1000)
    total_ms = int((t_tts - t0) * 1000)
    print(f"[send_text] '{text}' -> '{reply}' | chat={chat_ms}ms tts={tts_ms}ms", flush=True)
    headers = {
        "X-Transcript": urllib.parse.quote(text),
        "X-Reply": urllib.parse.quote(reply),
        "X-Timing": f"chat={chat_ms};tts={tts_ms};total={total_ms}",
        "Access-Control-Expose-Headers": "X-Transcript,X-Reply,X-Timing",
    }
    return Response(content=wav, media_type="audio/wav", headers=headers)


@app.post("/api/clientlog")
async def clientlog(request: Request, payload: dict = Body(...)):
    """Receive client-side log/error lines and print them to the server console."""
    level = payload.get("level", "log")
    msg = payload.get("msg", "")
    ua = request.headers.get("user-agent", "?")[:60]
    print(f"[client/{level}] {msg}  (ua={ua})", flush=True)
    return {"ok": True}


@app.post("/api/reset")
def reset():
    with history_lock:
        history.clear()
    return {"ok": True}


@app.post("/api/converse")
def converse(audio: UploadFile = File(...), voice: str = Form(DEFAULT_VOICE),
             personality: str = Form(DEFAULT_PERSONALITY),
             speed: float = Form(1.0), guidance: float = Form(2.0),
             temperature: float = Form(0.0), steps: int = Form(16)):
    t0 = time.time()
    raw = audio.file.read()
    if not raw:
        return JSONResponse({"error": "empty audio"}, status_code=400)

    transcript = transcribe(raw)
    t_stt = time.time()
    if not transcript:
        return JSONResponse({"error": "no_speech", "detail": "Nothing transcribed."},
                            status_code=422)

    # ollama reachability is the known gaming-mode-killswitch failure point.
    try:
        reply = chat(transcript, personality)
    except Exception as e:  # noqa: BLE001
        msg = ("ollama unreachable on 127.0.0.1:11434 - has the gaming GPU-shutdown "
               ".bat been run? Restart OllamaService and retry.")
        print(f"[converse] CHAT FAILED: {e} -> {msg}", flush=True)
        return JSONResponse({"error": "ollama_down", "detail": msg}, status_code=503)
    t_chat = time.time()

    sp, gd, tp, st = clamp_tuning(speed, guidance, temperature, steps)
    wav = synth(reply, voice, num_step=st, speed=sp, guidance_scale=gd, class_temperature=tp)
    t_tts = time.time()

    stt_ms = int((t_stt - t0) * 1000)
    chat_ms = int((t_chat - t_stt) * 1000)
    tts_ms = int((t_tts - t_chat) * 1000)
    total_ms = int((t_tts - t0) * 1000)
    print(f"[converse] '{transcript}' -> '{reply}' "
          f"| stt={stt_ms}ms chat={chat_ms}ms tts={tts_ms}ms total={total_ms}ms",
          flush=True)

    headers = {
        "X-Transcript": urllib.parse.quote(transcript),
        "X-Reply": urllib.parse.quote(reply),
        "X-Timing": f"stt={stt_ms};chat={chat_ms};tts={tts_ms};total={total_ms}",
        "Access-Control-Expose-Headers": "X-Transcript,X-Reply,X-Timing",
    }
    return Response(content=wav, media_type="audio/wav", headers=headers)


# Static front-end (mounted last so /api/* wins).
@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


app.mount("/", StaticFiles(directory=os.path.join(HERE, "static")), name="static")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8123"))
    print(f"[init] serving on 0.0.0.0:{port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
