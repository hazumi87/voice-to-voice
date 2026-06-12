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
import contextlib
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
from fastapi.middleware.cors import CORSMiddleware

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
    {"id": "sg_uncle", "label": "Singaporean Uncle", "group": "Characters",
     "system": "You are a friendly Singaporean uncle who speaks gentle, natural Singlish. Use "
               "Singlish lightly: AT MOST ONE particle (like 'lah', 'lor', 'leh') in a sentence, and "
               "NOT in every sentence - many sentences should have none. The rhythm and word order "
               "carry the accent more than the particles do. Occasionally a phrase like 'can or not' "
               "or 'where got' is fine, but sparingly. Be warm and matter-of-fact, like a real uncle "
               "at the kopitiam - understated, not a caricature."},

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

# Sample status-report paragraphs for the prototype (pick instead of retyping). A spread
# from plain to technical to terse to long, so reword can be tested against real-shaped text.
STATUS_PRESETS = [
    {"id": "plain", "label": "Plain status",
     "text": "Where we landed — the seam itself is done. Aurora accepted the whole event and "
             "command surface, and I ratified their answers. The only open items are integration "
             "mechanics on my side, not the contract."},
    {"id": "technical", "label": "Technical (stack / latency)",
     "text": "Service is green on port eighty-two twenty-one. Health returns loaded true with "
             "twenty-one voices. The synthesize endpoint renders a twenty-four kilohertz mono wav "
             "in about one and a half seconds on the four-eighty, and the clone prompt is cached "
             "after first use so repeat calls skip re-encoding."},
    {"id": "terse", "label": "Terse one-liner",
     "text": "Seam's done, contract ratified, only integration left on my side."},
    {"id": "long", "label": "Long multi-sentence",
     "text": "Here's where we are. The registration path is wired end to end: you pick a clip, "
             "name it, and it lands in the catalog as a voice. The consumer pulls it on demand and "
             "caches the encoded prompt, so the engine stays stateless and relocatable. Aurora "
             "signed off on the whole event and command surface, and I ratified their answers. "
             "What's left is integration mechanics on my side, plus deciding how the picker filters "
             "voices once there are more than a handful. Nothing blocking, just sequencing."},
    {"id": "blocker", "label": "Blocker / escalation",
     "text": "I'm blocked on the port registry write. The file is root owned and the service "
             "account has no sudo for it, so neither the shell nor the broker can touch it. I need "
             "a root capable actor to make the edit before I can move forward."},
    {"id": "milestone", "label": "Milestone done",
     "text": "Milestone hit. OmniVoice is live as the default speak engine, the voice prototype is "
             "up on the tablet, and the asset library rundoc is written for the agent. Calling it."},
]

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
# STT runs on CPU/int8 by DEFAULT - deliberately off the GPU. The GPU Whisper
# (cuda/float16) is the instance that has hung this service: a CUDA stall or
# OOM-that-hangs under the shared gpu_lock would take TTS down with it. The
# asset-library proves base/int8 on CPU is rock-solid, and the VRPC CPU handles
# base.en/int8 for short utterances in well under a second - fast enough for the
# live iPad loop. Keeping STT off the GPU also means it no longer contends with
# OmniVoice for gpu_lock: STT (CPU) and TTS (GPU) run in parallel.
# Set STT_DEVICE_PREF=cuda to force the GPU build back (A/B only).
_stt_pref = os.environ.get("STT_DEVICE_PREF", "cpu").lower()
from faster_whisper import WhisperModel
if _stt_pref == "cuda":
    print("[init] loading faster-whisper (base.en, cuda/float16) [forced] ...", flush=True)
    try:
        stt_model = WhisperModel("base.en", device="cuda", compute_type="float16")
        STT_DEVICE = "cuda/float16"
    except Exception as e:  # noqa: BLE001
        print(f"[init] cuda STT failed ({e}); falling back to CPU int8", flush=True)
        stt_model = WhisperModel("base.en", device="cpu", compute_type="int8")
        STT_DEVICE = "cpu/int8"
else:
    print("[init] loading faster-whisper (base.en, cpu/int8) ...", flush=True)
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
# Serializing GPU work is correct: one GPU + non-reentrant model state means two
# concurrent generate()/transcribe() calls would corrupt output or crash, not run
# faster. The DANGER is an op that HANGS while holding the lock (we've seen Whisper
# wedge) - it would block every later request forever and take the whole service
# dark. gpu_guard() is the insurance: a bounded acquire (503 instead of infinite
# wait) plus a watchdog that NAMES the in-flight op and warns if it runs long, so
# the next hang diagnoses itself instead of being another mystery restart.
gpu_lock = threading.Lock()

# STT now runs on CPU (see STT load below), so it must NOT share gpu_lock with TTS -
# that would serialize CPU STT behind GPU TTS for no reason. Its own lock keeps STT
# single-flight (faster-whisper isn't reentrant) while letting it run in parallel
# with OmniVoice on the GPU.
stt_lock = threading.Lock()

GPU_ACQUIRE_TIMEOUT_S = float(os.environ.get("GPU_ACQUIRE_TIMEOUT_S", "45"))  # max wait for the lock
GPU_WATCHDOG_WARN_S = float(os.environ.get("GPU_WATCHDOG_WARN_S", "20"))      # warn if an op runs past this
_gpu_inflight = {"op": None, "since": 0.0}  # what currently holds the lock (for /health + logs)


class GpuBusy(Exception):
    """The GPU lock could not be acquired within GPU_ACQUIRE_TIMEOUT_S."""


@contextlib.contextmanager
def gpu_guard(op: str):
    """Acquire gpu_lock with a timeout + a watchdog that logs slow/hung GPU ops.

    - Bounded acquire: if another op holds the GPU past GPU_ACQUIRE_TIMEOUT_S, raise
      GpuBusy (caller returns 503) instead of blocking this worker forever.
    - Watchdog: a daemon timer fires at GPU_WATCHDOG_WARN_S and logs WHICH op is
      still running and for how long. Repeats so a true hang leaves a clear trail.
    - Records the in-flight op so /health can report it without touching gpu_lock.
    """
    if not gpu_lock.acquire(timeout=GPU_ACQUIRE_TIMEOUT_S):
        held = _gpu_inflight["op"]
        held_for = (time.time() - _gpu_inflight["since"]) if _gpu_inflight["since"] else 0.0
        print(f"[gpu] BUSY: '{op}' waited {GPU_ACQUIRE_TIMEOUT_S:.0f}s; "
              f"'{held}' has held the GPU for {held_for:.0f}s", flush=True)
        raise GpuBusy(f"gpu busy: '{held}' in-flight {held_for:.0f}s")

    start = time.time()
    _gpu_inflight["op"] = op
    _gpu_inflight["since"] = start
    stop_watchdog = threading.Event()

    def _watch():
        n = 0
        while not stop_watchdog.wait(GPU_WATCHDOG_WARN_S):
            n += 1
            print(f"[gpu] SLOW: '{op}' still running after "
                  f"{time.time() - start:.0f}s (warn #{n})", flush=True)

    wd = threading.Thread(target=_watch, name=f"gpu-watchdog:{op}", daemon=True)
    wd.start()
    try:
        yield
    finally:
        stop_watchdog.set()
        dur = time.time() - start
        _gpu_inflight["op"] = None
        _gpu_inflight["since"] = 0.0
        gpu_lock.release()
        if dur >= GPU_WATCHDOG_WARN_S:
            print(f"[gpu] done '{op}' in {dur:.1f}s", flush=True)
# In-memory conversation history (single session prototype).
history = []
history_lock = threading.Lock()

app = FastAPI(title="voice-to-voice prototype")

# Browser consumers (e.g. the NUC asset-library audition player at http://hazwebserver)
# call /synthesize_ref cross-origin. A multipart POST triggers a CORS preflight (OPTIONS),
# so the service must answer it AND advertise Access-Control-Allow-Origin on the response.
# Endpoint returns raw WAV bytes with no cookies/credentials, so allow_origins=["*"] is safe.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(GpuBusy)
def _gpu_busy_handler(request: Request, exc: GpuBusy):
    # A GPU op held the lock past GPU_ACQUIRE_TIMEOUT_S. Return 503 (retryable) so
    # one slow/stuck op no longer cascades into a total outage - the worker is freed
    # and the service keeps answering. The speak relay treats a non-200 as "tier
    # unreachable" and falls back to browser, which is the right graceful degrade.
    return JSONResponse({"error": "gpu_busy", "detail": str(exc)}, status_code=503)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def transcribe(audio_bytes: bytes) -> str:
    """faster-whisper decodes mp4/AAC/webm/wav directly via bundled PyAV.
    Runs on CPU (default) under stt_lock - off the GPU lock, parallel to TTS."""
    with stt_lock:
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


# ---------------------------------------------------------------------------
# Personality REWORD transform (pull-side prototype)
#
# Distinct from chat(): chat() GENERATES a reply in-character (personality = the
# agent). reword() takes text the agent ALREADY wrote and rephrases it in a
# personality's voice BEFORE synthesis. This is the speak()-world mechanic: the
# agent says X, the pull side rewords X, then OmniVoice speaks it.
#
# `strength` is a spectrum, not a toggle:
#   none  -> speak verbatim, personality ignored
#   light -> subtle accent/dialect flavoring; SAME meaning + ~same length
#   full  -> rephrase fully in-character (slang, cadence); meaning preserved,
#            length may change
# Stateless (no history) — it's a transform, not a conversation.
# ---------------------------------------------------------------------------
REWORD_STRENGTH = {
    "light": (
        "Lightly adjust the following line to carry a SUBTLE flavor of the character's dialect "
        "and rhythm. Keep the SAME meaning, the same facts, and roughly the same length. Change "
        "as FEW words as possible - mostly word order and cadence, not vocabulary. Add at MOST one "
        "piece of dialect slang in the whole line, and only if it fits naturally; often add none. "
        "Do not literalize metaphors. Output ONLY the adjusted line, nothing else."
    ),
    "full": (
        "Rephrase the following line in the character's voice - their cadence, attitude, and some "
        "slang. Preserve the underlying meaning and facts; do not literalize figurative phrases. "
        "Use the character's slang TASTEFULLY and sparingly - at most a couple of dialect markers "
        "in the whole line, never one in every sentence. Sound authentic, not like a caricature. "
        "Output ONLY the rephrased line, nothing else."
    ),
}


def reword(text: str, personality_id: str = DEFAULT_PERSONALITY, strength: str = "full") -> str:
    """Rephrase already-written text in a personality's voice (pull-side transform)."""
    text = (text or "").strip()
    if not text or strength == "none":
        return text
    instr = REWORD_STRENGTH.get(strength, REWORD_STRENGTH["full"])
    persona = PERSONALITY_BY_ID.get(personality_id, PERSONALITY_BY_ID[DEFAULT_PERSONALITY])
    # The persona's own system prompt establishes WHO the character is; instr says
    # HOW hard to transform. VOICE_STYLE keeps it speech-shaped (no markdown, short).
    system = persona["system"] + " " + instr + VOICE_STYLE
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 200},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Editable prompts config (dev-panel pattern: edits persist to prompts.json,
# hot-reloaded into the running service; hardcoded values above are the defaults).
#
# The editable surface is exactly the 3 prompt component types you identified:
#   - strength.light / strength.full   (the 2 shared STRENGTH blocks)
#   - voice_style                      (the 1 shared VOICE_STYLE block)
#   - characters[<id>]                 (each personality's unique CHARACTER blurb)
# Saving overlays prompts.json onto the in-memory REWORD_STRENGTH / VOICE_STYLE /
# PERSONALITY_BY_ID so reword() + the anatomy endpoint immediately use the edits.
# ---------------------------------------------------------------------------
PROMPTS_PATH = os.path.join(HERE, "prompts.json")


def current_prompts() -> dict:
    """The current editable prompt strings (defaults + any saved overrides applied)."""
    return {
        "strength": {"light": REWORD_STRENGTH["light"], "full": REWORD_STRENGTH["full"]},
        "voice_style": VOICE_STYLE,
        "characters": [
            {"id": p["id"], "label": p["label"], "group": p.get("group", "Other"),
             "system": p["system"]}
            for p in PERSONALITIES
        ],
    }


def apply_prompts(cfg: dict):
    """Overlay a prompts dict onto the live in-memory strings (hot-reload)."""
    global VOICE_STYLE
    st = cfg.get("strength") or {}
    for k in ("light", "full"):
        if isinstance(st.get(k), str) and st[k].strip():
            REWORD_STRENGTH[k] = st[k]
    if isinstance(cfg.get("voice_style"), str) and cfg["voice_style"].strip():
        VOICE_STYLE = cfg["voice_style"]
    for c in (cfg.get("characters") or []):
        p = PERSONALITY_BY_ID.get(c.get("id"))
        if p and isinstance(c.get("system"), str) and c["system"].strip():
            p["system"] = c["system"]


def load_prompts_override():
    """At startup: if prompts.json exists, apply it over the hardcoded defaults."""
    if os.path.isfile(PROMPTS_PATH):
        try:
            apply_prompts(json.load(open(PROMPTS_PATH, encoding="utf-8")))
            print(f"[prompts] applied overrides from {PROMPTS_PATH}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[prompts] failed to load {PROMPTS_PATH}: {e}", flush=True)


def save_prompts(cfg: dict):
    """Persist edits to prompts.json AND hot-reload them into the running service."""
    apply_prompts(cfg)                       # live first
    with open(PROMPTS_PATH, "w", encoding="utf-8") as f:
        json.dump(current_prompts(), f, ensure_ascii=False, indent=2)


load_prompts_override()


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
    with gpu_guard("tts.create_clone_prompt"):
        return tts_model.create_voice_clone_prompt(
            ref_audio=(wav_np, TTS_SR), ref_text=ref_text, preprocess_prompt=True
        )


# Built-in reference clips for the prototype (working/ dir). iPad-friendly: pick by
# name instead of uploading. Prompts are cached so a clip is only encoded once.
WORKING_DIR = os.path.join(HERE, "working")
_refclip_cache = {}  # name -> clone prompt


def list_ref_clips():
    """Wav files in working/ that can be used as built-in prototype reference voices."""
    try:
        return sorted(f for f in os.listdir(WORKING_DIR)
                      if f.lower().endswith(".wav") and not f.startswith("_"))
    except FileNotFoundError:
        return []


def ref_clip_prompt(name: str):
    """Load + cache a clone prompt for a working/ reference clip (encode once)."""
    name = os.path.basename(name)  # no path traversal
    if name in _refclip_cache:
        return _refclip_cache[name]
    path = os.path.join(WORKING_DIR, name)
    if not os.path.isfile(path):
        return None
    wav, sr = sf.read(path)
    clip = np.asarray(wav, dtype=np.float32)
    if clip.ndim > 1:
        clip = clip.mean(axis=1)
    clip = trim_edges(clip)
    _b = io.BytesIO()
    sf.write(_b, clip, TTS_SR, format="WAV")
    rtext = transcribe(_b.getvalue())
    prompt = build_clone_prompt(clip, rtext)
    _refclip_cache[name] = prompt
    print(f"[proto] cached ref clip '{name}' ref_text='{rtext[:60]}...'", flush=True)
    return prompt


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
    with stt_lock:  # CPU STT, off the GPU lock (see transcribe())
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
    with gpu_guard(f"tts.generate[{voice_id}]"):
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


def synth_with_prompt(text: str, clone_prompt, num_step: int = 16, speed: float = 1.0,
                      guidance_scale: float = 2.0, class_temperature: float = 0.0) -> bytes:
    """Synthesize with an ad-hoc clone prompt (e.g. an uploaded wav not registered as a voice).
    Used by the prototype's ad-hoc wav+stt path so you can audition any clip without saving it."""
    common = dict(num_step=num_step, speed=speed, guidance_scale=guidance_scale,
                  class_temperature=class_temperature)
    with gpu_guard("tts.generate.adhoc_prompt"):
        audios = tts_model.generate(
            text=text, language="English", voice_clone_prompt=clone_prompt, **common
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


@app.post("/api/voices/register")
def register_custom_voice(
    audio: UploadFile = File(...),
    name: str = Form(...),
    ref_text: str = Form(...),
    voice_id: str = Form(None),
):
    """Register a custom voice from a PRE-TRANSCODED clip + a SUPPLIED transcript.

    This is the library-sourced path: the asset-library already transcodes the WAV
    and stores the ref_text, so we skip Whisper entirely. That matters - the STT step
    is the one that has hung the GPU; bypassing it removes that failure mode for
    library voices. The only GPU work here is build_clone_prompt (guarded).

    Pass voice_id to make registration idempotent/replaceable (re-registering the
    same id overwrites cleanly). Omit it to mint one from the label.
    """
    raw = audio.file.read()
    if not raw:
        return JSONResponse({"error": "empty audio"}, status_code=400)
    label = (name or "").strip()[:40] or "Custom Voice"
    ref_text = (ref_text or "").strip()
    if not ref_text:
        return JSONResponse({"error": "no_ref_text",
                             "detail": "ref_text is required for the no-STT register path."}, status_code=422)

    # Decode the (already clean) clip via PyAV -> mono float32 @ TTS_SR. No Whisper.
    try:
        wav = decode_audio_to_wave(raw)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": "decode_failed", "detail": str(e)}, status_code=400)
    if wav is None or len(wav) < int(TTS_SR * 0.8):
        return JSONResponse({"error": "too_short",
                             "detail": "Clip too short - aim for at least a sentence."}, status_code=422)
    # Library clips are pre-trimmed, but a light edge fade is cheap insurance against
    # a residual start/stop transient leaking into the cloned voice.
    wav = trim_edges(wav, lead_s=0.04, trail_s=0.04, fade_s=0.02)

    vid = (voice_id or "").strip() or (
        "cust_" + (re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")[:24] or "voice")
        + "_" + str(int(time.time()))
    )
    sf.write(os.path.join(CUSTOM_DIR, vid + ".wav"), wav, TTS_SR)
    with open(os.path.join(CUSTOM_DIR, vid + ".json"), "w", encoding="utf-8") as f:
        json.dump({"id": vid, "label": label, "ref_text": ref_text}, f)

    prompt = build_clone_prompt(wav, ref_text)  # guarded GPU op
    entry = {"id": vid, "label": label, "tags": "Custom", "custom": True}
    with custom_lock:
        # Replace any existing entry with the same id (idempotent re-register).
        custom_voices[:] = [v for v in custom_voices if v["id"] != vid]
        custom_voices.append(entry)
        custom_prompts[vid] = prompt
    print(f"[custom] registered (no-STT) '{label}' ({vid}) ref_text='{ref_text[:60]}'", flush=True)
    return {"voice": entry, "ref_text": ref_text, "skipped_stt": True}


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


@app.post("/api/prototype/speak")
def prototype_speak(
    text: str = Form(...),
    personality: str = Form(DEFAULT_PERSONALITY),
    strength: str = Form("full"),           # none | light | full
    voice: str = Form(DEFAULT_VOICE),        # preset id or saved custom id (ignored if wav uploaded)
    ref_text: str = Form(""),                # transcript for an uploaded wav (skips STT if given)
    speed: float = Form(1.0),
    guidance: float = Form(2.0),
    temperature: float = Form(0.0),
    steps: int = Form(32),
    wav: UploadFile = File(None),            # optional ad-hoc reference clip (e.g. kim-huat.wav)
    ref_clip: str = Form(""),                # OR a built-in working/ clip name (iPad-friendly, no upload)
):
    """Pull-side prototype: reword `text` in a personality (strength none/light/full),
    then speak it in the chosen voice. Voice can be a preset, a saved custom voice, or an
    ad-hoc uploaded wav (with ref_text, else auto-STT). Returns audio; the reworded text +
    metadata ride in response headers so the test form can show what was actually said.

    This is the experiment surface for: verbatim vs. reword  x  voice/accent  x  personality.
    """
    src = (text or "").strip()
    if not src:
        return JSONResponse({"error": "empty_text"}, status_code=400)
    if strength not in ("none", "light", "full"):
        strength = "full"

    # 1) TRANSFORM (pull-side): reword the agent's words in-character.
    try:
        spoken = reword(src, personality, strength)
    except Exception as e:  # noqa: BLE001 — ollama down etc.; fall back to verbatim
        return JSONResponse({"error": "reword_failed", "detail": str(e)}, status_code=503)

    sp, gd, tp, st = clamp_tuning(speed, guidance, temperature, steps)

    # 2) VOICE: built-in working/ clip (ref_clip), else ad-hoc uploaded clip, else
    #    a registered preset/custom id.
    try:
        if ref_clip:
            prompt = ref_clip_prompt(ref_clip)
            if prompt is None:
                return JSONResponse({"error": "unknown_ref_clip", "clip": ref_clip}, status_code=404)
            audio = synth_with_prompt(spoken, prompt, num_step=st, speed=sp,
                                      guidance_scale=gd, class_temperature=tp)
            used_voice = f"clip:{os.path.basename(ref_clip)}"
        elif wav is not None:
            raw = wav.file.read()
            if not raw:
                return JSONResponse({"error": "empty_wav"}, status_code=400)
            clip = decode_audio_to_wave(raw)
            if clip is None:
                return JSONResponse({"error": "decode_failed"}, status_code=400)
            clip = trim_edges(clip)
            rtext = (ref_text or "").strip()
            if not rtext:
                _b = io.BytesIO()
                sf.write(_b, clip, TTS_SR, format="WAV")
                rtext = transcribe(_b.getvalue())
            prompt = build_clone_prompt(clip, rtext)
            audio = synth_with_prompt(spoken, prompt, num_step=st, speed=sp,
                                      guidance_scale=gd, class_temperature=tp)
            used_voice = "adhoc-wav"
        else:
            with custom_lock:
                is_custom = voice in custom_prompts
            if voice not in VOICE_BY_ID and not is_custom:
                voice = DEFAULT_VOICE
            audio = synth(spoken, voice, num_step=st, speed=sp,
                          guidance_scale=gd, class_temperature=tp)
            used_voice = voice
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": "synth_failed", "detail": str(e)}, status_code=500)

    # Reworded text in a header (URL-encoded; headers must be latin-1 safe).
    return Response(content=audio, media_type="audio/wav", headers={
        "Cache-Control": "no-store",
        "x-spoken-text": urllib.parse.quote(spoken),
        "x-source-text": urllib.parse.quote(src),
        "x-personality": personality,
        "x-strength": strength,
        "x-voice": used_voice,
    })


@app.get("/api/prototype/refclips")
def prototype_refclips():
    """Built-in reference clips (working/ wavs) the prototype page can pick without uploading."""
    return {"clips": list_ref_clips()}


@app.get("/api/prototype/status_presets")
def prototype_status_presets():
    """Pre-written status-report paragraphs to test reword against (plain..technical..long)."""
    return {"presets": STATUS_PRESETS}


@app.get("/api/prototype/prompts")
def get_prompts():
    """Current editable prompt blocks (the 2 strength blocks, voice-style, per-character blurbs)."""
    return current_prompts()


@app.post("/api/prototype/prompts")
def post_prompts(cfg: dict = Body(...)):
    """Save edited prompts to prompts.json and hot-reload them live (no restart)."""
    try:
        save_prompts(cfg)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": "save_failed", "detail": str(e)}, status_code=500)
    return {"ok": True, "saved_to": PROMPTS_PATH}


@app.get("/api/prototype/prompt")
def prototype_prompt(personality: str = DEFAULT_PERSONALITY, strength: str = "full"):
    """Expose the EXACT reword prompt assembly for a given character + strength, so the
    prototype page can show how the system prompt comes together (transparency, not a guess).

    Mirrors reword() exactly:
        none  -> NO llm call; text is spoken verbatim (no prompt at all).
        else  -> system = CHARACTER + ' ' + STRENGTH + VOICE_STYLE ; user = the text.
    """
    persona = PERSONALITY_BY_ID.get(personality, PERSONALITY_BY_ID[DEFAULT_PERSONALITY])
    if strength == "none":
        return {
            "strength": "none",
            "calls_llm": False,
            "note": "strength=none: no LLM call is made. The text is spoken VERBATIM.",
            "components": [],
            "system_prompt": None,
            "model": OLLAMA_MODEL,
        }
    strength_text = REWORD_STRENGTH.get(strength, REWORD_STRENGTH["full"])
    system = persona["system"] + " " + strength_text + VOICE_STYLE
    return {
        "strength": strength,
        "calls_llm": True,
        "model": OLLAMA_MODEL,
        "components": [
            {"label": "CHARACTER", "source": "PERSONALITY_BY_ID[id]['system']", "text": persona["system"]},
            {"label": "STRENGTH", "source": f"REWORD_STRENGTH['{strength}']", "text": strength_text},
            {"label": "VOICE_STYLE", "source": "VOICE_STYLE (always appended)", "text": VOICE_STYLE.strip()},
        ],
        "assembly": "system = CHARACTER + ' ' + STRENGTH + VOICE_STYLE   |   user = <the text>",
        "system_prompt": system,
    }


@app.post("/synthesize_ref")
def synthesize_ref(
    text: str = Form(...),                    # the line to speak
    ref_text: str = Form(""),                 # transcript of the ref clip (auto-STT if omitted)
    speed: float = Form(1.0),
    guidance: float = Form(2.0),
    temperature: float = Form(0.0),
    steps: int = Form(32),
    wav: UploadFile = File(...),              # the reference clip (a voice's ref.wav)
):
    """Ad-hoc reference synth — the CLEAN player contract for consumers (asset-library, etc.).

    Hand OmniVoice a reference clip + its transcript + a line to speak; get WAV bytes back.
    Stateless: nothing is stored, no registration, no cache key. This is the (a)/(b) playback
    path from the voice-library rundoc — "test a clip" and "audition a registered voice" both
    POST the voice's ref.wav + ref_text here. The CONSUMER routes the returned bytes (e.g. a
    browser <audio> element); OmniVoice never plays audio or knows a "target".

    multipart form:
        wav        (file, required)  reference clip (any format PyAV decodes; mono ~3-15s ideal)
        text       (required)        the line to synthesize in that voice
        ref_text   (optional)        transcript of the clip; auto-transcribed if omitted
        speed/guidance/temperature/steps (optional) tuning, clamped to safe ranges
    returns: audio/wav bytes, header x-synth-device.
    """
    line = (text or "").strip()
    if not line:
        return JSONResponse({"error": "empty_text"}, status_code=400)
    raw = wav.file.read()
    if not raw:
        return JSONResponse({"error": "empty_wav"}, status_code=400)
    try:
        clip = decode_audio_to_wave(raw)
        if clip is None:
            return JSONResponse({"error": "decode_failed"}, status_code=400)
        clip = trim_edges(clip)
        rtext = (ref_text or "").strip()
        if not rtext:
            _b = io.BytesIO()
            sf.write(_b, clip, TTS_SR, format="WAV")
            rtext = transcribe(_b.getvalue())
        sp, gd, tp, st = clamp_tuning(speed, guidance, temperature, steps)
        prompt = build_clone_prompt(clip, rtext)
        audio = synth_with_prompt(line[:2000], prompt, num_step=st, speed=sp,
                                  guidance_scale=gd, class_temperature=tp)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": "synth_failed", "detail": str(e)}, status_code=500)
    return Response(content=audio, media_type="audio/wav",
                    headers={"Cache-Control": "no-store", "x-synth-device": "cuda:omnivoice"})


@app.post("/synthesize")
def synthesize(payload: dict = Body(...)):
    """neutts-compatible TTS endpoint so the speak MCP (agent-speech-relay) can treat
    voice-to-voice as a drop-in synth tier (engine=omnivoice).

    Contract mirrors neutts-synth's POST /synthesize: JSON in, raw WAV bytes out.
    Body: {text, voice?, speed?, guidance?, temperature?, steps?}. `voice` is an
    OmniVoice voice id (preset like 'f_us' or a saved custom voice); unknown/absent
    falls back to DEFAULT_VOICE so a bad voice name never fails the speak path.
    The x-synth-device header lets the relay log which device rendered it.
    """
    text = (payload.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "empty_text"}, status_code=400)
    voice = payload.get("voice") or DEFAULT_VOICE
    with custom_lock:
        is_custom = voice in custom_prompts
    if voice not in VOICE_BY_ID and not is_custom:
        voice = DEFAULT_VOICE  # tolerate unknown voice names from the relay
    # Optional generation tuning, clamped to safe ranges (same as /api/preview).
    sp, gd, tp, st = clamp_tuning(
        float(payload.get("speed", 1.0)),
        float(payload.get("guidance", 2.0)),
        float(payload.get("temperature", 0.0)),
        int(payload.get("steps", 32)),
    )
    wav = synth(text[:2000], voice, num_step=st, speed=sp,
                guidance_scale=gd, class_temperature=tp)
    return Response(content=wav, media_type="audio/wav",
                    headers={"Cache-Control": "no-store",
                             "x-synth-device": f"cuda:omnivoice"})


@app.get("/health")
def health_probe():
    """Lightweight liveness probe for harbor supervision (mirrors neutts /health).

    `loaded` + `voices` are present so the speak MCP's synthViaTier() health gate
    (which requires `loaded` truthy) accepts this service as a synth tier.
    """
    with custom_lock:
        voice_ids = [v["id"] for v in VOICES] + list(custom_prompts.keys())
    # Report the in-flight GPU op (if any) so a slow/hung op is visible from harbor
    # and the speak relay without having to read the logs. Never touches gpu_lock.
    inflight = _gpu_inflight["op"]
    gpu = {"busy": inflight is not None}
    if inflight is not None:
        gpu["op"] = inflight
        gpu["held_s"] = round(time.time() - _gpu_inflight["since"], 1)
    return {"ok": True, "loaded": True, "service": "voice-to-voice",
            "engine": "omnivoice", "device": "cuda:omnivoice",
            "tts_sr": TTS_SR, "voices": voice_ids, "gpu": gpu}


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
    port = int(os.environ.get("PORT", "8221"))
    print(f"[init] serving on 0.0.0.0:{port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
