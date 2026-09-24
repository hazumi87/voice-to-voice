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

# Reduce CUDA allocator fragmentation BEFORE torch is imported. On a contended GPU the
# OmniVoice load needs a large contiguous block; expandable_segments lets the allocator
# grow segments instead of failing to place one big block. Must be set pre-import.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---- CRASH DIAGNOSTICS (temporary) -----------------------------------------------------
# The TTS crash dies at the C/CUDA level with no Python traceback. Capture the actual death:
#  - CUDA_LAUNCH_BLOCKING makes CUDA errors surface synchronously AT the failing op (not async).
#  - TORCH_SHOW_CPP_STACKTRACES adds the C++ stack from torch.
#  - faulthandler dumps a native traceback (all threads) on a fatal signal (SIGABRT/SIGSEGV)
#    to working/crash_trace.log, so we see exactly where it dies. Slows GPU a bit — remove
#    CUDA_LAUNCH_BLOCKING once the crash is captured.
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")
import faulthandler
_HERE0 = os.path.dirname(os.path.abspath(__file__))
try:
    _crashf = open(os.path.join(_HERE0, "working", "crash_trace.log"), "a", buffering=1)
    _crashf.write(f"\n==== server start {time.strftime('%Y-%m-%d %H:%M:%S')} ====\n")
    faulthandler.enable(file=_crashf, all_threads=True)
except Exception:  # noqa: BLE001
    faulthandler.enable(all_threads=True)
# ----------------------------------------------------------------------------------------

# ---- THE "HANG" FIX: never let a log write block the event loop --------------------------
# Root cause (proven with py-spy on a wedged process): the ONLY blocking frame was
# logging.flush() on the uvicorn event-loop thread. v2v's stdout/stderr was a pipe the
# supervisor never drains; OmniVoice's tqdm progress bars flood it every render, the ~64KB
# Windows pipe buffer fills, and the next write blocks FOREVER. uvicorn runs the whole app
# on one asyncio thread, so a blocked access-log write wedges the entire server (port stays
# open, VRAM held, /health -> 000). This is NOT a CUDA/VRAM deadlock -- no thread was in
# generate()/torch/cuda. Two-part fix: (1) silence the tqdm flood; (2) route stdout/stderr
# to a FILE at the OS fd level (covers native writes too) -- a file write doesn't block on a
# full buffer the way an unread pipe does, so the loop can never wedge on a log line.
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
if os.environ.get("V2V_LOG_TO_FILE", "1") == "1":
    try:
        _logf = open(os.path.join(_HERE0, "working", "server_v2v.log"),
                     "a", buffering=1, encoding="utf-8", errors="replace")
        _logf.write(f"\n==== server start {time.strftime('%Y-%m-%d %H:%M:%S')} ====\n")
        os.dup2(_logf.fileno(), 1)   # raw stdout fd -> file (native libs writing fd 1)
        os.dup2(_logf.fileno(), 2)   # raw stderr fd -> file (tqdm, native libs writing fd 2)
        sys.stdout = _logf           # Python-level stdout (print, logging StreamHandler)
        sys.stderr = _logf           # Python-level stderr
    except Exception as _e:  # noqa: BLE001 — if redirect fails, keep default streams
        print(f"[init] stdout/stderr file redirect failed: {_e}", flush=True)
# ----------------------------------------------------------------------------------------

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

# OmniVoice's resident footprint is ~8GB VRAM. On a 16GB card shared with Ollama,
# from_pretrained can OOM. Loading it at import time meant that OOM crashed the whole
# process -> harbor restarted it -> the partial CUDA alloc leaked -> next start had even
# less VRAM -> a self-reinforcing crash-loop that took the ENTIRE site down (including the
# static /avatar mount, which needs no GPU at all). So the load is now NON-FATAL: if it
# fails the web server still starts and serves everything else; TTS lazily retries on the
# next synth via ensure_tts() once VRAM frees. tts_model is None until loaded.
TTS_SR = 24000              # OmniVoice sampling rate; refreshed from the model on load
tts_model = None
_tts_load_error = None
_tts_load_lock = threading.RLock()   # reentrant: ensure_tts -> rebuild -> build_clone_prompt
# Renders since the current model instance loaded. Reset to 0 on each (re)load so we can
# tell, per render, whether THIS was the cold first generate after a load — the state that
# historically produced the wrong/hollow first render. Logged with every synth.
_renders_since_load = 0
_warmup_paths = ""          # which generate paths the post-load warmup actually exercised


TTS_MIN_FREE_VRAM_GB = 10.5  # OmniVoice resident ~8GB; require headroom (Ollama evicted first)


def _free_vram_gb():
    """Free GPU VRAM in GB via torch (no subprocess). None if unavailable."""
    try:
        free, _total = torch.cuda.mem_get_info()
        return free / (1024 ** 3)
    except Exception:  # noqa: BLE001
        return None


def _evict_ollama():
    """Ask Ollama to unload its model from VRAM (keep_alive=0) so OmniVoice has clear
    headroom for its ~8GB load. Ollama is the main VRAM contender; a resident Ollama
    model during from_pretrained is what pushes us into the hard-abort zone. Best-effort
    with a brief settle wait. Ollama transparently reloads on the next chat/reword call."""
    try:
        body = json.dumps({"model": OLLAMA_MODEL, "keep_alive": 0}).encode()
        req = urllib.request.Request("http://127.0.0.1:11434/api/generate",
                                     data=body, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10).read()
        print("[tts] evicted Ollama model to free VRAM for OmniVoice load", flush=True)
        time.sleep(2.0)  # let the CUDA allocator actually release before we measure/alloc
    except Exception as e:  # noqa: BLE001
        print(f"[tts] ollama eviction skipped: {e}", flush=True)


def _load_tts_model():
    """Attempt to load OmniVoice onto the GPU. Returns True on success, False otherwise.
    NEVER lets the process die: a CUDA OOM during from_pretrained can be a HARD abort
    (not a catchable Python exception), so we GATE on free VRAM first and simply refuse
    to attempt the load when there isn't enough headroom — that's what prevents the
    crash-loop, not a try/except. The server runs fine without TTS (avatar/STT/chat
    don't need it); synth routes 503 until VRAM frees and a later call loads it."""
    global tts_model, TTS_SR, _tts_load_error
    if tts_model is not None:
        return True
    _evict_ollama()  # free VRAM before measuring + allocating — Ollama reloads on demand
    free = _free_vram_gb()
    if free is not None and free < TTS_MIN_FREE_VRAM_GB:
        _tts_load_error = (f"insufficient VRAM: {free:.1f}GB free < {TTS_MIN_FREE_VRAM_GB}GB needed "
                           f"(free VRAM, e.g. unload Ollama, then retry)")
        print(f"[tts] load skipped — {_tts_load_error}", flush=True)
        return False
    try:
        m = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
        tts_model = m
        TTS_SR = m.sampling_rate
        _tts_load_error = None
        print(f"[init] TTS ready, sr={TTS_SR}", flush=True)
        return True
    except Exception as e:  # noqa: BLE001
        _tts_load_error = str(e)
        print(f"[tts] OmniVoice load failed: {e}", flush=True)
        return False


# DEFERRED by default: do NOT load OmniVoice at import time. Loading it at startup meant a
# VRAM OOM there crashed the process -> harbor restart-loop -> whole site (incl. /avatar) down.
# The avatar + phonetics prototypes read only stored data and need no GPU at all, so the server
# must always start clean. TTS loads lazily on the first synth via ensure_tts(). Set EAGER_TTS=1
# to restore startup loading (only safe when VRAM is known-free).
if os.environ.get("EAGER_TTS") == "1":
    _load_tts_model()
else:
    print("[init] OmniVoice load DEFERRED (lazy on first synth); server starts GPU-free", flush=True)

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
# In-memory conversation history, PARTITIONED PER SPEAKER. Keyed by speaker id
# (from speaker_id.identify); unrecognized utterances share the "guest" thread.
# This stops overlapping users (e.g. Eric + Zachary) from contaminating each
# other's context. {key: [ {role, content}, ... ]}.
histories = {}
history_lock = threading.Lock()


def _history_for(key):
    """Get (creating if needed) the message list for a speaker key. Caller holds lock."""
    return histories.setdefault(key or "guest", [])


def _resolve_speaker_voice(sel):
    """A speaker's saved selection -> (voice_id|None, style|None).
    'char:<id>' -> that character's voice + style (NOT bio); a plain voice id -> (id, None)."""
    if not sel:
        return None, None
    if sel.startswith("char:"):
        cid = sel[5:]
        ch = next((c for c in CHARACTERS if c["id"] == cid), None)
        if ch:
            return (ch.get("voice") or None), ((ch.get("style") or "").strip() or None)
        return None, None
    return sel, None

import speaker_id  # ECAPA speaker identification (closed-set personalization)

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


def chat(user_text: str, personality_id: str = DEFAULT_PERSONALITY,
         speaker_name: str = None, history_key: str = "guest",
         style: str = None) -> str:
    persona = PERSONALITY_BY_ID.get(personality_id, PERSONALITY_BY_ID[DEFAULT_PERSONALITY])
    system = persona["system"] + VOICE_STYLE
    # Personalization: if speaker ID recognized who is talking, tell the agent so it
    # can address them by name. Identification only — never gate behavior on this.
    if speaker_name:
        system += (f"\n\nYou are speaking with {speaker_name}. Address them by their name, "
                   f"{speaker_name}, naturally when it fits — do not overuse it.")
    else:
        # Unknown speaker: don't let the model invent/guess a name (it was
        # hallucinating "Eric" for unrecognized speakers).
        system += ("\n\nYou do not know who you are speaking with. Do NOT address them by "
                   "any name and do NOT guess a name.")
    # If this speaker is mapped to a CHARACTER, fold in that character's STYLE only
    # (cadence/attitude) — NOT its bio/facts — so replies take on the character's voice.
    if style:
        system += f"\n\nSpeak in this style: {style}"
    # Read/append only THIS speaker's thread so users don't cross-contaminate, and
    # cap it to the most recent turns so context stays small.
    with history_lock:
        hist = _history_for(history_key)
        hist.append({"role": "user", "content": user_text})
        recent = hist[-(CHAT_HISTORY_TURNS * 2):]
        messages = [{"role": "system", "content": system}] + list(recent)
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
    reply = _strip_stage_directions(data["message"]["content"].strip())
    with history_lock:
        _history_for(history_key).append({"role": "assistant", "content": reply})
    return reply


_STAGE_DIRECTION_RE = re.compile(r"\s*\*[^*\n]{1,80}\*\s*")


def _strip_stage_directions(text: str) -> str:
    """Remove asterisk-delimited stage directions (*giggles maniacally*, *ahem*)
    from a reply that will be SPOKEN by TTS. The character prompts already forbid
    them, but models imitate their own history harder than they obey the system
    prompt (relapse observed 2026-09-22 minutes after the prompt fix), so this is
    the deterministic guarantee. Stripping BEFORE the history append matters as
    much as before synthesis — clean history stops re-teaching the habit."""
    cleaned = _STAGE_DIRECTION_RE.sub(" ", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned if cleaned else text


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
# CONSUMER ASSEMBLY (character-builder prototype)
#
# A library CHARACTER stores identity: bio (who they are + facts) + style (tone +
# speech patterns) + a default reword strength + voice + tuning. A CONSUMER picks a
# modality and assembles the system prompt. The modality INTRO is owned by the
# consumer (this code), NOT stored in the character:
#   - reword = a TRANSFORM. Restyle an already-written line. Needs style heavily,
#     facts rarely; carries the strength block. (bio optional - test toggle.)
#   - chat   = GENERATION. The model IS the character, answering as them. Needs
#     bio + style fully; the strength block is meaningless (no source line) -> omitted.
# Identity (bio+style) is shared; the intro + strength are what differ.
# ---------------------------------------------------------------------------
REWORD_INTRO = (
    "You are re-voicing a single line of text as {name}. Rewrite the line in this "
    "character's voice - keep its meaning and facts; change only the wording, cadence "
    "and attitude. Do NOT answer it, do NOT add new content - only restyle the given line."
)
CHAT_INTRO = (
    "You are {name}. You are having a spoken back-and-forth with the user. Stay fully "
    "in character, speak as {name}, and draw on what you know about yourself when relevant."
)

# Chat keeps the bio safe from front-truncation by giving Ollama real headroom and
# capping how many prior turns ride along. 8192 ctx is trivial for a 3B on the 4080.
CHAT_NUM_CTX = 8192
CHAT_HISTORY_TURNS = 12   # most recent user/assistant messages kept


def _ollama_chat(messages, num_predict=200, num_ctx=None, temperature=0.7):
    """One stateless Ollama chat call. Returns the assistant text (stripped)."""
    options = {"temperature": temperature, "num_predict": num_predict}
    if num_ctx:
        options["num_ctx"] = num_ctx
    payload = json.dumps({
        "model": OLLAMA_MODEL, "messages": messages,
        "stream": False, "options": options,
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["message"]["content"].strip()


def assemble_reword_system(name, bio, style, strength, include_bio=False):
    """Build the reword (transform) system prompt from explicit character fields."""
    intro = REWORD_INTRO.format(name=(name or "the character").strip())
    parts = [intro]
    if include_bio and (bio or "").strip():
        parts.append(bio.strip())
    if (style or "").strip():
        parts.append(style.strip())
    parts.append(REWORD_STRENGTH.get(strength, REWORD_STRENGTH["full"]))
    return " ".join(parts) + VOICE_STYLE


def assemble_chat_system(name, bio, style):
    """Build the chat (generation) system prompt - bio + style, no strength block."""
    intro = CHAT_INTRO.format(name=(name or "the character").strip())
    parts = [intro]
    if (bio or "").strip():
        parts.append(bio.strip())
    if (style or "").strip():
        parts.append(style.strip())
    return " ".join(parts) + VOICE_STYLE


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


# ---------------------------------------------------------------------------
# Characters: named bundles that pair a VOICE with a PERSONALITY (+ default
# reword strength). Picking "Singaporean Uncle" sets voice=UncleLo and
# personality=sg_uncle in one move. Config-driven via characters.json so new
# characters are a pure JSON edit - no server changes.
#
# A character may also carry its OWN inline `system` prompt. If present we
# register (or override) the referenced personality with it at load time, so a
# brand-new character can ship its prompt in characters.json without touching
# the hardcoded PERSONALITIES list. `personality` then defaults to the char id.
# ---------------------------------------------------------------------------
CHARACTERS_PATH = os.path.join(HERE, "characters.json")
CHARACTERS = []  # [{id,label,voice,personality,strength}]


def load_characters():
    """Load character (voice+personality) bundles from characters.json."""
    global CHARACTERS
    if not os.path.isfile(CHARACTERS_PATH):
        CHARACTERS = []
        return
    try:
        data = json.load(open(CHARACTERS_PATH, encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"[characters] failed to load {CHARACTERS_PATH}: {e}", flush=True)
        CHARACTERS = []
        return
    out = []
    for c in (data.get("characters") or []):
        cid = c.get("id")
        if not cid:
            continue
        pid = c.get("personality") or cid
        strength = c.get("strength") if c.get("strength") in ("none", "light", "full") else "full"
        # v2 split fields; fall back to the legacy single `system` blob (-> treated as bio).
        bio = c.get("bio")
        style = c.get("style") or ""
        legacy_system = c.get("system")
        if bio is None and isinstance(legacy_system, str):
            bio = legacy_system
        bio = bio or ""
        # The personality `system` consumed by reword()/chat() = bio + style joined, so
        # the existing /api/prototype/speak path keeps working unchanged.
        combined = (bio + ("\n\n" + style if style.strip() else "")).strip()
        if combined:
            if pid in PERSONALITY_BY_ID:
                PERSONALITY_BY_ID[pid]["system"] = combined
            else:
                p = {"id": pid, "label": c.get("label") or c.get("name") or pid,
                     "group": "Characters", "system": combined}
                PERSONALITIES.append(p)
                PERSONALITY_BY_ID[pid] = p
        tuning = c.get("tuning") or {}
        out.append({
            "id": cid,
            "title": c.get("title") or c.get("label") or cid,
            "label": c.get("label") or c.get("name") or c.get("title") or cid,
            "name": c.get("name") or c.get("label") or cid,
            "voice": c.get("voice", DEFAULT_VOICE),
            "personality": pid,
            "strength": strength,
            "bio": bio,
            "style": style,
            "split_synth": bool(c.get("split_synth", False)),
            "tuning": {
                "speed": float(tuning.get("speed", 1.0)),
                "guidance": float(tuning.get("guidance", 2.0)),
                "temperature": float(tuning.get("temperature", 0.0)),
                "steps": int(tuning.get("steps", 32)),
            },
            "schema_version": c.get("schema_version", 2 if (c.get("bio") or c.get("style")) else 1),
        })
    CHARACTERS = out
    print(f"[characters] loaded {len(out)} character(s) from {CHARACTERS_PATH}", flush=True)


# Budgets for character text fields (token estimates; client mirrors these).
BIO_BUDGET_TOKENS = 600
STYLE_BUDGET_TOKENS = 400


def _est_tokens(s: str) -> int:
    """Cheap, conservative token estimate (chars/4) - matches the client-side gate."""
    return (len(s or "") + 3) // 4


def save_character(c: dict) -> dict:
    """Append-or-replace a character (by id) in characters.json and hot-reload.

    Mirrors save_prompts(): write file, then reload in-memory so the new/updated
    character is live immediately (no restart).
    """
    title = (c.get("title") or "").strip()
    name = (c.get("name") or "").strip()
    if not title:
        raise ValueError("title is required")
    cid = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") or "character"
    bio = (c.get("bio") or "").strip()
    style = (c.get("style") or "").strip()
    # Defense-in-depth budget check (the UI also gates Save).
    if _est_tokens(bio) > BIO_BUDGET_TOKENS:
        raise ValueError(f"bio over budget ({_est_tokens(bio)}/{BIO_BUDGET_TOKENS} tokens)")
    if _est_tokens(style) > STYLE_BUDGET_TOKENS:
        raise ValueError(f"style over budget ({_est_tokens(style)}/{STYLE_BUDGET_TOKENS} tokens)")
    strength = c.get("strength") if c.get("strength") in ("none", "light", "full") else "full"
    tn = c.get("tuning") or {}
    entry = {
        "schema_version": 2,
        "id": cid,
        "title": title,
        "name": name or title,
        "label": name or title,
        "voice": (c.get("voice") or DEFAULT_VOICE).strip(),
        "bio": bio,
        "style": style,
        "strength": strength,
        "split_synth": bool(c.get("split_synth", False)),
        "tuning": {
            "speed": float(tn.get("speed", 1.0)),
            "guidance": float(tn.get("guidance", 2.0)),
            "temperature": float(tn.get("temperature", 0.0)),
            "steps": int(tn.get("steps", 32)),
        },
    }
    # Load existing file (preserve _meta + other characters), upsert by id.
    doc = {"characters": []}
    if os.path.isfile(CHARACTERS_PATH):
        try:
            doc = json.load(open(CHARACTERS_PATH, encoding="utf-8")) or doc
        except Exception:  # noqa: BLE001
            doc = {"characters": []}
    chars = doc.get("characters") or []
    chars = [x for x in chars if x.get("id") != cid]
    chars.append(entry)
    doc["characters"] = chars
    with open(CHARACTERS_PATH, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    load_characters()   # hot-reload
    return entry


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
    if tts_model is None:
        raise RuntimeError(f"TTS unavailable (OmniVoice not loaded): {_tts_load_error}")
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
    ensure_tts()
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
    """List saved custom voices from disk and build their clone prompts.

    Listing is DECOUPLED from prompt-building: a voice is always registered (so it shows
    in /api/voices) even before OmniVoice is loaded. The GPU clone prompt is built here
    only when the model is up; with TTS deferred it's built lazily by ensure_tts() on the
    first synth. (Before this, a missing model made build_clone_prompt raise and the voice
    silently vanished from the list — only presets showed.)"""
    listed = []
    for fn in sorted(os.listdir(CUSTOM_DIR)):
        if not fn.endswith(".json"):
            continue
        try:
            meta = json.load(open(os.path.join(CUSTOM_DIR, fn), encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"[init] failed to read custom voice {fn}: {e}", flush=True)
            continue
        vid = meta.get("id")
        if not vid:
            continue
        label = meta.get("label") or vid
        # Always list the voice so the UI shows it immediately.
        listed.append({"id": vid, "label": label, "tags": "Custom", "custom": True})
        if tts_model is None:
            continue  # prompt built lazily once the model loads (ensure_tts)
        try:
            wav, _ = sf.read(os.path.join(CUSTOM_DIR, vid + ".wav"))
            custom_prompts[vid] = build_clone_prompt(np.asarray(wav, dtype=np.float32), meta["ref_text"])
            print(f"[init] loaded custom voice '{label}' ({vid})", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[init] prompt build deferred/failed for {vid}: {e}", flush=True)
    custom_voices[:] = listed  # replace in one assignment (no empty-list window)


def _post_load_setup():
    """Run once right after the model loads: rebuild voice-clone + ref-clip prompts (skipped
    while the model was down) and warm BOTH generate paths. Reused by ensure_tts() (lazy load)
    and the background pre-warm thread so both code paths get identical, fully-warm voices.
    Must be called while holding _tts_load_lock."""
    try:
        custom_voices.clear()
        custom_prompts.clear()
        _refclip_cache.clear()
        load_custom_voices()
        print("[tts] model loaded + voice prompts rebuilt", flush=True)
        # WARM-UP: OmniVoice's FIRST generate() after load mis-applies conditioning (the
        # first render comes out wrong/hollow). It has TWO generate paths — clone
        # (voice_clone_prompt=) and instruct (instruct=) — and warming one does NOT warm the
        # other. Warm BOTH so whichever path the first real render takes is already warm.
        # Record which paths ran so a render log can show whether warmup covered it.
        global _renders_since_load, _warmup_paths
        _renders_since_load = 0
        warmed = []
        try:
            if custom_prompts:
                _warm = next(iter(custom_prompts.values()))
                with gpu_guard("tts.warmup.clone"):
                    tts_model.generate(text="warm up.", language="English",
                                       voice_clone_prompt=_warm, num_step=8)
                warmed.append("clone")
        except Exception as we:  # noqa: BLE001
            print(f"[tts] warmup(clone) skipped: {we}", flush=True)
        try:
            _instruct = VOICE_BY_ID[DEFAULT_VOICE]["instruct"]
            with gpu_guard("tts.warmup.instruct"):
                tts_model.generate(text="warm up.", language="English",
                                   instruct=_instruct, num_step=8)
            warmed.append("instruct")
        except Exception as we:  # noqa: BLE001
            print(f"[tts] warmup(instruct) skipped: {we}", flush=True)
        _warmup_paths = "+".join(warmed) if warmed else "none"
        print(f"[tts] warmup done (paths: {_warmup_paths})", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[tts] voice prompt rebuild after load failed: {e}", flush=True)


def ensure_tts():
    """Guarantee OmniVoice is loaded before a GPU TTS op; raise (caught -> 503) if it
    can't load. Lazily recovers after a non-fatal startup OOM: once VRAM frees, the next
    synth loads the model and rebuilds the voice-clone prompts that were skipped while
    it was down. Serialized by a reentrant lock so concurrent synths load it once."""
    if tts_model is not None:
        return
    with _tts_load_lock:
        if tts_model is not None:
            return
        if not _load_tts_model():
            raise RuntimeError(f"TTS unavailable (OmniVoice not loaded): {_tts_load_error}")
        _post_load_setup()


def prewarm_tts_bg():
    """Background pre-warm: load + warm OmniVoice in a daemon thread so the server reports
    healthy IMMEDIATELY (GPU-free — the avatar/phonetics pages need no GPU) while the model
    loads off to the side. By the time someone prompts, the model is resident and warmed, so
    the FIRST render is fast and on the correct voice — without the ~30s startup block that
    eager loading imposes. Serialized via _tts_load_lock with ensure_tts(), so a prompt that
    lands mid-load just waits on the same in-flight load (no double load)."""
    def _run():
        try:
            with _tts_load_lock:
                if tts_model is not None:
                    return
                print("[init] pre-warm: loading OmniVoice in background...", flush=True)
                if _load_tts_model():
                    _post_load_setup()
                    print("[init] pre-warm complete — first render will be fast", flush=True)
                else:
                    print(f"[init] pre-warm: load failed ({_tts_load_error}); "
                          "will retry lazily on first synth", flush=True)
        except Exception as e:  # noqa: BLE001 — pre-warm must never take the server down
            print(f"[init] pre-warm thread error: {e}", flush=True)
    threading.Thread(target=_run, name="tts-prewarm", daemon=True).start()


def _ensure_custom_prompt(voice_id: str):
    """Build a custom voice's clone prompt on demand if it wasn't already cached. Guarantees
    the FIRST synth after a (re)load uses the real cloned voice instead of falling back to the
    default instruct voice — the 'first render is default, second is the real voice' bug."""
    with custom_lock:
        if voice_id in custom_prompts:
            return custom_prompts[voice_id]
    mp = os.path.join(CUSTOM_DIR, voice_id + ".json")
    wp = os.path.join(CUSTOM_DIR, voice_id + ".wav")
    if not (os.path.isfile(mp) and os.path.isfile(wp)):
        return None
    try:
        meta = json.load(open(mp, encoding="utf-8"))
        wav, _ = sf.read(wp)
        prompt = build_clone_prompt(np.asarray(wav, dtype=np.float32), meta["ref_text"])
        with custom_lock:
            custom_prompts[voice_id] = prompt
        print(f"[tts] on-demand built clone prompt for {voice_id}", flush=True)
        return prompt
    except Exception as e:  # noqa: BLE001
        print(f"[tts] on-demand clone build failed for {voice_id}: {e}", flush=True)
        return None


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
    ensure_tts()
    with custom_lock:
        clone = custom_prompts.get(voice_id)
        is_known_custom = any(v["id"] == voice_id for v in custom_voices)
    # If it's a registered custom voice whose prompt isn't cached yet, build it NOW so the
    # first render uses the real cloned voice instead of falling back to the default.
    if clone is None and is_known_custom:
        clone = _ensure_custom_prompt(voice_id)
    common = dict(num_step=num_step, speed=speed, guidance_scale=guidance_scale,
                  class_temperature=class_temperature)
    path = "clone" if clone is not None else "instruct"
    global _renders_since_load
    _renders_since_load += 1
    cold = _renders_since_load == 1     # first real generate since this model instance loaded
    # One self-explaining line per render: which voice actually rendered, which generate
    # PATH it took, whether it was the cold first-generate-since-load, and whether the
    # warmup covered that path. When a render sounds wrong, this says why without guessing:
    # e.g. path=instruct cold=True while you asked for a custom voice => the voice-guard
    # bug resurfaced; cold=True warmup=clone path=instruct => warmup missed this path.
    warm_ok = path in _warmup_paths.split("+")
    diag = (f"voice={voice_id} requested_clone={is_known_custom} path={path} "
            f"cold={cold} renders_since_load={_renders_since_load} "
            f"warmup_paths={_warmup_paths} warm_covered={warm_ok} "
            f"steps={num_step} speed={speed} guidance={guidance_scale} temp={class_temperature}")
    print(f"[synth] {diag}", flush=True)
    try:
        with open(os.path.join(HERE, "working", "character_speak.log"), "a", encoding="utf-8") as _f:
            _f.write(f"{time.strftime('%H:%M:%S')} [synth] {diag}\n")
    except Exception:  # noqa: BLE001 — logging must never break a render
        pass
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
    ensure_tts()
    common = dict(num_step=num_step, speed=speed, guidance_scale=guidance_scale,
                  class_temperature=class_temperature)
    with gpu_guard("tts.generate.adhoc_prompt"):
        audios = tts_model.generate(
            text=text, language="English", voice_clone_prompt=clone_prompt, **common
        )
    buf = io.BytesIO()
    sf.write(buf, audios[0], TTS_SR, format="WAV")
    return buf.getvalue()


SPLIT_GAP_MS = 350  # silence inserted between chunks for split-synth pauses


def synth_split(text: str, synth_fn, gap_ms: int = SPLIT_GAP_MS) -> bytes:
    """Render `text` in chunks split on ellipses, stitching real silence between them.

    OmniVoice ignores punctuation for pacing (measured: ~0.1s per '...'), so the only
    way to get an audible pause is to break the utterance and insert silence at the
    script level. `synth_fn(chunk) -> WAV bytes` does the per-chunk synthesis (voice or
    clip). Falls back to a single call when there is no split point. NOTE: each chunk is
    synthed as a standalone sentence, so intonation resets at every pause — intended for
    halting/fragmented characters (e.g. Slow Chad), not smooth ones."""
    chunks = [c.strip() for c in re.split(r"\.{3,}|…", text) if c.strip()]
    if len(chunks) <= 1:
        return synth_fn(text)
    voiced, sr = [], TTS_SR
    for ch in chunks:
        # A single short fragment can occasionally make OmniVoice emit an empty array
        # (raises on an internal max()). Don't let one bad chunk kill the whole line —
        # skip it and keep the others.
        try:
            data, sr = sf.read(io.BytesIO(synth_fn(ch)), dtype="float32")
        except Exception as e:  # noqa: BLE001
            print(f"[split-synth] chunk failed, skipping: {ch!r} ({e})", flush=True)
            continue
        if data.size:
            voiced.append(data)
    if not voiced:
        return synth_fn(text)  # everything failed → fall back to one-shot
    gap = np.zeros(int(sr * gap_ms / 1000), dtype=voiced[0].dtype)
    stitched = []
    for i, data in enumerate(voiced):
        stitched.append(data)
        if i < len(voiced) - 1:
            stitched.append(gap)
    buf = io.BytesIO()
    sf.write(buf, np.concatenate(stitched), sr, format="WAV")
    return buf.getvalue()


def find_character(key: str):
    """Resolve a caller-supplied character name to a CHARACTERS entry.
    Matches id/title/label/name case-insensitively; None when unknown."""
    k = (key or "").strip().lower()
    if not k:
        return None
    for c in CHARACTERS:
        if k in (c["id"].lower(), c["title"].lower(), c["label"].lower(), c["name"].lower()):
            return c
    return None


def postprocess_wav(wav_bytes: bytes, out_sr, pad_ms: int) -> bytes:
    """Resample / pad a rendered WAV for delivery targets that need a specific wire
    format (the speak relay's device leg wants 48 kHz + A2DP-wake padding so the thin
    Pi endpoint never resamples — the spare compute lives on this box). Decode to
    float, polyphase-resample, pad head+tail with silence, re-encode 16-bit PCM WAV.
    Returns the input untouched when there is nothing to do."""
    if not out_sr and not pad_ms:
        return wav_bytes
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    target = int(out_sr or sr)
    if target != sr:
        from math import gcd  # lazy: only this path needs them
        from scipy.signal import resample_poly
        g = gcd(target, sr)
        data = resample_poly(data, target // g, sr // g).astype(np.float32)
    if pad_ms:
        pad = np.zeros(int(target * pad_ms / 1000), dtype=np.float32)
        data = np.concatenate([pad, data, pad])
    buf = io.BytesIO()
    sf.write(buf, data, target, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
load_custom_voices()  # rebuild saved custom voices at startup

# Speaker identification: load ECAPA + the enrolled voiceprints. Guarded — if it
# fails (e.g. model download), v2v still serves; identify() just returns 'unknown'.
try:
    speaker_id.load()
except Exception as e:  # noqa: BLE001
    print(f"[init] speaker-id load failed (continuing without ID): {e}", flush=True)
load_characters()     # voice+personality bundles (after voices/personalities exist)

# TTS load strategy (now that the "hang" is fixed and the deferred fallback exists):
#   TTS_PREWARM=1 (default) -> background pre-warm: server is healthy immediately AND the
#                              first render is fast (model loads + warms off-thread).
#   TTS_PREWARM=0           -> pure lazy: server starts GPU-free; first render pays the load.
#   EAGER_TTS=1 (legacy)    -> blocking load at import (above); ~30s startup, no prewarm.
# Override per-launch via env in the harbor start command.
if os.environ.get("TTS_PREWARM", "1") == "1" and os.environ.get("EAGER_TTS") != "1":
    prewarm_tts_bg()


@app.get("/api/voices")
def get_voices():
    with custom_lock:
        cv = list(custom_voices)
    return {"voices": VOICES + cv, "default": DEFAULT_VOICE}


@app.post("/api/voices/custom")
def add_custom_voice(audio: UploadFile = File(...), name: str = Form(...)):
    try:
        ensure_tts()  # cloning needs the GPU encoder
    except RuntimeError as e:
        return JSONResponse({"error": "tts_unavailable", "detail": str(e)}, status_code=503)
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
    try:
        ensure_tts()  # registering a voice needs the GPU encoder
    except RuntimeError as e:
        return JSONResponse({"error": "tts_unavailable", "detail": str(e)}, status_code=503)
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


@app.get("/api/characters")
def get_characters():
    """Character bundles: each pairs a voice id + personality id (+ default strength)."""
    return {"characters": CHARACTERS, "default": (CHARACTERS[0]["id"] if CHARACTERS else None),
            "budgets": {"bio": BIO_BUDGET_TOKENS, "style": STYLE_BUDGET_TOKENS}}


@app.post("/api/characters")
def post_character(c: dict = Body(...)):
    """Save a character (title/name/voice/bio/style/strength/tuning) to characters.json."""
    try:
        entry = save_character(c)
    except ValueError as e:
        return JSONResponse({"error": "invalid", "detail": str(e)}, status_code=422)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": "save_failed", "detail": str(e)}, status_code=500)
    return {"ok": True, "id": entry["id"], "saved_to": CHARACTERS_PATH}


@app.post("/api/prototype/character_speak")
def character_speak(
    mode: str = Form("reword"),              # reword | chat
    name: str = Form(""),
    bio: str = Form(""),
    style: str = Form(""),
    strength: str = Form("full"),            # reword only
    include_bio: str = Form("false"),        # reword only: include bio in the transform
    voice: str = Form(DEFAULT_VOICE),
    text: str = Form(...),                   # the line (reword) or the user message (chat)
    history: str = Form(""),                 # chat only: JSON [{role,content},...]
    speed: float = Form(1.0),
    guidance: float = Form(2.0),
    temperature: float = Form(0.0),
    steps: int = Form(32),
    ref_clip: str = Form(""),                # optional built-in working/ clip
    split_synth: str = Form("false"),        # split on '...' and stitch silence for real pauses
):
    """Consumer test-bench: assemble a system prompt from raw character fields per
    modality, run the LLM, then speak the result. Returns audio + the assembled
    system prompt (header) so the UI can show exactly what was sent.
    """
    src = (text or "").strip()
    if not src:
        return JSONResponse({"error": "empty_text"}, status_code=400)
    mode = mode if mode in ("reword", "chat") else "reword"
    if strength not in ("none", "light", "full"):
        strength = "full"

    # 1) TRANSFORM or GENERATE.
    try:
        if mode == "chat":
            system = assemble_chat_system(name, bio, style)
            msgs = [{"role": "system", "content": system}]
            try:
                prior = json.loads(history) if history else []
            except Exception:  # noqa: BLE001
                prior = []
            if isinstance(prior, list):
                msgs += [m for m in prior[-CHAT_HISTORY_TURNS:]
                         if isinstance(m, dict) and m.get("role") and m.get("content")]
            msgs.append({"role": "user", "content": src})
            spoken = _ollama_chat(msgs, num_predict=300, num_ctx=CHAT_NUM_CTX)
        else:
            inc = str(include_bio).lower() in ("1", "true", "yes", "on")
            if strength == "none":
                system = assemble_reword_system(name, bio, style, strength, inc)
                spoken = src  # verbatim, no LLM call
            else:
                system = assemble_reword_system(name, bio, style, strength, inc)
                spoken = _ollama_chat(
                    [{"role": "system", "content": system}, {"role": "user", "content": src}],
                    num_predict=200, num_ctx=CHAT_NUM_CTX)
    except Exception as e:  # noqa: BLE001 — ollama down etc.
        return JSONResponse({"error": "llm_failed", "detail": str(e)}, status_code=503)

    sp, gd, tp, st = clamp_tuning(speed, guidance, temperature, steps)
    do_split = str(split_synth).lower() in ("1", "true", "yes", "on")
    # Strip a comma before Singlish/sentence particles — OmniVoice pauses + mis-intones on
    # the comma ("cannot, lah" -> wrong). Deterministic; complements the style instruction.
    spoken = re.sub(r",\s+(lah|leh|lor|mah|meh|sia|hor|liao|la)\b", r" \1", spoken, flags=re.IGNORECASE)
    # Strip stage-direction "actions" the small model emits despite being told not to
    # (e.g. *stuttering*, *beatboxing noise*, (sighs)) — TTS reads them literally.
    spoken = re.sub(r"\*[^*\n]*\*", " ", spoken)            # *action*
    spoken = re.sub(r"\((?:[^()\n]{0,40})\)", " ", spoken)  # (short stage direction)
    spoken = re.sub(r"\s{2,}", " ", spoken).strip()
    # Log what the LLM actually produced so pause/style behavior is inspectable after
    # the fact (header x-spoken-text isn't logged; rendered clips aren't saved). Timestamped
    # so it can be correlated with an external hardware/VRAM log around a crash.
    ndots = spoken.count("...") + spoken.count(chr(0x2026))
    ts = time.strftime("%H:%M:%S")
    rec = f"{ts} mode={mode} split={int(do_split)} voice={voice} ellipses={ndots} spoken={spoken!r}"
    print(f"[character_speak] {rec}", flush=True)
    try:
        with open(os.path.join(HERE, "working", "character_speak.log"), "a", encoding="utf-8") as _f:
            _f.write(rec + "\n")
    except Exception:  # noqa: BLE001 — logging must never break a render
        pass

    # 2) VOICE: built-in clip, else a registered preset/custom id.
    try:
        if ref_clip:
            prompt = ref_clip_prompt(ref_clip)
            if prompt is None:
                return JSONResponse({"error": "unknown_ref_clip", "clip": ref_clip}, status_code=404)
            fn = lambda t: synth_with_prompt(t, prompt, num_step=st, speed=sp,  # noqa: E731
                                             guidance_scale=gd, class_temperature=tp)
            audio = synth_split(spoken, fn) if do_split else fn(spoken)
            used_voice = f"clip:{os.path.basename(ref_clip)}"
        else:
            # Validate against the registered-voice METADATA list (custom_voices), NOT the
            # lazy custom_prompts cache. On a cold first render the model isn't loaded yet so
            # custom_prompts is empty — checking it would mis-classify a real cust_ voice as
            # unknown and silently fall back to DEFAULT_VOICE (rendering the default voice at
            # the character's clone-tuned params -> "default + corrupted" first render).
            # synth() lazily builds the clone prompt for a known custom id via ensure_tts().
            with custom_lock:
                is_custom = any(v["id"] == voice for v in custom_voices)
            if voice not in VOICE_BY_ID and not is_custom:
                voice = DEFAULT_VOICE
            fn = lambda t: synth(t, voice, num_step=st, speed=sp,  # noqa: E731
                                 guidance_scale=gd, class_temperature=tp)
            audio = synth_split(spoken, fn) if do_split else fn(spoken)
            used_voice = voice
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": "synth_failed", "detail": str(e)}, status_code=500)

    return Response(content=audio, media_type="audio/wav", headers={
        "Cache-Control": "no-store",
        "x-spoken-text": urllib.parse.quote(spoken),
        "x-reply-text": urllib.parse.quote(spoken if mode == "chat" else ""),
        "x-system-prompt": urllib.parse.quote(system),
        "x-mode": mode,
        "x-voice": used_voice,
        "x-split": "1" if do_split else "0",
    })


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
            # Validate against the registered-voice METADATA list (custom_voices), NOT the
            # lazy custom_prompts cache. On a cold first render the model isn't loaded yet so
            # custom_prompts is empty — checking it would mis-classify a real cust_ voice as
            # unknown and silently fall back to DEFAULT_VOICE (rendering the default voice at
            # the character's clone-tuned params -> "default + corrupted" first render).
            # synth() lazily builds the clone prompt for a known custom id via ensure_tts().
            with custom_lock:
                is_custom = any(v["id"] == voice for v in custom_voices)
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
    Body: {text, voice?, character?, paraphrase?, speed?, guidance?, temperature?,
    steps?, sr?, pad_ms?}.
    - voice: an OmniVoice voice id (preset like 'f_us' or a saved custom voice);
      unknown/absent falls back to DEFAULT_VOICE so a bad voice name never fails
      the speak path.
    - character: a saved character id/name (e.g. 'jerma') -> that character's voice
      + saved tuning (explicit voice/tuning fields in the payload still override).
      Unknown character is a 404 with the known list, NOT a silent default-voice
      fallback: device-cue callers need a wrong name to fail loudly, not to ship a
      cue in the wrong voice.
    - paraphrase: reserved for a future in-character reword() pass. Accepted today
      so callers can already send it; currently ALWAYS renders verbatim and the
      response reports x-paraphrased: false.
    - sr: output sample rate in Hz (8000-48000). E.g. 48000 for the bluealsa/A2DP
      device leg; resampled server-side from the native 24 kHz render.
    - pad_ms: silence (0-2000 ms) prepended AND appended — A2DP sinks wake slowly
      and clip the first word without lead-in.
    The x-synth-device header lets the relay log which device rendered it.
    """
    text = (payload.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "empty_text"}, status_code=400)
    char = None
    char_key = (payload.get("character") or "").strip()
    if char_key:
        char = find_character(char_key)
        if char is None:
            return JSONResponse({"error": "unknown_character", "character": char_key,
                                 "known": [c["id"] for c in CHARACTERS]},
                                status_code=404)
    voice = payload.get("voice") or (char["voice"] if char else None) or DEFAULT_VOICE
    with custom_lock:
        is_custom = voice in custom_prompts or any(v["id"] == voice for v in custom_voices)
    if voice not in VOICE_BY_ID and not is_custom:
        voice = DEFAULT_VOICE  # tolerate unknown voice names from the relay
    paraphrase = bool(payload.get("paraphrase", False))
    if paraphrase:
        # Contract placeholder: the reword() pass isn't wired into this endpoint yet.
        print("[synthesize] paraphrase requested but not enabled — rendering verbatim", flush=True)
    tuning = char["tuning"] if char else {}
    try:
        # Optional generation tuning, clamped to safe ranges (same as /api/preview).
        sp, gd, tp, st = clamp_tuning(
            float(payload.get("speed", tuning.get("speed", 1.0))),
            float(payload.get("guidance", tuning.get("guidance", 2.0))),
            float(payload.get("temperature", tuning.get("temperature", 0.0))),
            int(payload.get("steps", tuning.get("steps", 32))),
        )
        out_sr = payload.get("sr")
        if out_sr is not None:
            out_sr = max(8000, min(48000, int(out_sr)))
        pad_ms = max(0, min(2000, int(payload.get("pad_ms", 0) or 0)))
    except (TypeError, ValueError) as e:
        return JSONResponse({"error": "invalid_param", "detail": str(e)}, status_code=400)
    wav = synth(text[:2000], voice, num_step=st, speed=sp,
                guidance_scale=gd, class_temperature=tp)
    wav = postprocess_wav(wav, out_sr, pad_ms)
    headers = {"Cache-Control": "no-store",
               "x-synth-device": "cuda:omnivoice",
               "x-voice": voice,
               "x-sample-rate": str(out_sr or TTS_SR),
               "x-paraphrased": "false"}
    if char is not None:
        headers["x-character"] = char["id"]
    return Response(content=wav, media_type="audio/wav", headers=headers)


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
    out = {"ok": True, "loaded": tts_model is not None, "service": "voice-to-voice",
           "engine": "omnivoice", "device": "cuda:omnivoice",
           "tts_sr": TTS_SR, "voices": voice_ids, "gpu": gpu}
    if tts_model is None:
        out["tts_error"] = _tts_load_error  # TTS down but server up (avatar/STT/chat still work)
    return out


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
        "turns": sum(len(h) for h in histories.values()) // 2,
    }


def _stt_log(msg):
    line = f"{time.strftime('%H:%M:%S')} [stt] {msg}"
    print(line, flush=True)
    try:
        with open(os.path.join(HERE, "working", "character_speak.log"), "a", encoding="utf-8") as _f:
            _f.write(line + "\n")
    except Exception:  # noqa: BLE001
        pass


@app.post("/api/stt")
def stt(audio: UploadFile = File(...)):
    """Transcribe one audio segment to text (no chat, no history). For Compose mode.

    Heavily logged: the 'received' line is written BEFORE transcribe and the 'done' line
    AFTER, both flushed. If a crash happens during STT we'll see 'received' with no 'done',
    isolating the voice-input path as the trigger (vs TTS)."""
    raw = audio.file.read()
    if not raw:
        return JSONResponse({"error": "empty audio"}, status_code=400)
    _stt_log(f"received {len(raw)} bytes -> transcribing...")
    t0 = time.time()
    try:
        text = transcribe(raw)
    except Exception as e:  # noqa: BLE001
        _stt_log(f"FAILED after {time.time()-t0:.2f}s: {e!r}")
        return JSONResponse({"error": "stt_failed", "detail": str(e)}, status_code=500)
    _stt_log(f"done in {time.time()-t0:.2f}s -> {text!r}")
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
        histories.clear()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Speaker identification — enrollment CRUD + test (closed-set personalization).
# Voiceprints live in speakers/ (gitignored). identify() also runs inside
# /api/converse on every device utterance.
# ---------------------------------------------------------------------------
@app.get("/api/speakers")
def speakers_list():
    return JSONResponse({"ready": speaker_id.is_ready(), "speakers": speaker_id.list_speakers()})


@app.post("/api/speakers/enroll")
def speakers_enroll(name: str = Form(...), audio: UploadFile = File(...)):
    entry, err = speaker_id.enroll(name, audio.file.read())
    if err:
        return JSONResponse({"error": err}, status_code=400)
    return JSONResponse({"ok": True, "speaker": entry})


@app.post("/api/speakers/add_clip")
def speakers_add_clip(id: str = Form(...), audio: UploadFile = File(...)):
    entry, err = speaker_id.add_clip(id, audio.file.read())
    if err:
        return JSONResponse({"error": err}, status_code=400)
    return JSONResponse({"ok": True, "speaker": entry})


@app.post("/api/speakers/rename")
def speakers_rename(id: str = Form(...), name: str = Form(...)):
    entry, err = speaker_id.rename(id, name)
    if err:
        return JSONResponse({"error": err}, status_code=400)
    return JSONResponse({"ok": True, "speaker": entry})


@app.post("/api/speakers/delete")
def speakers_delete(id: str = Form(...)):
    ok = speaker_id.delete(id)
    return JSONResponse({"ok": ok})


@app.post("/api/speakers/voice")
def speakers_set_voice(id: str = Form(...), voice: str = Form("")):
    """Associate a TTS voice with a speaker (agent replies in their voice)."""
    entry, err = speaker_id.set_voice(id, voice)
    if err:
        return JSONResponse({"error": err}, status_code=400)
    return JSONResponse({"ok": True, "speaker": entry})


@app.post("/api/speakers/identify")
def speakers_identify(audio: UploadFile = File(...)):
    """Test identification on an uploaded/recorded clip (for the Enroll tab)."""
    name, conf, detail, _sid = speaker_id.identify(audio.file.read())
    return JSONResponse({"speaker": name, "confidence": round(conf, 3), "detail": detail})


# ---------------------------------------------------------------------------
# GUEST ENROLLMENT DIALOG — deterministic state machine (single satellite)
#
# When a device utterance is NOT identified, the agent can run a short, fully
# DETERMINISTIC sub-dialog: ask the guest's name, match it (plain string match,
# NO LLM) to an enrolled speaker, then capture a fresh DEVICE-MIC clip into that
# speaker's voiceprint so future identification improves (closes the cross-mic
# gap that makes enrolled users score 'unknown'). The LLM only ever RESTYLES the
# phrasing — it never decides what happens (see memory deterministic-action-llm-
# styling). Only one dialog is in flight at a time (one physical device, one
# speaker), so a single lock-guarded state object is correct for the prototype.
#
#   States:  normal -> awaiting_name -> awaiting_clip(sid) -> normal
#   The bridge re-opens the mic when a reply carries 'X-Followup-Listen: 1'.
#
# The guest voice/style is the DEFAULT until the user gives a recognized name;
# from the moment of the match onward every line speaks in THAT speaker's voice +
# character STYLE — including the "give me a voice sample" request.
# ---------------------------------------------------------------------------
GUEST_FLOW_TTL = 60.0          # a dialog left hanging this long auto-resets to normal
GUEST_FLOW_MAX_CHAIN = 4       # cap re-arm hops so a confused guest can't loop forever
_guest_flow = {"state": "normal", "sid": None, "name": None, "expires": 0.0, "chain": 0}
_guest_flow_lock = threading.Lock()

# Spoken intents (the deterministic WHAT), each as (FLOURISH, CORE):
#  - FLOURISH: a short greeting that is safe to restyle in-character (style layer).
#  - CORE: the actionable instruction, spoken FAITHFULLY (NOT reworded) — a small
#    LLM will happily riff a "talk for ten seconds" request away if allowed to
#    restyle it (measured: a strong comedic persona dropped the instruction entirely).
#    The CORE is still spoken in the person's VOICE; only its phrasing is fixed.
GF_ASK_NAME = ("",
               "I'm not quite sure who I'm talking to. If you're already enrolled, just "
               "tell me your name and I'll get reacquainted.")
GF_MATCHED = ("Thanks, {name}!",
              "Let me get a fresh sample of your voice so I recognize you next time. Just "
              "keep talking to me for about ten seconds — tell me what you've been up to today.")
GF_CLIP_OK = ("Perfect, {name}!",
              "I've got a clearer sample of your voice now, so I should recognize you much "
              "more reliably from here on.")
GF_CLIP_SHORT = ("",
                 "I only caught a moment there, {name}. Could you keep talking to me for a "
                 "few more seconds so I can get a good sample of your voice?")
GF_NO_MATCH = ("",
               "Hmm, I don't have anyone enrolled by that name yet. No problem — we can keep "
               "chatting, and you can enroll a voice on the setup page any time.")
# Spoken ONLY when add_clip genuinely failed (retries exhausted or a hard error). Never
# claim success on failure — that was the bug that told Eric "Perfect!" over a silent room.
GF_CLIP_FAIL = ("",
                "I couldn't quite get a clean sample that time, {name} — no worries, we can "
                "try again later. I'll keep chatting with you as usual for now.")

# An unknown speaker only gets interrogated if they ASK about their identity (so we
# don't pester every borderline-unknown utterance during ordinary guest chat).
_IDENTITY_CUES = (
    "who am i", "do you know who i am", "do you know me", "you don't know me",
    "you don't recognize", "don't you recognize", "recognize me", "recognize my voice",
    "identify my voice", "identify me", "know who i am", "it's me", "this is me",
    "remember me", "who is this", "who am i talking", "you know me", "guess who",
)

SYS_STYLE_INSTR = (
    "You are LIGHTLY re-voicing a spoken system line to flavor it with a character's tone. "
    "This line is functional: it may ASK the listener to do something (say their name, keep "
    "talking for a number of seconds, etc.). You MUST preserve every instruction, question, "
    "request, name and number from the original — do not drop them, do not replace them with "
    "a joke or a tangent, do not answer the line yourself. Keep about the same length. Change "
    "only word choice and rhythm to match the style. Output ONLY the re-voiced line."
)


def _gf_reset_locked():
    _guest_flow.update(state="normal", sid=None, name=None, expires=0.0, chain=0)


def _gf_get():
    """Current flow state (a copy), auto-expiring a stale dialog. Lock-guarded."""
    with _guest_flow_lock:
        if _guest_flow["state"] != "normal" and time.time() > _guest_flow["expires"]:
            _gf_reset_locked()
        return dict(_guest_flow)


def _gf_set(state, sid=None, name=None, chain=0):
    with _guest_flow_lock:
        _guest_flow.update(state=state, sid=sid, name=name, chain=chain,
                           expires=time.time() + GUEST_FLOW_TTL)


def _gf_reset():
    with _guest_flow_lock:
        _gf_reset_locked()


def _looks_like_identity_query(t):
    tl = (t or "").lower()
    return any(c in tl for c in _IDENTITY_CUES)


def _match_enrolled_name(transcript):
    """Plain whole-word match of a transcript against enrolled speaker names.
    Returns (sid, name) or (None, None). Deterministic — NO LLM. Matches on the
    first token of each enrolled name ('Eric Havir' -> 'eric')."""
    words = set(re.findall(r"[a-z0-9]+", (transcript or "").lower()))
    if not words:
        return None, None
    for sp in speaker_id.list_speakers():
        nm = (sp.get("name") or "").strip().lower()
        if nm and nm.split()[0] in words:
            return sp["id"], sp["name"]
    return None, None


# A restyle is REJECTED (fall back to verbatim) if it leaks meta-instruction words,
# rambles, or drops a required token. Small models given a strong comedic persona will
# regenerate a short greeting into a tangent ("I'm not Oliver, I'm Pat...") or echo the
# transform framing ("I'm ready to revoice that line") — we never speak those.
_STYLE_BAD_TOKENS = ("revoice", "re-voice", "original line", "the line", "as an ai",
                     "instruction", "i'm not ", "i am not ", "sure, here", "here is",
                     "here's the")


def _style_line(text, style, must_keep=None):
    """Restyle a short deterministic line with a speaker's STYLE (cadence only, no bio).
    Returns the text VERBATIM if there's no style, styling fails, or the result fails a
    quality guard — the literal WHAT must always survive, and we never speak garbage."""
    text = (text or "").strip()
    if not text or not style:
        return text
    try:
        # Low temperature + a preservation-first instruction: this is a TRANSFORM, not
        # generation. The character STYLE is tone guidance only.
        system = f"{SYS_STYLE_INSTR}\n\nCharacter tone to apply: {style}{VOICE_STYLE}"
        out = (_ollama_chat(
            [{"role": "system", "content": system}, {"role": "user", "content": text}],
            num_predict=120, temperature=0.3,
        ) or "").strip().strip('"')
        low = out.lower()
        bad = (
            not out
            or len(out) > max(48, len(text) * 3)            # rambled past a flourish
            or any(tok in low for tok in _STYLE_BAD_TOKENS)  # meta-leak / refusal
            or any(k.lower() not in low for k in (must_keep or []))  # dropped the name
        )
        if bad:
            print(f"[guestflow] restyle rejected ('{out[:60]}'); verbatim", flush=True)
            return text
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[guestflow] style failed ({e!r}); using verbatim", flush=True)
        return text


def _gf_voice_response(line, name, voice_id, style, transcript, followup, t0,
                       speaker_label="guest", spk_conf=0.0):
    """Compose (styled flourish + faithful core) -> synth a deterministic dialog line
    and return the HTTP Response (mirrors converse()'s headers, adds X-Followup-Listen
    for the bridge re-arm). Only the flourish is restyled; the core is spoken as-written."""
    flourish, core = line
    nm = name or "there"
    flourish = flourish.format(name=nm) if flourish else ""
    core = core.format(name=nm) if core else ""
    styled = _style_line(flourish, style, must_keep=[nm] if name else None) if flourish else ""
    spoken = (styled + " " + core).strip()
    wav = synth(spoken, voice_id or DEFAULT_VOICE, num_step=16, speed=1.0,
                guidance_scale=2.0, class_temperature=0.0)
    total_ms = int((time.time() - t0) * 1000)
    print(f"[guestflow] heard='{transcript}' -> '{spoken}' "
          f"| as={speaker_label} followup={int(bool(followup))} | {total_ms}ms", flush=True)
    headers = {
        "X-Transcript": urllib.parse.quote(transcript or ""),
        "X-Reply": urllib.parse.quote(spoken),
        "X-Speaker": urllib.parse.quote(speaker_label),
        "X-Speaker-Confidence": f"{spk_conf:.3f}",
        "X-Followup-Listen": "1" if followup else "0",
        "X-Timing": f"total={total_ms}",
        "Access-Control-Expose-Headers":
            "X-Transcript,X-Reply,X-Speaker,X-Speaker-Confidence,X-Followup-Listen,X-Timing",
    }
    return Response(content=wav, media_type="audio/wav", headers=headers)


# Device recognition sessions (Eric, 2026-09-21): once a speaker is recognized on a
# device at full threshold, their NEXT turns on that device within the TTL accept at
# the lower sticky floor (only while they remain the top match — see
# speaker_id.identify) so mid-conversation score dips don't flip the reply voice
# back and forth. In-process state is fine at crawl: one engine instance owns all
# devices; this moves to the NUC state server with the rest of session state.
_device_sessions: dict = {}
DEVICE_SESSION_TTL_S = 1800         # 30 min sliding window; each recognized turn extends it
STICKY_THRESHOLD = 0.30             # session speaker floor; noise ~0.05-0.09, impostor max 0.30


@app.post("/api/converse")
def converse(audio: UploadFile = File(...), voice: str = Form(DEFAULT_VOICE),
             personality: str = Form(DEFAULT_PERSONALITY),
             speed: float = Form(1.0), guidance: float = Form(2.0),
             temperature: float = Form(0.0), steps: int = Form(16),
             device: str = Form("")):
    # `device`: originating endpoint id (va_bridge sends it on every request per the
    # home-automation contract, 2026-09-20). Keys the sticky recognition session
    # below; fuller per-device state moves to the NUC state server later.
    t0 = time.time()
    raw = audio.file.read()
    if not raw:
        return JSONResponse({"error": "empty audio"}, status_code=400)

    flow = _gf_get()

    # --- awaiting_clip: THIS utterance is the enrollment sample (no STT, no chat).
    # Route the raw audio straight into the matched speaker's voiceprint.
    if flow["state"] == "awaiting_clip" and flow["sid"]:
        sid, name = flow["sid"], (flow["name"] or "there")
        vvoice, vstyle = _resolve_speaker_voice(speaker_id.get_voice(sid))
        entry, err = speaker_id.add_clip(sid, raw)
        if err:
            chain = flow["chain"] + 1
            if "too short" in err and chain <= GUEST_FLOW_MAX_CHAIN:
                _gf_set("awaiting_clip", sid=sid, name=name, chain=chain)
                return _gf_voice_response(GF_CLIP_SHORT, name, vvoice, vstyle,
                                          "(enrollment clip)", True, t0, speaker_label=name)
            # Retries exhausted, or a hard error -> bow out HONESTLY. Do NOT fall through to
            # GF_CLIP_OK: nothing was saved, so claiming success would be a lie (and would
            # leave the user thinking their print improved when it didn't).
            print(f"[guestflow] add_clip gave up for {sid}: {err}", flush=True)
            _gf_reset()
            return _gf_voice_response(GF_CLIP_FAIL, name, vvoice, vstyle,
                                      "(enrollment clip)", False, t0, speaker_label=name)
        print(f"[guestflow] add_clip OK for '{name}' ({sid}); "
              f"now {entry['clips']} clips", flush=True)
        _gf_reset()
        return _gf_voice_response(GF_CLIP_OK, name, vvoice, vstyle,
                                  "(enrollment clip)", False, t0, speaker_label=name)

    # Everything else needs a transcript.
    transcript = transcribe(raw)
    t_stt = time.time()
    if not transcript:
        if flow["state"] != "normal":
            _gf_reset()      # a blank reply mid-dialog shouldn't strand the state
        return JSONResponse({"error": "no_speech", "detail": "Nothing transcribed."},
                            status_code=422)

    # --- awaiting_name: match the spoken name to an enrolled speaker (no LLM) ----
    if flow["state"] == "awaiting_name":
        sid, name = _match_enrolled_name(transcript)
        if sid:
            vvoice, vstyle = _resolve_speaker_voice(speaker_id.get_voice(sid))
            _gf_set("awaiting_clip", sid=sid, name=name, chain=flow["chain"] + 1)
            return _gf_voice_response(GF_MATCHED, name, vvoice, vstyle,
                                      transcript, True, t0, speaker_label=name)
        _gf_reset()          # unknown name -> bow out gracefully, don't loop
        return _gf_voice_response(GF_NO_MATCH, None, DEFAULT_VOICE, None, transcript,
                                  False, t0)

    # Speaker identification (closed-set, personalization only). Runs after STT for
    # crawl simplicity — it's ~100-300ms vs seconds of total, and never raises. If
    # recognized, the name is handed to the agent for personalization; 'unknown'
    # changes nothing. Concurrent-with-STT is a later optimization (blueprint A/B).
    sess = _device_sessions.get(device) if device else None
    sticky_sid = (sess["sid"] if sess and (time.time() - sess["t"]) < DEVICE_SESSION_TTL_S
                  else None)
    speaker, spk_conf, spk_detail, spk_id = speaker_id.identify(
        raw, sticky_sid=sticky_sid, sticky_floor=STICKY_THRESHOLD)
    if device and spk_id:
        # Any successful identification (full or sticky) opens/extends the session.
        _device_sessions[device] = {"sid": spk_id, "t": time.time()}
    spk_name = speaker if speaker and speaker != "unknown" else None
    # Per-speaker conversation thread (stable by id; unknown -> shared "guest").
    history_key = spk_id or "guest"
    # This speaker's saved selection -> reply voice (+ character style if a character).
    spk_sel = speaker_id.get_voice(spk_id) if spk_id else ""
    spk_voice, spk_style = _resolve_speaker_voice(spk_sel)
    if spk_voice:
        voice = spk_voice

    # --- normal state, GUEST (unidentified): maybe OFFER the enrollment dialog ---
    # Self-identify by name -> jump straight to clip capture in THEIR voice+style.
    # Ask about identity -> ask their name. Otherwise answer normally (don't
    # interrogate every borderline-unknown utterance — that would be maddening).
    if spk_id is None:
        sid, name = _match_enrolled_name(transcript)
        if sid:
            vvoice, vstyle = _resolve_speaker_voice(speaker_id.get_voice(sid))
            _gf_set("awaiting_clip", sid=sid, name=name, chain=1)
            return _gf_voice_response(GF_MATCHED, name, vvoice, vstyle,
                                      transcript, True, t0, speaker_label=name,
                                      spk_conf=spk_conf)
        if _looks_like_identity_query(transcript):
            _gf_set("awaiting_name", chain=1)
            return _gf_voice_response(GF_ASK_NAME, None, DEFAULT_VOICE, None, transcript,
                                      True, t0, spk_conf=spk_conf)
        # else: fall through to the normal guest reply below.

    # --- DEVELOPMENT MODE (voice channel P3, docs/voice-channel.md §4.2) ---------
    # A device whose table entry says mode=development forwards the RAW transcript to
    # the briefing-table engine instead of the chat LLM, gated on speaker identity,
    # and speaks a short ack. It never falls back to chat on failure (you asked for
    # the table; getting a chatbot instead would be maddening). Local intents above
    # (enrollment, "who am I") never reach here, so they never reach the engine.
    if (_voice_devices().get(device) or {}).get("mode") == "development":
        return _dev_turn(transcript, device, spk_id, spk_name, spk_conf, spk_detail,
                         voice, t0, t_stt)

    # ollama reachability is the known gaming-mode-killswitch failure point.
    try:
        reply = chat(transcript, personality, speaker_name=spk_name,
                     history_key=history_key, style=spk_style)
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
          f"| spk={speaker}({spk_conf:.2f},{spk_detail}) dev={device or '-'} "
          f"| stt={stt_ms}ms chat={chat_ms}ms tts={tts_ms}ms total={total_ms}ms",
          flush=True)

    headers = {
        "X-Transcript": urllib.parse.quote(transcript),
        "X-Reply": urllib.parse.quote(reply),
        "X-Speaker": urllib.parse.quote(speaker or "unknown"),
        # Stable enrollment id (survives renames) — ring-colour hints key on this,
        # per the home-automation LED contract (relay t:7d41c9e2, 2026-09-21).
        "X-Speaker-Sid": spk_id or "",
        "X-Speaker-Confidence": f"{spk_conf:.3f}",
        "X-Followup-Listen": "0",   # normal replies never re-arm the mic (dialog only)
        "X-Timing": f"stt={stt_ms};chat={chat_ms};tts={tts_ms};total={total_ms}",
        "Access-Control-Expose-Headers":
            "X-Transcript,X-Reply,X-Speaker,X-Speaker-Sid,X-Speaker-Confidence,"
            "X-Followup-Listen,X-Timing",
    }
    return Response(content=wav, media_type="audio/wav", headers=headers)



# ---------------------------------------------------------------------------
# VOICE CHANNEL — briefing-table ⇄ voice-to-voice (contract: docs/voice-channel.md in
# briefing-table, relay thread t:330a518a, 2026-09-22).
#
# P1 = the OUTBOUND leg only: an engine pushes SHORT text for a device; we synthesize
# it in the speaker's assigned voice, play it on the device that spoke (via the Dot
# bridge's announce ingress on loopback) and answer TRUTHFULLY only after playback
# finished. Bearer-authenticated and FAIL-CLOSED: no tokens configured = nobody in.
# There is deliberately NO loopback exemption — Tailscale Serve proxies every remote
# caller onto 127.0.0.1, so "loopback" carries no trust here (the engine learned this
# the hard way; see their auth.isLocalHookDelivery note).
#
# We NEVER re-route: a device that is unknown/offline/busy is reported as such and
# the engine's router decides where the message goes next.
# ---------------------------------------------------------------------------
import datetime
import hmac
import secrets
import urllib.error

VOICE_AUTH_PATH = os.path.join(HERE, "voice_auth.json")        # git-ignored; {tokens:{name:token}}
# Sealed-delivery landing zone: one file per caller, `voice_auth.d/<caller>.token`,
# holding the raw token (the exec-bridge's `file` target writes exactly that). The
# sealed transport only runs NUC -> VRPC, so a NUC-side caller mints its own bearer,
# age-seals it to this host and harbor deliver_sealed drops it here. Git-ignored.
VOICE_AUTH_DIR = os.path.join(HERE, "voice_auth.d")
VOICE_DEVICES_PATH = os.path.join(HERE, "voice_devices.json")  # committed device table
VOICE_TEXT_MAX = 500              # chars; over = 413 too_long (we never truncate silently)
VOICE_DELIVER_TIMEOUT_S = 30.0    # our playback cap; the engine waits 45 s on its side
VOICE_REPLY_CACHE_TTL_S = 600.0   # replyId idempotency window
DEV_TURN_MIN_CONF = 0.50          # P3: min speaker confidence to FORWARD a dev turn

_json_hot_cache: dict = {}        # path -> (mtime, data)


def _load_json_hot(path: str, default):
    """Read a JSON file, re-reading only when its mtime changes (hot-reloadable config)."""
    try:
        m = os.path.getmtime(path)
    except OSError:
        return default
    ent = _json_hot_cache.get(path)
    if ent and ent[0] == m:
        return ent[1]
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"[voice] cannot read {os.path.basename(path)}: {e!r}", flush=True)
        return default
    _json_hot_cache[path] = (m, data)
    return data


def _voice_devices() -> dict:
    return ((_load_json_hot(VOICE_DEVICES_PATH, {}) or {}).get("devices") or {})


def _voice_auth(request: Request):
    """Bearer check against voice_auth.json. Returns the token's owner name, else None."""
    hdr = request.headers.get("authorization", "")
    if not hdr.lower().startswith("bearer "):
        return None
    tok = hdr[7:].strip()
    if not tok:
        return None
    tokens = dict((_load_json_hot(VOICE_AUTH_PATH, {}) or {}).get("tokens") or {})
    try:
        for fn in os.listdir(VOICE_AUTH_DIR):
            if fn.endswith(".token"):
                with open(os.path.join(VOICE_AUTH_DIR, fn), "r", encoding="utf-8") as f:
                    t = f.read().strip()
                if t:
                    tokens[fn[:-6]] = t     # a sealed-delivered file wins over the json
    except OSError:
        pass
    for name, t in tokens.items():
        if t and hmac.compare_digest(str(t), tok):
            return name
    return None


def _voice_unauth():
    return JSONResponse({"delivered": False, "error": "unauthorized"}, status_code=401)


_earcon_cache: dict = {}


def _earcon_wav() -> bytes:
    """Two short rising tones (~260 ms) then 150 ms of silence, at the TTS sample rate.
    Meaning: "this is a pushed message, not an answer to what you just said". Generated
    in-process once; no asset file to lose."""
    sr = TTS_SR
    b = _earcon_cache.get(sr)
    if b:
        return b

    def tone(freq, ms, amp=0.18):
        n = int(sr * ms / 1000)
        t = np.arange(n) / sr
        fade_in = np.minimum(1.0, t / 0.010)
        fade_out = np.minimum(1.0, (t[-1] - t) / 0.025) if n else np.ones(0)
        return amp * np.sin(2 * np.pi * freq * t) * fade_in * fade_out

    y = np.concatenate([tone(784, 110), tone(1175, 150),
                        np.zeros(int(sr * 0.15))]).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    b = buf.getvalue()
    _earcon_cache[sr] = b
    return b


def _concat_wavs(parts) -> bytes:
    """Concatenate WAV byte blobs (same sample rate) into one mono WAV."""
    arrs, sr = [], None
    for p in parts:
        y, s = sf.read(io.BytesIO(p), dtype="float32")
        if getattr(y, "ndim", 1) > 1:
            y = y.mean(axis=1)
        if sr is None:
            sr = s
        elif s != sr:
            raise ValueError(f"sample-rate mismatch {s} != {sr}")
        arrs.append(y)
    buf = io.BytesIO()
    sf.write(buf, np.concatenate(arrs), sr, format="WAV")
    return buf.getvalue()


_voice_dev_locks: dict = {}
_voice_dev_locks_guard = threading.Lock()


def _voice_dev_lock(dev_id: str) -> threading.Lock:
    with _voice_dev_locks_guard:
        return _voice_dev_locks.setdefault(dev_id, threading.Lock())


_voice_reply_cache: dict = {}     # replyId -> (t, status, body)
_voice_reply_cache_lock = threading.Lock()


def _bridge_base(dev: dict) -> str:
    u = dev.get("announce_url") or ""
    return u.split("/announce/", 1)[0]


def _push_to_device(dev: dict, wav: bytes, start_conversation: bool, timeout: float):
    """Hand a WAV to the device's adapter. Only the Dot bridge adapter exists in P1.
    Returns (http_status, body) with body always carrying delivered + error."""
    if dev.get("adapter", "bridge") != "bridge":
        return 502, {"delivered": False, "error": "adapter_unsupported",
                     "adapter": dev.get("adapter")}
    url = (f"{dev['announce_url']}?start_conversation={1 if start_conversation else 0}"
           f"&timeout={int(timeout)}")
    headers = {"Content-Type": "audio/wav"}
    if dev.get("announce_token"):
        headers["X-Announce-Token"] = dev["announce_token"]
    req = urllib.request.Request(url, data=wav, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout + 10) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read() or b"{}")
        except Exception:  # noqa: BLE001
            body = {}
        if not isinstance(body, dict) or "error" not in body:
            body = {"delivered": False, "error": "playback_failed", "detail": f"http {e.code}"}
        return e.code, body
    except Exception as e:  # noqa: BLE001 — refused/timeout: the bridge (and so the device) is off
        return 502, {"delivered": False, "error": "device_offline", "detail": repr(e)}


# Contract status codes (docs/voice-channel.md §4.4): every failure body is
# {delivered:false, error}; the status just lets a dumb client branch.
_VOICE_ERR_STATUS = {
    "device_unknown": 404, "device_offline": 502, "device_busy": 409,
    "playback_failed": 502, "synth_unavailable": 503, "too_long": 413,
    "unauthorized": 401, "adapter_unsupported": 502, "empty_wav": 500,
    "unknown_character": 404, "cooldown": 429, "internal": 500,
}


@app.get("/api/voice/devices")
def voice_devices(request: Request):
    """Device table + live connection state (best-effort, from each bridge's /devices)."""
    who = _voice_auth(request)
    if not who:
        return _voice_unauth()
    devs = _voice_devices()
    live: dict = {}
    for base in {_bridge_base(d) for d in devs.values() if d.get("adapter", "bridge") == "bridge"}:
        if not base:
            continue
        try:
            with urllib.request.urlopen(f"{base}/devices", timeout=3) as r:
                for row in (json.loads(r.read()) or {}).get("devices", []):
                    live[row["id"]] = row
        except Exception as e:  # noqa: BLE001
            print(f"[voice] bridge {base} unreachable for /devices: {e!r}", flush=True)
    out = []
    for dev_id, d in devs.items():
        row = live.get(dev_id) or {}
        out.append({"device": dev_id, "kind": d.get("kind"), "room": d.get("room") or "",
                    "mode": d.get("mode", "chat"), "adapter": d.get("adapter", "bridge"),
                    "connected": row.get("connected"), "busy": row.get("busy")})
    return {"devices": out, "caller": who}


_voice_cooldowns: dict = {}       # cooldownKey -> last delivered time
_voice_cooldowns_lock = threading.Lock()


@app.post("/api/voice/deliver")
def voice_deliver(request: Request, payload: dict = Body(...)):
    """Engine -> v2v: speak `text` on `device`. See docs/voice-channel.md §4.4.

    Body: {device, text, channelId?, name?, inReplyTo?, replyId?, sid?, character?,
           expectsReply?, paraphrase?: off|subtle|full, vendor?,
           cooldownKey?, cooldownSeconds?}
    `character` (explicit voice, e.g. a Home Assistant announcement) beats `sid`.
    `cooldownKey` + `cooldownSeconds`: at most one delivery per key per window; a
    repeat inside the window answers 429 {delivered:false, error:"cooldown"} without
    speaking (the door announcer's "say it once" rule, enforced here, not in HA).
    200 {delivered:true, played_ms, device, ...} only AFTER playback finished.
    Otherwise {delivered:false, error} with a matching status (see _VOICE_ERR_STATUS)."""
    who = _voice_auth(request)
    if not who:
        return _voice_unauth()
    status, body = _voice_deliver_core(payload, who)
    return JSONResponse(body, status_code=status)


def _voice_deliver_core(payload: dict, who: str):
    """The deliver pipeline minus HTTP auth: validate -> voice -> synth -> push.
    Returns (status, body). Shared by the HTTP endpoint and the MQTT announcer."""
    t0 = time.time()
    device = str(payload.get("device") or "").strip()
    text = str(payload.get("text") or "").strip()
    reply_id = str(payload.get("replyId") or "").strip()
    if not device:
        return 400, {"delivered": False, "error": "bad_request", "field": "device"}
    if not text:
        return 400, {"delivered": False, "error": "bad_request", "field": "text"}
    if len(text) > VOICE_TEXT_MAX:
        return 413, {"delivered": False, "error": "too_long", "max": VOICE_TEXT_MAX,
                             "len": len(text)}

    # Idempotency: the same replyId never plays twice; it gets the first outcome back.
    if reply_id:
        with _voice_reply_cache_lock:
            now = time.time()
            for k in [k for k, v in _voice_reply_cache.items()
                      if now - v[0] > VOICE_REPLY_CACHE_TTL_S]:
                _voice_reply_cache.pop(k, None)
            hit = _voice_reply_cache.get(reply_id)
        if hit:
            body = dict(hit[2])
            body["replayed"] = True
            return hit[1], body

    dev = _voice_devices().get(device)
    if not dev:
        return 404, {"delivered": False, "error": "device_unknown", "device": device}

    # Cooldown: "announce once, then stay quiet for a while" for callers whose trigger
    # can repeat (a door lock reporting several unlock transitions, a buggy relay).
    cd_key = str(payload.get("cooldownKey") or "").strip()
    try:
        cd_s = float(payload.get("cooldownSeconds") or 0)
    except (TypeError, ValueError):
        cd_s = 0.0
    if cd_key and cd_s > 0:
        with _voice_cooldowns_lock:
            last = _voice_cooldowns.get(cd_key, 0.0)
        left = cd_s - (time.time() - last)
        if left > 0:
            print(f"[voice] cooldown '{cd_key}' from={who}: {left:.0f}s left, not speaking",
                  flush=True)
            return 429, {"delivered": False, "error": "cooldown", "cooldownKey": cd_key,
                         "retryAfterS": int(left), "device": device}

    sid = str(payload.get("sid") or "").strip()
    expects_reply = bool(payload.get("expectsReply", False))
    para = str(payload.get("paraphrase") or "off").strip().lower()
    if para not in ("off", "subtle", "full"):
        para = "off"

    # Voice = an explicit `character` (a saved character id/name: its voice + style +
    # tuning; unknown = loud 404, never a silent default — same rule as /synthesize),
    # else the speaker's saved selection for `sid`, else the default voice.
    # paraphrase=off speaks the caller's words VERBATIM in that voice.
    char_key = str(payload.get("character") or "").strip()
    ch = None
    if char_key:
        ch = find_character(char_key)
        if ch is None:
            return 404, {"delivered": False, "error": "unknown_character",
                                 "character": char_key,
                                 "known": [c["id"] for c in CHARACTERS]}
        sel = f"char:{ch['id']}"
    else:
        sel = speaker_id.get_voice(sid) if sid else ""
    voice, style = _resolve_speaker_voice(sel)
    voice = voice or DEFAULT_VOICE
    if ch is None and sel.startswith("char:"):
        ch = next((c for c in CHARACTERS if c["id"] == sel[5:]), None)
    tn = (ch or {}).get("tuning") or {}
    sp, gd, tp, st = clamp_tuning(float(tn.get("speed", 1.0)), float(tn.get("guidance", 2.0)),
                                  float(tn.get("temperature", 0.0)), int(tn.get("steps", 16)))
    spoken = text
    if para != "off" and style:
        try:
            if para == "subtle":
                spoken = _style_line(text, style)          # cadence only, quality-guarded
            else:
                system = assemble_reword_system((ch or {}).get("name", ""),
                                                (ch or {}).get("bio", ""), style, "full")
                spoken = _ollama_chat([{"role": "system", "content": system},
                                       {"role": "user", "content": text}],
                                      num_predict=200) or text
        except Exception as e:  # noqa: BLE001 — never let styling block delivery
            print(f"[voice] paraphrase={para} failed ({e!r}); verbatim", flush=True)
            spoken = text
    spoken = _strip_stage_directions(spoken) or text

    try:
        wav = synth(spoken, voice, num_step=st, speed=sp, guidance_scale=gd,
                    class_temperature=tp)
    except GpuBusy as e:
        return 503, {"delivered": False, "error": "synth_unavailable",
                             "detail": str(e)}
    except Exception as e:  # noqa: BLE001
        print(f"[voice] synth failed: {e!r}", flush=True)
        return 503, {"delivered": False, "error": "synth_unavailable",
                             "detail": repr(e)}
    try:
        wav = _concat_wavs([_earcon_wav(), wav])
    except Exception as e:  # noqa: BLE001 — earcon is a nicety; the text must still play
        print(f"[voice] earcon concat failed ({e!r}); playing text only", flush=True)
    t_synth = time.time()

    # One delivery at a time per device (the bridge also serializes; this keeps our
    # own workers from piling up behind a slow playback).
    lock = _voice_dev_lock(device)
    if not lock.acquire(timeout=VOICE_DELIVER_TIMEOUT_S):
        status, body = 409, {"delivered": False, "error": "device_busy",
                             "detail": "another delivery is in progress"}
    else:
        try:
            status, body = _push_to_device(dev, wav, expects_reply, VOICE_DELIVER_TIMEOUT_S)
        finally:
            lock.release()
    if not body.get("delivered"):
        status = _VOICE_ERR_STATUS.get(body.get("error", ""), status if status >= 400 else 502)
    t_done = time.time()
    body.update({
        "device": device, "replyId": reply_id or None,
        "inReplyTo": payload.get("inReplyTo"), "channelId": payload.get("channelId"),
        "spoken": spoken, "voice": voice, "paraphrase": para, "expectsReply": expects_reply,
        "caller": who,
        "timing": {"synth_ms": int((t_synth - t0) * 1000),
                   "deliver_ms": int((t_done - t_synth) * 1000),
                   "total_ms": int((t_done - t0) * 1000)},
    })
    print(f"[voice] deliver from={who} dev={device} name={payload.get('name')!r} "
          f"sid={sid or '-'} voice={voice} para={para} expects={int(expects_reply)} "
          f"-> {'OK' if body.get('delivered') else body.get('error')} "
          f"played={body.get('played_ms', '-')}ms synth={body['timing']['synth_ms']}ms "
          f"total={body['timing']['total_ms']}ms | {spoken!r}", flush=True)
    if reply_id:
        with _voice_reply_cache_lock:
            _voice_reply_cache[reply_id] = (time.time(), status, body)
    if body.get("delivered") and cd_key:
        with _voice_cooldowns_lock:
            _voice_cooldowns[cd_key] = time.time()
    return status, body


# ---------------------------------------------------------------------------
# DEVELOPMENT MODE — the INBOUND leg (P3). Engine config lives in voice_devices.json
# under "voice_engine"; the bearer v2v PRESENTS to the engine is read per call from
# secrets/briefing-table-engine.token (sealed-delivered; never a caller key here).
# ---------------------------------------------------------------------------
# Eric's ruling (2026-09-24, live test): hands-free means NOTHING is held. The engine
# no longer holds on draft/busy, and a 202 must never produce a hold cue here. The old
# "you have a draft open" / "they're busy" lines are retired, not just unused.
DEV_ACK_LINES = {
    "sent": "Sent to {name}.",
    "no_channel": "No channel is open.",
    "not_you": "I'm not sure that's you.",
    "channel_gone": "That channel isn't open.",
    "engine_down": "The table isn't answering.",
}
_CTRL_RE = re.compile("[" + "".join(chr(c) for c in range(0, 32)) + chr(127) + "]")
_dev_ack_cache: dict = {}         # (voice, line) -> wav; acks repeat, synth once
_dev_ack_lock = threading.Lock()


def _engine_cfg() -> dict:
    return ((_load_json_hot(VOICE_DEVICES_PATH, {}) or {}).get("voice_engine") or {})


def _engine_token() -> str:
    path = _engine_cfg().get("token_path") or os.path.join("secrets", "briefing-table-engine.token")
    if not os.path.isabs(path):
        path = os.path.join(HERE, path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def _engine_inbound(body: dict, timeout: float):
    """POST the utterance to the engine. Returns (http_status|None, dict)."""
    url = _engine_cfg().get("inbound_url")
    if not url:
        return None, {"error": "engine_unconfigured"}
    headers = {"Content-Type": "application/json"}
    tok = _engine_token()
    if tok:
        headers["Authorization"] = "Bearer " + tok
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers,
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read() or b"{}")
            return r.status, (data if isinstance(data, dict) else {})
    except urllib.error.HTTPError as e:
        try:
            data = json.loads(e.read() or b"{}")
        except Exception:  # noqa: BLE001
            data = {}
        return e.code, (data if isinstance(data, dict) else {})
    except Exception as e:  # noqa: BLE001 — refused, DNS, timeout: the engine is unreachable
        return None, {"error": "unreachable", "detail": repr(e)}


# Per-utterance journal (Eric, 2026-09-24: "logs, concrete not inferred" — both sides
# must be able to reconstruct a turn by utteranceId). One JSON line per dev-mode turn,
# appended to working/voice_turns.jsonl and mirrored onto the agent-share so the NUC
# engine agent can read it with fetch_share("voice-to-voice/voice_turns.jsonl").
VOICE_JOURNAL_PATH = os.path.join(HERE, "working", "voice_turns.jsonl")
VOICE_JOURNAL_MIRROR = r"F:\agent-share\voice-to-voice\voice_turns.jsonl"
_voice_journal_lock = threading.Lock()


def _voice_journal(entry: dict):
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    with _voice_journal_lock:
        for path in (VOICE_JOURNAL_PATH, VOICE_JOURNAL_MIRROR):
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line)
            except OSError as e:
                print(f"[voice-journal] write failed {path}: {e!r}", flush=True)


def _dev_ack_wav(line: str, voice: str) -> bytes:
    key = (voice or DEFAULT_VOICE, line)
    with _dev_ack_lock:
        wav = _dev_ack_cache.get(key)
    if wav is None:
        wav = synth(line, voice or DEFAULT_VOICE, num_step=16)
        with _dev_ack_lock:
            if len(_dev_ack_cache) > 64:
                _dev_ack_cache.clear()
            _dev_ack_cache[key] = wav
    return wav


def _dev_turn(transcript, device, spk_id, spk_name, spk_conf, spk_detail, voice, t0, t_stt):
    """One development-mode turn: speaker gate -> engine inbound -> spoken ack.
    Every outcome is audible; the X-Route header says which one for the P4 miss log."""
    utt_id = f"v2v-{device}-{int(time.time() * 1000)}-{secrets.token_hex(2)}"
    # Contract §3 + amendment 6: one line, single spaces, NO C0/DEL control characters
    # (the engine answers 400 on any). Whisper can emit tabs/newlines; normalise all.
    transcript = " ".join(_CTRL_RE.sub(" ", transcript or "").split())
    match = "sticky" if str(spk_detail or "").startswith("sticky") else "full"
    status, resp, name = None, {}, None
    if not spk_id or spk_conf < DEV_TURN_MIN_CONF:
        outcome, line = "refused_speaker", DEV_ACK_LINES["not_you"]
    else:
        cfg = _engine_cfg()
        body = {
            "text": transcript, "device": device, "speaker": spk_name or "",
            "sid": spk_id, "confidence": round(float(spk_conf), 3), "match": match,
            "utteranceId": utt_id,
            "replyTo": cfg.get("reply_to") or "https://vrpc-3.tail567253.ts.net/api/voice/deliver",
        }
        status, resp = _engine_inbound(body, float(cfg.get("timeout_s", 8)))
        name = resp.get("name") or "the table"
        if status == 202:
            # Any 202 is "sent" — even if the engine still reports held (transition
            # window while its hold removal lands). The held flag is journaled.
            outcome, line = "sent", DEV_ACK_LINES["sent"].format(name=name)
        elif status == 409:
            outcome, line = "no_channel", DEV_ACK_LINES["no_channel"]
        elif status == 403:
            outcome, line = "refused_engine", DEV_ACK_LINES["not_you"]
        elif status == 404:
            outcome, line = "channel_gone", DEV_ACK_LINES["channel_gone"]
        else:
            outcome, line = "engine_down", DEV_ACK_LINES["engine_down"]
    t_eng = time.time()
    wav = _dev_ack_wav(line, voice)
    t_tts = time.time()
    print(f"[dev-turn] dev={device} spk={spk_name or 'unknown'}({spk_conf:.2f},{match}) "
          f"-> {outcome} status={status} name={name!r} utt={utt_id} "
          f"| stt={int((t_stt - t0) * 1000)}ms engine={int((t_eng - t_stt) * 1000)}ms "
          f"ack={int((t_tts - t_eng) * 1000)}ms | heard={transcript!r} said={line!r} "
          f"| engine_body={json.dumps(resp, ensure_ascii=False)[:300]}", flush=True)
    _voice_journal({
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds"),
        "utteranceId": utt_id, "device": device, "mode": "development",
        "text": transcript, "speaker": spk_name or "", "sid": spk_id or "",
        "confidence": round(float(spk_conf), 3), "match": match,
        "engine_status": status, "engine_body": resp, "outcome": outcome, "cue": line,
        "timing_ms": {"stt": int((t_stt - t0) * 1000), "engine": int((t_eng - t_stt) * 1000),
                      "ack": int((t_tts - t_eng) * 1000), "total": int((t_tts - t0) * 1000)},
    })
    headers = {
        "X-Transcript": urllib.parse.quote(transcript),
        "X-Reply": urllib.parse.quote(line),
        "X-Speaker": urllib.parse.quote(spk_name or "unknown"),
        "X-Speaker-Sid": spk_id or "",
        "X-Speaker-Confidence": f"{spk_conf:.3f}",
        "X-Followup-Listen": "0",
        "X-Route": f"dev:{outcome}",
        "X-Utterance-Id": utt_id,
        "X-Timing": f"stt={int((t_stt - t0) * 1000)};engine={int((t_eng - t_stt) * 1000)};"
                    f"ack={int((t_tts - t_eng) * 1000)};total={int((t_tts - t0) * 1000)}",
        "Access-Control-Expose-Headers":
            "X-Transcript,X-Reply,X-Speaker,X-Speaker-Sid,X-Speaker-Confidence,"
            "X-Followup-Listen,X-Route,X-Utterance-Id,X-Timing",
    }
    return Response(content=wav, media_type="audio/wav", headers=headers)


@app.post("/api/voice/devices/{device}/mode")
def voice_device_mode(device: str, request: Request, payload: dict = Body(...)):
    """Dev-panel toggle: set a device's mode (chat|development). Persists to
    voice_devices.json (the file is the config; this is just its editor)."""
    who = _voice_auth(request)
    if not who:
        return _voice_unauth()
    mode = str(payload.get("mode") or "").strip().lower()
    if mode not in ("chat", "development"):
        return JSONResponse({"error": "bad_request", "field": "mode",
                             "allowed": ["chat", "development"]}, status_code=400)
    with open(VOICE_DEVICES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    dev = (data.get("devices") or {}).get(device)
    if not dev:
        return JSONResponse({"error": "device_unknown", "device": device}, status_code=404)
    prev = dev.get("mode", "chat")
    dev["mode"] = mode
    tmp = VOICE_DEVICES_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp, VOICE_DEVICES_PATH)
    print(f"[voice] device {device} mode {prev} -> {mode} (by {who})", flush=True)
    return {"device": device, "mode": mode, "previous": prev}


# ---------------------------------------------------------------------------
# MQTT ANNOUNCER — Home Assistant (and anything else on the house broker) can make a
# device speak without an HTTP client: publish the deliver body as JSON on
# `voice_mqtt.topic`; the outcome is published on `<ack_topic>/<replyId>` so an
# automation can wait for it and fall back only when v2v is genuinely unreachable
# (no ack at all). Why MQTT: HA's rest_command needs configuration.yaml, which its
# REST token cannot write; mqtt.publish is a built-in service. The broker is the NUC
# mosquitto (anonymous on the LAN) — same trust level as every other home event.
# Config lives in voice_devices.json under "voice_mqtt"; absent = announcer off.
# ---------------------------------------------------------------------------
def _voice_mqtt_cfg() -> dict:
    return ((_load_json_hot(VOICE_DEVICES_PATH, {}) or {}).get("voice_mqtt") or {})


def _voice_mqtt_thread():
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        print("[voice-mqtt] paho-mqtt not installed; announcer off", flush=True)
        return
    cfg = _voice_mqtt_cfg()
    if not cfg.get("broker"):
        print("[voice-mqtt] no voice_mqtt.broker in voice_devices.json; announcer off",
              flush=True)
        return
    topic = cfg.get("topic", "home/voice/announce")
    ack_topic = cfg.get("ack_topic", "home/voice/announce/ack")

    def on_connect(client, userdata, flags, reason_code, properties=None):
        client.subscribe(topic, qos=1)
        print(f"[voice-mqtt] connected {cfg['broker']}:{cfg.get('port', 1883)} "
              f"subscribed {topic}", flush=True)

    def on_message(client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8", "replace") or "{}")
        except Exception as e:  # noqa: BLE001
            print(f"[voice-mqtt] bad payload on {msg.topic}: {e!r}", flush=True)
            return
        if not isinstance(payload, dict):
            return
        reply_id = str(payload.get("replyId") or f"mqtt-{int(time.time() * 1000)}")
        payload["replyId"] = reply_id
        # Run OFF the network thread so a slow synth/playback never stalls the client
        # loop (keepalives would lapse and the broker would drop us mid-announce).
        def _run():
            try:
                status, body = _voice_deliver_core(payload, "mqtt")
            except Exception as e:  # noqa: BLE001
                status, body = 500, {"delivered": False, "error": "internal", "detail": repr(e)}
            body["status"] = status
            try:
                client.publish(f"{ack_topic}/{reply_id}", json.dumps(body), qos=1)
            except Exception as e:  # noqa: BLE001
                print(f"[voice-mqtt] ack publish failed: {e!r}", flush=True)
        threading.Thread(target=_run, name=f"voice-mqtt:{reply_id}", daemon=True).start()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="voice-to-voice")
    client.on_connect = on_connect
    client.on_message = on_message
    client.reconnect_delay_set(min_delay=2, max_delay=60)
    while True:
        try:
            client.connect(cfg["broker"], int(cfg.get("port", 1883)), keepalive=30)
            client.loop_forever(retry_first_connection=True)
        except Exception as e:  # noqa: BLE001
            print(f"[voice-mqtt] connection lost/failed: {e!r}; retry in 10s", flush=True)
            time.sleep(10)


threading.Thread(target=_voice_mqtt_thread, name="voice-mqtt", daemon=True).start()


# Static front-end (mounted last so /api/* wins).
@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


# Speech-to-visual prototypes (built Vite app). Mounted BEFORE the catch-all "/"
# so /avatar/* resolves to the avatar SPA build. html=True serves index.html for
# the app root. If the build is missing, the mount is skipped (dev still runs the
# app on :5173 via `npm run dev`).
_AVATAR_DIST = os.path.join(HERE, "avatar", "web", "dist")
if os.path.isdir(_AVATAR_DIST):
    app.mount("/avatar", StaticFiles(directory=_AVATAR_DIST, html=True), name="avatar")


app.mount("/", StaticFiles(directory=os.path.join(HERE, "static")), name="static")


def _start_hc_heartbeat(interval=60):
    """Self-ping harbor's injected HC_PING_URL so harbor shows green (live dead-man).
    No-op when HC_PING_URL is unset (e.g. run outside harbor)."""
    import urllib.request as _u
    url = os.environ.get("HC_PING_URL")
    if not url:
        return
    def _loop():
        while True:
            try:
                _u.urlopen(url, timeout=5).read()
            except Exception:
                pass
            time.sleep(interval)
    threading.Thread(target=_loop, daemon=True, name="hc-heartbeat").start()
    print(f"[hc] heartbeat -> {url} every {interval}s", flush=True)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8221"))
    print(f"[init] serving on 0.0.0.0:{port}", flush=True)
    _start_hc_heartbeat()
    # access_log=False: the per-request access log line was the exact frame that wedged the
    # event loop (py-spy: send -> logging.flush). Our endpoints already log what we need to
    # working/*.log, so drop uvicorn's per-request chatter -- less volume, and the hot path
    # no longer logs from the loop thread at all.
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning", access_log=False)
