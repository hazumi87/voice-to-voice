# Voice-to-Voice Prototype — Architecture & Integration Handoff

**Status:** working prototype (prove-it). Runs entirely on the VRPC, reachable from the iPad over Tailscale.
**Author:** built with Claude in `F:\Code\voice-to-voice`. **For:** architect, to integrate the voice loop into other app surfaces.
**Last updated:** 2026-06-07.

---

## 1. What it is

A spoken-conversation web app. The user taps a mic, speaks, and the agent replies in synthesized speech. Capture modes, selectable voices (incl. user-cloned custom voices), selectable agent personalities, live transcript, and per-request voice tuning.

**Feature set delivered:**
- **STT → LLM → TTS** round-trip, all local to the VRPC.
- **3 capture modes:** Push-to-talk, Auto (VAD auto-stop on pause), Compose (multi-segment: each pause queues a transcribed segment, manual Send fires the whole thing).
- **21 preset voices** via OmniVoice *voice-design* (gender/age/pitch/accent) — no reference WAVs needed. Each has an audio **preview**.
- **Custom (cloned) voices:** record in-browser (with a live timer) or upload a clip; auto-transcribed; saved/named; persist across restarts; previewable + deletable. Reading-prompt tabs (Stella / Rainbow / Harvard) are shown for the user to read while recording.
- **18 agent personalities** in 3 groups (Characters / By Generation / By Decade) — just system prompts.
- **Voice tuning** (per request, persisted): Speed, Expressiveness (guidance), Liveliness (temperature), Quality (diffusion steps), with safe ranges. Plus an editable **Test phrase** box to audition any text in the current voice + tuning.
- **Live waveform** (Web Audio analyser), **Cancel** (discard recording), **Stop** (halt playback), **New session** (wipe memory+transcript+draft).
- **Two-pane UI:** controls left, live chat transcript right.
- **Conversation memory:** server-side history sent on every turn; cleared only by New session / reset.
- **Client→server logging:** browser log lines + uncaught errors POST to the server console (iOS debugging).
- Settings (mode, voice, personality, tuning, test text) persisted in `localStorage`.

---

## 2. Runtime architecture

```
iPad browser (Safari/Chrome — both WebKit on iOS)
   │  HTTPS (TLS terminated by Tailscale)
   ▼
Tailscale serve  https://vrpc-3.tail567253.ts.net  (port 443, tailnet-only)
   │  proxy → loopback
   ▼
FastAPI / uvicorn   0.0.0.0:8123   (server.py)   ── serves static frontend + JSON/audio API
   ├── STT   faster-whisper base.en  (CUDA/float16, in-process)         ─┐
   ├── CHAT  HTTP → ollama 127.0.0.1:11434  (llama3.2:3b)                │ all on the VRPC,
   └── TTS   OmniVoice k2-fsa/OmniVoice (CUDA/float16, in-process)       ─┘ single GPU (RTX 4080)
              ├─ voice-design (preset voices, instruct string)
              └─ voice-clone  (custom voices, reusable clone prompt)
```

STT and TTS run **in the FastAPI process** (models loaded once at startup, serialized by a single
`threading.Lock` since they share the GPU). ollama runs as a **separate Windows service**, called over
local HTTP. The browser never talks to ollama directly.

**Per-mode request flow:**
- **Push-to-talk / Auto:** browser records audio → `POST /api/converse` (audio + voice + personality + tuning) → server STT+chat+TTS → reply WAV (transcript/reply/timing in headers).
- **Compose:** each segment → `POST /api/stt` (audio → text, queued client-side). On Send → `POST /api/send_text` (joined text + voice + personality + tuning) → chat+TTS → reply WAV.

---

## 3. Component inventory — where things live

| Component | Where | Notes |
|---|---|---|
| App code (`server.py`, `static/`) | **Project folder** `F:\Code\voice-to-voice` | the only hand-written code |
| Custom voice store (`custom_voices/`) | **Project folder** `F:\Code\voice-to-voice\custom_voices` | per voice: `<id>.wav` (trimmed reference) + `<id>.json` (label, ref_text) |
| Python venv + all pip deps | **Project folder** `F:\Code\voice-to-voice\.venv` | torch+cu128, omnivoice, faster-whisper, av (PyAV), fastapi, uvicorn… |
| OmniVoice weights (`k2-fsa/OmniVoice`, ~3.1 GB, incl. higgs audio tokenizer) | **PC user cache** `C:\Users\erich\.cache\huggingface\hub` | downloaded on first run; shared HF cache |
| faster-whisper `base.en` (~141 MB) | **PC user cache** `…\models--Systran--faster-whisper-base.en` | ctranslate2 format |
| LLM `llama3.2:3b` | **System (ollama store)** `%USERPROFILE%\.ollama` | pulled via `ollama pull`; `llama3.1` also present |
| ollama runtime | **System service** `OllamaService` @ 127.0.0.1:11434 | auto-starts at boot, localhost-only |
| HTTPS exposure | **Tailscale config** (not in project) | `tailscale serve` 443 → 127.0.0.1:8123, tailnet-only |

**Key point for the architect:** the *code* + *custom voices* are self-contained in the project folder, but the
base *models* live in the shared HF cache and the *LLM* lives in ollama — neither is in the project folder.
A move/clone of the project folder re-downloads the HF models on first run (custom voices come along; base
models don't unless the HF cache is copied too).

---

## 4. API contract

Base URL (tailnet): `https://vrpc-3.tail567253.ts.net`  ·  Local: `http://127.0.0.1:8123`

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/voices` | — | `{voices:[{id,label,tags,custom?}], default}` (presets + custom) |
| GET | `/api/personalities` | — | `{personalities:[{id,label,group}], default}` |
| GET | `/api/preview` | query: `voice`, `text?`, `speed?`, `guidance?`, `temperature?`, `steps?` | `audio/wav` sample (uses `text` if given, else default line) |
| POST | `/api/converse` | multipart: `audio`, `voice`, `personality`, `speed`, `guidance`, `temperature`, `steps` | `audio/wav` reply. Headers: `X-Transcript`, `X-Reply`, `X-Timing` |
| POST | `/api/stt` | multipart: `audio` | `{text}` (transcribe only, no chat/history) |
| POST | `/api/send_text` | form: `text`, `voice`, `personality`, `speed`, `guidance`, `temperature`, `steps` | `audio/wav` reply. Same headers as converse |
| POST | `/api/voices/custom` | multipart: `audio`, `name` | `{voice:{id,label,tags,custom}, ref_text}` — clones + saves a custom voice |
| POST | `/api/voices/custom_delete` | form: `voice` | `{ok}` — removes voice + its files |
| POST | `/api/reset` | — | clears server conversation history |
| GET | `/api/health` | — | `{stt, tts_sr, ollama, ollama_model, turns}` |
| POST | `/api/clientlog` | json: `{level,msg}` | echoes browser logs to server console |

Audio in: server STT (faster-whisper + bundled PyAV) decodes **iPad `audio/mp4`/AAC**, webm/opus, ogg, wav
directly — **no ffmpeg needed**. Custom-voice clips are decoded to a waveform with **PyAV** (`av`) and resampled
to 24 kHz. Audio out: 24 kHz mono WAV.

Error contract: ollama-down → **503** `{error:"ollama_down", detail}` (gaming-mode killswitch). No-speech → **422**.

---

## 5. Custom voice (cloning) pipeline

OmniVoice clones a voice from a short reference clip + its transcript (separate from voice-design presets).

`POST /api/voices/custom` does:
1. **Decode** the uploaded/recorded clip → mono float32 @ 24 kHz (PyAV; handles iPad mp4/AAC).
2. **Auto-transcribe** with faster-whisper, capturing **segment timestamps**.
3. **Window-select for quality:** keep only whole sentences up to **~13s** (`CLONE_MAX_S`), cut on a sentence
   boundary using the segment timestamps. This lets the user read a long passage comfortably while the clone
   always uses a reference inside OmniVoice's quality window (recommended **3–10s**, degrades >20s).
4. **Tap-trim:** strip the leading/trailing finger-tap (the screen tap when starting/stopping a recording) and
   fade the edges, so the tap doesn't leak into synthesized replies. If the whole clip was kept, a larger
   trailing trim removes the stop-tap; if the tail was already discarded by windowing, just a fade.
5. **Save** `<id>.wav` (trimmed reference) + `<id>.json` (label, ref_text), build a reusable
   `VoiceClonePrompt`, and add it to the in-memory registry. **Reloaded at startup** so voices persist.

Custom voices appear in `/api/voices` with `custom:true`, work in all modes/personalities, support preview,
and honor tuning. `synth()` branches: if the voice id is in `custom_prompts`, it calls
`generate(voice_clone_prompt=…)`, else `generate(instruct=…)`.

**Front-end (Add custom voice modal):** reading-prompt tabs (Stella / Rainbow / Harvard) above the controls;
name field; **Record sample** with a live **timer** (30s safety cap; user taps stop) or **Upload file**;
playback to check; **Create voice**. Trash icon on custom rows deletes.

---

## 6. Voice tuning (per request)

Threaded through `synth()` and exposed on `/api/converse`, `/api/send_text`, `/api/preview`. Persisted in
`localStorage`; a collapsible **Tuning** panel has sliders + a **Reset** + the **Test phrase** box.

| UI name | OmniVoice param | Safe slider range | Server clamp | Default |
|---|---|---|---|---|
| Speed | `speed` | 0.8–1.3 | 0.7–1.4 | 1.0 |
| Expressiveness | `guidance_scale` | 1.5–3.5 | 1.0–4.0 | 2.0 |
| Liveliness | `class_temperature` | 0–0.6 | 0–0.8 | 0.0 |
| Quality | `num_step` | 16–48 | 12–48 | 16 |

Ranges are deliberately kept in the high-quality zone; the server **clamps** all values (extremes degrade or
destabilize output). Saved client values are also clamped into range on load. The **Test phrase** box sends
editable text to `/api/preview?text=…` with the current voice + tuning — pure TTS audition, no LLM.

---

## 7. Resource footprint (RTX 4080, 16 GB)

- OmniVoice float16: **~2.0 GB** VRAM. Load ~2 s.
- faster-whisper base.en: small (~hundreds of MB on GPU).
- Coexists with the live **neutts** service (~4.8 GB). Observed total with everything loaded: **~9.7 GB used / ~6.4 GB free.**
- **Latency (warm, server-side):** STT ~160 ms · chat ~200 ms (llama3.2:3b) · TTS ~430 ms (num_step=16) → **~0.8–1.0 s total** per turn. Felt latency adds browser record + network. Higher `num_step` / lower `speed` increases TTS time.

---

## 8. Config knobs (server.py unless noted)

- `OLLAMA_MODEL` = `"llama3.2:3b"`.
- `VOICES` — preset list (id/label/tags/instruct). Instruct grammar: gender · age · pitch · style(whisper) · accent (american/british/australian/canadian/indian/korean/portuguese/russian/japanese/chinese).
- `PERSONALITIES` — id/label/group/system. `VOICE_STYLE` appended to every persona (keeps replies short & TTS-safe).
- `clamp_tuning()` — the safe tuning ranges (table above).
- `CLONE_MAX_S` = 13.0 — custom-voice reference window length.
- `PREVIEW_TEXT` — default audition line.
- `num_step` — 16 (chat), 32 (preview default), slider-controlled per request.
- VAD tuning (`static/main.js`): `VOICE_RMS` 0.030, `SILENCE_HANG_MS` 1200, `MAX_REC_MS` 20000, `MIN_HOLD_MS` 350, `SAMPLE_MAX_MS` 30000 (custom-voice recorder cap).
- STT model: `WhisperModel("base.en","cuda","float16")` — bump to `small.en` for accuracy.

---

## 9. Networking / access / iOS gotchas

- **HTTPS is mandatory for the mic.** iOS `getUserMedia` only works on a secure context. Served via `tailscale serve` with a valid `.ts.net` cert. Use the bare hostname URL (port 443) — **not** the IP and **not** `:8123` (plain HTTP → "invalid response").
- **No Windows firewall rule needed** — Tailscale proxies to loopback.
- **iOS mic re-prompts every page load.** All iOS browsers are WebKit; the per-site mic grant resets per session. Only **Safari** persists it (Website Settings → Microphone → Allow) or via **Add to Home Screen** (standalone PWA). Chrome-iOS has no persistent allow.
- **iOS recording bug worked around:** a fresh `getUserMedia` stream per recording (reused streams record 0 bytes on iOS); `MediaRecorder.start(200)` timeslice flushes chunks.
- **iOS audio playback:** the `<audio>` element is "unlocked" inside the mic-tap gesture (silent clip) so replies autoplay after the async fetch.

---

## 10. Known limitations

- Single global conversation session (one `history` list) — **not multi-user/multi-session.**
- GPU work serialized by one lock — fine for one user, queues under concurrency.
- Child voices are the least stable voice-design output; very long custom reads degrade clones (mitigated by the 13s windowing).
- No streaming TTS — reply audio is generated fully then sent.
- ollama gaming-mode `.bat` kills `OllamaService`; while down, chat returns 503.
- Nothing harbor-supervised; server launched manually (`start.bat`).

---

## 11. Integration guidance (for other app surfaces)

The three legs are cleanly separable — the API is already factored for reuse:

- **Reuse as-is:** call `POST /api/converse` (audio→audio) or split `POST /api/stt` + `POST /api/send_text`. Plain multipart/JSON; embed in a webview or native client. Tuning params are optional (sane defaults).
- **To make it a real service:**
  1. Replace the single global `history` with per-session state (session id → history dict) — the only real blocker to multi-user. Custom voices are already global/shared by design.
  2. Wrap model load + uvicorn under harbor supervision; pin the venv python path and `HF_HUB_DISABLE_SYMLINKS=1`.
  3. Decide TTS engine: this prototype uses **OmniVoice** (voice-design + cloning). Production runs **neutts** on `:8220`. Consolidate or run both (VRAM allows it today).
  4. Consider sentence-wise streaming TTS for snappier feel.
  5. Move STT off the shared FastAPI process if GPU contention bites.
- **Data, not code:** preset voices = OmniVoice instruct strings; personalities = LLM system prompts; custom voices = saved clip+transcript clone prompts in `custom_voices/`. All portable.
- **Frontend patterns worth lifting:** iOS mic-unlock, fresh-stream-per-recording, MIME via `isTypeSupported`, the VAD loop, the clone-window selection (Whisper-timestamp sentence boundary + tap trim), and tuning persistence — all in `static/main.js` + `server.py`.

---

## 12. File manifest (project folder)

```
F:\Code\voice-to-voice\
├── server.py            # FastAPI app: models, all endpoints, STT/chat/TTS, cloning, tuning
├── static\
│   ├── index.html       # two-pane UI, voice modal, custom-voice modal, tuning panel, styles
│   └── main.js          # recording, VAD, waveform, modes, transcript, previews, cloning UI, tuning
├── custom_voices\       # saved cloned voices: <id>.wav (trimmed ref) + <id>.json (label, ref_text)
├── start.bat            # launcher (sets HF env, runs venv python server.py)
├── smoke_test.py        # standalone OmniVoice install-gate test
├── README.md            # quick usage
├── ARCHITECTURE.md      # this file
├── TASKS.md             # build log
├── server.log           # runtime log (incl. mirrored client logs)
├── .venv\               # python env + all deps (NOT base models)
└── *.wav                # throwaway test artifacts (safe to delete)
```

## 13. How to run

```bat
start.bat
```
(or `.venv\Scripts\python.exe server.py`). Binds `0.0.0.0:8123`. `tailscale serve` fronts it on
`https://vrpc-3.tail567253.ts.net/`. ollama (`OllamaService`) must be running. Open the HTTPS URL on the iPad.
