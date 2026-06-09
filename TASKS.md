# TASKS

## 2026-06-06 — Voice-to-voice prototype built end-to-end
- Probed VRPC: RTX 4080 (~13 GB free per torch), Store-shim Python 3.12.10, ollama service up.
  No tiny model pulled; chose llama3.1 (8B, already present).
- Created `.venv` from the Store python; installed torch 2.11.0+cu128, torchaudio, omnivoice 0.1.5,
  faster-whisper 1.2.1. CUDA verified. **No wheel-hunt** — install gate passed clean.
- Install gate smoke test (GO): OmniVoice loads in ~2s, +1962 MiB VRAM float16, 13 GB headroom.
  Voice-design synth RTF 0.16 (5.05s audio / 0.83s), 24 kHz. No reference WAV needed.
- Built `server.py` (FastAPI/uvicorn, one port 8123): /api/voices, /api/converse, /api/reset,
  /api/health. STT+TTS guarded by a GPU lock; ollama called server-side.
- Built front-end (`static/index.html` + `main.js`): push-to-talk mic, voice radio buttons,
  status indicator, debug console. MIME picked via MediaRecorder.isTypeSupported (iPad = audio/mp4).
- Full-loop test (smoke WAV → converse): **total 1555ms** (stt 466 / chat 362 / tts 726).
- Fronted with `tailscale serve` HTTPS at https://vrpc-3.tail567253.ts.net/ (valid cert →
  iOS mic works). No firewall rule needed (serve proxies via loopback).

### Decisions
- FastAPI serves static front-end directly (no Vite/Node) — simplest throwaway.
- Voices via OmniVoice voice-design instruct presets, not cloned reference WAVs.
- HF symlink footgun fixed with HF_HUB_DISABLE_SYMLINKS=1.

### Issues / flags
- `tailscale serve` replaced a pre-existing **Funnel** that pointed to :5173 (likely a prior
  Vite iteration). Changed public Funnel → tailnet-only serve. Reversible.
- ollama gaming-mode killswitch remains the known failure point for the chat leg.
