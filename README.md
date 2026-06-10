# Voice-to-Voice Prototype (VRPC, throwaway)

A standalone spoken-only web app: tap mic → speak → hear a reply. No text I/O.
Built to prove the end-to-end voice loop on the VRPC. Not production, not harbor-supervised.

## The loop (all local to the VRPC)
```
iPad mic (push-to-talk)
  → STT   faster-whisper base.en (cuda/float16, decodes iPad mp4/AAC via PyAV)
  → CHAT  ollama llama3.1 @ 127.0.0.1:11434 (server-side only)
  → TTS   OmniVoice k2-fsa/OmniVoice (cuda/float16, voice-design mode, no ref WAV)
  → audio plays back on the iPad
```

## Run
```
start.bat              # or: .venv\Scripts\python.exe server.py
```
Server listens on `0.0.0.0:8221`. `tailscale serve` fronts it over HTTPS.

## Open on the iPad
**https://vrpc-3.tail567253.ts.net/**
HTTPS is required so iOS Safari grants the mic. The tailscale cert is valid, so no warning.

## Voices (radio buttons)
Voice-design presets (gender/age/pitch/accent) — no reference audio needed. Edit `VOICES` in `server.py`.
OmniVoice supports accents: american, british, australian, canadian, indian, korean, portuguese, russian, japanese, chinese.

## Endpoints
- `GET  /api/voices`   — preset list
- `POST /api/converse` — multipart audio + `voice` → reply WAV (transcript/reply/timing in headers)
- `POST /api/reset`    — clear conversation history
- `GET  /api/health`   — component status

## Notes / gotchas
- **ollama gaming-mode killswitch:** if speech transcribes but no reply is spoken, ollama is down.
  Restart `OllamaService`. `/api/converse` returns a clear 503 in this case.
- Conversation history is in-memory, single session. Use the Debug → Reset button to clear.
- Deviation from the handoff: served from FastAPI (one port) instead of Vite, to keep the
  throwaway minimal. No Node dependency.
- `tailscale serve` here replaced a pre-existing Funnel that pointed to :5173.
