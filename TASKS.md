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

## 2026-06-09 — Graduated to a harbor-supervised service
- Added `GET /health` route for harbor probing (mirrors neutts `/health`).
- `git init` + `.gitignore` (excludes .venv, *.wav, server.log, custom_voices/) + remote
  `git@github.com:hazumi87/voice-to-voice.git`; committed + pushed `main`.
- Registered in harbor: added a `voice-to-voice` entry to `F:\code\harbor\services.json`
  mirroring `neutts-synth` (abs venv python, `args:[server.py]`, gpu:true, autostart:true,
  git block, health `http://localhost:8123/health`, env: PYTHONUTF8 / HF_HUB_DISABLE_SYMLINKS=1 /
  HF_HOME pinned to the shared cache / PORT=8123). Did NOT touch neutts or other services.
- `harbor reload` (added voice-to-voice, all else unchanged) → `harbor start` → up in 8s.
  Verified: /health 200, TTS round-trip OK, GPU coexists with neutts (~10.5 GB used / 5.5 free),
  Tailscale HTTPS front still 200. `harbor restart` → self-recovered in 1s (supervision proof:
  process is harbor-owned, survives VS Code closing; autostart brings it up on boot).
- Registered the project in `hazumi87/project-registry` (projects.yaml) on a cross-surface
  branch (PENDING MERGE by NUC canonical agent, per branch_required policy).

### Decisions (2026-06-09)
- Mirrored the neutts-synth registration mechanism exactly (edit services.json + harbor reload).
- Omitted the Healthchecks `hc` block — must not invent a uuid; several services run without one.
  harbor shows purple "running" (process-alive), same as neutts. **Follow-up:** provision a real hc uuid.
- Kept the single global conversation `history` (fine for single user; multi-user is a noted follow-up).
- Kept :8123 + Tailscale HTTPS (iPad mic needs the secure context).

## 2026-06-09 — Port moved 8123 → 8221 (NUC services band) + harbor entry de-git'd
- Checked NUC port registry (`/opt/dev/infrastructure/ports.json`): 8200–8299 = "services"
  band (internal infra). neutts-synth is 8220; **8221 is the first free slot** (not registered,
  not listening locally or on NUC). Moved off the lone 8123 outlier into the band with its sibling.
- Edited `PORT` default in `server.py` (8123 → 8221) + all docs (ARCHITECTURE.md ×4, README.md).
  Frontend uses relative URLs — no client change needed.
- **harbor services.json entry revised** (handed to NUC agent, not applied from VRPC):
  - Removed the `git` block — VRPC is canonical; harbor never needs to pull (commit is already in
    the tree). Harbor just supervises/restarts. (Removing the pull also kills the dirty-file class
    of failure.)
  - `PORT` env → 8221. `autostart: true` retained. `gpu: true`, `/health` retained.
- **COMPANION STEP (host config, NOT done here):** `tailscale serve` currently proxies 443 →
  127.0.0.1:**8123**. Must be repointed to **8221** or the iPad HTTPS front breaks. Tailscale
  config is host-level (not in repo) — Eric / the NUC-side flow must repoint it when the new
  services.json goes live.

### Decisions (2026-06-09, port move)
- Moved into the NUC-registry "services" band (8221) for consistency with neutts (8220) and to
  stop 8123 being a registry-drift outlier. Registry confirms 8221 free on both NUC and VRPC.
- Dropped harbor `git` block: canonical-on-VRPC means pull is redundant and a foot-gun.
- Speak-MCP rollout (agreed): OmniVoice added as a *selectable* engine first (neutts stays
  default), tested via explicit `engine: omnivoice`, promoted to default only after Eric approves.

## 2026-06-16 — Resolved the recurring "hang"; voice fixes; TTS pre-warm; postmortem doc
- **Root-caused the "crash"**: it was a HANG = blocked stdout write wedging the uvicorn
  event loop (py-spy: only blocking frame was logging.flush(), zero CUDA frames). NOT a
  CUDA/VRAM deadlock. Fixed (be10e93): TQDM_DISABLE, os.dup2 stdout/stderr -> file,
  uvicorn access_log=False. Verified stable across many voices with both models co-resident.
- Fixed cold first-render default+corrupted voice: guard checked custom_prompts (lazy GPU
  cache) instead of custom_voices (metadata) -> cust_ id fell back to DEFAULT on the unwarmed
  instruct path (086d9de); dual-path warmup + per-render [synth] diagnostics (2135be8).
- Added background TTS pre-warm (07b398c): TTS_PREWARM=1 default -> healthy immediately +
  fast first render; TTS_PREWARM=0 lazy; EAGER_TTS=1 legacy blocking.
- talkback (814c745): VAD auto-stop on pause; fixed chat reply auto-play on iOS (prime
  shared #player in-gesture, play reply through it).
- Reverted the NVIDIA per-program Sysmem Fallback override (was never relevant; global stays
  Driver Default for graceful VRAM spill).
- **Key decision**: when a server "hangs", run `py-spy dump --pid` BEFORE theorizing. py-spy
  now installed in the venv. Full postmortem + combine-prototypes watch-list in DISCOVERIES.md.

## 2026-07-06 — speak() device targets + character/paraphrase contract (KB cbad834f6fcf491b)
- /synthesize gained character (voice+tuning, loud 404 on unknown), paraphrase (verbatim placeholder), sr (server-side resample, 48k for A2DP wire), pad_ms (sink-wake silence) — commit 265a946; also split out the prior session's guest-enrollment honesty fix as 6c46257
- agent-speech-relay (NUC, commit 1bef658): target={kind:'device', id:station} pushes wire-ready WAV to HEARTHOS_PLAY_URL callback; device speech never falls back — structured error codes so HearthOS fails over to its stored cue clips (synth_unavailable = VRPC down)
- Key decisions: push-not-pull (relay stays loopback-only), character speech never degrades to Web Speech, delivered:true only after station playback completes
- Awaiting HearthOS: play callback endpoint → HEARTHOS_PLAY_URL into relay PM2 env

## 2026-09-20 — va_bridge multi-device refactor for EchoMuse Dot (contract t:7d41c9e2)
- Contract with home-automation ratified + Eric-approved: Echo Dot 2 (EchoMuse, amonet v1.1.0/FireOS 5) as second voice endpoint; seam = ESPHome native API socket; full contract in F:\agent-share\relay\home-automation--voice-to-voice\
- va_bridge.py refactored: device roster in va_bridge_devices.json (static name); per-device task with capped-exponential reconnect (EchoMuse per-device port vanishes when Dot disconnects); mDNS discovery matches instance prefix `echomuse-` AND port range 16001-16999 (never service type alone — their BLE proxy shares _esphomelib._tcp on 17001+); device id sent on every /api/converse POST; per-device reply URLs /reply/<id>.wav (+legacy /reply.wav); per-device reply_gain (3.0 Atom Echo / 1.0 Dot until tuned)
- server.py /api/converse: accepts+logs `device` form field; per-device session state deferred to the NUC state-server milestone (ratified)
- Key decisions: keep full HA event superset (EchoMuse uses STT_END{text} to stop its mic feed — trimming would reintroduce their issue #343); ERROR→RUN_END kept (handled as dead turn); end-of-speech decision stays bridge-side (codified in contract)
- Smoke-tested: compile OK, HTTP up on :8222, both device tasks retry correctly with no hardware present. Real pairing awaits their controller.

## 2026-09-21 — Dot first live turns + audible failure cues
- Dot attached: mDNS matched echomuse-lf0964130r0g at 192.168.1.48:16001, connected+subscribed first try; atom-echo-bridge autostart ON. First two live turns ran the full pipe (Dot spoke a reply in the room).
- Defect found, theirs: utterance audio truncated at front (turn 2 "jarvis what time is it" arrived as "is it."; turn 1 5.12s → empty transcript → 422). Evidence posted on relay; fix = controller pre-roll from wake timestamp.
- Added audible failure cues: on converse 422 (no speech) / other failure, serve pre-rendered cue_no_speech.wav / cue_engine_fail.wav via the NORMAL TTS_END path (deliberately no ERROR event first — controller's ERROR handler sets turn waiters and could race the fetch). Falls back to silent ERROR+RUN_END if cue files missing.
- Known infra bug hit: nuc-exec-bridge drops command arguments (echo returns bare newline) — do not trust its results; reported to Eric.

## 2026-09-21 — M1 COMPLETE: Echo Dot live voice endpoint
- On-device wake (their flip) fixed front-truncation; full 5.2s sentences arrive intact.
- Eric enrolled through the Dot mic via the voice-only flow (print now 5 clips/54.7s); next turn identified Eric(0.60) and replied in his Jerma character voice. Full acceptance path proven: wake -> capture -> ECAPA reco -> personalized reply -> Dot playback, all local.
- Barge-in/'Thank you.' incident root-caused: controller barge (likely self-hearing, their AEC investigation) + Whisper hallucination on near-silence; mitigated by d67c719 anti-hallucination gate.
- Next: M2 latency (LLM dominates, 10-15s totals); their AEC + rings answers; custom wake phrase theirs.

## 2026-09-21 — Recognition robustness: best-clip matching + device sticky sessions
- speaker_id: identify() now scores each speaker by their BEST single clip, not the mean centroid — cross-mic enrollment (browser/Atom Echo/Dot) no longer dilutes; Eric measured 0.5498 vs 0.55 threshold downstairs under the centroid.
- Device sticky sessions (Eric's design): once recognized at full threshold on a device, next turns within 300s accept the SAME top-matching speaker down to 0.40 — stops mid-conversation voice flip-flop. Sticky never applies to a non-top-match; fresh sessions need full threshold; noise scores ~0.05-0.09.
- /api/converse log line now carries dev=<device id> (bridge sends it since 921ff94; server restart finally picked the field up).

## 2026-09-22T20:03:04Z — Voice channel P1: /api/voice/deliver + bridge announce ingress (briefing-table seam)
- Contract negotiated + ratified with the briefing-table agent on the cross-host relay
  (F:\agent-share\relay\briefing-table--voice-to-voice, t:330a518a); consensus doc is
  docs/voice-channel.md in briefing-table (vk-1704), awaiting Eric's approval.
- server.py: bearer auth (voice_auth.json, git-ignored, hot-reload, fail-closed, NO loopback
  exemption), GET /api/voice/devices, POST /api/voice/deliver (speaker-sid -> character voice,
  paraphrase off|subtle|full, earcon prefix, replyId idempotency, per-device lock, truthful
  played_ms after playback, structured errors + status map). voice_devices.json device table.
  tools/mint_voice_token.py.
- va_bridge.py: live-bridge registry, run/busy tracking (_run_end helper owns every RUN_END),
  announce() with wait-for-idle + start_conversation, POST /announce/<device> (loopback-only
  unless announce_token), GET /devices.
- Gate PASSED via https://vrpc-3.tail567253.ts.net: 401/404/413 paths, live delivery on the
  Dot (played_ms 4936, synth 2.6 s), replay of the same replyId returned cached, no re-play.
- Decision (both agents, Eric to confirm): v2v stays canonical on the VRPC; this work first.

## 2026-09-23T00:12:05Z — Door announcements: HA Welcome Home automations -> Echo Dot (MQTT announcer + cooldown)
- /api/voice/deliver: explicit `character` param (733afa0); `cooldownKey`/`cooldownSeconds` -> 429 cooldown,
  no speech; deliver core split from the HTTP wrapper (7e2ab5c).
- MQTT announcer thread in server.py: subscribes home/voice/announce on NUC mosquitto (100.110.14.59:1883),
  runs the deliver core as caller "mqtt", acks on home/voice/announce/ack/<replyId>. Config: voice_devices.json
  `voice_mqtt`. paho-mqtt added to the venv. tools/mqtt_announce_test.py = the HA-shaped round trip.
- Home Assistant (HAOS box 192.168.1.66; edited as claude-ops over ssh hazwebserver-claude-ops with the
  provisioned HA_TOKEN per the `home-assistant` skill, installed on this host at user scope): the five
  "Welcome Home <name>" automations now do: delay 2s -> if person matches -> mqtt.publish deliver body
  ("<Name> is home.", character jerma, cooldownKey door-<slug>, 600s) -> wait_for_trigger on the ack (30s)
  -> AEOTec doorbell param 6 ONLY if no ack (v2v unreachable). Originals backed up on the NUC at
  ~claude-ops/ha-backups/<id>.<ts>.json. Verified: MQTT round trip (spoke, then 429 on repeat) and a real
  HA trigger with the person sensor mocked to Emily -> Dot said "Emily is home." (played 2280 ms).
- Why MQTT not rest_command: HA's REST token cannot write configuration.yaml; mqtt.publish is built in.

## 2026-09-23T00:58:36Z — Voice channel P3: development-mode branch in /api/converse (inbound leg)
- Device table `mode: development` (toggle: POST /api/voice/devices/<id>/mode, bearer) routes a turn
  STT -> speaker gate (DEV_TURN_MIN_CONF 0.50, unknown never forwards) -> POST engine inbound
  (voice_engine in voice_devices.json; bearer read per call from secrets/briefing-table-engine.token)
  -> spoken ack (cached per voice+line). Outcomes, all audible: sent / held / no_channel / refused_speaker /
  refused_engine (403) / channel_gone (404) / engine_down. X-Route + X-Utterance-Id headers; [dev-turn] log
  line per turn for the P4 miss log. Never falls back to the chat LLM. Local intents (enrollment,
  identity) return before the branch and never reach the engine.
- tools/engine_inbound_stub.py stands in for the engine's P2 endpoint. Gate PASSED against it with real
  audio: Eric's Dot-mic enrollment clip -> dev:sent (engine 141 ms); Tina's clip -> dev:refused_engine;
  synthesized voice (0.10) -> dev:refused_speaker; stub down -> dev:engine_down. Device restored to chat.
