# Discoveries — Runtime & Server (voice-to-voice)

Durable record of bugs hit, what was tried, and what actually fixed them. Read this before
expanding the server or combining prototypes — several of these failures are *integration*
failures that will resurface when streams are merged. (`avatar/DISCOVERIES.md` is the
visualization-stream equivalent.)

---

## 1. The "crash" was a HANG, and the hang was a blocked stdout write (RESOLVED)

**Symptom (recurred for ~20 fix-attempts over hours):** after a TTS render the server stopped
responding — `/health` → 000 — but the process stayed alive, port 8221 stayed LISTENING, and VRAM
stayed held (~14 GB). It always hit during/just after a TTS render, and looked exactly like a
CUDA/VRAM deadlock.

**What we tried that did NOT fix it (all targeted a crash/OOM that wasn't happening):**
- `f35d741` make OmniVoice load non-fatal (stop a presumed crash-loop)
- `e362c32` defer OmniVoice load to first synth + VRAM precheck
- `b030812` evict Ollama (`keep_alive=0`) before loading OmniVoice
- `e08ae3b` `device_map=cuda:0` + `expandable_segments`
- Toggling the NVIDIA **CUDA Sysmem Fallback Policy** per-program for python (this never even
  applied cleanly — the venv launches through the WindowsApps *store* python, not the venv exe,
  so the per-program override pointed at the wrong binary; global stayed Driver Default the whole
  time). Reverted — it was never relevant.

**Root cause (proven, not theorized):** ran `py-spy dump` on the wedged process. The **only**
blocking frame was `logging.flush()` on the uvicorn asyncio event-loop thread (the per-request
access-log send). **Zero** threads were in `tts_model.generate` / torch / cuda. Mechanism:
- v2v's stdout/stderr is a pipe the supervisor (harbor) never drains.
- OmniVoice's `tqdm` progress bars flood that pipe on every render.
- The ~64 KB Windows pipe buffer fills; the next write **blocks forever**.
- uvicorn runs the whole app on ONE asyncio thread, so a blocked log write freezes the entire
  server. Port stays open, VRAM stays held → looks like a GPU hang.

**Fix (`be10e93`):**
- `TQDM_DISABLE=1` + `HF_HUB_DISABLE_PROGRESS_BARS=1` — kill the progress-bar flood.
- Redirect stdout/stderr to `working/server_v2v.log` at the **OS fd level** (`os.dup2`) so even
  native writes go to a file. A file write can't block on a full buffer the way an unread pipe
  can. (`V2V_LOG_TO_FILE=0` to opt out.)
- `uvicorn.run(access_log=False, log_level="warning")` — removes the exact wedging call.

**Verified:** drove many voices/characters back-to-back (the sequence that reliably wedged it) with
both models co-resident at 14 GB — 33+ renders, zero hangs.

**Lessons for the future:**
- **Any** long-lived server launched under a supervisor that pipes-but-doesn't-drain stdout is at
  risk. If we add another chatty library (or merge a prototype that prints a lot), it can refill
  this gap. Keep stdout/stderr redirected to a file, keep tqdm/progress bars disabled.
- When a process "hangs," **`py-spy dump --pid <pid>` FIRST**, before any theory. py-spy is
  installed in the venv. The wedged stack names the cause; guessing cost ~20 rounds here.
- Per-render diagnostics live in `working/character_speak.log` as `[synth] ...` lines
  (voice, generate path clone|instruct, cold/first-since-load, warmup coverage, tuning).

---

## 2. First render = default voice + "long pipe" corruption (RESOLVED)

**Symptom:** the FIRST render after a (re)start came out as the default voice AND sounded
hollow/corrupted ("spoken in a long pipe"); the second render onward was the correct voice.

**Two bugs, one trigger:**
1. **Wrong list checked.** The prototype synth endpoints validated the requested voice against
   `custom_prompts` (the lazy GPU clone-prompt cache, empty on a cold render) instead of
   `custom_voices` (the metadata list, always populated at startup). So a valid `cust_` id looked
   "unknown" on a cold render and was rewritten to `DEFAULT_VOICE` *before* `synth()` ran. Fixed
   `086d9de` — check `custom_voices` like `synth()` already does; `synth()` then lazily builds the
   clone prompt via `ensure_tts()`.
2. **Warmup missed the path actually used.** OmniVoice mis-renders its first `generate()` after a
   load. It has two paths — clone (`voice_clone_prompt=`) and instruct (`instruct=`) — and warming
   one doesn't warm the other. The old warmup only warmed clone; bug #1 forced the real render onto
   the unwarmed instruct path → cold artifact ("pipe"). Fixed `2135be8` — warm BOTH paths; log
   `warm_covered` per render.

**Note:** `speed`/`guidance` are passed natively into OmniVoice's `generate()` (not a post-hoc
resample), so they are NOT a corruption source — an early wrong guess.

---

## 3. Custom voices vanished from the list under deferred TTS (RESOLVED)

`build_clone_prompt` raised when the model wasn't loaded, and the voice was appended only *after*
that call — so with deferred load, voices silently dropped from `/api/voices`. Fixed `2e41a06` —
**list from metadata first**, build the GPU prompt only if the model is up (else lazily later).
Listing is decoupled from prompt-building.

---

## 4. TTS load strategy & fast first response (current design)

The deferred load was introduced during the crash panic (§1) and made the first render slow
(~30 s: model load + warmup). Now that the hang is fixed, load strategy is a free choice,
controlled by env on the harbor launch:

| Env | Behavior | Trade-off |
|-----|----------|-----------|
| `TTS_PREWARM=1` (default) | Load + warm OmniVoice in a **background daemon thread** | Server healthy immediately (GPU-free, avatar/phonetics don't wait); first render fast | 
| `TTS_PREWARM=0` | Pure lazy — load on first synth | First render pays the ~30 s load |
| `EAGER_TTS=1` | Blocking load at import | ~30 s startup; legacy |

Pre-warm is serialized with `ensure_tts()` via `_tts_load_lock` (a prompt mid-load waits on the
same load — no double load). `_post_load_setup()` is shared by lazy load and pre-warm so both get
identical prompt-rebuild + dual-path warmup. (`07b398c`)

---

## 5. Other deterministic server-side text fixes (in `character_speak`)

The small chat model (llama3.2:3b) ignores several persona instructions; we strip server-side:
- `*actions*` / short `(stage directions)` — TTS reads them literally (e.g. Jerma's
  `*beatboxing noise*`). Also removed un-vocalizable traits from styles.
- Comma before Singlish particles (`lah/leh/lor/...`) — OmniVoice pauses + mis-intones on the
  comma; strip it so the particle attaches to the preceding word.

---

## Watch-list for COMBINING PROTOTYPES (verbal × visual × future)

- **stdout discipline (§1):** any merged component that prints heavily, or any new progress-bar
  library, can refill the pipe gap. Keep the file redirect + tqdm-off. Re-check with `py-spy` on
  any new hang.
- **Single event-loop:** uvicorn serves everything on one asyncio thread. Any *synchronous*
  blocking call on a request path (a slow file write, a blocking subprocess, a lock) freezes the
  whole server — not just that request. Long/blocking work belongs in a thread (like pre-warm) or
  an async API.
- **VRAM co-residence:** OmniVoice (~8 GB) + Ollama (~3.4 GB) ≈ 11.4 GB live together fine on the
  16 GB card. Adding a third GPU model (e.g. a visual/3D pipeline) needs a VRAM budget check; keep
  the NVIDIA Sysmem Fallback at **Driver Default** (graceful spill, not hard OOM).
- **Cold-start warmup (§2):** any new generate path or new model needs its own warmup, or its first
  output will be cold. Watch the `[synth] warm_covered=` flag.
- **Lazy vs metadata lists (§2/§3):** when a resource has a "registered" list and a "built/cached"
  list, validate against the registered one and build lazily. Don't gate visibility on the cache.
