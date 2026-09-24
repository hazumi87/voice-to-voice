"""
ESPHome <-> v2v bridge (NO Home Assistant, NO new AI).

This is a thin PROTOCOL ADAPTER. A voice device speaks the ESPHome native API
(binary protobuf); v2v speaks HTTP. This connects to each device as an
aioesphomeapi client, receives the streamed utterance after the wake word, and
hands it to the EXISTING v2v /api/converse (STT->LLM->TTS). The reply WAV is
served back over HTTP for the device side to fetch by URL.

All STT/LLM/TTS stays in v2v. This file adds only the translation glue.

MULTI-DEVICE (2026-09-20, home-automation contract t:7d41c9e2): devices come
from va_bridge_devices.json (static name, same dir). Two discovery modes:
  - "static": fixed host(:port), e.g. a real ESPHome Atom Echo on :6053.
  - "mdns":   an EchoMuse-emulated satellite. Per-device ports start at 16001
              and are NEVER 6053; the listener only exists while the physical
              Dot is connected to the controller, so discovery re-runs on every
              reconnect. Match the mDNS INSTANCE NAME prefix "echomuse-" AND
              the 16001-16999 port range — never the service type alone: the
              EchoMuse BLE proxy also registers _esphomelib._tcp on 17001+.
EchoMuse peers are PLAINTEXT ONLY (their frame protocol actively rejects a
Noise preamble and closes). We connect with legacy password auth ("" default),
which is plaintext — same as the Atom Echo always used.

Event contract (ratified): emit the standard HA superset unchanged —
RUN_START, STT_START, STT_VAD_END, STT_END{text}, INTENT_START, INTENT_END,
TTS_START, TTS_END{url}, RUN_END. EchoMuse's handler has no else branch (extras
fall through silently) and STT_END{text} is actively USED to stop its mic feed,
so do NOT trim. ERROR{code,message}->RUN_END is handled as a dead turn.
End-of-speech decision is OURS (webrtcvad + energy gate below); a controller
end=True chunk is honored as a hard stop.

Verified against aioesphomeapi 45.x (subscribe_voice_assistant / handle_audio /
send_voice_assistant_event). Run on the VRPC:  python va_bridge.py
"""
import asyncio
import audioop
import io
import json
import os
import random
import time
import wave
from urllib.parse import unquote

import requests
import webrtcvad
from aiohttp import web

from aioesphomeapi import APIClient
from aioesphomeapi.model import (
    VoiceAssistantAudioSettings,
    VoiceAssistantCommandFlag,
    VoiceAssistantEventType as EV,
)

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "va_bridge_devices.json")

MIC_RATE = 16000                  # devices stream 16kHz mono 16-bit PCM (raw)

# Server-side end-of-speech detection: the device streams continuously and
# relies on us to decide when the user stopped. webrtcvad is robust at the low
# SNR these mics produce (noise floor ~50 RMS) where a fixed RMS threshold
# can't separate speech from room noise.
VAD_FRAME_MS = 20
VAD_FRAME_BYTES = MIC_RATE * 2 * VAD_FRAME_MS // 1000     # 16k*2B*20ms = 640B
VAD_AGGRESSIVENESS = 2            # 0..3
# Turn timing DEFAULTS; each device may override them in va_bridge_devices.json with
# "silence_hang_s", "no_speech_timeout_s", "max_utterance_s" (Eric, 2026-09-24: a
# development-mode user may dictate for a while, so the Dot runs with a long cap).
SILENCE_HANG = 0.8               # seconds of non-speech after speech => end
NO_SPEECH_TIMEOUT = 10.0          # give up if no speech ever detected
MAX_UTTER = 15.0                  # hard cap (backstop against runaway background noise)
# Echo-settle guard for the followup re-arm (guest enrollment capture). After we play a
# reply via the announce path and re-open the mic (start_conversation=True), the loud
# reply + listen chirp can still be ringing in the room — on the Atom Echo the mic and
# speaker sit inches apart on one shared I2S bus, and at 15dB amp gain the bleed is
# enough to self-trigger the VAD. That captured ~1s of the device's OWN audio, then
# closed the turn on silence BEFORE the user spoke (every enrollment clip came in at
# 1.0s -> "too short" loop). So for the run that immediately follows a followup, drop
# the first FOLLOWUP_GUARD_S of audio: let the speaker tail/echo decay so only the
# user's real speech is captured.
FOLLOWUP_GUARD_S = 0.9
# Listen cue: mic audio is dropped while the beep plays (until the device confirms it
# finished), never longer than LISTEN_CUE_MAX_S; LISTEN_CUE_TAIL_S covers the room decay.
LISTEN_CUE_MAX_S = 2.0
LISTEN_CUE_TAIL_S = 0.08

# Energy gate to ignore far-field background talk (e.g. a TV) that webrtcvad would
# otherwise score as continuous speech and never let the turn end. A frame only
# counts as "still talking" if it's speech AND loud enough relative to THIS
# speaker's peak level — so a close, loud user keeps the turn open, but quiet
# room/TV audio doesn't. Adaptive (frac of running peak) so soft talkers aren't cut.
# Measured on the Dot mic 2026-09-23 (Eric's enrollment clips vs a silent capture):
# speech frames p10~100-770 / median ~1800 / p90 3000-5000; room noise 44-98. The old
# floor of 60 sat INSIDE the noise band, so after the NUC/VRPC reboot three frames of
# hiss counted as "speech started" the instant the stream opened, the 0.8 s silence
# hang closed the turn before Eric spoke, and Whisper hallucinated a phrase from the
# empty clip ("Okay", "check.", "Stay warm."). 200 is 2x above the loudest noise
# frame and ~9x below median speech.
VOICED_ABS_FLOOR = 200           # absolute rms floor; below this is never "voiced"
VOICED_PEAK_FRAC = 0.18         # ...and must clear this fraction of the running peak
SPEECH_START_FRAMES = 6         # voiced 20 ms frames (120 ms) before a turn counts as speech

# Reconnect backoff per device (C7: the EchoMuse per-device port is torn down
# whenever the physical Dot disconnects — never assume-always-listening).
BACKOFF_MIN = 3.0
BACKOFF_MAX = 60.0

MDNS_SERVICE = "_esphomelib._tcp.local."

# Per-device latest reply WAV, served at /reply/<device_id>.wav. One slot per
# device is correct for the batch (M1) contract: one turn in flight per device.
_reply_store: dict[str, bytes] = {}
_last_reply_device: str | None = None    # for the legacy /reply.wav route

# Live bridges by device id (P1 voice-channel, 2026-09-22): the /announce ingress
# needs the CONNECTED VoiceBridge for a device to push a reply that arrives minutes
# after the turn that asked for it. A device with no entry here is offline.
_bridges: dict[str, "VoiceBridge"] = {}


def _wav_seconds(b: bytes) -> float:
    """Duration of a WAV payload (0.0 if unparseable). Used to estimate how long the
    device is busy playing a TTS_END reply we can't await."""
    try:
        with wave.open(io.BytesIO(b), "rb") as w:
            return w.getnframes() / float(w.getframerate() or 1)
    except Exception:  # noqa: BLE001
        return 0.0

# Audible failure cues (pre-rendered, static files next to this script). A dead
# turn used to be SILENT (ERROR+RUN_END) and Eric read the silence as "still
# thinking" (first live test, 2026-09-21). When a cue exists we instead serve it
# through the NORMAL TTS_END path — deliberately NOT sending ERROR first, because
# the controller's ERROR handler sets its turn waiters and could race the fetch.
_BRIDGE_DIR = os.path.dirname(os.path.abspath(__file__))
def _load_cue(name: str) -> bytes | None:
    try:
        with open(os.path.join(_BRIDGE_DIR, name), "rb") as f:
            return f.read()
    except OSError:
        return None
_CUE_NO_SPEECH = _load_cue("cue_no_speech.wav")      # engine 422: nothing transcribed
_CUE_LISTEN = _load_cue("cue_listen.wav")        # soft bass beep = "speak now" (mic open)
_CUE_ENGINE_FAIL = _load_cue("cue_engine_fail.wav")  # engine down/other failure


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not cfg.get("devices"):
        raise SystemExit(f"no devices configured in {CONFIG_PATH}")
    return cfg


def amplify_wav(wav_bytes: bytes, factor: float) -> bytes:
    if factor == 1.0:
        return wav_bytes
    with wave.open(io.BytesIO(wav_bytes), "rb") as r:
        params = r.getparams()
        frames = r.readframes(r.getnframes())
    louder = audioop.mul(frames, params.sampwidth, factor)   # multiplies + clips
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setparams(params)
        w.writeframes(louder)
    return out.getvalue()


def pcm_to_wav(pcm: bytes, rate: int = MIC_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


class VoiceBridge:
    def __init__(self, client: APIClient, dev: dict, cfg: dict):
        self.client = client
        self.dev_id = dev["id"]
        self.reply_gain = float(dev.get("reply_gain", 1.0))
        self.converse_url = cfg["converse_url"]
        self.reply_url = (f"http://{cfg['server_public_ip']}:{cfg['server_port']}"
                          f"/reply/{self.dev_id}.wav")
        self._buf = bytearray()
        self._loop = asyncio.get_event_loop()
        self._audio_chunks = 0
        self._processing = False
        self._watchdog = None
        self._vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self._frame_rem = bytearray()    # leftover bytes not yet a full VAD frame
        self._speech_frames = 0
        self._peak_rms = 0.0             # running loudness peak for the energy gate
        self._next_run_guard = 0.0       # echo-settle guard to apply to the NEXT run
        self._guard_until = 0.0          # monotonic time until which to drop input audio
        # Voice-channel delivery (P1): a pushed reply must never play over a capture or
        # over another reply. _run_active spans RUN_START..RUN_END; _busy_until covers
        # the TTS_END playback we hand the device by URL and cannot await.
        self._run_active = False
        self._busy_until = 0.0
        self._announce_lock = asyncio.Lock()
        self.announce_key = f"{self.dev_id}-announce"
        self.announce_url = (f"http://{cfg['server_public_ip']}:{cfg['server_port']}"
                             f"/reply/{self.announce_key}.wav")
        # Thinking-filler library (Eric, 2026-09-21): short in-character utterances
        # ("umm...", "let me think...") played via the announce path DURING the
        # engine round trip, so the think-gap isn't dead silence. Plays only after
        # capture ends (never over the user's speech) and no filler contains the
        # wake phrase (phrase-free playback proven barge-safe even pre-AEC-fix).
        # Listen cue (Eric, 2026-09-24): a soft bass beep announced the moment the mic
        # opens, so the user knows when to speak without watching the ring. Per-device
        # "listen_cue": false disables it. Mic audio is dropped until the device reports
        # the cue finished, so the beep is never in the clip and can't trip speech-start.
        self.listen_cue = bool(dev.get("listen_cue", True))
        self.silence_hang = float(dev.get("silence_hang_s", SILENCE_HANG))
        self.no_speech_timeout = float(dev.get("no_speech_timeout_s", NO_SPEECH_TIMEOUT))
        self.max_utter = float(dev.get("max_utterance_s", MAX_UTTER))
        self.cue_key = f"{self.dev_id}-cue"
        self.cue_url = (f"http://{cfg['server_public_ip']}:{cfg['server_port']}"
                        f"/reply/{self.cue_key}.wav")
        self._fillers: list[bytes] = []
        self._last_filler = -1
        self.filler_key = f"{self.dev_id}-filler"
        self.filler_url = (f"http://{cfg['server_public_ip']}:{cfg['server_port']}"
                           f"/reply/{self.filler_key}.wav")
        fset = dev.get("filler_set", "")
        if fset:
            fdir = os.path.join(_BRIDGE_DIR, "fillers", fset)
            if os.path.isdir(fdir):
                for fn in sorted(os.listdir(fdir)):
                    if fn.endswith(".wav"):
                        try:
                            with open(os.path.join(fdir, fn), "rb") as f:
                                raw = f.read()
                            self._fillers.append(amplify_wav(raw, self.reply_gain))
                        except Exception:  # noqa: BLE001
                            pass
            self._log(f"[filler] set '{fset}': {len(self._fillers)} clip(s)")

    def _log(self, msg: str):
        print(f"[{self.dev_id}] {msg}", flush=True)

    def _run_end(self, busy_wav: bytes | None = None):
        """Send RUN_END and mark the pipeline run closed. If we just handed the device a
        reply WAV by URL (TTS_END path), estimate how long it will be playing so a
        pushed announcement waits instead of talking over it."""
        self.client.send_voice_assistant_event(EV.VOICE_ASSISTANT_RUN_END, {})
        self._run_active = False
        if busy_wav is not None:
            self._busy_until = self._loop.time() + _wav_seconds(busy_wav) + 0.5

    def busy(self) -> bool:
        return self._run_active or self._loop.time() < self._busy_until

    async def announce(self, wav: bytes, start_conversation: bool = False,
                       timeout: float = 30.0) -> dict:
        """Push a WAV to the device via the announce path and report TRUTHFULLY:
        {delivered:true, played_ms} only after playback finished; otherwise
        {delivered:false, error}. Waits (up to `timeout`) for any capture/playback in
        progress to end first; never interrupts the user. Serialized per device."""
        t_wait0 = self._loop.time()
        while self.busy():
            if self._loop.time() - t_wait0 > timeout:
                return {"delivered": False, "error": "device_busy"}
            await asyncio.sleep(0.1)
        async with self._announce_lock:
            try:
                _reply_store[self.announce_key] = amplify_wav(wav, self.reply_gain)
            except Exception:  # noqa: BLE001
                _reply_store[self.announce_key] = wav
            if start_conversation:
                # The reply asked a question: the mic re-opens right after playback,
                # so arm the echo-settle guard exactly as the enrollment followup does.
                self._next_run_guard = FOLLOWUP_GUARD_S
            t0 = self._loop.time()
            try:
                await self.client.send_voice_assistant_announcement_await_response(
                    media_id=self.announce_url, timeout=timeout,
                    start_conversation=start_conversation)
            except Exception as e:  # noqa: BLE001
                self._log(f"[announce] FAILED ({e!r})")
                self._next_run_guard = 0.0
                return {"delivered": False, "error": "playback_failed", "detail": repr(e)}
            played_ms = int((self._loop.time() - t0) * 1000)
            self._log(f"[announce] played {len(wav)}B in {played_ms}ms "
                      f"start_conversation={int(start_conversation)}")
            return {"delivered": True, "played_ms": played_ms}

    async def handle_start(self, conversation_id, flags, audio_settings, wake_word_phrase):
        self._log(f"[start] conv={conversation_id} flags={VoiceAssistantCommandFlag(flags)!r} "
                  f"wake='{wake_word_phrase}'")
        self._buf = bytearray()
        self._frame_rem = bytearray()
        self._audio_chunks = 0
        self._speech_frames = 0
        self._peak_rms = 0.0
        self._processing = False
        self._heard_speech = False
        self._run_active = True
        now = self._loop.time()
        self._start_t = now
        self._last_voice_t = now
        # Consume any echo-settle guard armed by the preceding followup re-arm.
        self._guard_until = now + self._next_run_guard
        if self._next_run_guard:
            self._log(f"[guard] dropping first {self._next_run_guard:.1f}s "
                      f"(echo-settle after followup)")
        self._next_run_guard = 0.0
        self.client.send_voice_assistant_event(EV.VOICE_ASSISTANT_RUN_START, {})
        self.client.send_voice_assistant_event(EV.VOICE_ASSISTANT_STT_START, {})
        if self.listen_cue and _CUE_LISTEN:
            # Hold the mic closed (drop audio) until the cue has played; capped so a
            # cue that never confirms can't eat the whole turn.
            self._guard_until = max(self._guard_until, now + LISTEN_CUE_MAX_S)
            asyncio.create_task(self._play_listen_cue())
        if self._watchdog:
            self._watchdog.cancel()
        self._watchdog = asyncio.create_task(self._end_watchdog())
        return 0   # 0/None => API-audio path (audio arrives via handle_audio; no UDP)

    async def _play_listen_cue(self):
        """Beep = 'speak now'. Announced over the open run like the filler is; when the
        device confirms playback the guard is released and the clip starts clean."""
        t0 = self._loop.time()
        try:
            _reply_store[self.cue_key] = _CUE_LISTEN
            await self.client.send_voice_assistant_announcement_await_response(
                media_id=self.cue_url, timeout=5.0)
            self._log(f"[cue] listen beep played in {int((self._loop.time() - t0) * 1000)}ms")
        except Exception as e:  # noqa: BLE001
            self._log(f"[cue] listen beep skipped ({e!r})")
        finally:
            # Release the guard: the user may already be talking over the beep's tail.
            self._guard_until = min(self._guard_until, self._loop.time() + LISTEN_CUE_TAIL_S)
            self._start_t = self._loop.time()    # no-speech timeout counts from "speak now"
            self._last_voice_t = self._start_t

    async def handle_audio(self, audio: bytes, audio2=None):
        if self._processing:
            return
        now = self._loop.time()
        # Echo-settle guard: drop input while the speaker tail/echo from a just-played
        # followup reply is still ringing, so it can't self-trigger the VAD and close the
        # turn before the user speaks. Drop BEFORE buffering so the echo isn't in the clip.
        if now < self._guard_until:
            return
        self._audio_chunks += 1
        self._buf.extend(audio)
        # Reframe the stream into fixed 20ms frames for webrtcvad.
        self._frame_rem.extend(audio)
        while len(self._frame_rem) >= VAD_FRAME_BYTES:
            frame = bytes(self._frame_rem[:VAD_FRAME_BYTES])
            del self._frame_rem[:VAD_FRAME_BYTES]
            try:
                is_speech = self._vad.is_speech(frame, MIC_RATE)
            except Exception:  # noqa: BLE001
                is_speech = False
            # Energy gate: only count a frame as "still talking" if it's both speech
            # and loud relative to this speaker's peak — rejects far-field TV/room
            # talk so the turn ends ~SILENCE_HANG after the close user stops.
            r = audioop.rms(frame, 2)
            if r > self._peak_rms:
                self._peak_rms = r
            loud_enough = r >= max(VOICED_ABS_FLOOR, VOICED_PEAK_FRAC * self._peak_rms)
            voiced = is_speech and loud_enough
            if voiced:
                self._speech_frames += 1
                self._last_voice_t = now
                if not self._heard_speech and self._speech_frames >= SPEECH_START_FRAMES:
                    self._heard_speech = True
                    self._log("[audio] speech started")
        if self._audio_chunks % 25 == 0:
            rms = audioop.rms(audio, 2)
            self._log(f"[audio] {self._audio_chunks} chunks, {len(self._buf)}B, rms={rms}")

    async def _end_watchdog(self):
        # End-of-speech = amplitude silence after speech (the device streams until told
        # otherwise, so WE decide when the user stopped — ratified: our side is the
        # decider; a controller end=True is a hard stop that lands in handle_stop).
        try:
            while True:
                await asyncio.sleep(0.1)
                if self._processing:
                    return
                now = self._loop.time()
                if not self._heard_speech:
                    if now - self._start_t > self.no_speech_timeout:
                        self._log("[watchdog] no speech detected — aborting")
                        await self._abort()
                        return
                    continue
                silence = now - self._last_voice_t
                if silence > self.silence_hang or (now - self._start_t) > self.max_utter:
                    why = "silence" if silence > self.silence_hang else "maxlen"
                    self._log(f"[watchdog] end-of-utterance ({why}, {len(self._buf)}B)")
                    await self._end_and_process()
                    return
        except asyncio.CancelledError:
            pass

    async def _abort(self):
        self._processing = True
        self.client.send_voice_assistant_event(EV.VOICE_ASSISTANT_STT_VAD_END, {})
        self._run_end()

    async def handle_stop(self, server_side: bool):
        n = len(self._buf)
        self._log(f"[stop] server_side={server_side} bytes={n} ({n/32000.0:.2f}s)")
        await self._end_and_process()

    def _dump_capture(self, pcm: bytes):
        try:
            d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
            last = os.path.join(d, "last_capture.wav")
            prev = os.path.join(d, "previous_capture.wav")
            if os.path.exists(last):
                os.replace(last, prev)
            with wave.open(last, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(MIC_RATE)
                w.writeframes(pcm)
            self._log(f"[capture] {len(pcm)}B ({len(pcm) / (MIC_RATE * 2):.2f}s, "
                      f"rms={audioop.rms(pcm, 2) if pcm else 0}) -> last_capture.wav")
        except Exception as e:  # noqa: BLE001 — never let a debug dump break a turn
            self._log(f"[capture] dump failed: {e!r}")

    async def _end_and_process(self):
        if self._processing:
            return
        self._processing = True
        if self._watchdog:
            self._watchdog.cancel()
        # Tell the device the user stopped talking so it stops streaming mic audio.
        self.client.send_voice_assistant_event(EV.VOICE_ASSISTANT_STT_VAD_END, {})
        n = len(self._buf)
        # Debug transparency: every captured turn lands at working/atom_echo/
        # last_capture.wav (previous_capture.wav = the one before), so a "it didn't
        # hear me" report can be diagnosed from the raw mic audio instead of rms guesses.
        self._dump_capture(bytes(self._buf))
        if n < 1600:
            self._log(f"[end] too little audio ({n}B), aborting")
            self._run_end()
            return
        # Anti-hallucination gate: if the whole turn never produced VAD-qualifying
        # speech, don't send it to STT — Whisper invents phrases ("Thank you.") on
        # near-silence, and a self-triggered barge-in turn is exactly such audio
        # (seen live 2026-09-21: 2.56s at rms 103-332 -> hallucinated turn). End
        # SILENTLY: no user spoke, so no one is waiting, and a spoken cue here
        # could re-trigger the mic in a loop.
        if not self._heard_speech:
            self._log(f"[end] no qualifying speech ({n}B, controller-ended) — "
                      f"dropping turn silently (anti-hallucination)")
            self._run_end()
            return
        pcm = bytes(self._buf)
        self._buf = bytearray()
        asyncio.create_task(self._process(pcm))

    async def _serve_cue(self, cue: bytes):
        """Speak a stock failure cue through the normal turn-completion path, so a
        failed turn is audible instead of silent. Completes the required event
        sequence (STT_VAD_END was already sent by _end_and_process)."""
        global _last_reply_device
        try:
            _reply_store[self.dev_id] = amplify_wav(cue, self.reply_gain)
        except Exception:  # noqa: BLE001
            _reply_store[self.dev_id] = cue
        _last_reply_device = self.dev_id
        self.client.send_voice_assistant_event(EV.VOICE_ASSISTANT_INTENT_START, {})
        self.client.send_voice_assistant_event(EV.VOICE_ASSISTANT_INTENT_END, {})
        self.client.send_voice_assistant_event(EV.VOICE_ASSISTANT_TTS_START, {})
        self.client.send_voice_assistant_event(
            EV.VOICE_ASSISTANT_TTS_END, {"url": self.reply_url})
        self._run_end(busy_wav=_reply_store.get(self.dev_id))
        self._log(f"[cue] served failure cue via {self.reply_url}")

    async def _play_filler(self):
        """Announce one random filler clip while the engine thinks. Best-effort:
        any failure just means silence, exactly what we had before."""
        try:
            idx = random.randrange(len(self._fillers))
            if len(self._fillers) > 1 and idx == self._last_filler:
                idx = (idx + 1) % len(self._fillers)
            self._last_filler = idx
            _reply_store[self.filler_key] = self._fillers[idx]
            await self.client.send_voice_assistant_announcement_await_response(
                media_id=self.filler_url, timeout=15.0)
            self._log(f"[filler] played clip #{idx}")
        except Exception as e:  # noqa: BLE001
            self._log(f"[filler] skipped ({e!r})")

    async def _process(self, pcm: bytes):
        global _last_reply_device
        wav = pcm_to_wav(pcm)

        # Kick the filler concurrently with the engine round trip; await it before
        # delivering the real reply so the two playbacks never overlap.
        filler_task = None
        if self._fillers:
            filler_task = asyncio.create_task(self._play_filler())

        def _post():
            return requests.post(
                self.converse_url,
                files={"audio": ("utterance.wav", wav, "audio/wav")},
                data={"device": self.dev_id},   # A6: device id on every request
                timeout=120,
            )
        try:
            resp = await self._loop.run_in_executor(None, _post)
            resp.raise_for_status()
        except Exception as e:  # noqa: BLE001
            self._log(f"[converse] FAILED: {e!r}")
            if filler_task:
                await asyncio.gather(filler_task, return_exceptions=True)
            status = getattr(getattr(e, "response", None), "status_code", None)
            cue = _CUE_NO_SPEECH if status == 422 else _CUE_ENGINE_FAIL
            if cue is not None:
                await self._serve_cue(cue)
            else:
                # No cue file on disk — fall back to the ratified silent dead turn
                # (ERROR unblocks the controller's waiters).
                self.client.send_voice_assistant_event(
                    EV.VOICE_ASSISTANT_ERROR,
                    {"code": "converse_failed", "message": str(e)})
                self._run_end()
            return

        if filler_task:
            await asyncio.gather(filler_task, return_exceptions=True)

        transcript = unquote(resp.headers.get("X-Transcript", ""))
        reply_text = unquote(resp.headers.get("X-Reply", ""))
        # Ring-hint contract (relay t:7d41c9e2): who the engine identified, carried
        # as extra data keys on TTS_START. The controller's event handler ignores
        # unknown keys, so this is safe against any controller version; theirs maps
        # sid -> scene colour. speaker="unknown"/sid="" means unidentified.
        spk = unquote(resp.headers.get("X-Speaker", "unknown"))
        spk_sid = resp.headers.get("X-Speaker-Sid", "")
        # The server flags a reply that expects an immediate spoken answer (the guest
        # enrollment sub-dialog: "tell me your name" / "talk for 10s"). We honor it with
        # the device's native continue-conversation, so the mic re-opens with NO wake
        # word and NO button press — a voice-only protocol. On EchoMuse this rides
        # VoiceAssistantAnnounce + start_conversation, which is only advertised when the
        # device reports the `mic` capability — home-automation verifies at pairing; the
        # try/except below degrades to a lost followup (wake word needed) if absent.
        followup = resp.headers.get("X-Followup-Listen", "0") == "1"
        try:
            _reply_store[self.dev_id] = amplify_wav(resp.content, self.reply_gain)
        except Exception as e:  # noqa: BLE001
            self._log(f"[amp] failed ({e!r}), serving original")
            _reply_store[self.dev_id] = resp.content
        _last_reply_device = self.dev_id
        self._log(f"[converse] heard='{transcript}' reply='{reply_text}' "
                  f"wav={len(resp.content)}B (x{self.reply_gain} gain) followup={followup}")

        self.client.send_voice_assistant_event(EV.VOICE_ASSISTANT_STT_END, {"text": transcript})
        self.client.send_voice_assistant_event(EV.VOICE_ASSISTANT_INTENT_START, {})
        self.client.send_voice_assistant_event(EV.VOICE_ASSISTANT_INTENT_END, {})

        if followup:
            # Close THIS pipeline run without playing TTS through it (so the reply isn't
            # played twice), then use the announce path: it plays reply.wav AND re-opens
            # the mic (start_conversation=True) for the user's answer. The device then
            # starts a fresh voice_assistant run (flags=0, no wake word) -> handle_start.
            self._run_end()
            # Arm the echo-settle guard for the capture run the device is about to start,
            # so the loud reply we're about to play doesn't self-trigger that run's mic.
            self._next_run_guard = FOLLOWUP_GUARD_S
            try:
                await self.client.send_voice_assistant_announcement_await_response(
                    media_id=self.reply_url, timeout=20.0, start_conversation=True)
                self._log(f"[followup] announced {self.reply_url} + re-opened mic (voice-only)")
            except Exception as e:  # noqa: BLE001
                self._log(f"[followup] announce/continue failed: {e!r}")
        else:
            self.client.send_voice_assistant_event(
                EV.VOICE_ASSISTANT_TTS_START,
                {"text": reply_text, "speaker": spk, "sid": spk_sid})
            self.client.send_voice_assistant_event(
                EV.VOICE_ASSISTANT_TTS_END, {"url": self.reply_url})
            self._run_end(busy_wav=_reply_store.get(self.dev_id))
            self._log(f"[tts] handed url {self.reply_url} to device")


async def serve_reply_for(request: web.Request) -> web.Response:
    dev_id = request.match_info["device_id"]
    wav = _reply_store.get(dev_id)
    if wav is None:
        return web.Response(status=404)
    return web.Response(body=wav, content_type="audio/wav")


def _ingress_allowed(request: web.Request, cfg: dict) -> bool:
    """The announce ingress can make the Dot say anything, so it is loopback-only
    (v2v on this box calls it) unless the roster configures `announce_token`, in which
    case a matching X-Announce-Token header is accepted from anywhere. :8222 is NOT
    behind Tailscale Serve, so a loopback check here is meaningful (unlike v2v's :8221)."""
    tok = (cfg.get("announce_token") or "").strip()
    if tok and request.headers.get("X-Announce-Token", "") == tok:
        return True
    return (request.remote or "") in ("127.0.0.1", "::1")


async def announce_for(request: web.Request) -> web.Response:
    """POST /announce/<device_id>?start_conversation=0|1&timeout=30  body: WAV bytes.
    Pushes the WAV to the device via the announce path. Truthful, synchronous reply:
    {delivered:true, played_ms} after playback, else {delivered:false, error}."""
    cfg = request.app["cfg"]
    if not _ingress_allowed(request, cfg):
        return web.json_response({"delivered": False, "error": "unauthorized"}, status=401)
    dev_id = request.match_info["device_id"]
    bridge = _bridges.get(dev_id)
    if bridge is None:
        known = any(d["id"] == dev_id for d in cfg.get("devices", []))
        return web.json_response(
            {"delivered": False, "error": "device_offline" if known else "device_unknown"},
            status=404)
    wav = await request.read()
    if len(wav) < 100:
        return web.json_response({"delivered": False, "error": "empty_wav"}, status=400)
    sc = request.query.get("start_conversation", "0") in ("1", "true", "yes")
    try:
        timeout = min(max(float(request.query.get("timeout", "30")), 1.0), 60.0)
    except ValueError:
        timeout = 30.0
    res = await bridge.announce(wav, start_conversation=sc, timeout=timeout)
    status = 200 if res.get("delivered") else (409 if res.get("error") == "device_busy" else 502)
    res["device"] = dev_id
    return web.json_response(res, status=status)


async def devices_list(request: web.Request) -> web.Response:
    """GET /devices -> roster with live connection state (for v2v's /api/voice/devices)."""
    cfg = request.app["cfg"]
    out = []
    for d in cfg.get("devices", []):
        b = _bridges.get(d["id"])
        out.append({"id": d["id"], "enabled": bool(d.get("enabled", True)),
                    "connected": b is not None,
                    "busy": bool(b.busy()) if b is not None else None})
    return web.json_response({"devices": out})


async def serve_reply_legacy(request: web.Request) -> web.Response:
    # Back-compat for anything still fetching /reply.wav: latest reply, any device.
    if _last_reply_device is None or _last_reply_device not in _reply_store:
        return web.Response(status=404)
    return web.Response(body=_reply_store[_last_reply_device],
                        content_type="audio/wav")


def _start_hc_heartbeat(interval=60):
    """Self-ping harbor's injected HC_PING_URL so harbor shows green (live dead-man).
    No-op when HC_PING_URL is unset (e.g. run outside harbor)."""
    import threading
    import urllib.request
    url = os.environ.get("HC_PING_URL")
    if not url:
        return
    def _loop():
        while True:
            try:
                urllib.request.urlopen(url, timeout=5).read()
            except Exception:  # noqa: BLE001
                pass
            time.sleep(interval)
    threading.Thread(target=_loop, daemon=True, name="hc-heartbeat").start()
    print(f"[hc] heartbeat -> {url} every {interval}s", flush=True)


def _mdns_find(prefix: str, exact: str, port_min: int, port_max: int,
               timeout: float = 6.0):
    """Browse _esphomelib._tcp for an EchoMuse satellite. BLOCKING — run in executor.

    Match the instance NAME prefix AND the voice port range, never the service
    type alone: the EchoMuse BLE proxy registers the same service type on 17001+.
    An exact instance name (from config, once known) pins a specific device.
    Returns (instance, addr, port) or None.
    """
    from zeroconf import Zeroconf, ServiceBrowser, ServiceListener

    found: dict[str, tuple[str, int]] = {}

    class _Listener(ServiceListener):
        def add_service(self, zc, type_, name):
            try:
                info = zc.get_service_info(type_, name, 3000)
            except Exception:  # noqa: BLE001
                info = None
            if info and info.parsed_addresses() and info.port:
                found[name] = (info.parsed_addresses()[0], info.port)
        def update_service(self, zc, type_, name):
            pass
        def remove_service(self, zc, type_, name):
            pass

    zc = Zeroconf()
    try:
        ServiceBrowser(zc, MDNS_SERVICE, _Listener())
        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(0.25)
            for name, (addr, port) in list(found.items()):
                instance = name.split("." + MDNS_SERVICE)[0].rstrip(".")
                if not (port_min <= port <= port_max):
                    continue
                if exact:
                    if instance == exact:
                        return instance, addr, port
                elif instance.startswith(prefix):
                    return instance, addr, port
        return None
    finally:
        zc.close()


def _resolve(host):
    """OS-resolve a '<name>.local' mDNS host to an IPv4 address.

    WHY this is not just socket.gethostbyname: on Windows, Python's
    socket.gethostbyname() does NOT use the mDNS / Windows DNS Client path for
    '.local' names -- it goes straight to the C library resolver and raises
    (or is flaky) for mDNS. Only the Windows DNS Client / .NET resolver
    (System.Net.Dns.GetHostAddresses) reliably resolves '<name>.local'.
    aioesphomeapi's own zeroconf resolver is also intermittently flaky here, so
    we pre-resolve to an IP and hand aioesphomeapi the IP.

    Resolution order: (a) Windows .NET resolver via a short powershell call,
    (b) socket.gethostbyname, (c) the host unchanged. Never raises; returns the
    host on total failure so the caller can still try."""
    import socket
    import subprocess
    import re

    if not host.endswith(".local"):
        return host

    # (a) Windows .NET / DNS Client resolver -- the only thing that reliably
    # resolves mDNS '.local' names on Windows.
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "[System.Net.Dns]::GetHostAddresses('%s') | "
             "ForEach-Object { $_.IPAddressToString }" % host],
            capture_output=True, text=True, timeout=3,
        ).stdout
        for line in out.splitlines():
            m = re.match(r"^\s*(\d{1,3}(?:\.\d{1,3}){3})\s*$", line)
            if m:
                return m.group(1)
    except Exception:  # noqa: BLE001 - powershell missing/timeout/etc.
        pass

    # (b) Plain stdlib resolver (works on non-Windows; sometimes works here).
    try:
        return socket.gethostbyname(host)
    except OSError:
        pass

    # (c) Give up gracefully; let aioesphomeapi try the raw hostname.
    return host


async def run_device(dev: dict, cfg: dict):
    """Own one device: discover -> connect -> subscribe -> park -> reconnect.

    Retry forever with capped exponential backoff (reset on a successful
    connect). For mdns devices the per-device port is torn down whenever the
    physical device leaves the controller, so discovery re-runs every attempt.
    """
    loop = asyncio.get_event_loop()
    dev_id = dev["id"]
    backoff = BACKOFF_MIN
    while True:
        client = None
        disconnected = asyncio.Event()

        async def _on_stop(expected: bool):
            # Fires when the device link drops. Without this the loop parked on
            # a never-set Event and never recovered from a post-connect drop.
            print(f"[{dev_id}] [api] connection lost (expected={expected})", flush=True)
            disconnected.set()

        try:
            if dev.get("discovery") == "mdns":
                hit = await loop.run_in_executor(
                    None, _mdns_find,
                    dev.get("mdns_instance_prefix", "echomuse-"),
                    dev.get("mdns_instance", ""),
                    int(dev.get("mdns_port_min", 16001)),
                    int(dev.get("mdns_port_max", 16999)))
                if hit is None:
                    print(f"[{dev_id}] [mdns] no matching satellite advertised "
                          f"(device offline or not paired yet); retry in {backoff:.0f}s",
                          flush=True)
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, BACKOFF_MAX)
                    continue
                instance, addr, port = hit
                print(f"[{dev_id}] [mdns] matched '{instance}' at {addr}:{port}", flush=True)
            else:
                addr = _resolve(dev["host"])
                port = int(dev.get("port", 6053))

            client = APIClient(address=addr, port=port,
                               password=dev.get("password", ""))
            await client.connect(on_stop=_on_stop, login=True)
            print(f"[{dev_id}] [api] connected to {addr}:{port}", flush=True)
            backoff = BACKOFF_MIN          # success resets the backoff
            bridge = VoiceBridge(client, dev, cfg)
            client.subscribe_voice_assistant(
                handle_start=bridge.handle_start,
                handle_stop=bridge.handle_stop,
                handle_audio=bridge.handle_audio,
            )
            print(f"[{dev_id}] [api] subscribed to voice_assistant (API-audio)", flush=True)
            _bridges[dev_id] = bridge      # announce ingress can reach this device now
            await disconnected.wait()      # park until the link drops, then reconnect
            print(f"[{dev_id}] [api] link dropped — reconnecting in {BACKOFF_MIN:.0f}s",
                  flush=True)
            await asyncio.sleep(BACKOFF_MIN)
        except Exception as e:  # noqa: BLE001
            print(f"[{dev_id}] [api] connect/run failed: {e!r}; retry in {backoff:.0f}s",
                  flush=True)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, BACKOFF_MAX)
        finally:
            _bridges.pop(dev_id, None)     # offline until the next successful subscribe
            if client is not None:
                try:
                    await client.disconnect()
                except Exception:  # noqa: BLE001
                    pass


async def main():
    cfg = load_config()
    _start_hc_heartbeat()
    app = web.Application()
    app["cfg"] = cfg
    app.router.add_get("/reply/{device_id}.wav", serve_reply_for)
    app.router.add_get("/reply.wav", serve_reply_legacy)
    app.router.add_post("/announce/{device_id}", announce_for)
    app.router.add_get("/devices", devices_list)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", cfg["server_port"]).start()
    print(f"[http] reply WAVs served on {cfg['server_public_ip']}:{cfg['server_port']} "
          f"(/reply/<device>.wav)", flush=True)

    enabled = [d for d in cfg["devices"] if d.get("enabled", True)]
    print(f"[bridge] {len(enabled)} device(s): " +
          ", ".join(f"{d['id']}({d.get('discovery','static')})" for d in enabled),
          flush=True)
    await asyncio.gather(*(run_device(d, cfg) for d in enabled))


if __name__ == "__main__":
    asyncio.run(main())
