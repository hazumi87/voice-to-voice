"""
ESPHome <-> v2v bridge (NO Home Assistant, NO new AI).

This is a thin PROTOCOL ADAPTER. The Atom Echo's voice path speaks the ESPHome
native API (binary, port 6053); v2v speaks HTTP. This connects to the device as
an aioesphomeapi client, receives the streamed utterance after the on-device
wake word, and hands it to the EXISTING v2v /api/converse (STT->Ollama->TTS).
The reply WAV is served back over HTTP for the device's media_player to fetch.

All STT/LLM/TTS stays in v2v. This file adds only the translation glue.

Verified against aioesphomeapi 45.x (subscribe_voice_assistant / handle_audio /
send_voice_assistant_event). Run on the VRPC:  python va_bridge.py
"""
import asyncio
import audioop
import io
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

DEVICE_HOST = "atom-echo-1.local"  # mDNS name — survives DHCP lease changes
                                   # (was hardcoded .19; device moved to .18 after a reboot)
DEVICE_PORT = 6053
CONVERSE_URL = "http://127.0.0.1:8221/api/converse"   # existing v2v endpoint

# Must be the VRPC LAN IP, reachable FROM the device (not 127.0.0.1).
SERVER_PUBLIC_IP = "192.168.1.35"
SERVER_PORT = 8222

MIC_RATE = 16000                  # device streams 16kHz mono 16-bit PCM (raw)

# Server-side end-of-speech detection: the device streams continuously and
# relies on us to decide when the user stopped. webrtcvad is robust at the low
# SNR this mic produces (noise floor ~50 RMS) where a fixed RMS threshold can't
# separate speech from room noise.
VAD_FRAME_MS = 20
VAD_FRAME_BYTES = MIC_RATE * 2 * VAD_FRAME_MS // 1000     # 16k*2B*20ms = 640B
VAD_AGGRESSIVENESS = 2            # 0..3
SILENCE_HANG = 0.8               # seconds of non-speech after speech => end
NO_SPEECH_TIMEOUT = 10.0          # give up if no speech ever detected
MAX_UTTER = 15.0                  # hard cap (backstop against runaway background noise)

# Energy gate to ignore far-field background talk (e.g. a TV) that webrtcvad would
# otherwise score as continuous speech and never let the turn end. A frame only
# counts as "still talking" if it's speech AND loud enough relative to THIS
# speaker's peak level — so a close, loud user keeps the turn open, but quiet
# room/TV audio doesn't. Adaptive (frac of running peak) so soft talkers aren't cut.
VOICED_ABS_FLOOR = 60            # absolute rms floor; below this is never "voiced"
VOICED_PEAK_FRAC = 0.18         # ...and must clear this fraction of the running peak

REPLY_GAIN = 3.0                  # boost for the external MAX98357 amp+speaker.
                                  # Was 5.0 for the QUIET onboard speaker (x5 hard-clips
                                  # -> rattle); 3.0 is louder than 1.5 without the buzz.
                                  # Lower toward 1.5 if it starts distorting.

_latest_reply_wav: bytes | None = None


def amplify_wav(wav_bytes: bytes, factor: float) -> bytes:
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
    def __init__(self, client: APIClient):
        self.client = client
        self._buf = bytearray()
        self._loop = asyncio.get_event_loop()
        self._audio_chunks = 0
        self._processing = False
        self._watchdog = None
        self._vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self._frame_rem = bytearray()    # leftover bytes not yet a full VAD frame
        self._speech_frames = 0
        self._peak_rms = 0.0             # running loudness peak for the energy gate

    async def handle_start(self, conversation_id, flags, audio_settings, wake_word_phrase):
        print(f"[start] conv={conversation_id} flags={VoiceAssistantCommandFlag(flags)!r} "
              f"wake='{wake_word_phrase}'", flush=True)
        self._buf = bytearray()
        self._frame_rem = bytearray()
        self._audio_chunks = 0
        self._speech_frames = 0
        self._peak_rms = 0.0
        self._processing = False
        self._heard_speech = False
        now = self._loop.time()
        self._start_t = now
        self._last_voice_t = now
        self.client.send_voice_assistant_event(EV.VOICE_ASSISTANT_RUN_START, {})
        self.client.send_voice_assistant_event(EV.VOICE_ASSISTANT_STT_START, {})
        if self._watchdog:
            self._watchdog.cancel()
        self._watchdog = asyncio.create_task(self._end_watchdog())
        return 0   # 0/None => API-audio path (audio arrives via handle_audio)

    async def handle_audio(self, audio: bytes, audio2=None):
        if self._processing:
            return
        self._audio_chunks += 1
        self._buf.extend(audio)
        # Reframe the stream into fixed 20ms frames for webrtcvad.
        self._frame_rem.extend(audio)
        now = self._loop.time()
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
                if not self._heard_speech and self._speech_frames >= 3:
                    self._heard_speech = True
                    print("[audio] speech started", flush=True)
        if self._audio_chunks % 25 == 0:
            rms = audioop.rms(audio, 2)
            print(f"[audio] {self._audio_chunks} chunks, {len(self._buf)}B, rms={rms}", flush=True)

    async def _end_watchdog(self):
        # End-of-speech = amplitude silence after speech (device streams forever,
        # so WE decide when the user stopped). Also caps on no-speech / max length.
        try:
            while True:
                await asyncio.sleep(0.1)
                if self._processing:
                    return
                now = self._loop.time()
                if not self._heard_speech:
                    if now - self._start_t > NO_SPEECH_TIMEOUT:
                        print("[watchdog] no speech detected — aborting", flush=True)
                        await self._abort()
                        return
                    continue
                silence = now - self._last_voice_t
                if silence > SILENCE_HANG or (now - self._start_t) > MAX_UTTER:
                    why = "silence" if silence > SILENCE_HANG else "maxlen"
                    print(f"[watchdog] end-of-utterance ({why}, {len(self._buf)}B)", flush=True)
                    await self._end_and_process()
                    return
        except asyncio.CancelledError:
            pass

    async def _abort(self):
        self._processing = True
        self.client.send_voice_assistant_event(EV.VOICE_ASSISTANT_STT_VAD_END, {})
        self.client.send_voice_assistant_event(EV.VOICE_ASSISTANT_RUN_END, {})

    async def handle_stop(self, server_side: bool):
        n = len(self._buf)
        print(f"[stop] server_side={server_side} bytes={n} ({n/32000.0:.2f}s)", flush=True)
        await self._end_and_process()

    async def _end_and_process(self):
        if self._processing:
            return
        self._processing = True
        if self._watchdog:
            self._watchdog.cancel()
        # Tell the device the user stopped talking so it stops streaming mic audio.
        self.client.send_voice_assistant_event(EV.VOICE_ASSISTANT_STT_VAD_END, {})
        n = len(self._buf)
        if n < 1600:
            print(f"[end] too little audio ({n}B), aborting", flush=True)
            self.client.send_voice_assistant_event(EV.VOICE_ASSISTANT_RUN_END, {})
            return
        pcm = bytes(self._buf)
        self._buf = bytearray()
        asyncio.create_task(self._process(pcm))

    async def _process(self, pcm: bytes):
        global _latest_reply_wav
        wav = pcm_to_wav(pcm)

        def _post():
            return requests.post(
                CONVERSE_URL,
                files={"audio": ("utterance.wav", wav, "audio/wav")},
                timeout=120,
            )
        try:
            resp = await self._loop.run_in_executor(None, _post)
            resp.raise_for_status()
        except Exception as e:  # noqa: BLE001
            print(f"[converse] FAILED: {e!r}", flush=True)
            self.client.send_voice_assistant_event(
                EV.VOICE_ASSISTANT_ERROR, {"code": "converse_failed", "message": str(e)})
            self.client.send_voice_assistant_event(EV.VOICE_ASSISTANT_RUN_END, {})
            return

        transcript = unquote(resp.headers.get("X-Transcript", ""))
        reply_text = unquote(resp.headers.get("X-Reply", ""))
        # The server flags a reply that expects an immediate spoken answer (the guest
        # enrollment sub-dialog: "tell me your name" / "talk for 10s"). We honor it with
        # the device's native continue-conversation, so the mic re-opens with NO wake
        # word and NO button press — a voice-only protocol.
        followup = resp.headers.get("X-Followup-Listen", "0") == "1"
        try:
            _latest_reply_wav = amplify_wav(resp.content, REPLY_GAIN)
        except Exception as e:  # noqa: BLE001
            print(f"[amp] failed ({e!r}), serving original", flush=True)
            _latest_reply_wav = resp.content
        print(f"[converse] heard='{transcript}' reply='{reply_text}' "
              f"wav={len(resp.content)}B (x{REPLY_GAIN} gain) followup={followup}", flush=True)

        self.client.send_voice_assistant_event(EV.VOICE_ASSISTANT_STT_END, {"text": transcript})
        self.client.send_voice_assistant_event(EV.VOICE_ASSISTANT_INTENT_START, {})
        self.client.send_voice_assistant_event(EV.VOICE_ASSISTANT_INTENT_END, {})
        url = f"http://{SERVER_PUBLIC_IP}:{SERVER_PORT}/reply.wav"

        if followup:
            # Close THIS pipeline run without playing TTS through it (so the reply isn't
            # played twice), then use the announce path: it plays reply.wav AND re-opens
            # the mic (start_conversation=True) for the user's answer. The device then
            # starts a fresh voice_assistant run (flags=0, no wake word) -> handle_start.
            self.client.send_voice_assistant_event(EV.VOICE_ASSISTANT_RUN_END, {})
            try:
                await self.client.send_voice_assistant_announcement_await_response(
                    media_id=url, timeout=20.0, start_conversation=True)
                print(f"[followup] announced {url} + re-opened mic (voice-only)", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[followup] announce/continue failed: {e!r}", flush=True)
        else:
            self.client.send_voice_assistant_event(
                EV.VOICE_ASSISTANT_TTS_START, {"text": reply_text})
            self.client.send_voice_assistant_event(EV.VOICE_ASSISTANT_TTS_END, {"url": url})
            self.client.send_voice_assistant_event(EV.VOICE_ASSISTANT_RUN_END, {})
            print(f"[tts] handed url {url} to device", flush=True)


async def serve_reply(request: web.Request) -> web.Response:
    if _latest_reply_wav is None:
        return web.Response(status=404)
    return web.Response(body=_latest_reply_wav, content_type="audio/wav")


def _start_hc_heartbeat(interval=60):
    """Self-ping harbor's injected HC_PING_URL so harbor shows green (live dead-man).
    No-op when HC_PING_URL is unset (e.g. run outside harbor)."""
    import os
    import threading
    import time
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


async def main():
    _start_hc_heartbeat()
    app = web.Application()
    app.router.add_get("/reply.wav", serve_reply)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", SERVER_PORT).start()
    print(f"[http] reply.wav served on {SERVER_PUBLIC_IP}:{SERVER_PORT}", flush=True)

    # Connect in a retry loop so a transient mDNS hiccup or an offline device
    # never crash-exits the process (harbor would mark it down and the HC
    # heartbeat would stop). aioesphomeapi's own zeroconf resolver can time out
    # on '<name>.local', so pre-resolve to an IP via the OS resolver (Windows
    # resolves mDNS) and hand aioesphomeapi the IP.
    while True:
        client = None
        disconnected = asyncio.Event()

        async def _on_stop(expected: bool):
            # Fires when the device link drops. The pico's wifi is flaky
            # (WinError 121/64 mid-session); without this the loop parked on a
            # never-set Event and never recovered from a post-connect drop.
            print(f"[api] connection lost (expected={expected})", flush=True)
            disconnected.set()

        try:
            addr = _resolve(DEVICE_HOST)
            client = APIClient(address=addr, port=DEVICE_PORT, password="")
            await client.connect(on_stop=_on_stop, login=True)
            print(f"[api] connected to {addr}:{DEVICE_PORT}", flush=True)
            bridge = VoiceBridge(client)
            client.subscribe_voice_assistant(
                handle_start=bridge.handle_start,
                handle_stop=bridge.handle_stop,
                handle_audio=bridge.handle_audio,
            )
            print("[api] subscribed to voice_assistant (API-audio). Say 'okay nabu'.", flush=True)
            await disconnected.wait()      # park until the link drops, then reconnect
            print("[api] link dropped — reconnecting in 3s", flush=True)
            await asyncio.sleep(3)
        except Exception as e:  # noqa: BLE001
            print(f"[api] connect/run failed: {e!r}; retrying in 5s", flush=True)
            await asyncio.sleep(5)
        finally:
            if client is not None:
                try:
                    await client.disconnect()
                except Exception:  # noqa: BLE001
                    pass


def _resolve(host):
    """OS-resolve a '<name>.local' mDNS host to an IPv4 address.

    WHY this is not just socket.gethostbyname: on Windows, Python's
    socket.gethostbyname() does NOT use the mDNS / Windows DNS Client path for
    '.local' names -- it goes straight to the C library resolver and raises
    (or is flaky) for mDNS. Only the Windows DNS Client / .NET resolver
    (System.Net.Dns.GetHostAddresses) reliably resolves '<name>.local'.
    aioesphomeapi's own zeroconf resolver is also intermittently flaky here, so
    we pre-resolve to an IP and hand it the IP.

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


if __name__ == "__main__":
    asyncio.run(main())
