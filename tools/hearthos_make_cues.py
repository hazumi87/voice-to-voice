"""Render HearthOS system voice cues via voice-to-voice /synthesize (OmniVoice).

Pipeline per cue:
  synth (24 kHz mono float WAV) -> trim edge silence -> resample 48 kHz ->
  normalize to -16 LUFS integrated (BS.1770 K-weighted, gated), peak capped
  at -1 dBFS -> pad 150 ms silence head/tail -> write 16-bit PCM WAV + 128k MP3.

Usage: python make_cues.py <cues.json> <outdir>
  cues.json: [{"key": "bt_connected", "text": "Speaker connected."}, ...]
"""
import io
import json
import sys
import urllib.request

import numpy as np
import soundfile as sf
from scipy.signal import lfilter, resample_poly

SYNTH_URL = "http://127.0.0.1:8221/synthesize"
VOICE = "cust_jerma_1781246722"
TUNING = {"speed": 1.0, "guidance": 2.0, "temperature": 0.0, "steps": 32}
TARGET_SR = 48000
TARGET_LUFS = -16.0
PEAK_CEIL = 10 ** (-1.0 / 20.0)  # -1 dBFS
PAD_MS = 150
TRIM_THRESH_DB = -45.0  # edge-silence trim threshold (relative to peak)


def synth(text: str) -> tuple[np.ndarray, int]:
    body = json.dumps({"text": text, "voice": VOICE, **TUNING}).encode()
    req = urllib.request.Request(
        SYNTH_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        wav_bytes = r.read()
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float64", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def trim_edges(x: np.ndarray, sr: int) -> np.ndarray:
    peak = np.max(np.abs(x)) or 1.0
    thresh = peak * 10 ** (TRIM_THRESH_DB / 20.0)
    # 10 ms RMS envelope so single-sample clicks don't defeat the trim
    win = max(1, sr // 100)
    env = np.sqrt(np.convolve(x * x, np.ones(win) / win, mode="same"))
    idx = np.where(env > thresh)[0]
    if len(idx) == 0:
        return x
    return x[idx[0]:idx[-1] + 1]


# --- ITU-R BS.1770-4 integrated loudness (mono) ---
def _k_weight(x: np.ndarray, sr: int) -> np.ndarray:
    # Stage 1: high-shelf (head effects). Coefficients per BS.1770 for 48 kHz;
    # we only measure at 48 kHz so the fixed coefficients are exact.
    assert sr == 48000
    b1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]
    a1 = [1.0, -1.69065929318241, 0.73248077421585]
    # Stage 2: high-pass (RLB)
    b2 = [1.0, -2.0, 1.0]
    a2 = [1.0, -1.99004745483398, 0.99007225036621]
    return lfilter(b2, a2, lfilter(b1, a1, x))


def integrated_lufs(x: np.ndarray, sr: int) -> float:
    y = _k_weight(x, sr)
    block = int(0.400 * sr)
    hop = int(0.100 * sr)
    if len(y) < block:
        ms = np.mean(y * y)
        return -0.691 + 10 * np.log10(ms + 1e-12)
    blocks = np.array([np.mean(y[i:i + block] ** 2)
                       for i in range(0, len(y) - block + 1, hop)])
    lk = -0.691 + 10 * np.log10(blocks + 1e-12)
    # absolute gate -70 LUFS, then relative gate -10 LU
    abs_gated = blocks[lk > -70.0]
    if len(abs_gated) == 0:
        return -70.0
    rel_thresh = -0.691 + 10 * np.log10(np.mean(abs_gated) + 1e-12) - 10.0
    gated = blocks[lk > max(-70.0, rel_thresh)]
    if len(gated) == 0:
        gated = abs_gated
    return -0.691 + 10 * np.log10(np.mean(gated) + 1e-12)


def process(x: np.ndarray, sr: int) -> np.ndarray:
    x = trim_edges(x, sr)
    if sr != TARGET_SR:
        from math import gcd
        g = gcd(TARGET_SR, sr)
        x = resample_poly(x, TARGET_SR // g, sr // g)
    lufs = integrated_lufs(x, TARGET_SR)
    gain = 10 ** ((TARGET_LUFS - lufs) / 20.0)
    peak = np.max(np.abs(x)) or 1.0
    gain = min(gain, PEAK_CEIL / peak)  # never exceed -1 dBFS
    x = x * gain
    pad = np.zeros(int(TARGET_SR * PAD_MS / 1000))
    return np.concatenate([pad, x, pad])


def write_mp3(x: np.ndarray, path: str) -> None:
    import av
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with av.open(path, "w") as container:
        stream = container.add_stream("mp3", rate=TARGET_SR)
        stream.bit_rate = 128_000
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray(pcm.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = TARGET_SR
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def main() -> None:
    cues_path, outdir = sys.argv[1], sys.argv[2]
    with open(cues_path, encoding="utf-8") as f:
        cues = json.load(f)
    for cue in cues:
        key, text = cue["key"], cue["text"]
        print(f"[{key}] synth: {text!r}", flush=True)
        raw, sr = synth(text)
        x = process(raw, sr)
        dur = len(x) / TARGET_SR
        lufs = integrated_lufs(x, TARGET_SR)
        # Eric's naming: plain <event-key>.wav (no cue_ prefix)
        wav_path = f"{outdir}/{key}.wav"
        sf.write(wav_path, x, TARGET_SR, subtype="PCM_16")
        write_mp3(x, f"{outdir}/{key}.mp3")
        print(f"[{key}] wrote {wav_path}  {dur:.2f}s  {lufs:.1f} LUFS", flush=True)


if __name__ == "__main__":
    main()
