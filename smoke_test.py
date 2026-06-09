"""OmniVoice install-gate smoke test.
Loads k2-fsa/OmniVoice, measures load time + VRAM, renders ONE voice-design clip
(no reference WAV needed), reports synth time / RTF, and prints VRAM coexistence.
"""
import time
import torch
import soundfile as sf
from omnivoice import OmniVoice

DEV = "cuda"


def vram_mb():
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024 / 1024, total / 1024 / 1024


used_before, total = vram_mb()
print(f"[vram] before load: {used_before:.0f} / {total:.0f} MiB used "
      f"(neutts + others already resident)")

t0 = time.time()
model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=DEV, dtype=torch.float16)
load_s = time.time() - t0
used_after, _ = vram_mb()
print(f"[load] model loaded in {load_s:.1f}s")
print(f"[vram] after load:  {used_after:.0f} MiB used  (+{used_after - used_before:.0f} MiB for OmniVoice)")
print(f"[vram] headroom remaining: {total - used_after:.0f} MiB")

TEXT = "Hey there. This is a quick test of the OmniVoice text to speech engine running on the VRPC."

# Voice-design mode: instruct string, no reference audio. This is the radio-button path.
t0 = time.time()
audios = model.generate(
    text=TEXT,
    language="English",
    instruct="female, young adult, american accent",
    num_step=16,  # fast preset
)
synth_s = time.time() - t0
audio = audios[0]
dur_s = len(audio) / model.sampling_rate
rtf = synth_s / dur_s
print(f"[synth] sr={model.sampling_rate} Hz  audio_dur={dur_s:.2f}s  "
      f"synth_time={synth_s:.2f}s  RTF={rtf:.2f}")

used_peak, _ = vram_mb()
print(f"[vram] after synth: {used_peak:.0f} MiB used")

sf.write("smoke_out.wav", audio, model.sampling_rate)
print("[done] wrote smoke_out.wav")
