#!/usr/bin/env python3
import sys, os, traceback
LOG = os.path.expanduser("~/Library/Logs/AudioTranscriber_test.log")
def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{msg}\n")

log("=== Test starting ===")
log(f"test file: {os.path.isfile('/tmp/test_audio.wav')}")

log("--- Import faster_whisper ---")
from faster_whisper import WhisperModel
log("OK imported")

log("--- Check assets ---")
from faster_whisper.utils import get_assets_path
assets = get_assets_path()
log(f"Assets: {assets} exists:{os.path.isdir(assets)}")
if os.path.isdir(assets):
    for fn in os.listdir(assets):
        fp = os.path.join(assets, fn)
        log(f"  {fn} ({os.path.getsize(fp)} bytes)")

log("--- Load model ---")
model = WhisperModel("large-v3", device="cpu", compute_type="int8", cpu_threads=4, num_workers=2)
log("OK model loaded")

log("--- Transcribe ---")
segments, info = model.transcribe("/tmp/test_audio.wav", beam_size=5, best_of=5, 
    temperature=0.0, vad_filter=True, 
    vad_parameters=dict(min_silence_duration_ms=500, threshold=0.5))
log(f"OK language:{info.language} prob:{info.language_probability:.2%}")
for i, seg in enumerate(segments):
    if i >= 3: break
    log(f"  [{seg.start:.1f}s-{seg.end:.1f}s] {seg.text.strip()[:80]}")
log("=== Test complete ===")
