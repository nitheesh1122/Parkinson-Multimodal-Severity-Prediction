"""Record a speech sample for downstream Parkinson's biomarker extraction."""

from pathlib import Path

import sounddevice as sd
from scipy.io.wavfile import write


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = PROJECT_ROOT / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

fs = 44100
duration = 5

print("Speak now...")

recording = sd.rec(
    int(duration * fs),
    samplerate=fs,
    channels=1,
    dtype='int16'
)

sd.wait()

output_path = SAMPLES_DIR / "voice.wav"
write(str(output_path), fs, recording)

print(f"Recording saved to {output_path}")
