"""Extract speech biomarkers from the recorded voice sample."""

from pathlib import Path

import parselmouth
from parselmouth.praat import call
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VOICE_SAMPLE = PROJECT_ROOT / "samples" / "voice.wav"
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
OUTPUT_CSV_PATH = PROCESSED_DIR / "features.csv"

# ----------------------------
# Load the recorded voice
# ----------------------------

sound = parselmouth.Sound(str(VOICE_SAMPLE))

# ----------------------------
# Pitch
# ----------------------------

pitch = sound.to_pitch()

pitch_values = pitch.selected_array["frequency"]
pitch_values = pitch_values[pitch_values != 0]

mean_pitch = pitch_values.mean()

# ----------------------------
# Intensity
# ----------------------------

intensity = sound.to_intensity()

mean_intensity = intensity.values.mean()

# ----------------------------
# HNR
# ----------------------------

harmonicity = sound.to_harmonicity()

hnr = harmonicity.values.mean()

# ----------------------------
# Point Process
# ----------------------------

pointProcess = call(sound, "To PointProcess (periodic, cc)", 75, 500)

# =====================================================
#                JITTER FEATURES
# =====================================================

jitter_local = call(
    pointProcess,
    "Get jitter (local)",
    0,
    0,
    0.0001,
    0.02,
    1.3
)

jitter_absolute = call(
    pointProcess,
    "Get jitter (local, absolute)",
    0,
    0,
    0.0001,
    0.02,
    1.3
)

rap = call(
    pointProcess,
    "Get jitter (rap)",
    0,
    0,
    0.0001,
    0.02,
    1.3
)

ppq5 = call(
    pointProcess,
    "Get jitter (ppq5)",
    0,
    0,
    0.0001,
    0.02,
    1.3
)

ddp = call(
    pointProcess,
    "Get jitter (ddp)",
    0,
    0,
    0.0001,
    0.02,
    1.3
)

# =====================================================
#                SHIMMER FEATURES
# =====================================================

shimmer_local = call(
    [sound, pointProcess],
    "Get shimmer (local)",
    0,
    0,
    0.0001,
    0.02,
    1.3,
    1.6
)

shimmer_db = call(
    [sound, pointProcess],
    "Get shimmer (local_dB)",
    0,
    0,
    0.0001,
    0.02,
    1.3,
    1.6
)

apq3 = call(
    [sound, pointProcess],
    "Get shimmer (apq3)",
    0,
    0,
    0.0001,
    0.02,
    1.3,
    1.6
)

apq5 = call(
    [sound, pointProcess],
    "Get shimmer (apq5)",
    0,
    0,
    0.0001,
    0.02,
    1.3,
    1.6
)

apq11 = call(
    [sound, pointProcess],
    "Get shimmer (apq11)",
    0,
    0,
    0.0001,
    0.02,
    1.3,
    1.6
)

dda = call(
    [sound, pointProcess],
    "Get shimmer (dda)",
    0,
    0,
    0.0001,
    0.02,
    1.3,
    1.6
)

# =====================================================
# Save everything
# =====================================================

features = {
    "Pitch": mean_pitch,
    "Intensity": mean_intensity,
    "HNR": hnr,

    "Jitter(%)": jitter_local,
    "Jitter(Abs)": jitter_absolute,
    "RAP": rap,
    "PPQ5": ppq5,
    "DDP": ddp,

    "Shimmer": shimmer_local,
    "Shimmer(dB)": shimmer_db,
    "APQ3": apq3,
    "APQ5": apq5,
    "APQ11": apq11,
    "DDA": dda,
}

df = pd.DataFrame([features])

print("\nExtracted Features\n")
print(df)

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_CSV_PATH, index=False)

print(f"\nFeatures saved to {OUTPUT_CSV_PATH}")