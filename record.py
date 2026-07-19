import sounddevice as sd
from scipy.io.wavfile import write

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

write("voice.wav", fs, recording)

print("Recording Saved")
