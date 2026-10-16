"""Record audio from the microphone and display its waveform, spectrum, and spectrogram."""

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

DURATION = 5  # seconds
SAMPLE_RATE = 44100  # Hz

OCTAVE_BANDS = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]  # Hz


def octave_label(freq):
    return f"{freq / 1000:g}k" if freq >= 1000 else f"{freq:g}"

print(f"Recording {DURATION}s of audio...")
audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float64")
sd.wait()
print("Done recording.")

audio = audio.flatten()
times = np.arange(len(audio)) / SAMPLE_RATE

spectrum = np.abs(np.fft.rfft(audio))
freqs = np.fft.rfftfreq(len(audio), d=1 / SAMPLE_RATE)

fig, (ax_wave, ax_spectrum, ax_specgram) = plt.subplots(3, 1, figsize=(9, 9))

ax_wave.plot(times, audio)
ax_wave.set_title("Waveform")
ax_wave.set_xlabel("Time (s)")
ax_wave.set_ylabel("Amplitude")

ax_spectrum.plot(freqs, spectrum)
ax_spectrum.set_title("Spectrum")
ax_spectrum.set_xlabel("Frequency (Hz)")
ax_spectrum.set_ylabel("Magnitude")

Sxx, spec_freqs, bins = mlab.specgram(audio, Fs=SAMPLE_RATE)
spec_img = ax_specgram.pcolormesh(bins, spec_freqs, 10 * np.log10(Sxx + 1e-12), shading="auto", cmap="magma")
ax_specgram.set_title("Spectrogram")
ax_specgram.set_xlabel("Time (s)")
ax_specgram.set_ylabel("Frequency (octave band)")
ax_specgram.set_yscale("log", base=2)
bands_in_range = [f for f in OCTAVE_BANDS if f <= SAMPLE_RATE / 2]
ax_specgram.set_ylim(bands_in_range[0], SAMPLE_RATE / 2)
ax_specgram.set_yticks(bands_in_range)
ax_specgram.set_yticklabels([octave_label(f) for f in bands_in_range])
ax_specgram.minorticks_off()
fig.colorbar(spec_img, ax=ax_specgram, label="Intensity (dB)")

fig.tight_layout()
plt.show()
