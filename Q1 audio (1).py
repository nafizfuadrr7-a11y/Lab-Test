import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Load the audio file
# Ensure 'speech.wav' is in the same folder as this script
samplerate, data = wavfile.read('voice.wav')

# If the file has two channels (stereo), convert it to one (mono)
if len(data.shape) > 1:
    data = data[:, 0]

# Normalize the data (Scale amplitude to range [-1, 1])
# This makes min/max amplitude easier to read
data_norm = data / np.max(np.abs(data))

# --- Criteria (a): Metadata Analysis ---
num_samples = len(data_norm)
duration = num_samples / samplerate
max_amp = np.max(data_norm)
min_amp = np.min(data_norm)

print(f"--- Audio Signal Metadata ---")
print(f"Sampling Rate:   {samplerate} Hz")
print(f"Total Samples:   {num_samples}")
print(f"Duration:        {duration:.3f} seconds")
print(f"Max Amplitude:   {max_amp:.2f}")
print(f"Min Amplitude:   {min_amp:.2f}")

# Setup the plotting area
plt.figure(figsize=(10, 8))

# --- Criteria (b): Time-Domain Waveform ---
time_axis = np.linspace(0, duration, num_samples)

plt.subplot(2, 1, 1)
plt.plot(time_axis, data_norm, color='teal', linewidth=0.5)
plt.title("Waveform (Time Domain)")
plt.xlabel("Time (seconds)")
plt.ylabel("Normalized Amplitude")
plt.grid(alpha=0.3)

# --- Criteria (c): Frequency-Domain (FFT) ---
# Compute FFT
fft_out = np.fft.fft(data_norm)
# Get the frequencies for the x-axis
freqs = np.fft.fftfreq(num_samples, 1/samplerate)

# We only care about the positive frequencies (first half of the array)
num_half = num_samples // 2
positive_freqs = freqs[:num_half]
magnitude = np.abs(fft_out[:num_half])



plt.subplot(2, 1, 2)
plt.plot(positive_freqs, magnitude, color='darkorange')
plt.title("Magnitude Spectrum (Frequency Domain)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 5000)  # Zooming into 0-5kHz as human speech lives here
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()