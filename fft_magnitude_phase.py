#%%
import numpy as np
import matplotlib.pyplot as plt

#%%# Create a sample signal
sampling_rate = 1000  # Hz
duration = 1  # second
t = np.linspace(0, duration, sampling_rate * duration)
signal = 3 * np.sin(2 * np.pi * 5 * t) + 2 * np.sin(2 * np.pi * 10 * t + np.pi/4)

# plot example
fig, ax = plt.subplots()
ax.plot(t, signal)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.set_title('Example Signal')
plt.show()


#%% Compute FFT
fft_result = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(signal), 1/sampling_rate)

# Extract real and imaginary parts
real_part = np.real(fft_result)
imag_part = np.imag(fft_result)

# Calculate magnitude and phase
magnitude = np.abs(fft_result)
phase = np.angle(fft_result)

#%% Verify the conversions
# From real/imag to magnitude/phase
calculated_magnitude = np.sqrt(real_part**2 + imag_part**2)
calculated_phase = np.arctan2(imag_part, real_part)

# From magnitude/phase to real/imag
calculated_real = magnitude * np.cos(phase)
calculated_imag = magnitude * np.sin(phase)

# Plot only positive frequencies up to Nyquist frequency
positive_freq_mask = (frequencies > 0) & (frequencies <= sampling_rate/2)
pos_frequencies = frequencies[positive_freq_mask]
pos_magnitude = magnitude[positive_freq_mask]
pos_phase = phase[positive_freq_mask]

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(pos_frequencies, pos_magnitude)
plt.title('Magnitude Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

plt.subplot(2, 1, 2)
plt.plot(pos_frequencies, pos_phase)
plt.title('Phase Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (radians)')

plt.tight_layout()
plt.show()