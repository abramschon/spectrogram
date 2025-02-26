#%%
import numpy as np
import scipy.signal as signal
import torch
import time
from typing import Tuple, List
import matplotlib.pyplot as plt

#%%
def main():
    #%%
    original_len = 1000
    target_len = 2000
    n_trials = 100
    # Generate test signal
    t, test_signal = generate_test_signal(original_len)
    
    # Initialize timing lists
    numpy_times = []
    scipy_times = []
    pytorch_times = []
    
    # Run multiple trials
    for i in range(n_trials):
        # NumPy
        time_numpy, resampled_numpy = numpy_resample(test_signal, original_len, target_len)
        
        # SciPy
        time_scipy, resampled_scipy = scipy_resample(test_signal, target_len)
        
        # PyTorch
        time_pytorch, resampled_pytorch = pytorch_resample(test_signal, target_len)

        # discard initial runs
        if i == 0:
            continue
        
        # Append times
        numpy_times.append(time_numpy)
        scipy_times.append(time_scipy)
        pytorch_times.append(time_pytorch)
    
    # Print average times
    print(f"\nAverage execution times over {n_trials} trials:")
    print(f"NumPy:   {np.mean(numpy_times)*1000:.2f} ms")
    print(f"SciPy:   {np.mean(scipy_times)*1000:.2f} ms")
    print(f"PyTorch: {np.mean(pytorch_times)*1000:.2f} ms")
    
    # Plot results
    t_resampled = np.linspace(0, 10, target_len)
    
    plt.figure(figsize=(12, 8))
    
    # Original signal
    plt.subplot(4, 1, 1)
    plt.plot(t, test_signal, label='Original')
    plt.title('Original Signal')
    
    # NumPy resampled
    plt.subplot(4, 1, 2)
    plt.plot(t_resampled, resampled_numpy, label='NumPy')
    plt.title('NumPy Resampled')
    
    # SciPy resampled
    plt.subplot(4, 1, 3)
    plt.plot(t_resampled, resampled_scipy, label='SciPy')
    plt.title('SciPy Resampled')
    
    # PyTorch resampled
    plt.subplot(4, 1, 4)
    plt.plot(t_resampled, resampled_pytorch, label='PyTorch')
    plt.title('PyTorch Resampled')
    
    plt.tight_layout()
    plt.show()


    #%% Compare methods with different parameters
    print("Testing upsampling (1000 -> 2000 points)")
    compare_methods(original_len=1000, target_len=2000)
    
    print("\nTesting downsampling (2000 -> 1000 points)")
    compare_methods(original_len=2000, target_len=1000)

#%%
def generate_test_signal(n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a test signal with multiple frequency components."""
    t = np.linspace(0, 10, n_points)
    # Create a signal with multiple frequency components
    rng = np.random.default_rng()
    for _ in range(10):
        freq = rng.uniform(0, 10)
        signal += np.sin(2 * np.pi * freq * t)

    return t, signal

def numpy_resample(x: np.ndarray, original_len: int, target_len: int) -> Tuple[float, np.ndarray]:
    """Resample using NumPy's FFT."""
    start_time = time.time()
    
    # Compute FFT
    fft = np.fft.fft(x)
    
    # If upsampling, pad with zeros; if downsampling, truncate
    if target_len > original_len:
        pad_len = target_len - original_len
        fft_padded = np.pad(fft, (0, pad_len))
    else:
        fft_padded = fft[:target_len]
    
    # Compute IFFT and adjust amplitude
    resampled = np.real(np.fft.ifft(fft_padded))
    resampled *= target_len / original_len
    
    elapsed_time = time.time() - start_time
    return elapsed_time, resampled

def scipy_resample(x: np.ndarray, target_len: int) -> Tuple[float, np.ndarray]:
    """Resample using SciPy's resample function."""
    start_time = time.time()
    resampled = signal.resample(x, target_len)
    elapsed_time = time.time() - start_time
    return elapsed_time, resampled

def pytorch_resample(x: np.ndarray, target_len: int) -> Tuple[float, np.ndarray]:
    """Resample using PyTorch's FFT."""
    # Convert to PyTorch tensor
    x_tensor = torch.from_numpy(x).float()
    
    start_time = time.time() # time after conversion
    
    # Compute FFT
    fft = torch.fft.fft(x_tensor)
    
    # If upsampling, pad with zeros; if downsampling, truncate
    if target_len > len(x):
        pad_len = target_len - len(x)
        fft_padded = torch.nn.functional.pad(fft, (0, pad_len))
    else:
        fft_padded = fft[:target_len]
    
    # Compute IFFT and adjust amplitude
    resampled = torch.fft.ifft(fft_padded).real
    resampled *= target_len / len(x)
    
    elapsed_time = time.time() - start_time
    return elapsed_time, resampled.numpy()

def compare_methods(original_len: int = 1000, target_len: int = 2000, n_trials: int = 100) -> None:
    """Compare different resampling methods."""
    # Generate test signal
    t, test_signal = generate_test_signal(original_len)
    
    # Initialize timing lists
    numpy_times = []
    scipy_times = []
    pytorch_times = []
    
    # Run multiple trials
    for _ in range(n_trials):
        # NumPy
        time_numpy, resampled_numpy = numpy_resample(test_signal, original_len, target_len)
        numpy_times.append(time_numpy)
        
        # SciPy
        time_scipy, resampled_scipy = scipy_resample(test_signal, target_len)
        scipy_times.append(time_scipy)
        
        # PyTorch
        time_pytorch, resampled_pytorch = pytorch_resample(test_signal, target_len)
        pytorch_times.append(time_pytorch)
    
    # Print average times
    print(f"\nAverage execution times over {n_trials} trials:")
    print(f"NumPy:   {np.mean(numpy_times)*1000:.2f} ms")
    print(f"SciPy:   {np.mean(scipy_times)*1000:.2f} ms")
    print(f"PyTorch: {np.mean(pytorch_times)*1000:.2f} ms")
    
    # Plot results
    t_resampled = np.linspace(0, 10, target_len)
    
    plt.figure(figsize=(12, 8))
    
    # Original signal
    plt.subplot(4, 1, 1)
    plt.plot(t, test_signal, label='Original')
    plt.title('Original Signal')
    plt.grid(True)
    
    # NumPy resampled
    plt.subplot(4, 1, 2)
    plt.plot(t_resampled, resampled_numpy, label='NumPy')
    plt.title('NumPy Resampled')
    plt.grid(True)
    
    # SciPy resampled
    plt.subplot(4, 1, 3)
    plt.plot(t_resampled, resampled_scipy, label='SciPy')
    plt.title('SciPy Resampled')
    plt.grid(True)
    
    # PyTorch resampled
    plt.subplot(4, 1, 4)
    plt.plot(t_resampled, resampled_pytorch, label='PyTorch')
    plt.title('PyTorch Resampled')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

#%%
if __name__ == "__main__":
    main()
