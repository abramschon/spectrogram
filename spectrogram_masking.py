#%% Imports
import torch
import numpy as np
import einops
import matplotlib.pyplot as plt
from typing import Tuple


#%%
def main():
    #%% Generate an example accelerometer signal
    secs=3
    freq=20
    t, x = generate_test_accelerometer(secs=secs, freq=freq, n_sines=3)
    x_tensor = torch.from_numpy(x)

    # Plot the example
    fig, ax = plt.subplots()
    ax.plot(t, x)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration (m/s^2)')
    ax.set_title('Example Accelerometer Signal')
    plt.show()

    #%% Get the spectrogram
    x_spec = time_to_spectrogram( # shape: T, C, F, P 
        x_tensor, 
        n_fft=freq, 
        hop_length=freq//2,
        phase=True,
        stack_axes=False)
    
    x_mag = time_to_spectrogram( # shape: T, C, F, P=1
        x_tensor, 
        n_fft=freq, 
        hop_length=freq//2,
        phase=False,
        stack_axes=False)

    #%% Plot the real and imaginary parts
    # Real part is the first entry of last dim
    fig, axes = plt.subplots(3, 1)
    for c in range(3):
        axes[c].imshow(x_spec[:,c,:,0])
        axes[c].set_xlabel('Frequency (Hz)')
        axes[c].set_ylabel('Time')

    plt.show()

    # Plot the imaginary part
    # Imaginary part is the second entry of last dim
    fig, axes = plt.subplots(3, 1)
    for c in range(3):
        axes[c].imshow(x_spec[:,c,:,1])
        axes[c].set_xlabel('Frequency (Hz)')
        axes[c].set_ylabel('Time')

    plt.show()

    # Plot the magnitude
    fig, axes = plt.subplots(3, 1)
    for c in range(3):
        axes[c].imshow(x_mag[:,c,:,0])
        axes[c].set_xlabel('Frequency (Hz)')
        axes[c].set_ylabel('Time')

    #%% Check that the inverse works for the spectrogram with phase information
    x_time = spectrogram_to_time(
        x_spec, 
        n_fft=freq, 
        hop_length=freq//2)
    
    # Plot the original and reconstructed signals
    fig, axes = plt.subplots(2, 1)
    axes[0].plot(t, x)
    axes[0].set_title('Original Signal')
    axes[1].plot(t, x_time)
    axes[1].set_title('Reconstructed Signal')
    plt.show()

    # Check that the inverse works for the spectrogram without phase information
    x_mag_time = spectrogram_to_time(
        x_mag, 
        n_fft=freq, 
        hop_length=freq//2)
    fig, ax = plt.subplots()
    ax.plot(t, x_mag_time)
    ax.set_title('Reconstructed Signal (Magnitude Only)')
    plt.show()


    #%% Mask the spectrogram
    x_spec_mask_time, mask = randomly_mask_tensor(x_spec, p=0.5, width=1, axis=0, prob=0.9)
    fig, axes = plt.subplots(3, 1)
    for c in range(3):
        axes[c].imshow(x_spec_mask_time[:,c,:,0])
        axes[c].set_xlabel('Frequency (Hz)')
        axes[c].set_ylabel('Time')  
    plt.show()

    x_masked_time = spectrogram_to_time(
        x_spec_mask_time, 
        n_fft=freq, 
        hop_length=freq//2
    )
    
    fig, axes = plt.subplots(2, 1)
    axes[0].plot(t, x)
    axes[0].set_title('Original Signal')
    axes[1].plot(t, x_masked_time)
    axes[1].set_title('Reconstructed masked Signal')
    plt.show()


    #%% Mask along frequency
    x_spec_mask_freq, mask = randomly_mask_tensor(x_spec, p=0.5, width=1, axis=2, prob=0.9)
    if len(mask) == mask.sum():
        print("None of the values were masked")

    fig, axes = plt.subplots(3, 1)
    for c in range(3):
        axes[c].imshow(x_spec_mask_freq[:,c,:,0])
        axes[c].set_xlabel('Frequency (Hz)')
        axes[c].set_ylabel('Time')  
    plt.show()

    x_masked_freq = spectrogram_to_time(
        x_spec_mask_freq, 
        n_fft=freq, 
        hop_length=freq//2
    )
    
    fig, axes = plt.subplots(2, 1)
    axes[0].plot(t, x)
    axes[0].set_title('Original Signal')
    axes[1].plot(t, x_masked_freq)
    axes[1].set_title('Reconstructed Masked Signal')
    plt.show()


#%% Functions
def generate_test_accelerometer(secs: int=10, 
                                freq: int=30, 
                                n_sines: int=3,
                                min_freq: int=0,
                                max_freq: int=1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a test signal with multiple frequency components."""    
    n_points = int(secs * freq)
    t = np.linspace(0, secs, n_points)
    x = np.zeros((n_points, 3))
    # Create a signal with multiple frequency components
    rng = np.random.default_rng()
    for c in range(3): # 3 channels
        for _ in range(n_sines):
            freq = rng.uniform(min_freq, max_freq)
            x[:, c] += np.sin(2 * np.pi * freq * t)

    return t, x

def time_to_spectrogram(x, n_fft, hop_length, window=None, phase=False, stack_axes=True):
    """
    Convert time-domain signals to spectrograms
    
    Parameters:
    -----------
    x : torch.Tensor
        Input tensor of shape [sequence_length, num_channels]
    n_fft : int
        FFT window size
    hop_length : int
        FFT window shift
    window : torch.Tensor, optional
        Window function (default: Hann window)
    phase : bool, optional
        Whether to include phase information (default: False)
    stack_axes : bool, optional
        Whether to stack all axes (default: True)
        
    Returns:
    --------
    torch.Tensor
        Spectrogram tensor
    """
    if window is None:
        window = torch.hann_window(n_fft)
        
    # Reshape for STFT: [sequence_length, num_channels] -> [num_channels, sequence_length]
    x = einops.rearrange(x, 'S C -> C S')
    
    # Compute STFT
    x = torch.stft(
        input=x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True
    )  # Shape: [num_channels, num_bins, num_frames]
    
    if phase:
        # Convert complex to cartesian (real and imaginary parts)
        x = complex_to_cartesian(x)
    else:
        # Convert complex to magnitude only
        x = complex_to_magnitude(x, expand=True)
    
    if stack_axes:
        # Stack all spectrograms and put time dim first:
        # [num_channels, num_bins, num_frames, stft_parts] -> [num_frames, num_channels x num_bins x stft_parts]
        x = einops.rearrange(x, 'C F T P -> T (C F P)')
    else:
        # Keep axes separate but put time first
        x = einops.rearrange(x, 'C F T P -> T C F P')
        
    return x

def spectrogram_to_time(x, n_fft, hop_length, window=None):
    """
    Convert spectrograms back to time-domain signals using inverse STFT
    
    Parameters:
    -----------
    x : torch.Tensor
        Input spectrogram tensor of shape [num_frames, num_channels, num_bins, 2]
        where the last dimension contains real and imaginary parts
    n_fft : int
        FFT window size
    hop_length : int
        FFT window shift
    window : torch.Tensor, optional
        Window function (default: Hann window)
        
    Returns:
    --------
    torch.Tensor
        Time-domain signal tensor of shape [sequence_length, num_channels]
    """
    if window is None:
        window = torch.hann_window(n_fft)
    
    # Rearrange dimensions to match torch.istft requirements
    # [num_frames, num_channels, num_bins, 2] -> [num_channels, num_bins, num_frames, 2]
    x = einops.rearrange(x, 'T C F P -> C F T P')
    
    if x.shape[-1] == 1:
        # If only magnitude is provided, assume phase is zero
        x_complex = torch.complex(x[..., 0], torch.zeros_like(x[..., 0]))
    else:
        # Convert from real/imaginary to complex
        x_real = x[..., 0]
        x_imag = x[..., 1]
        x_complex = torch.complex(x_real, x_imag)
    x_time = torch.istft(
        input=x_complex,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
    )  # Shape: [num_channels, sequence_length]
    
    # Rearrange back to [sequence_length, num_channels]
    x_time = einops.rearrange(x_time, 'C S -> S C')
    
    return x_time

def complex_to_cartesian(x):
    """Convert complex tensor to cartesian (real and imaginary parts)"""
    real = x.real.unsqueeze(-1)
    imag = x.imag.unsqueeze(-1)
    return torch.cat([real, imag], dim=-1)

def complex_to_magnitude(x, expand=True):
    """Convert complex tensor to magnitude"""
    mag = torch.abs(x)
    if expand:
        return mag.unsqueeze(-1)
    return mag

def randomly_mask_tensor(x, p, width=1, axis=0, prob=0.9, limit_range=None, flip_prob=0.0):
    """
    Zero out random slices along an axis
    
    Parameters:
    -----------
    x : torch.Tensor
        Input tensor
    p : float
        Fraction of values to zero out
    width: int
        Number of contiguous zeros in the mask
    axis: int
        Tensor axis to mask along
    prob: float
        Probability of masking (default: 0.9)
    limit_range: tuple(int, int) or None
        Range allowed to be masked (default: None)
    flip_prob: float
        Probability of flipping the mask (default: 0.0)
        
    Returns:
    --------
    torch.Tensor
        Masked tensor
    torch.Tensor
        Binary mask (1 = kept, 0 = masked)
    """
    import random
    
    v = torch.rand(())
    total_mask = torch.ones(x.shape[axis])
    
    if v < prob:
        shape = x.shape
        if limit_range is None:
            length = shape[axis]
            n = length - width
        else:
            n = limit_range.copy()
            n[1] = n[1] - width
            length = n[1] - n[0]
            
        ratio = length / width
        num_starts = int(p * ratio)
        
        # Sample start indices
        starts = torch.tensor(random.sample(range(n), num_starts))
        
        # Generate contiguous mask
        mask = gen_contiguous_mask(shape[axis], starts, width)
        
        if torch.rand(()) < flip_prob:
            mask = torch.flip(mask, dims=(0,))
            
        # Apply mask
        x = mask_tensor(x, mask=mask, axis=axis)
        total_mask *= mask
        
    return x, total_mask

def gen_contiguous_mask(length, starts, width=1, dtype=torch.float32):
    """
    Generate a binary mask with contiguous sets of zeros
    
    Parameters:
    -----------
    length: int
        Length of resulting mask vector
    starts: torch.Tensor[int]
        Start indices of zero-segments
    width: int
        Size of the zero-segments
    dtype: torch datatype
        Datatype of output mask
        
    Returns:
    --------
    mask : torch.Tensor<length>
        Binary mask (1 = kept, 0 = masked)
    """
    # Generate indices for the positions that will be masked
    indices = []
    for start in starts:
        for i in range(width):
            indices.append(start + i)
    
    indices = torch.tensor(indices).to(torch.int64)
    
    # Create a mask with ones
    hits = torch.zeros(length)
    
    # Set positions to mask as 1
    if len(indices) > 0:
        hits.scatter_(0, indices, torch.ones(len(indices)))
    
    # The mask should be 1 wherever nothing was masked
    return (hits == 0).to(dtype)

def mask_tensor(x, mask, axis=0):
    """
    Mask a tensor with a provided mask along a given axis
    
    Parameters:
    ----------
    x : torch.Tensor
        The tensor to mask
    mask : torch.Tensor
        Multiplicative mask vector (must match x along masking axis)
    axis : int
        The axis to apply the mask along
        
    Returns:
    -------
    torch.Tensor
        Masked tensor
    """
    shape = x.shape
    new_shape = [length if i == axis else 1 for i, length in enumerate(shape)]
    mask = torch.reshape(mask, new_shape)
    return x * mask
# %%
