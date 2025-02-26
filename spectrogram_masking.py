#%% Imports
import torch
import numpy as np
import einops
import matplotlib.pyplot as plt
from typing import Tuple


#%%
def main():
    #%%
    secs=10
    freq=20
    t, x = generate_test_accelerometer(secs=secs, freq=freq)
    x_tensor = torch.from_numpy(x)

    #%% Plot the example
    fig, ax = plt.subplots()
    ax.plot(t, x)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration (m/s^2)')
    ax.set_title('Example Accelerometer Signal')
    plt.show()

    #%%
    x_spec = time_to_spectrogram(
        x_tensor, 
        n_fft=freq, 
        hop_length=freq//2)

    #%%




#%% Functions
def generate_test_accelerometer(secs: int=10, freq: int=30) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a test signal with multiple frequency components."""    
    n_points = int(secs * freq)
    t = np.linspace(0, secs, n_points)
    x = np.zeros((n_points, 3))
    # Create a signal with multiple frequency components
    rng = np.random.default_rng()
    for c in range(3): # 3 channels
        for _ in range(10):
            freq = rng.uniform(0, 10)
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
        center=False,
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
    rank = len(shape)
    new_shape = [length if i == axis else 1 for i, length in enumerate(shape)]
    mask = torch.reshape(mask, new_shape)
    return x * mask
# %%
