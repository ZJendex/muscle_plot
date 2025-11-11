import numpy as np
from scipy.signal import get_window, spectrogram, butter, filtfilt
from scipy.signal import detrend
def get_phase_to_displacement(complex_signal):
    """
    Convert unwrapped phase to radial displacement:
        disp = phase * c / (4*pi*Fc)
    """
    c = 3e8
    phase = np.unwrap(np.angle(complex_signal))
    phase = detrend(phase)
    displacement = phase * c / (4 * np.pi * 77e9)
    return displacement

def bandpass_filter(data, cutoff, fs, order=5):
    """Apply bandpass filter to signal."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='bandpass', analog=False)
    return filtfilt(b, a, data)

def second_derivative(x, h):
    """
    Seven-point stencil approximation of the second derivative for stability.
    """
    y = np.zeros_like(x)
    for i in range(3, len(x) - 3):
        y[i] = (4*x[i] + (x[i+1]+x[i-1]) - 2*(x[i+2]+x[i-2]) - (x[i+3]+x[i-3])) / (16*h*h)
    return y

def stft_spectrogram(y, fs, window_size):
    """
    Compute and (optionally) plot STFT spectrogram.
    Returns:
        t (np.ndarray): time axis in seconds
        f (np.ndarray): frequency axis in Hz
        p_db (np.ndarray): power spectrogram in dB
        Sxx (np.ndarray): power spectral density values
    """
    # Use Hann window, same as MATLAB "hann"
    window = get_window("hann", window_size, fftbins=True)
    # FFT length = 2 * window size (zero-padding factor of 2)
    nfft = window_size * 2
    # Overlap = 50% of window size
    noverlap = window_size // 2

    # Compute spectrogram (SciPy automatically returns one-sided spectrum [0..fs/2])
    f, t, Sxx = spectrogram(
        x=y,
        fs=fs,
        window=window,
        nperseg=window_size,
        noverlap=noverlap,
        nfft=nfft,
        detrend=False,
        scaling="density",
        mode="psd"  # "psd" gives power spectrum like MATLAB
    )

    # Convert power to dB (avoid log(0) with small epsilon)
    eps = np.finfo(float).eps
    p_db = 10.0 * np.log10(Sxx + eps)

    # Limit frequency range [0.1 Hz, 20000 Hz] as in MATLAB code
    min_freq = 0.1
    max_freq = min(20000.0, f.max() if f.size else 20000.0)
    freq_idx = (f >= min_freq) & (f <= max_freq)
    f = f[freq_idx]
    p_db = p_db[freq_idx, :]

    return t, f, p_db, Sxx

import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

def get_peak_to_peak_envelope(signal, distance=None, prominence=None, interpolation='cubic'):
    """
    Calculate peak-to-peak envelope of a signal.
    
    Parameters:
    -----------
    signal : np.ndarray
        Input signal (1D array)
    distance : int, optional
        Minimum number of samples between consecutive peaks.
        If None, automatically set to len(signal)//100
    prominence : float, optional
        Minimum prominence of peaks. If None, automatically set based on signal std.
    interpolation : str, default='cubic'
        Type of interpolation ('linear', 'cubic', 'quadratic')
        
    Returns:
    --------
    upper_envelope : np.ndarray
        Upper envelope of the signal (same length as input)
    lower_envelope : np.ndarray
        Lower envelope of the signal (same length as input)
    p2p_envelope : np.ndarray
        Peak-to-peak envelope (upper - lower, same length as input)
    """
    # Auto-set parameters if not provided
    if distance is None:
        distance = max(1, len(signal) // 100)
    if prominence is None:
        prominence = np.std(signal) * 0.1
    
    # Find peaks (local maxima)
    peaks_idx, _ = find_peaks(signal, distance=distance, prominence=prominence)
    
    # Find valleys (local minima) by inverting signal
    valleys_idx, _ = find_peaks(-signal, distance=distance, prominence=prominence)
    
    # Add first and last points to ensure full coverage
    if len(peaks_idx) > 0:
        if peaks_idx[0] != 0:
            peaks_idx = np.concatenate([[0], peaks_idx])
        if peaks_idx[-1] != len(signal) - 1:
            peaks_idx = np.concatenate([peaks_idx, [len(signal) - 1]])
    else:
        peaks_idx = np.array([0, len(signal) - 1])
    
    if len(valleys_idx) > 0:
        if valleys_idx[0] != 0:
            valleys_idx = np.concatenate([[0], valleys_idx])
        if valleys_idx[-1] != len(signal) - 1:
            valleys_idx = np.concatenate([valleys_idx, [len(signal) - 1]])
    else:
        valleys_idx = np.array([0, len(signal) - 1])
    
    # Interpolate upper envelope
    x_axis = np.arange(len(signal))
    interp_kind = interpolation if len(peaks_idx) > 3 else 'linear'
    upper_interp = interp1d(peaks_idx, signal[peaks_idx], kind=interp_kind, 
                           bounds_error=False, fill_value='extrapolate')
    upper_envelope = upper_interp(x_axis)
    
    # Interpolate lower envelope
    interp_kind = interpolation if len(valleys_idx) > 3 else 'linear'
    lower_interp = interp1d(valleys_idx, signal[valleys_idx], kind=interp_kind,
                           bounds_error=False, fill_value='extrapolate')
    lower_envelope = lower_interp(x_axis)
    
    # Calculate peak-to-peak envelope
    p2p_envelope = upper_envelope - lower_envelope
    
    return upper_envelope, lower_envelope, p2p_envelope