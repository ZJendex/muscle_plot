import numpy as np
from scipy.signal import find_peaks

def extract_motion_peaks(p_db, t, peak_prominence=2.5, peak_dominant_threshold=2):
    """
    Replicates MATLAB logic:
    - Sum STFT dB power across the lower half of the frequency band.
    - Normalize to [0,10].
    - Overlay this energy envelope curve on the spectrogram.
    - Detect peaks in the envelope as potential motion regions.
    Returns:
        peak_times, energy_envelope_norm
    """
    energy_envelope = np.sum(p_db, axis=0)  # sum across frequency -> 1D time series

    # Normalize to [0,10] (same scaling as MATLAB code)
    e_min = np.min(energy_envelope)
    e_max = np.max(energy_envelope)
    if np.isclose(e_max, e_min):
        energy_envelope_norm = np.zeros_like(energy_envelope)
    else:
        energy_envelope_norm = (energy_envelope - e_min) / (e_max - e_min) * 10.0

    # Detect peaks in normalized energy (same threshold as MATLAB)
    peaks, peak_values = find_peaks(energy_envelope_norm, prominence=peak_prominence)
    peak_times = np.array([])
    if len(peaks) > 0:
        for i in range(len(peaks)):
            if peak_values["prominences"][i] > energy_envelope_norm.mean()*peak_dominant_threshold:
                peak_times = np.append(peak_times, t[peaks[i]])


    return peak_times, energy_envelope_norm