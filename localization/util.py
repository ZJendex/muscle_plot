import numpy as np
from scipy.signal.windows import gaussian
# util.py
import os
import re
import glob
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Literal
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, get_window, find_peaks

def gaussian_window(N, fs, sigma_sec):
    """
    Generate Gaussian window.
    """
    std_samples = sigma_sec * fs
    return gaussian(N, std_samples, sym=True)

def analyzeMuscleData(y, t, motion_points, fs, if_chaos=False, motion_cutoff=0.2):
    """
    Extract muscle features (RMS, MPF, MF) from signal chunk.

    Parameters
    ----------
    y : np.ndarray
        Input signal chunk (1D).
    t : np.ndarray
        Time vector corresponding to y.
    motion_points : np.ndarray
        Time points to exclude (artifacts).
    fs : float
        Sampling frequency (Hz).
    if_chaos : bool
        Whether to compute chaos features.
    Returns
    -------
    rms_val : float
        Root mean square value (scaled).
    mpf_val : float
        Mean power frequency (Hz).
    mf_val : float
        Median frequency (Hz).
    chaos_higuchi_fd : float
        Higuchi fractal dimension.
    chaos_lle : float
        Local linear embedding.
    chaos_sampen : float
        Sample entropy.
    chaos_corr_dim : float
        Correlation dimension.
    """

    N = len(y)

    # Apply Gaussian window
    sigma_sec = 0.3
    w = gaussian_window(N, fs, sigma_sec)
    y_win = y * w

    # Exclude motion artifacts
    t = np.asarray(t).flatten()
    motion_points = np.asarray(motion_points).flatten()
    motion_points = motion_points + t[0]
    is_close = np.any(np.abs(t[:, None] - motion_points[None, :]) <= motion_cutoff, axis=1)

    # 1) RMS (exclude motion points)
    rms_val = np.sqrt(np.mean(y_win[~is_close] ** 2)) * 100

    # 2) Power spectrum
    fft_vals = np.fft.fft(y_win)
    P2 = np.abs(fft_vals / N) ** 2
    P1 = P2[: N // 2 + 1]
    if len(P1) > 2:
        P1[1:-1] *= 2
    f = fs * np.arange(len(P1)) / N

    mpf_val = np.sum(f * P1) / np.sum(P1)  # Mean power frequency

    # 3) Median frequency
    cumsum_power = np.cumsum(P1)
    total_power = np.sum(P1)
    half_power = total_power / 2
    idx_median = np.argmax(cumsum_power >= half_power)
    mf_val = f[idx_median]

    # 4) Chaos features

    chaos_features = None

    features = {
        "rms": rms_val,
        "mpf": mpf_val,
        "mf": mf_val,
        "chaos_higuchi_fd": chaos_features["HiguchiFD"] if chaos_features else None,
        "chaos_lle": chaos_features["LLE"] if chaos_features else None,
        "chaos_sampen": chaos_features["SampEn"] if chaos_features else None,
        "chaos_corr_dim": chaos_features["CorrDim"] if chaos_features else None
    }

    return features

# File name pattern: {Sensor}_capture_YYYYMMDD_HHMMSSmmm.bin
FNAME_RE = re.compile(r'^(IMU|LoadCell|Radar)_capture_(\d{8})_(\d{9})\.bin$')
@dataclass(frozen=True)
class CaptureInfo:
    """Simple container for a discovered capture file."""
    sensor: str
    path: str
    start_ms: int  # Epoch milliseconds parsed from the file name

class SensorTrioAligner:
    """
    Discover IMU/LoadCell/Radar files in a folder and compute start-time shifts.

    select:
        "latest"             -> pick the latest file for each sensor (default; matches your script)
        "nearest_to_radar"   -> pick the IMU/LoadCell file whose start time is nearest to the chosen Radar file
    """

    def __init__(
        self,
        data_dir: str,
        select: Literal["latest", "nearest_to_radar"] = "latest",
        name_re: re.Pattern[str] = FNAME_RE,
    ) -> None:
        self.data_dir = os.path.abspath(data_dir)
        self.select = select
        self.name_re = name_re

        self.imu: Optional[CaptureInfo] = None
        self.lc: Optional[CaptureInfo] = None
        self.rd: Optional[CaptureInfo] = None
        self.camera: Optional[CaptureInfo] = None

    # ---------- parsing utilities ----------
    def _parse_start_ms_from_name(self, path: str) -> Optional[int]:
        """
        Parse start time from file name:
            <Sensor>_capture_YYYYMMDD_HHMMSSmmm.bin
        Treats the timestamp as local time and returns epoch milliseconds.
        """
        base = os.path.basename(path)
        m = self.name_re.match(base)
        if not m:
            return None
        _sensor, yyyymmdd, hhmmssmmm = m.groups()

        yyyy = int(yyyymmdd[0:4])
        mm = int(yyyymmdd[4:6])
        dd = int(yyyymmdd[6:8])

        hh = int(hhmmssmmm[0:2])
        mi = int(hhmmssmmm[2:4])
        ss = int(hhmmssmmm[4:6])
        mmm = int(hhmmssmmm[6:9])

        # Naive local time is fine for relative differences
        dt = datetime(yyyy, mm, dd, hh, mi, ss, mmm * 1000)
        return int(dt.timestamp() * 1000)

    def _parse_start_ms_from_camera_folder(self, folder_name: str) -> Optional[int]:
        """
        Parse start time from camera folder name:
            camera_capture_YYYYMMDD_HHMMSSmmm
        Treats the timestamp as local time and returns epoch milliseconds.
        """
        camera_re = re.compile(r'^camera_capture_(\d{8})_(\d{9})$')
        m = camera_re.match(folder_name)
        if not m:
            return None
        yyyymmdd, hhmmssmmm = m.groups()

        yyyy = int(yyyymmdd[0:4])
        mm = int(yyyymmdd[4:6])
        dd = int(yyyymmdd[6:8])

        hh = int(hhmmssmmm[0:2])
        mi = int(hhmmssmmm[2:4])
        ss = int(hhmmssmmm[4:6])
        mmm = int(hhmmssmmm[6:9])

        # Naive local time is fine for relative differences
        dt = datetime(yyyy, mm, dd, hh, mi, ss, mmm * 1000)
        return int(dt.timestamp() * 1000)

    def _collect_for_sensor(self, sensor: str) -> List[CaptureInfo]:
        out: List[CaptureInfo] = []
        if sensor == "camera":
            folders = []
            for root, dirs, files in os.walk(self.data_dir):
                for dir_name in dirs:
                    if dir_name.startswith("camera_capture_"):
                        folders.append(os.path.join(root, dir_name))
            for folder in folders:
                folder_name = os.path.basename(folder)  # e.g. "camera_capture_20250928_183844291"
                ms = self._parse_start_ms_from_camera_folder(folder_name)
                if ms is not None:
                    out.append(CaptureInfo(sensor=sensor, path=folder, start_ms=ms))
        else:
            files = glob.glob(os.path.join(self.data_dir, f"{sensor}_capture_*.bin"))
            for p in files:
                ms = self._parse_start_ms_from_name(p)
                if ms is not None:
                    out.append(CaptureInfo(sensor=sensor, path=p, start_ms=ms))
        
        print(out)
        return out


    # ---------- discovery ----------
    def discover(self) -> Tuple[CaptureInfo, CaptureInfo, CaptureInfo, CaptureInfo]:
        """
        Discover one file per sensor and populate self.imu / self.lc / self.rd.

        Returns:
            (imu, lc, rd)
        """
        imu_list = self._collect_for_sensor("IMU")
        lc_list = self._collect_for_sensor("LoadCell")
        rd_list = self._collect_for_sensor("Radar")
        camera_list = self._collect_for_sensor("camera")

        if not imu_list:
            raise FileNotFoundError("No IMU_capture_*.bin found or unparsable names.")
        if not lc_list:
            raise FileNotFoundError("No LoadCell_capture_*.bin found or unparsable names.")
        if not rd_list:
            raise FileNotFoundError("No Radar_capture_*.bin found or unparsable names.")
        if not camera_list:
            raise FileNotFoundError("No camera_capture_* folder found or unparsable names.")

        # Always choose a Radar first (default: latest)
        rd = max(rd_list, key=lambda x: x.start_ms)

        if self.select == "latest":
            imu = max(imu_list, key=lambda x: x.start_ms)
            lc = max(lc_list, key=lambda x: x.start_ms)
            camera = max(camera_list, key=lambda x: x.start_ms)
        elif self.select == "nearest_to_radar":
            imu = min(imu_list, key=lambda x: abs(x.start_ms - rd.start_ms))
            lc = min(lc_list, key=lambda x: abs(x.start_ms - rd.start_ms))
            camera = min(camera_list, key=lambda x: abs(x.start_ms - rd.start_ms))
        else:
            raise ValueError(f"Unknown select='{self.select}'")

        self.imu, self.lc, self.rd, self.camera = imu, lc, rd, camera

        print("[Auto-Discover] Picked trio:")
        print("  IMU     :", os.path.basename(imu.path))
        print("  LoadCell:", os.path.basename(lc.path))
        print("  Radar   :", os.path.basename(rd.path))
        print("  Camera  :", os.path.basename(camera.path))
        return imu, lc, rd, camera

    # ---------- alignment ----------
    def compute_time_shift_ms(
        self,
        reference: Literal["Radar", "IMU", "LoadCell"] = "Radar"
    ) -> Dict[str, int]:
        """
        Compute start-time shifts in milliseconds relative to the chosen reference.

        Positive shift => the sensor starts LATER than the reference.
        Negative shift => the sensor starts EARLIER than the reference.
        """
        if not (self.imu and self.lc and self.rd and self.camera):
            raise RuntimeError("Call discover() before compute_time_shift_ms().")

        ref_map = {"Radar": self.rd, "IMU": self.imu, "LoadCell": self.lc, "Camera": self.camera}
        if reference not in ref_map:
            raise ValueError("reference must be one of 'Radar', 'IMU', 'LoadCell', 'Camera'")

        ref = ref_map[reference]
        return {
            "IMU_vs_ref": self.imu.start_ms - ref.start_ms,
            "LoadCell_vs_ref": self.lc.start_ms - ref.start_ms,
            "Radar_vs_ref": self.rd.start_ms - ref.start_ms,
            "Camera_vs_ref": self.camera.start_ms - ref.start_ms,
        }

    def summary(self, reference: Literal["Radar", "IMU", "LoadCell", "Camera"] = "Radar") -> str:
        """Build a human-readable summary line with shifts."""
        shifts = self.compute_time_shift_ms(reference=reference)
        return (
            f"[Time Alignment (ref={reference})] "
            f"IMU: {shifts['IMU_vs_ref']:+d} ms, "
            f"LoadCell: {shifts['LoadCell_vs_ref']:+d} ms, "
            f"Radar: {shifts['Radar_vs_ref']:+d} ms, "
            f"Camera: {shifts['Camera_vs_ref']:+d} ms"
        )

# ---------- convenient procedural wrapper (optional) ----------
def discover_bin_trio(data_dir: str) -> Tuple[CaptureInfo, CaptureInfo, CaptureInfo, CaptureInfo]:
    """Helper to mirror your previous function name; returns (imu, lc, rd, camera)."""
    return SensorTrioAligner(data_dir).discover()

def compute_time_shift_ms(
    imu: CaptureInfo, lc: CaptureInfo, rd: CaptureInfo, camera: CaptureInfo
) -> Tuple[int, int, int]:
    """
    Backward-compatible helper returning (imu_vs_radar_ms, lc_vs_radar_ms, camera_vs_radar_ms).
    """
    return imu.start_ms - rd.start_ms, lc.start_ms - rd.start_ms, camera.start_ms - rd.start_ms

def crop_by_time(timestamps, values, t_window, inclusive="both"):
    """
    Crop data based on a time window, keeping all arrays aligned with timestamps.

    Args:
        timestamps : 1D sequence (list / np.array / pd.Series)
            Time values (must be same unit as t_window).
        *arrays : sequences
            One or more arrays with the same length as timestamps.
            Each will be cropped according to the time window.
        t_window : tuple (t_start, t_end)
            Start and end of the time window.
        inclusive : str, optional
            How to handle boundaries:
                'both'    -> include [t_start, t_end]
                'left'    -> include [t_start, t_end)
                'right'   -> include (t_start, t_end]
                'neither' -> include (t_start, t_end)

    Returns:
        mask : np.ndarray (boolean array)
            Mask of selected indices.
        cropped : list
            [cropped_timestamps, cropped_array1, cropped_array2, ...]
    """
    t = np.asarray(timestamps)

    t_start, t_end = t_window
    if t_start > t_end:
        raise ValueError("t_window start must be <= end")

    # Build mask depending on inclusivity
    if inclusive == "both":
        mask = (t >= t_start) & (t <= t_end)
    elif inclusive == "left":
        mask = (t >= t_start) & (t < t_end)
    elif inclusive == "right":
        mask = (t > t_start) & (t <= t_end)
    elif inclusive == "neither":
        mask = (t > t_start) & (t < t_end)
    else:
        raise ValueError("inclusive must be 'both', 'left', 'right', or 'neither'")

    # Always include cropped timestamps as the first element
    out = [t[mask]]

    # Apply the same mask to each array
    a = np.asarray(values)
    out.append(a[mask])

    return mask, out

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

def chunk_data(y, t, fs, chunk_size=1.0, step_size=1.0):
    """
    Chunk data into segments of a given size.
    """
    ys = []
    ts = []
    win_size = int(chunk_size * fs)
    step_size = int(step_size * fs)
    # ensure y and t are the same length
    if len(y) != len(t):
        raise ValueError("y and t must be the same length")
    # ensure after chunking, the last chunk is not full, drop the last chunk
    for i in range(0, len(y), step_size):
        if i + win_size > len(y):
            break
        ys.append(y[i:i+win_size])
        ts.append(t[i:i+win_size])
    return np.array(ys), np.array(ts)

def extract_motion_peaks(y, fs, window_size, peak_prominence=2.5):
    """
    Replicates MATLAB logic:
    - Sum STFT dB power across the lower half of the frequency band.
    - Normalize to [0,10].
    - Overlay this energy envelope curve on the spectrogram.
    - Detect peaks in the envelope as potential motion regions.
    Returns:
        t, f, p_db, peak_times, energy_envelope_norm
    """
    t, f, p_db, Sxx = stft_spectrogram(y, fs, window_size)

    if f.size == 0 or t.size == 0:
        return t, f, p_db, np.array([]), np.array([])  # no valid spectrum

    # MATLAB: energy_envelope = sum(p_db(1:end/2, :), 1)
    # In SciPy, frequencies are [0..fs/2], so "first half" = f <= max/2
    half_mask = f <= (f.max() * 0.5)
    p_half = p_db[half_mask, :]
    if p_half.size == 0:
        p_half = p_db  # fallback if empty

    energy_envelope = np.sum(p_half, axis=0)  # sum across frequency -> 1D time series

    # Normalize to [0,10] (same scaling as MATLAB code)
    e_min = np.min(energy_envelope)
    e_max = np.max(energy_envelope)
    if np.isclose(e_max, e_min):
        energy_envelope_norm = np.zeros_like(energy_envelope)
    else:
        energy_envelope_norm = (energy_envelope - e_min) / (e_max - e_min) * 10.0

    # Overlay curve is plotted at the top of the spectrogram
    top_freq = f.max()
    low_freq = f.min()
    overlay_y = top_freq - 0.05 * (top_freq - low_freq) * energy_envelope_norm

    # Detect peaks in normalized energy (same threshold as MATLAB)
    peaks, _ = find_peaks(energy_envelope_norm, prominence=peak_prominence)
    if energy_envelope_norm.mean() < 4: # is valid only if most of the signal is quiet [need further wild motion detection]
        peak_times = t[peaks] if peaks.size > 0 else np.array([])
    else:
        peak_times = np.array([])


    return t, f, p_db, peak_times, energy_envelope_norm

# ---------- CFAR utilities ----------
def num_training_cells(Tr, Gr, Ta, Ga):
    """Compute number of training cells given window sizes."""
    Wr = 2*(Tr+Gr) + 1
    Wa = 2*(Ta+Ga) + 1
    GWr = 2*Gr + 1
    GWa = 2*Ga + 1
    return Wr*Wa - GWr*GWa

def alpha_from_pfa(M, pfa):
    """Scale factor for CA-CFAR threshold."""
    return M * ((pfa)**(-1.0/M) - 1.0)

def to_power_domain(H, domain='dB'):
    """
    Convert heatmap to linear power domain for CFAR thresholding.
    domain: 'linear_power' | 'linear_amplitude' | 'dB'
    """
    H = np.asarray(H, dtype=float)
    if domain == 'linear_power':
        return H
    elif domain == 'linear_amplitude':
        return H**2
    elif domain == 'dB':
        return 10.0**(H/10.0)
    else:
        raise ValueError("domain must be one of {'linear_power','linear_amplitude','dB'}")

def cfar_2d(heatmap, Tr=3, Gr=1, Ta=1, Ga=1, pfa=8e-2, domain="dB"):
    """
    2D CA-CFAR detector on heatmap (range x angle).
    
    Parameters:
    -----------
    heatmap : 2D ndarray
        Input radar heatmap.
    Tr, Gr : int
        Training and guard cells along range dimension.
    Ta, Ga : int
        Training and guard cells along angle dimension.
    pfa : float
        Desired probability of false alarm.
    
    Returns:
    --------
    detections : list of (range_idx, angle_idx, value)
        List of detected bright bins.
    """

    heatmap = np.array(heatmap)
    heatmap = to_power_domain(heatmap, domain)
    num_range, num_angle = heatmap.shape
    detections = []

    # scaling factor from false alarm rate
    M = num_training_cells(Tr, Gr, Ta, Ga)
    alpha = alpha_from_pfa(M, pfa)

    for r in range(Tr+Gr, num_range-(Tr+Gr)):
        for a in range(Ta+Ga, num_angle-(Ta+Ga)):
            # window boundaries
            r_start, r_end = r-(Tr+Gr), r+(Tr+Gr)+1
            a_start, a_end = a-(Ta+Ga), a+(Ta+Ga)+1
            window = heatmap[r_start:r_end, a_start:a_end].copy()

            # zero out CUT+guard region
            window[Tr:Tr+2*Gr+1, Ta:Ta+2*Ga+1] = 0

            # estimate local noise level
            noise_level = np.sum(window) / np.count_nonzero(window)

            # threshold
            threshold = alpha * noise_level

            if heatmap[r, a] > threshold:
                detections.append((r, a))

    # do not print to another line
    # print(f"Detections (range, angle, value) where r_lo < {Tr+Gr} and a_lo < {Ta+Ga}: {detections}", end="\r")
    return detections