import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
import random
import torch
from sklearn.metrics import pairwise_distances
from joblib import parallel_backend
import nolds
from pyentrp import entropy as pyentrp_entropy

def compute_recurrence_plot(signal, m, tau, percentage):
    """
    Compute recurrence plot using time-delay embedding.

    Parameters
    ----------
    signal : np.ndarray
        Input signal (1D array)
    m : int
        Embedding dimension
    tau : int
        Time delay (in samples)
    percentage : float
        Percentage of points to consider as recurrent (threshold percentile)

    Returns
    -------
    recurrence_matrix : np.ndarray
        Binary recurrence matrix
    """
    # Create time-delay embedding
    n_points = len(signal) - (m - 1) * tau

    if n_points < 10:
        raise ValueError("Signal too short for given m and tau")

    # Build m-dimensional embedding
    embedding = np.zeros((n_points, m))
    for i in range(m):
        embedding[:, i] = signal[i * tau : i * tau + n_points]

    # normalize the embedding
    embedding = StandardScaler().fit_transform(embedding)

    # Compute pairwise distances
    distances = squareform(pdist(embedding, metric='euclidean'))

    # Determine threshold based on percentile
    threshold = np.percentile(distances, percentage)

    # Create binary recurrence matrix
    recurrence_matrix = (distances <= threshold).astype(int)    

    return recurrence_matrix, distances

def compute_rp_cpu_parallel(signal, m, tau, percentage, n_jobs=-1):
    n_points = len(signal) - (m - 1) * tau
    if n_points < 10:
        raise ValueError("Signal too short")

    # Embedding + normalization
    embedding = np.column_stack([signal[i*tau:i*tau+n_points] for i in range(m)])
    embedding = StandardScaler().fit_transform(embedding)

    # Parallel pairwise distances (uses OpenMP)
    with parallel_backend('threading'):
        D = pairwise_distances(embedding, metric='euclidean', n_jobs=n_jobs)

    eps = np.percentile(D, percentage)
    RP = (D <= eps).astype(np.uint8)
    return RP, D

def raq_measures(rp_matrix, l_min=2, v_min=2):
    """
    Calculate RQA measures that match pyunicorn results exactly.
    
    Parameters
    ----------
    rp_matrix : np.ndarray
        Binary recurrence matrix (0s and 1s)
    l_min : int, optional
        Minimum diagonal line length for DET calculation (default: 2)
    v_min : int, optional
        Minimum vertical line length for LAM calculation (default: 2)
    
    Returns
    -------
    dict
        Dictionary containing RQA measures
    """
    import numpy as np
    
    # Ensure binary matrix
    rp = (rp_matrix > 0).astype(int)
    N = rp.shape[0]
    
    # Calculate Recurrence Rate (RR)
    total_points = N * N
    recurrence_points = np.sum(rp)
    RR = recurrence_points / total_points
    
    # Find diagonal lines (excluding main diagonal)
    diagonal_lines = []
    visited = np.zeros_like(rp, dtype=bool)
    
    for i in range(N):
        for j in range(N):
            if rp[i, j] == 1 and not visited[i, j] and i != j:  # Exclude main diagonal
                # Find the longest diagonal line starting from (i,j)
                line_length = 1
                k = 1
                while (i + k < N and j + k < N and rp[i + k, j + k] == 1):
                    line_length += 1
                    k += 1
                
                # Mark all points in this line as visited
                for l in range(line_length):
                    if i + l < N and j + l < N:
                        visited[i + l, j + l] = True
                
                if line_length >= 1:
                    diagonal_lines.append(line_length)
    
    # Find vertical lines
    vertical_lines = []
    visited_vert = np.zeros_like(rp, dtype=bool)
    
    for i in range(N):
        for j in range(N):
            if rp[i, j] == 1 and not visited_vert[i, j]:
                # Find the longest vertical line starting from (i,j)
                line_length = 1
                k = 1
                while (i + k < N and rp[i + k, j] == 1):
                    line_length += 1
                    k += 1
                
                # Mark all points in this line as visited
                for l in range(line_length):
                    if i + l < N:
                        visited_vert[i + l, j] = True
                
                if line_length >= 1:
                    vertical_lines.append(line_length)
    
    # Filter lines by minimum length
    diagonal_lines_filtered = [l for l in diagonal_lines if l >= l_min]
    vertical_lines_filtered = [l for l in vertical_lines if l >= v_min]
    
    # Calculate DET (Determinism)
    if diagonal_lines:
        # Count frequency of each diagonal line length
        all_unique_lengths, all_counts = np.unique(diagonal_lines, return_counts=True)
        all_P_l = dict(zip(all_unique_lengths, all_counts))
        
        # Count frequency of diagonal lines >= l_min
        if diagonal_lines_filtered:
            unique_lengths, counts = np.unique(diagonal_lines_filtered, return_counts=True)
            P_l = dict(zip(unique_lengths, counts))
            
            numerator = sum(l * P_l[l] for l in P_l.keys())
            denominator = sum(l * all_P_l[l] for l in all_P_l.keys())
            DET = numerator / denominator if denominator > 0 else 0
        else:
            DET = 0
    else:
        DET = 0
    
    # Calculate LAM (Laminarity)
    if vertical_lines:
        # Count frequency of each vertical line length
        all_unique_lengths, all_counts = np.unique(vertical_lines, return_counts=True)
        all_P_v = dict(zip(all_unique_lengths, all_counts))
        
        # Count frequency of vertical lines >= v_min
        if vertical_lines_filtered:
            unique_lengths, counts = np.unique(vertical_lines_filtered, return_counts=True)
            P_v = dict(zip(unique_lengths, counts))
            
            numerator = sum(v * P_v[v] for v in P_v.keys())
            denominator = sum(v * all_P_v[v] for v in all_P_v.keys())
            LAM = numerator / denominator if denominator > 0 else 0
        else:
            LAM = 0
    else:
        LAM = 0
    
    # Calculate ENTR (Entropy of diagonal lines) - using filtered lines
    if diagonal_lines_filtered:
        unique_lengths, counts = np.unique(diagonal_lines_filtered, return_counts=True)
        P_l = counts / np.sum(counts)  # Normalize to probabilities
        ENTR = -np.sum(P_l * np.log2(P_l + 1e-10))  # Add small epsilon to avoid log(0)
    else:
        ENTR = 0
    
    # Calculate V_ENTR (Entropy of vertical lines) - using filtered lines
    if vertical_lines_filtered:
        unique_lengths, counts = np.unique(vertical_lines_filtered, return_counts=True)
        P_v = counts / np.sum(counts)  # Normalize to probabilities
        V_ENTR = -np.sum(P_v * np.log2(P_v + 1e-10))  # Add small epsilon to avoid log(0)
    else:
        V_ENTR = 0
    
    # Calculate L (Average diagonal line length) - using filtered lines
    L = np.mean(diagonal_lines_filtered) if diagonal_lines_filtered else 0
    
    # Calculate TT (Average vertical line length - Trapping Time) - using filtered lines
    TT = np.mean(vertical_lines_filtered) if vertical_lines_filtered else 0
    
    # Calculate L_MAX (Maximum diagonal line length) - using filtered lines
    L_MAX = max(diagonal_lines_filtered) if diagonal_lines_filtered else 0
    
    # Calculate V_MAX (Maximum vertical line length) - using filtered lines
    V_MAX = max(vertical_lines_filtered) if vertical_lines_filtered else 0
    
    # Calculate DIV (Divergence = 1/L_MAX)
    DIV = 1.0 / L_MAX if L_MAX > 0 else 0
    
    # TRAP is the same as TT
    TRAP = TT
    
    return {
        'RR': RR,           # Recurrence Rate
        'DET': DET,         # Determinism
        'LAM': LAM,         # Laminarity
        'ENTR': ENTR,       # Entropy of diagonal lines
        'V_ENTR': V_ENTR,   # Entropy of vertical lines
        'L': L,             # Average diagonal line length
        'TT': TT,           # Average vertical line length (Trapping Time)
        'L_MAX': L_MAX,     # Maximum diagonal line length
        'V_MAX': V_MAX,     # Maximum vertical line length
        'DIV': DIV,         # Divergence
        'TRAP': TRAP        # Trapping time
    }

def compute_ami_tau(signal, fs, min_tau_ms=10, max_tau_ms=30, n_bins=20):
    """
    Compute optimal time delay (tau) using Average Mutual Information (AMI).

    The optimal tau is found at the first minimum of the AMI function.

    Parameters
    ----------
    signal : np.ndarray
        Input signal (1D array)
    fs : float
        Sampling frequency in Hz
    min_tau_ms : float
        Minimum tau to search in milliseconds
    max_tau_ms : float
        Maximum tau to search in milliseconds
    n_bins : int
        Number of bins for histogram

    Returns
    -------
    tau_optimal : int
        Optimal time delay in samples
    ami_values : np.ndarray
        AMI values for each tau
    tau_range : np.ndarray
        Range of tau values tested (in samples)
    """
    # Convert time range to samples
    min_tau_samples = int(min_tau_ms * fs / 1000)
    max_tau_samples = int(max_tau_ms * fs / 1000)

    # Ensure reasonable range
    min_tau_samples = max(1, min_tau_samples)
    max_tau_samples = min(len(signal) // 4, max_tau_samples)

    # Array to store AMI values
    tau_range = np.arange(min_tau_samples, max_tau_samples + 1)
    ami_values = np.zeros(len(tau_range))

    # Normalize signal to [0, 1] for better histogram binning
    signal_norm = (signal - signal.min()) / (signal.max() - signal.min() + 1e-10)

    for i, tau in enumerate(tau_range):
        # Create time-delayed versions
        x = signal_norm[:-tau]
        y = signal_norm[tau:]

        # Compute 2D histogram
        hist_2d, _, _ = np.histogram2d(x, y, bins=n_bins)

        # Compute marginal distributions
        px = np.sum(hist_2d, axis=1)
        py = np.sum(hist_2d, axis=0)

        # Normalize to probabilities
        hist_2d = hist_2d / (hist_2d.sum() + 1e-10)
        px = px / (px.sum() + 1e-10)
        py = py / (py.sum() + 1e-10)

        # Compute mutual information: I(X;Y) = H(X) + H(Y) - H(X,Y)
        h_x = entropy(px + 1e-10)
        h_y = entropy(py + 1e-10)
        h_xy = entropy(hist_2d.flatten() + 1e-10)

        ami_values[i] = h_x + h_y - h_xy

    # Find first local minimum
    tau_optimal = min_tau_samples
    for i in range(1, len(ami_values) - 1):
        if ami_values[i] < ami_values[i-1] and ami_values[i] < ami_values[i+1]:
            tau_optimal = tau_range[i]
            break

    # If no minimum found, use the tau with minimum AMI value
    if tau_optimal == min_tau_samples:
        tau_optimal = tau_range[np.argmin(ami_values)]

    return tau_optimal, ami_values, tau_range

def compute_fnn_m(signal, tau, max_m=10, R_tol=15.0, A_tol=2.0):
    """
    Compute optimal embedding dimension (m) using False Nearest Neighbors (FNN).

    Parameters
    ----------
    signal : np.ndarray
        Input signal (1D array)
    tau : int
        Time delay (from AMI)
    max_m : int
        Maximum embedding dimension to test
    R_tol : float
        Tolerance for distance criterion (default: 15.0)
    A_tol : float
        Tolerance for size criterion (default: 2.0)

    Returns
    -------
    m_optimal : int
        Optimal embedding dimension
    fnn_percentages : np.ndarray
        Percentage of false nearest neighbors for each m
    m_range : np.ndarray
        Range of m values tested
    """
    N = len(signal)
    m_range = np.arange(1, max_m + 1)
    fnn_percentages = np.zeros(len(m_range))

    for m_idx, m in enumerate(m_range):
        # Number of points in m-dimensional embedding
        n_points = N - (m + 1) * tau

        if n_points < 10:
            # Not enough points for this embedding
            fnn_percentages[m_idx] = np.nan
            continue

        # Build m-dimensional embedding
        embedding_m = np.zeros((n_points, m))
        for i in range(m):
            embedding_m[:, i] = signal[i*tau : i*tau + n_points]

        # Build (m+1)-dimensional embedding
        embedding_m1 = np.zeros((n_points, m + 1))
        for i in range(m + 1):
            embedding_m1[:, i] = signal[i*tau : i*tau + n_points]

        # For each point, find its nearest neighbor in m-dimensional space
        n_false_neighbors = 0

        for i in range(n_points):
            # Compute distances to all other points in m-dimensional space
            distances_m = np.linalg.norm(embedding_m - embedding_m[i], axis=1)

            # Find nearest neighbor (excluding itself)
            distances_m[i] = np.inf
            nn_idx = np.argmin(distances_m)
            dist_m = distances_m[nn_idx]

            # Compute distance in (m+1)-dimensional space
            dist_m1 = np.linalg.norm(embedding_m1[i] - embedding_m1[nn_idx])

            # Check if this is a false nearest neighbor
            # Criterion 1: Distance increase is too large
            if dist_m > 0:
                dist_increase_ratio = np.abs(dist_m1 - dist_m) / dist_m

                # Criterion 2: Distance is too large relative to signal size
                signal_std = np.std(signal)

                if dist_increase_ratio > R_tol or dist_m1 / signal_std > A_tol:
                    n_false_neighbors += 1

        fnn_percentages[m_idx] = 100 * n_false_neighbors / n_points

    # Find optimal m (first m where FNN% drops below threshold, e.g., 1%)
    m_optimal = max_m
    threshold = 1.0  # 1% threshold

    for m_idx, fnn_pct in enumerate(fnn_percentages):
        if not np.isnan(fnn_pct) and fnn_pct < threshold:
            m_optimal = m_range[m_idx]
            break

    return m_optimal, fnn_percentages, m_range

# ===========use pyentrp library to calculate chaos features===========
def calculate_sample_entropy(signal, m=2, r=0.2):
    """Calculate Sample Entropy (SampEn)."""
    try:
        r_abs = r * np.std(signal)
        return pyentrp_entropy.sample_entropy(signal, m, r_abs)[0]
    except:
        return 0.0

# ===========use nolds library to calculate chaos features===========
def calculate_correlation_dimension(signal, emb_dim=10, max_dim=5):
    """Calculate Correlation Dimension."""
    try:
        return nolds.corr_dim(signal, emb_dim=emb_dim)
    except:
        return 0.0

def calculate_dfa(signal):
    """Calculate Detrended Fluctuation Analysis (DFA) exponent."""
    try:
        return nolds.dfa(signal)
    except:
        return 0.0

def calculate_higuchi_fd(signal, k_max=10):
    """Calculate Higuchi Fractal Dimension."""
    try:
        return nolds.hurst_rs(signal)
    except:
        return 0.0

def calculate_lyapunov_rosenstein(signal, emb_dim=10, tau=1):
    """Calculate Maximum Lyapunov Exponent using Rosenstein's method."""
    try:
        return nolds.lyap_r(signal, emb_dim=emb_dim, lag=tau)
    except:
        return 0.0

def get_chaos_features(signal, tau, m=3):
    """Get chaos features."""
    return{
        'sample_entropy': calculate_sample_entropy(signal, m),
        'correlation_dimension': calculate_correlation_dimension(signal, m),
        'dfa': calculate_dfa(signal),
        'higuchi_fd': calculate_higuchi_fd(signal),
    }