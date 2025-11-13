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