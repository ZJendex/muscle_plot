# Spatial Localization Analysis Data - Usage Guide

## Overview

This guide explains how to use the data saved by `cc_spatial_localization_analyzeTT_PLOTTING_reuse.py`. The script analyzes radar signals at multiple spatial locations and saves all computed features for later reuse.

## Output Files

The script generates two types of files:

1. **PNG Image**: `phase_signal_all_{name_data}_{range_value}_{angle_value}.png`
   - Combined visualization of all locations
   
2. **Pickle Data**: `saved_data_{name_data}_{range_value}_{angle_value}.pkl`
   - All computed data for replotting and analysis

## Data Structure

The pickle file contains a dictionary with three main sections:

```python
saved_data = {
    'metadata': {...},           # Experiment metadata
    'radar_heatmap': <array>,    # Aggregated radar heatmap
    'location_data': {...}       # Per-location analysis results
}
```

### 1. Metadata Section

```python
metadata = {
    'name_data': str,          # Subject identifier (e.g., "1017_jerry")
    'folder_name': str,        # Folder name (e.g., "1017_jerry_ccMVIC70_r1")
    'range_value': int,        # Range bin value from filename
    'angle_value': int,        # Angle bin value from filename
    'locations': list,         # List of (range, angle) tuples analyzed
    'time_take': list,         # Time range analyzed [start, ..., end] in seconds
    'fs': float                # Sampling frequency (Hz)
}
```

### 2. Radar Heatmap

```python
radar_heatmap: numpy.ndarray
# Shape: (range_bins, angle_bins)
# Aggregated radar heatmap data from 5-20 range bins
# Already normalized (divided by 10000)
```

### 3. Location Data

Each location has a key `loc_{range_bin}_{angle_bin}` containing:

```python
location_data = {
    'loc_11_8': {
        'range_bin': int,              # Range bin coordinate
        'angle_bin': int,              # Angle bin coordinate
        'time_axis': numpy.ndarray,    # Time vector (seconds), cropped to 5-25s, starts at 0
        'radar_v': numpy.ndarray,      # Phase signal 2nd derivative (acceleration m/s²)
        'stft': {
            't': numpy.ndarray,        # STFT time vector
            'f': numpy.ndarray,        # STFT frequency vector (Hz)
            'p_db': numpy.ndarray,     # Power spectrum in dB (2D array)
            'Sxx': numpy.ndarray       # STFT magnitude (2D array)
        },
        'chaos_features': {
            'entr_values': numpy.ndarray,  # Entropy values per 1-second chunk
            'det_values': numpy.ndarray,   # Determinism values per 1-second chunk
            'entr_std': float,             # Standard deviation of entropy
            'det_std': float               # Standard deviation of determinism
        }
    },
    'loc_39_2': {...},
    'loc_16_11': {...},
    'loc_11_6': {...}
}
```

## How to Load and Use the Data

### Basic Loading

```python
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the saved data
file_path = "datasets/isotonic_analysis/DRAW_Results/{folder_name}/saved_data_{name}_{range}_{angle}.pkl"
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# Access metadata
print(f"Subject: {data['metadata']['name_data']}")
print(f"Sampling frequency: {data['metadata']['fs']} Hz")
print(f"Analyzed locations: {data['metadata']['locations']}")
```

### Example 1: Plot Phase Signal for a Specific Location

```python
# Select a location
loc_key = 'loc_11_8'  # Range bin 11, Angle bin 8
loc_data = data['location_data'][loc_key]

# Plot the phase signal
plt.figure(figsize=(12, 4))
plt.plot(loc_data['time_axis'], loc_data['radar_v'])
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.title(f"Phase Signal at Range={loc_data['range_bin']}, Angle={loc_data['angle_bin']}")
plt.grid(True)
plt.show()
```

### Example 2: Plot STFT Spectrogram

```python
loc_data = data['location_data']['loc_11_8']
stft = loc_data['stft']

plt.figure(figsize=(12, 6))
plt.pcolormesh(stft['t'], stft['f'], stft['p_db'], shading='auto')
plt.colorbar(label='Power (dB)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('STFT Spectrogram')
plt.ylim(0, 100)  # Focus on 0-100 Hz
plt.show()
```

### Example 3: Plot Chaos Features

```python
loc_data = data['location_data']['loc_11_8']
chaos = loc_data['chaos_features']

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plot Entropy
axes[0].plot(chaos['entr_values'], linewidth=2)
axes[0].set_ylabel('ENTR')
axes[0].set_title(f"Entropy (STD: {chaos['entr_std']:.3f})")
axes[0].grid(True)

# Plot Determinism
axes[1].plot(chaos['det_values'], linewidth=2)
axes[1].set_ylabel('DET')
axes[1].set_xlabel('Chunk Index (1-second chunks)')
axes[1].set_title(f"Determinism (STD: {chaos['det_std']:.3f})")
axes[1].grid(True)

plt.tight_layout()
plt.show()
```

### Example 4: Compare All Locations

```python
# Compare entropy standard deviation across all locations
fig, ax = plt.subplots(figsize=(10, 6))

locations = []
entr_stds = []

for loc_key, loc_data in data['location_data'].items():
    range_bin = loc_data['range_bin']
    angle_bin = loc_data['angle_bin']
    entr_std = loc_data['chaos_features']['entr_std']
    
    locations.append(f"({range_bin},{angle_bin})")
    entr_stds.append(entr_std)

ax.bar(locations, entr_stds)
ax.set_xlabel('Location (Range, Angle)')
ax.set_ylabel('Entropy Standard Deviation')
ax.set_title('Entropy Variability Across Locations')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### Example 5: Plot Radar Heatmap

```python
heatmap = data['radar_heatmap']
metadata = data['metadata']

plt.figure(figsize=(10, 8))
plt.imshow(heatmap, aspect='auto', origin='lower',
           extent=[0, metadata['angle_value'], 5, 20])
plt.colorbar(label='Intensity (normalized)')
plt.xlabel('Angle Bins')
plt.ylabel('Range Bins')
plt.title(f"Radar Heatmap - {metadata['name_data']}")

# Mark analyzed locations
for loc in metadata['locations']:
    plt.plot(loc[1], loc[0], 'r*', markersize=15)

plt.show()
```

### Example 6: Advanced - Reconstruct Full Plot

```python
def reconstruct_plot(data):
    """Recreate the original 4x4 subplot figure"""
    locations_data = data['location_data']
    
    fig, axes = plt.subplots(4, len(locations_data), figsize=(24, 16))
    
    for col_idx, (loc_key, loc_data) in enumerate(locations_data.items()):
        # Row 0: Phase signal
        axes[0, col_idx].plot(loc_data['time_axis'], loc_data['radar_v'])
        axes[0, col_idx].set_title(f"({loc_data['range_bin']}, {loc_data['angle_bin']})")
        
        # Row 1: STFT
        stft = loc_data['stft']
        axes[1, col_idx].pcolormesh(stft['t'], stft['f'], stft['p_db'])
        axes[1, col_idx].set_ylim(0, 100)
        
        # Row 2: Entropy
        chaos = loc_data['chaos_features']
        axes[2, col_idx].plot(chaos['entr_values'])
        axes[2, col_idx].set_ylim(1.5, 5)
        
        # Row 3: Determinism
        axes[3, col_idx].plot(chaos['det_values'])
        axes[3, col_idx].set_ylim(0.5, 1)
    
    plt.tight_layout()
    return fig

# Use the function
fig = reconstruct_plot(data)
plt.show()
```

## Data Processing Notes

### Time Axis
- Original data is cropped to **5-25 seconds** of the recording
- Time axis is **normalized to start at 0**
- Length depends on sampling frequency: `len(time_axis) = 20 * fs`

### Phase Signal (radar_v)
- Represents the **2nd derivative of phase** (acceleration)
- Units: **m/s²**
- Typical range: -100 to +100 m/s²

### STFT Parameters
- Window size: **256 samples**
- Sampling rate: **1000 Hz** (from metadata['fs'])
- Frequency range: **0-100 Hz** (typically displayed)

### Chaos Features
- Computed on **1-second chunks** of data
- **Embedding dimension (m)**: 3
- **Time delay (tau)**: Computed using AMI method
- **Recurrence threshold**: 10
- Number of chunks ≈ 20 (for 20-second signal)

## File Naming Convention

```
saved_data_{subject}_{range}_{angle}.pkl
```

Examples:
- `saved_data_1017_jerry_11_8.pkl`
- `saved_data_1024_freddy_12_8.pkl`

## Tips for Analysis

1. **Compare Locations**: Use `entr_std` to identify locations with high variability
2. **Fatigue Detection**: Compare early vs. late chunks in `entr_values` and `det_values`
3. **Frequency Analysis**: Use STFT data to identify dominant motion frequencies
4. **Spatial Patterns**: Use radar_heatmap to understand spatial distribution

## Troubleshooting

### Common Issues

**Q: KeyError when accessing location**
```python
# Check available locations first
print(data['location_data'].keys())
```

**Q: Array dimension mismatch**
```python
# Check array shapes
loc_data = data['location_data']['loc_11_8']
print(f"Time axis shape: {loc_data['time_axis'].shape}")
print(f"Radar signal shape: {loc_data['radar_v'].shape}")
```

**Q: Missing file**
```python
import os
# Check if file exists
file_path = "path/to/saved_data.pkl"
if os.path.exists(file_path):
    print("File found!")
else:
    print("File not found. Check path and filename.")
```

## Contact & Support

For questions about the data format or analysis methods, refer to:
- Original script: `cc_spatial_localization_analyzeTT_PLOTTING_reuse.py`
- Chaos features: `utils/chaos.py`
- Signal processing: `utils/signal.py`

