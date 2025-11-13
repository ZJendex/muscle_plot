# Localization Analysis Results

This directory contains the results from muscle localization analysis using radar data.

## 📂 Files

- `localization_error_CDF_YYYYMMDD_HHMMSS.png` - CDF plots showing localization errors
- `localization_results_YYYYMMDD_HHMMSS.pkl` - Python pickle file containing all analysis results

## 📖 How to Load the Pickle File

### Basic Loading

```python
import pickle

# Load the results
with open('localization_results_YYYYMMDD_HHMMSS.pkl', 'rb') as f:
    results_list = pickle.load(f)

# results_list is a list of dictionaries
print(f"Total samples: {len(results_list)}")
print(f"First result: {results_list[0]}")
```

### Data Structure

Each entry in `results_list` is a dictionary with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `name_data` | str | Subject identifier (format: MMDD_name) |
| `folder_name` | str | Source folder containing the analyzed data |
| `predict_location` | tuple | Predicted muscle location (range_bin, angle_bin) |
| `true_location` | tuple | Ground truth location (range_bin, angle_bin) |
| `distance` | float | Euclidean distance error in meters (Cartesian) |
| `range_difference` | float | Absolute difference in forward distance (x) in meters |
| `angle_difference` | float | Absolute difference in lateral distance (y) in meters |
| `range_bin_difference` | int | Absolute difference in range bin index |
| `angle_bin_difference` | int | Absolute difference in angle bin index |

### Example Usage

```python
import pickle
import numpy as np

# Load the results
with open('localization_results_YYYYMMDD_HHMMSS.pkl', 'rb') as f:
    results_list = pickle.load(f)

# Extract specific data
range_bin_diffs = [r['range_bin_difference'] for r in results_list]
angle_bin_diffs = [r['angle_bin_difference'] for r in results_list]
distances = [r['distance'] for r in results_list]

# Calculate statistics
mean_range_error = np.mean(range_bin_diffs)
mean_angle_error = np.mean(angle_bin_diffs)
mean_distance_error = np.mean(distances)

print(f"Mean range bin error: {mean_range_error:.3f} bins")
print(f"Mean angle bin error: {mean_angle_error:.3f} bins")
print(f"Mean distance error: {mean_distance_error:.3f} m")

# Filter by subject
subject = "1012_aditi"
subject_results = [r for r in results_list if r['name_data'] == subject]
print(f"\nResults for {subject}: {len(subject_results)} samples")

# Find outliers (distance > 4m)
outliers = [r for r in results_list if r['distance'] > 4]
print(f"\nOutliers (distance > 4m): {len(outliers)}")
for outlier in outliers:
    print(f"  - {outlier['name_data']}: {outlier['distance']:.3f}m")
```

### Reproduce the CDF Plot

```python
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the results
with open('localization_results_YYYYMMDD_HHMMSS.pkl', 'rb') as f:
    results_list = pickle.load(f)

# Extract data
range_bin_diffs = [r['range_bin_difference'] for r in results_list]
angle_bin_diffs = [r['angle_bin_difference'] for r in results_list]
distances = [r['distance'] for r in results_list]

# Calculate statistics
mean_range_bin_error = np.mean(range_bin_diffs)
mean_angle_bin_error = np.mean(angle_bin_diffs)
mean_distance_error = np.mean(distances)

# Create figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

# Plot CDF for range bin difference
ax1.plot(np.sort(range_bin_diffs), 
         np.arange(1, len(range_bin_diffs) + 1) / len(range_bin_diffs), 
         linewidth=2)
ax1.set_xlabel("Range Bin Difference", fontsize=12)
ax1.set_ylabel("CDF", fontsize=12)
ax1.set_title(f"CDF of Range Bin Difference\nMean Abs Error: {mean_range_bin_error:.3f} bins", 
              fontsize=14)
ax1.grid(True, alpha=0.3)

# Plot CDF for angle bin difference
ax2.plot(np.sort(angle_bin_diffs), 
         np.arange(1, len(angle_bin_diffs) + 1) / len(angle_bin_diffs), 
         linewidth=2, color='orange')
ax2.set_xlabel("Angle Bin Difference", fontsize=12)
ax2.set_ylabel("CDF", fontsize=12)
ax2.set_title(f"CDF of Angle Bin Difference\nMean Abs Error: {mean_angle_bin_error:.3f} bins", 
              fontsize=14)
ax2.grid(True, alpha=0.3)

# Plot CDF for distance
ax3.plot(np.sort(distances), 
         np.arange(1, len(distances) + 1) / len(distances), 
         linewidth=2, color='green')
ax3.set_xlabel("Distance (m)", fontsize=12)
ax3.set_ylabel("CDF", fontsize=12)
ax3.set_title(f"CDF of Distance\nMean Abs Error: {mean_distance_error:.3f} m", 
              fontsize=14)
ax3.grid(True, alpha=0.3)

plt.suptitle(f"Localization Error Analysis (Total samples: {len(results_list)})", 
             fontsize=16, y=1.02)
plt.tight_layout()
plt.show()
```

## 🔧 Radar Parameters

- **Range resolution:** 0.0422 m/bin
- **Angle FFT size:** 16
- **Range filter:** 7-15 bins (outliers excluded)
- **Top sorted entropy values:** 5

## 📊 Analysis Overview

The analysis evaluates muscle localization accuracy by comparing predicted locations (from entropy-based detection) with ground truth locations. Three metrics are computed:

1. **Range Bin Difference:** Discretized range error
2. **Angle Bin Difference:** Discretized angle error  
3. **Distance:** Euclidean distance error in Cartesian coordinates (meters)

The CDF plots show the cumulative distribution of these errors across all subjects and trials.

