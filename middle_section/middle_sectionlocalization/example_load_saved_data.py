"""
Example: How to Load and Use Saved Spatial Localization Data
==============================================================

This script demonstrates how to load and work with data saved by
cc_spatial_localization_analyzeTT_PLOTTING_reuse.py

For complete documentation, see README_spatial_localization_data.md
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================================
# CONFIGURATION: Update these paths for your data
# ============================================================================
name_data = "1017_jerry"
folder_name = "1017_jerry_ccMVIC70_r1"
range_value = 11
angle_value = 8

file_path = f"datasets\\isotonic_analysis\\DRAW_Results\\{folder_name}\\saved_data_{name_data}_{range_value}_{angle_value}.pkl"

# ============================================================================
# LOAD DATA
# ============================================================================
print("Loading saved data...")
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    print("Please update the configuration variables above.")
    exit(1)

with open(file_path, 'rb') as f:
    data = pickle.load(f)

print(f"✓ Loaded data for subject: {data['metadata']['name_data']}")
print(f"  Sampling frequency: {data['metadata']['fs']} Hz")
print(f"  Number of locations: {len(data['location_data'])}")
print(f"  Locations analyzed: {data['metadata']['locations']}")

# ============================================================================
# EXAMPLE 1: Inspect Available Locations
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 1: Available Locations and Their Properties")
print("="*70)

for loc_key, loc_data in data['location_data'].items():
    range_bin = loc_data['range_bin']
    angle_bin = loc_data['angle_bin']
    entr_std = loc_data['chaos_features']['entr_std']
    det_std = loc_data['chaos_features']['det_std']
    
    print(f"\n{loc_key}:")
    print(f"  Position: Range={range_bin}, Angle={angle_bin}")
    print(f"  Signal length: {len(loc_data['radar_v'])} samples")
    print(f"  Duration: {loc_data['time_axis'][-1]:.1f} seconds")
    print(f"  Entropy STD: {entr_std:.4f}")
    print(f"  Determinism STD: {det_std:.4f}")

# ============================================================================
# EXAMPLE 2: Plot Single Location - Phase Signal
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 2: Plotting Phase Signal for Location (11, 8)")
print("="*70)

loc_data = data['location_data']['loc_11_8']

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(loc_data['time_axis'], loc_data['radar_v'], linewidth=0.8, color='steelblue')
ax.set_xlabel('Time (s)', fontsize=11)
ax.set_ylabel('Acceleration (m/s²)', fontsize=11)
ax.set_title(f"Phase Signal at Range={loc_data['range_bin']}, Angle={loc_data['angle_bin']}", 
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim(-100, 100)
plt.tight_layout()
plt.savefig('example_output_phase_signal.png', dpi=150)
print("✓ Saved: example_output_phase_signal.png")
plt.close()

# ============================================================================
# EXAMPLE 3: Plot STFT Spectrogram
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 3: Plotting STFT Spectrogram")
print("="*70)

stft = loc_data['stft']

fig, ax = plt.subplots(figsize=(14, 6))
pcm = ax.pcolormesh(stft['t'], stft['f'], stft['p_db'], shading='auto', cmap='viridis')
plt.colorbar(pcm, ax=ax, label='Power (dB)')
ax.set_xlabel('Time (s)', fontsize=11)
ax.set_ylabel('Frequency (Hz)', fontsize=11)
ax.set_title('STFT Spectrogram', fontsize=13, fontweight='bold')
ax.set_ylim(0, 100)  # Focus on 0-100 Hz
plt.tight_layout()
plt.savefig('example_output_stft.png', dpi=150)
print("✓ Saved: example_output_stft.png")
plt.close()

# ============================================================================
# EXAMPLE 4: Plot Chaos Features Over Time
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 4: Plotting Chaos Features (ENTR and DET)")
print("="*70)

chaos = loc_data['chaos_features']

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Entropy
axes[0].plot(chaos['entr_values'], linewidth=2, color='orangered', marker='o', markersize=5)
axes[0].axhline(np.mean(chaos['entr_values']), color='gray', linestyle='--', 
                label=f'Mean: {np.mean(chaos["entr_values"]):.3f}')
axes[0].set_ylabel('Entropy (ENTR)', fontsize=11, fontweight='bold')
axes[0].set_title(f"Entropy Over Time (STD: {chaos['entr_std']:.4f})", 
                  fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend()
axes[0].set_ylim(1.5, 5)

# Determinism
axes[1].plot(chaos['det_values'], linewidth=2, color='darkblue', marker='o', markersize=5)
axes[1].axhline(np.mean(chaos['det_values']), color='gray', linestyle='--', 
                label=f'Mean: {np.mean(chaos["det_values"]):.3f}')
axes[1].set_ylabel('Determinism (DET)', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Chunk Index (1-second chunks)', fontsize=11)
axes[1].set_title(f"Determinism Over Time (STD: {chaos['det_std']:.4f})", 
                  fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend()
axes[1].set_ylim(0.5, 1.0)

plt.tight_layout()
plt.savefig('example_output_chaos_features.png', dpi=150)
print("✓ Saved: example_output_chaos_features.png")
plt.close()

# ============================================================================
# EXAMPLE 5: Compare All Locations
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 5: Comparing Entropy Variability Across All Locations")
print("="*70)

locations = []
entr_stds = []
det_stds = []

for loc_key, loc_data in data['location_data'].items():
    range_bin = loc_data['range_bin']
    angle_bin = loc_data['angle_bin']
    locations.append(f"({range_bin},{angle_bin})")
    entr_stds.append(loc_data['chaos_features']['entr_std'])
    det_stds.append(loc_data['chaos_features']['det_std'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Entropy STD comparison
axes[0].bar(locations, entr_stds, color='orangered', alpha=0.7)
axes[0].set_xlabel('Location (Range, Angle)', fontsize=11)
axes[0].set_ylabel('Entropy Standard Deviation', fontsize=11)
axes[0].set_title('Entropy Variability Across Locations', fontsize=12, fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(True, alpha=0.3, axis='y')

# Determinism STD comparison
axes[1].bar(locations, det_stds, color='darkblue', alpha=0.7)
axes[1].set_xlabel('Location (Range, Angle)', fontsize=11)
axes[1].set_ylabel('Determinism Standard Deviation', fontsize=11)
axes[1].set_title('Determinism Variability Across Locations', fontsize=12, fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('example_output_location_comparison.png', dpi=150)
print("✓ Saved: example_output_location_comparison.png")
plt.close()

# ============================================================================
# EXAMPLE 6: Plot Radar Heatmap with Analyzed Locations
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 6: Plotting Radar Heatmap with Analyzed Locations")
print("="*70)

heatmap = data['radar_heatmap']
metadata = data['metadata']

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(heatmap, aspect='auto', origin='lower',
               extent=[0, metadata['angle_value'], 5, 20],
               cmap='hot')
plt.colorbar(im, ax=ax, label='Intensity (normalized)')
ax.set_xlabel('Angle Bins', fontsize=11)
ax.set_ylabel('Range Bins', fontsize=11)
ax.set_title(f"Radar Heatmap - {metadata['name_data']}\nAnalyzed Locations Marked", 
             fontsize=13, fontweight='bold')

# Mark analyzed locations
for loc in metadata['locations']:
    ax.plot(loc[1], loc[0], 'c*', markersize=20, markeredgecolor='white', 
            markeredgewidth=1.5, label='Analyzed')

# Remove duplicate labels
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), fontsize=10)

plt.tight_layout()
plt.savefig('example_output_heatmap.png', dpi=150)
print("✓ Saved: example_output_heatmap.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("\nGenerated 5 example plots:")
print("  1. example_output_phase_signal.png - Time-domain phase signal")
print("  2. example_output_stft.png - Frequency spectrogram")
print("  3. example_output_chaos_features.png - Entropy and Determinism over time")
print("  4. example_output_location_comparison.png - Cross-location comparison")
print("  5. example_output_heatmap.png - Spatial heatmap with locations")
print("\nFor more examples and detailed documentation, see:")
print("  datasets/isotonic_analysis/README_spatial_localization_data.md")
print("="*70)

