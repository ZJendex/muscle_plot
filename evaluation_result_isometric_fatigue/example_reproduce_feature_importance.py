"""
Example script to reproduce feature importance plots from saved data
This demonstrates how to load and visualize feature importance data
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Configuration
output_folder = "results_isometric_fatigue\\radar_chaos_kfold_lastTry"

# ============================================================================
# Load feature importance data
# ============================================================================

print("Loading feature importance data...")
with open(f'{output_folder}/best_classifier_feature_importance.pkl', 'rb') as f:
    feature_importance_data = pickle.load(f)

print(f"\nClassifier: {feature_importance_data['classifier_name']}")
print(f"Number of folds: {feature_importance_data['n_folds']}")
print(f"Total features: {len(feature_importance_data['feature_names'])}")

# ============================================================================
# Method 1: Reproduce the exact plot (Top 20 features with error bars)
# ============================================================================

print("\n" + "="*80)
print("METHOD 1: Reproducing Top 20 Features Plot")
print("="*80)

# Extract data
feature_names = np.array(feature_importance_data['feature_names'])
importance_mean = np.array(feature_importance_data['importance_mean'])
importance_std = np.array(feature_importance_data['importance_std'])

# Get top 20 features
top_n = 20
sorted_indices = importance_mean.argsort()[-top_n:][::-1]
top_features = feature_names[sorted_indices]
top_mean = importance_mean[sorted_indices]
top_std = importance_std[sorted_indices]

# Create the plot
fig, ax = plt.subplots(figsize=(12, 10))
y_pos = np.arange(len(top_features))

ax.barh(y_pos, top_mean, xerr=top_std,
        color='green', alpha=0.7, capsize=5, ecolor='black')

ax.set_yticks(y_pos)
ax.set_yticklabels(top_features, fontsize=10)
ax.set_xlabel('Feature Importance (mean ± std)', fontsize=12)
ax.set_title(f'{feature_importance_data["classifier_name"]} - Top {top_n} Features\n' + 
            f'(Averaged across {feature_importance_data["n_folds"]} folds)', fontsize=14)
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(f'{output_folder}/example_reproduced_feature_importance.png', 
           dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_folder}/example_reproduced_feature_importance.png")

# ============================================================================
# Method 2: Print top features with statistics
# ============================================================================

print("\n" + "="*80)
print("METHOD 2: Top Features with Statistics")
print("="*80)

print(f"\n{'Rank':<6} {'Feature':<45} {'Mean':<12} {'Std':<12} {'CV':<10}")
print('-'*85)

for i in range(top_n):
    idx = sorted_indices[i]
    cv = importance_std[idx] / (importance_mean[idx] + 1e-10)
    print(f"{i+1:<6} {feature_names[idx]:<45} "
          f"{importance_mean[idx]:<12.6f} "
          f"{importance_std[idx]:<12.6f} "
          f"{cv:<10.3f}")

# ============================================================================
# Method 3: Per-fold importance (Heatmap)
# ============================================================================

print("\n" + "="*80)
print("METHOD 3: Per-Fold Feature Importance Heatmap")
print("="*80)

# Get importance per fold
importance_matrix = np.array(feature_importance_data['importance_per_fold'])  # (n_folds, n_features)

# Get top 7 features for heatmap (adjust as needed)
top_7_indices = importance_mean.argsort()[-7:][::-1]
top_7_features = feature_names[top_7_indices]
heatmap_data = importance_matrix[:, top_7_indices].T  # (7, n_folds)

# Create heatmap
fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')

# Set ticks and labels
n_folds = feature_importance_data['n_folds']
ax.set_xticks(np.arange(n_folds))
ax.set_xticklabels([f'Fold {i+1}' for i in range(n_folds)], fontsize=10)
ax.set_yticks(np.arange(7))
ax.set_yticklabels(top_7_features, fontsize=10)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Feature Importance', fontsize=11)

# Add values to heatmap
for i in range(7):
    for j in range(n_folds):
        text = ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                      ha="center", va="center", color="black", fontsize=9)

ax.set_title(f'{feature_importance_data["classifier_name"]} - Feature Importance Across Folds\n(Top 7 Features)', 
            fontsize=14, fontweight='bold')
ax.set_xlabel('Fold', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)

plt.tight_layout()
plt.savefig(f'{output_folder}/example_reproduced_heatmap.png', 
           dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_folder}/example_reproduced_heatmap.png")

# ============================================================================
# Method 4: Load from CSV (for Excel/Pandas users)
# ============================================================================

print("\n" + "="*80)
print("METHOD 4: Load from CSV")
print("="*80)

# Load the detailed CSV
df = pd.read_csv(f'{output_folder}/best_classifier_feature_importance_detailed.csv')

print("\nTop 10 features from CSV:")
print(df.head(10).to_string(index=False))

# You can also filter or sort as needed
stable_features = df[df['CV'] < 0.5].sort_values('Mean_Importance', ascending=False)
print(f"\nStable features (CV < 0.5): {len(stable_features)}")
print(stable_features.head(10).to_string(index=False))

# ============================================================================
# Method 5: Access pre-computed top 20 features
# ============================================================================

print("\n" + "="*80)
print("METHOD 5: Pre-computed Top 20 Features")
print("="*80)

print(f"\n{'Rank':<6} {'Feature':<45} {'Mean':<12} {'Std':<12}")
print('-'*75)

for i, feat_data in enumerate(feature_importance_data['top_20_features'], 1):
    print(f"{i:<6} {feat_data['feature_name']:<45} "
          f"{feat_data['mean_importance']:<12.6f} "
          f"{feat_data['std_importance']:<12.6f}")

# ============================================================================
# Method 6: Compare features across folds
# ============================================================================

print("\n" + "="*80)
print("METHOD 6: Feature Stability Analysis")
print("="*80)

# Calculate coefficient of variation for all features
cv_all = importance_std / (importance_mean + 1e-10)

# Find most stable features (low CV)
stable_indices = cv_all.argsort()[:10]
print("\nMost STABLE features (lowest CV):")
print(f"{'Feature':<45} {'Mean':<12} {'CV':<10}")
print('-'*67)
for idx in stable_indices:
    print(f"{feature_names[idx]:<45} {importance_mean[idx]:<12.6f} {cv_all[idx]:<10.3f}")

# Find most variable features (high CV)
variable_indices = cv_all.argsort()[-10:][::-1]
print("\nMost VARIABLE features (highest CV):")
print(f"{'Feature':<45} {'Mean':<12} {'CV':<10}")
print('-'*67)
for idx in variable_indices:
    print(f"{feature_names[idx]:<45} {importance_mean[idx]:<12.6f} {cv_all[idx]:<10.3f}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"✓ Feature importance data loaded successfully")
print(f"✓ Total features: {len(feature_names)}")
print(f"✓ Top feature: {top_features[0]}")
print(f"✓ Plots saved to: {output_folder}")
print("\nYou can now use this data for:")
print("  1. Publication figures")
print("  2. Feature selection")
print("  3. Interpretability analysis")
print("  4. Cross-validation stability analysis")

plt.show()

print("\n" + "="*80)
print("DONE!")
print("="*80)

