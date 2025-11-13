# Feature Importance: Quick Reference Guide

## Overview

The classification script now saves complete feature importance data, allowing you to:
- ✅ Reproduce the exact feature importance plot shown in your results
- ✅ Access per-fold importance values for stability analysis
- ✅ Load data in multiple formats (pickle, JSON, CSV)
- ✅ Create custom visualizations for publications

---

## Files Generated

After running `classify_fatigue_kfold_radar_PR_1111Transfer.py`, you'll find:

| File | Size | Use Case |
|------|------|----------|
| `best_classifier_feature_importance.pkl` | ~10KB | Complete data, fastest to load (Python only) |
| `best_classifier_feature_importance.json` | ~30KB | Human-readable, can open in text editor |
| `best_classifier_feature_importance_detailed.csv` | ~5KB | Excel-friendly, sortable, filterable |

---

## Quick Start

### 1. Reproduce the Exact Plot (Top 20 Features)

```python
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load data
with open('results_isometric_fatigue/radar_chaos_kfold_lastTry/best_classifier_feature_importance.pkl', 'rb') as f:
    data = pickle.load(f)

# Extract top 20
feature_names = np.array(data['feature_names'])
importance_mean = np.array(data['importance_mean'])
importance_std = np.array(data['importance_std'])

top_20_idx = importance_mean.argsort()[-20:][::-1]

# Plot
fig, ax = plt.subplots(figsize=(12, 10))
y_pos = np.arange(20)
ax.barh(y_pos, importance_mean[top_20_idx], 
        xerr=importance_std[top_20_idx],
        color='green', alpha=0.7, capsize=5, ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(feature_names[top_20_idx])
ax.set_xlabel('Feature Importance (mean ± std)')
ax.set_title(f'{data["classifier_name"]} - Top 20 Features')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('reproduced_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 2. Get Top Features as List

```python
import pickle

with open('results_isometric_fatigue/radar_chaos_kfold_lastTry/best_classifier_feature_importance.pkl', 'rb') as f:
    data = pickle.load(f)

# Pre-computed top 20
for i, feat in enumerate(data['top_20_features'], 1):
    print(f"{i:2d}. {feat['feature_name']:45s} {feat['mean_importance']:.6f} ± {feat['std_importance']:.6f}")
```

### 3. Load in Excel/Pandas

```python
import pandas as pd

df = pd.read_csv('results_isometric_fatigue/radar_chaos_kfold_lastTry/best_classifier_feature_importance_detailed.csv')

# View top features
print(df.head(20))

# Filter stable features (low coefficient of variation)
stable = df[df['CV'] < 0.3].sort_values('Mean_Importance', ascending=False)
print(stable)

# Export for publication
stable.head(10).to_csv('stable_important_features.csv', index=False)
```

---

## What's Inside the Data

### Main Fields

| Field | Type | Description |
|-------|------|-------------|
| `classifier_name` | str | Name of the classifier (e.g., "Logistic Regression") |
| `n_folds` | int | Number of cross-validation folds |
| `feature_names` | list | Names of all features |
| `importance_mean` | list | Mean importance across all folds |
| `importance_std` | list | Standard deviation across folds |
| `importance_cv` | list | Coefficient of variation (std/mean) |
| `importance_per_fold` | 2D array | Raw importance values per fold (n_folds × n_features) |
| `top_20_features` | list | Pre-computed top 20 with all statistics |

### Top Features Structure

Each feature in `top_20_features` contains:

```python
{
    'feature_name': 'time_zero_crossing_rate',
    'mean_importance': 0.753124,
    'std_importance': 0.123456,
    'cv': 0.163942,  # Coefficient of variation (lower = more stable)
    'min_importance': 0.623451,
    'max_importance': 0.891234
}
```

---

## Common Tasks

### Task 1: Find Most Stable Features

```python
import pickle
import numpy as np

with open('...best_classifier_feature_importance.pkl', 'rb') as f:
    data = pickle.load(f)

cv = np.array(data['importance_cv'])
mean_imp = np.array(data['importance_mean'])
feature_names = np.array(data['feature_names'])

# Get features with high importance AND low CV
stable_mask = (cv < 0.2) & (mean_imp > np.median(mean_imp))
stable_features = feature_names[stable_mask]

print(f"Found {len(stable_features)} stable features:")
for feat in stable_features:
    idx = np.where(feature_names == feat)[0][0]
    print(f"  {feat}: {mean_imp[idx]:.4f} (CV: {cv[idx]:.3f})")
```

### Task 2: Compare Feature Importance Across Folds

```python
import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('...best_classifier_feature_importance.pkl', 'rb') as f:
    data = pickle.load(f)

# Get per-fold data
importance_matrix = np.array(data['importance_per_fold'])  # Shape: (n_folds, n_features)
feature_names = np.array(data['feature_names'])
importance_mean = importance_matrix.mean(axis=0)

# Get top 5 features
top_5_idx = importance_mean.argsort()[-5:][::-1]
top_5_features = feature_names[top_5_idx]

# Plot per-fold importance
fig, ax = plt.subplots(figsize=(10, 6))
for i, feat_idx in enumerate(top_5_idx):
    ax.plot(range(1, data['n_folds']+1), 
            importance_matrix[:, feat_idx], 
            marker='o', label=feature_names[feat_idx])

ax.set_xlabel('Fold')
ax.set_ylabel('Feature Importance')
ax.set_title('Feature Importance Across Folds (Top 5 Features)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('importance_per_fold.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Task 3: Export Publication-Ready Table

```python
import pandas as pd

# Load CSV
df = pd.read_csv('...best_classifier_feature_importance_detailed.csv')

# Get top 10 with nice formatting
top_10 = df.head(10).copy()
top_10['Mean_Importance'] = top_10['Mean_Importance'].map('{:.4f}'.format)
top_10['Std_Importance'] = top_10['Std_Importance'].map('{:.4f}'.format)
top_10['CV'] = top_10['CV'].map('{:.3f}'.format)

# Rename for publication
top_10.columns = ['Feature Name', 'Mean', 'Std Dev', 'CV', 'Min', 'Max']

# Save as LaTeX
with open('feature_table.tex', 'w') as f:
    f.write(top_10.to_latex(index=False))

# Save as Markdown
with open('feature_table.md', 'w') as f:
    f.write(top_10.to_markdown(index=False))

print("✓ Exported to LaTeX and Markdown formats")
```

---

## Understanding the Metrics

### Importance Value
- Higher value = more important for classification
- For Logistic Regression: absolute value of coefficients
- For tree-based models: based on information gain/gini importance

### Standard Deviation (Std)
- Measures variation across folds
- High std = feature importance varies across subjects

### Coefficient of Variation (CV)
- CV = Std / Mean
- Lower is better (more stable)
- CV < 0.2: Very stable
- CV > 0.5: Highly variable

### Feature Selection Tips
1. **High Mean + Low CV**: Most reliable features
2. **High Mean + High CV**: Important but subject-dependent
3. **Low Mean**: Consider removing for model simplification

---

## Example Scripts Provided

### `reproduce_auc_plot.py`
- Loads all saved data including feature importance
- Generates ROC curves AND feature importance plots
- Run: `python reproduce_auc_plot.py`

### `example_reproduce_feature_importance.py`
- Comprehensive examples of 6 different methods
- Demonstrates all visualization types
- Run: `python example_reproduce_feature_importance.py`

---

## Integration with Your Paper

### Figures for Publication

1. **Feature Importance Bar Chart** (already saved by main script)
   - File: `feature_importance_logistic_regression_with_variance.png`
   - Shows top 20 features with error bars
   - Publication-ready at 300 DPI

2. **Feature Heatmap** (already saved)
   - File: `feature_importance_logistic_regression_heatmap.png`
   - Shows top 5 features across all folds

3. **Custom Plots** (use examples above)
   - Per-fold line plots
   - Stability analysis scatter plots
   - Feature comparison plots

### Tables for Publication

Generate directly from CSV:

```python
import pandas as pd

df = pd.read_csv('...best_classifier_feature_importance_detailed.csv')

# Format for publication
pub_table = df.head(15)[['Feature', 'Mean_Importance', 'Std_Importance', 'CV']]
pub_table.columns = ['Feature', 'Importance', 'Std Dev', 'CV']

# Round for readability
pub_table['Importance'] = pub_table['Importance'].round(4)
pub_table['Std Dev'] = pub_table['Std Dev'].round(4)
pub_table['CV'] = pub_table['CV'].round(3)

print(pub_table.to_latex(index=False))  # For LaTeX papers
print(pub_table.to_markdown(index=False))  # For README/docs
```

---

## Verification

To verify the saved data matches your original results:

```python
import pickle

# Load saved data
with open('...best_classifier_feature_importance.pkl', 'rb') as f:
    data = pickle.load(f)

# Check
print(f"Classifier: {data['classifier_name']}")
print(f"Folds: {data['n_folds']}")
print(f"Features: {len(data['feature_names'])}")
print(f"\nTop feature: {data['top_20_features'][0]['feature_name']}")
print(f"Top importance: {data['top_20_features'][0]['mean_importance']:.6f}")

# Compare with image you showed me:
# Should see: time_zero_crossing_rate, freq_low_high_ratio, nonlinear_rms as top features
```

---

## Need Help?

- **Full guide**: See `PREDICTION_SAVING_GUIDE.md`
- **Examples**: Run `python example_reproduce_feature_importance.py`
- **Reproduce all**: Run `python reproduce_auc_plot.py`

---

## Summary

**You now have:**
✅ Complete feature importance data saved in 3 formats  
✅ Per-fold importance values for stability analysis  
✅ Pre-computed top 20 features with statistics  
✅ Scripts to reproduce any plot or analysis  
✅ Publication-ready figures and tables  

**Everything needed to reproduce your feature importance plot and create comprehensive documentation for your research!** 🎉

