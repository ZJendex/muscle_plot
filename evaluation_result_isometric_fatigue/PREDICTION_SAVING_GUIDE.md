# Prediction Saving and Reproduction Guide

## Overview

The script `classify_fatigue_kfold_radar_PR_1111Transfer.py` has been updated to save all predictions and labels for each fold of the best classifier. This allows you to:

1. **Reproduce AUC plots** from saved predictions
2. **Analyze per-fold performance** in detail
3. **Generate comprehensive documentation** automatically
4. **Share results** without re-running expensive computations

---

## Files Generated

When you run `classify_fatigue_kfold_radar_PR_1111Transfer.py`, the following files will be saved in the output folder (`results_isometric_fatigue/radar_chaos_kfold_lastTry/`):

### 1. Prediction Files

| File | Format | Description |
|------|--------|-------------|
| `best_classifier_predictions_per_fold.pkl` | Pickle | Per-fold predictions with metadata (most complete) |
| `best_classifier_predictions_per_fold.json` | JSON | Same as above but human-readable |
| `best_classifier_all_predictions.csv` | CSV | Tabular format for easy inspection in Excel/Pandas |
| `best_classifier_overall_results.pkl` | Pickle | Overall aggregated results across all folds |

### 2. Feature Importance Files

| File | Format | Description |
|------|--------|-------------|
| `best_classifier_feature_importance.pkl` | Pickle | Complete feature importance data (most complete) |
| `best_classifier_feature_importance.json` | JSON | Same as above but human-readable |
| `best_classifier_feature_importance_detailed.csv` | CSV | All features with statistics |

### 3. Data Structure

#### Per-Fold Predictions (`best_classifier_predictions_per_fold.pkl`)

```python
{
    'classifier_name': 'Logistic Regression',
    'n_folds': 5,
    'feature_names': ['time_mav', 'time_waveform_length', ...],
    'folds': [
        {
            'fold_number': 1,
            'test_subjects': ['aditi', 'freddy'],
            'test_indices': [0, 1, 2, ...],
            'y_true': [0, 1, 0, ...],
            'y_pred': [0, 1, 1, ...],
            'y_pred_proba': [0.234, 0.876, 0.543, ...],
            'subjects': ['aditi', 'aditi', 'freddy', ...],
            'metrics': {
                'accuracy': 0.85,
                'precision': 0.82,
                'recall': 0.88,
                'f1': 0.85,
                'auc': 0.87
            }
        },
        # ... more folds
    ]
}
```

#### Overall Results (`best_classifier_overall_results.pkl`)

```python
{
    'classifier_name': 'Logistic Regression',
    'n_folds': 5,
    'y_true_all': [0, 1, 0, 1, ...],  # All true labels
    'y_pred_all': [0, 1, 1, 1, ...],  # All predictions
    'y_pred_proba_all': [0.23, 0.87, 0.54, ...],  # All probabilities
    'subjects_all': ['aditi', 'freddy', ...],  # Subject for each sample
    'overall_metrics': {
        'accuracy': 0.85,
        'precision': 0.82,
        'recall': 0.88,
        'f1': 0.85,
        'auc': 0.87
    }
}
```

#### Feature Importance (`best_classifier_feature_importance.pkl`)

```python
{
    'classifier_name': 'Logistic Regression',
    'n_folds': 5,
    'feature_names': ['time_mav', 'time_waveform_length', ...],
    'importance_per_fold': [[...], [...], ...],  # Shape: (n_folds, n_features)
    'importance_mean': [0.75, 0.68, 0.62, ...],  # Mean across folds
    'importance_std': [0.12, 0.15, 0.08, ...],   # Std across folds
    'importance_cv': [0.16, 0.22, 0.13, ...],    # Coefficient of variation
    'importance_min': [0.62, 0.51, 0.53, ...],   # Min across folds
    'importance_max': [0.89, 0.85, 0.71, ...],   # Max across folds
    'top_20_features': [
        {
            'feature_name': 'time_zero_crossing_rate',
            'mean_importance': 0.7531,
            'std_importance': 0.1234,
            'cv': 0.1639,
            'min_importance': 0.6234,
            'max_importance': 0.8912
        },
        # ... 19 more features
    ]
}
```

---

## Usage

### Option 1: Run the Reproduction Script

The easiest way to reproduce results:

```bash
python reproduce_auc_plot.py
```

This will:
- Load all saved predictions
- Print a comprehensive summary
- Reproduce the overall ROC curve
- Generate per-fold ROC curves
- Create a detailed README.md file

### Option 2: Load Data Manually

#### In Python:

```python
import pickle
import numpy as np
from sklearn.metrics import roc_curve, auc

# Load per-fold predictions
with open('results_isometric_fatigue/radar_chaos_kfold_lastTry/best_classifier_predictions_per_fold.pkl', 'rb') as f:
    data = pickle.load(f)

# Access data
print(f"Classifier: {data['classifier_name']}")
print(f"Number of folds: {data['n_folds']}")

# Access specific fold
fold_1 = data['folds'][0]
y_true_fold1 = np.array(fold_1['y_true'])
y_pred_proba_fold1 = np.array(fold_1['y_pred_proba'])

# Calculate ROC curve for fold 1
fpr, tpr, _ = roc_curve(y_true_fold1, y_pred_proba_fold1)
roc_auc = auc(fpr, tpr)
print(f"Fold 1 AUC: {roc_auc:.4f}")
```

#### Using Pandas (CSV):

```python
import pandas as pd

# Load predictions
df = pd.read_csv('results_isometric_fatigue/radar_chaos_kfold_lastTry/best_classifier_all_predictions.csv')

# View data
print(df.head())

# Filter by fold
fold_1_df = df[df['fold'] == 1]

# Filter by subject
aditi_df = df[df['subject'] == 'aditi']

# Calculate per-subject accuracy
subject_accuracy = df.groupby('subject')['correct'].mean()
print(subject_accuracy)

# Get misclassifications
misclassified = df[df['correct'] == False]
print(f"Total misclassified: {len(misclassified)}")
```

---

## Reproducing Plots

### Reproducing the AUC Plot

#### Method 1: Using the Helper Script

```python
import reproduce_auc_plot

# Configure output folder
output_folder = "results_isometric_fatigue\\radar_chaos_kfold_lastTry"

# Load predictions
data = reproduce_auc_plot.load_predictions(output_folder)
overall_data = reproduce_auc_plot.load_overall_results(output_folder)

# Plot overall ROC curve
fig, ax, auc_score = reproduce_auc_plot.plot_roc_curve_overall(
    overall_data['y_true_all'],
    overall_data['y_pred_proba_all'],
    data['classifier_name'],
    save_path=f'{output_folder}/my_reproduced_roc.png'
)

# Plot per-fold ROC curves
fig2, ax2, mean_auc, std_auc = reproduce_auc_plot.plot_roc_curves_per_fold(
    data['folds'],
    data['classifier_name'],
    save_path=f'{output_folder}/my_per_fold_roc.png'
)

print(f"Overall AUC: {auc_score:.4f}")
print(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
```

#### Method 2: Custom ROC Plot

```python
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Load overall results
with open('results_isometric_fatigue/radar_chaos_kfold_lastTry/best_classifier_overall_results.pkl', 'rb') as f:
    data = pickle.load(f)

# Extract data
y_true = np.array(data['y_true_all'])
y_pred_proba = np.array(data['y_pred_proba_all'])

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve - {data["classifier_name"]}')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig('my_custom_roc.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Reproducing Feature Importance Plots

#### Method 1: Using the Helper Script

```python
import reproduce_auc_plot

# Load feature importance data
feature_data = reproduce_auc_plot.load_feature_importance(output_folder)

# Print summary
reproduce_auc_plot.print_feature_importance_summary(feature_data, top_n=20)

# Plot top 20 features with error bars
fig, ax = reproduce_auc_plot.plot_feature_importance(
    feature_data,
    top_n=20,
    save_path='my_feature_importance.png'
)

# Plot heatmap of top 5 features across folds
fig2, ax2 = reproduce_auc_plot.plot_feature_importance_heatmap(
    feature_data,
    top_n=5,
    save_path='my_feature_heatmap.png'
)
```

#### Method 2: Custom Feature Importance Plot

```python
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load feature importance data
with open('results_isometric_fatigue/radar_chaos_kfold_lastTry/best_classifier_feature_importance.pkl', 'rb') as f:
    feature_data = pickle.load(f)

# Extract data
feature_names = np.array(feature_data['feature_names'])
importance_mean = np.array(feature_data['importance_mean'])
importance_std = np.array(feature_data['importance_std'])

# Get top 20 features
top_n = 20
sorted_indices = importance_mean.argsort()[-top_n:][::-1]
top_features = feature_names[sorted_indices]
top_mean = importance_mean[sorted_indices]
top_std = importance_std[sorted_indices]

# Create plot
fig, ax = plt.subplots(figsize=(12, 10))
y_pos = np.arange(len(top_features))

ax.barh(y_pos, top_mean, xerr=top_std,
        color='green', alpha=0.7, capsize=5, ecolor='black')

ax.set_yticks(y_pos)
ax.set_yticklabels(top_features, fontsize=10)
ax.set_xlabel('Feature Importance (mean ± std)', fontsize=12)
ax.set_title(f'{feature_data["classifier_name"]} - Top {top_n} Features\n' + 
            f'(Averaged across {feature_data["n_folds"]} folds)', fontsize=14)
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('my_custom_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()
```

#### Method 3: Quick Example Script

For a comprehensive example with multiple visualization methods:

```bash
python example_reproduce_feature_importance.py
```

This script demonstrates 6 different methods to load and visualize feature importance:
1. Reproduce the exact plot (Top 20 with error bars)
2. Print features with statistics
3. Per-fold importance heatmap
4. Load from CSV (for Excel/Pandas users)
5. Access pre-computed top 20 features
6. Feature stability analysis

---

## Advanced Analysis Examples

### Per-Subject Performance

```python
import pandas as pd

df = pd.read_csv('results_isometric_fatigue/radar_chaos_kfold_lastTry/best_classifier_all_predictions.csv')

# Calculate per-subject metrics
subject_stats = df.groupby('subject').agg({
    'correct': ['sum', 'count', 'mean'],
    'prediction_probability': 'mean'
}).round(4)

subject_stats.columns = ['Correct', 'Total', 'Accuracy', 'Avg_Confidence']
print(subject_stats)
```

### Per-Fold Confidence Analysis

```python
import pickle
import numpy as np

with open('results_isometric_fatigue/radar_chaos_kfold_lastTry/best_classifier_predictions_per_fold.pkl', 'rb') as f:
    data = pickle.load(f)

for fold in data['folds']:
    fold_num = fold['fold_number']
    proba = np.array(fold['y_pred_proba'])
    
    # Calculate confidence metrics
    mean_conf = np.mean(proba)
    low_conf = np.sum((proba > 0.4) & (proba < 0.6))  # uncertain predictions
    
    print(f"Fold {fold_num}:")
    print(f"  Mean confidence: {mean_conf:.4f}")
    print(f"  Uncertain predictions: {low_conf} / {len(proba)}")
```

### Misclassification Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results_isometric_fatigue/radar_chaos_kfold_lastTry/best_classifier_all_predictions.csv')

# Get misclassifications
misclassified = df[df['correct'] == False]

# Analyze confidence of misclassifications
plt.figure(figsize=(10, 6))
plt.hist(misclassified['prediction_probability'], bins=20, alpha=0.7, label='Misclassified')
plt.hist(df[df['correct'] == True]['prediction_probability'], bins=20, alpha=0.7, label='Correct')
plt.xlabel('Prediction Confidence')
plt.ylabel('Count')
plt.title('Confidence Distribution: Correct vs Misclassified')
plt.legend()
plt.savefig('confidence_analysis.png', dpi=300)
plt.show()
```

---

## Generating Documentation

The `reproduce_auc_plot.py` script automatically generates a comprehensive README.md file that includes:

- Overview of the classifier and cross-validation setup
- Overall performance metrics
- Per-fold results table
- List of all features used
- Instructions for reproducing results
- Class distribution statistics

To generate the README:

```bash
python reproduce_auc_plot.py
```

Or call the function directly:

```python
import reproduce_auc_plot

data = reproduce_auc_plot.load_predictions('results_isometric_fatigue/radar_chaos_kfold_lastTry')
overall_data = reproduce_auc_plot.load_overall_results('results_isometric_fatigue/radar_chaos_kfold_lastTry')

reproduce_auc_plot.generate_readme(data, overall_data, 'results_isometric_fatigue/radar_chaos_kfold_lastTry')
```

---

## Tips

1. **Pickle vs JSON**: Use pickle for Python analysis (faster, preserves types). Use JSON for viewing in text editors or sharing with non-Python tools.

2. **CSV for Excel**: The CSV file can be opened directly in Excel for quick inspection and filtering.

3. **Version Control**: The JSON and CSV files are text-based and work well with git diff.

4. **Backup**: Keep the pickle files as they contain the most complete data structure.

5. **Reproducibility**: The saved files allow you to reproduce all plots and analyses without re-running the expensive cross-validation.

---

## Troubleshooting

**Q: "FileNotFoundError: No predictions file found"**
- Make sure you've run `classify_fatigue_kfold_radar_PR_1111Transfer.py` first
- Check that the `output_folder` path is correct

**Q: "How do I change the output folder?"**
- Edit line 61 in `classify_fatigue_kfold_radar_PR_1111Transfer.py`:
  ```python
  output_folder = f"results_isometric_fatigue\\your_folder_name"
  ```
- Edit line 12 in `reproduce_auc_plot.py` to match

**Q: "Can I load data from a specific fold only?"**
- Yes! See the "Advanced Analysis Examples" section above

---

## Summary

The updated workflow is now:

1. **Run classification**: `python classify_fatigue_kfold_radar_PR_1111Transfer.py`
   - Saves 4 prediction files
   - Saves 3 feature importance files
   - Total: 7 data files for complete reproducibility

2. **Reproduce/analyze**: `python reproduce_auc_plot.py`
   - Generates ROC curves
   - Generates feature importance plots
   - Creates comprehensive README

3. **Explore feature importance**: `python example_reproduce_feature_importance.py`
   - Multiple visualization methods
   - Feature stability analysis
   - Export-ready figures

4. **Custom analysis**: Load the saved files and analyze as needed

This approach saves time and ensures complete reproducibility! 🎉

### Quick Reference: Files Saved

**Predictions (4 files):**
- `best_classifier_predictions_per_fold.pkl`
- `best_classifier_predictions_per_fold.json`
- `best_classifier_all_predictions.csv`
- `best_classifier_overall_results.pkl`

**Feature Importance (3 files):**
- `best_classifier_feature_importance.pkl`
- `best_classifier_feature_importance.json`
- `best_classifier_feature_importance_detailed.csv`

All files contain complete data needed to reproduce any plot or analysis from your results!

