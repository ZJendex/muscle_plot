# Plot and Data Reproduction Guide
## Isotonic Fatigue Classification - Complete Reference

This guide explains how to reproduce all plots and analysis from `classify_cc_fatigue_1111.py` using the saved data files.

---

## 📋 Table of Contents
1. [Quick Start](#quick-start)
2. [Saved Data Files](#saved-data-files)
3. [Reproduction Examples](#reproduction-examples)
4. [Common Tasks](#common-tasks)
5. [File Formats Reference](#file-formats-reference)
6. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### Step 1: Run the main script
```bash
python classify_cc_fatigue_1111.py
```

### Step 2: Reproduce all plots
```bash
python reproduce_plots_example.py
```

### Step 3: Access results
All files are in: `result_isotonic_fatigue/final_fatigue_prediction/chaos/`

---

## 📁 Saved Data Files

### Core Data Files

| File | Format | Purpose |
|------|--------|---------|
| `features_df.pkl` / `.csv` | Pickle/CSV | All extracted features for every sample |
| `reproduction_data.pkl` | Pickle | Labels, subjects, timestamps, feature names |
| `all_results_complete.pkl` | Pickle | Complete CV results with all metrics |

### ROC/AUC Data

| File | Format | Contents |
|------|--------|----------|
| `roc_curve_data.pkl` / `.json` | Pickle/JSON | FPR, TPR, AUC for all folds and classifiers |

### Feature Importance Data

| Model | Summary File | Per-Fold File |
|-------|-------------|---------------|
| **Logistic Regression** | `feature_importance_logistic_regression_summary.csv` | `feature_importance_logistic_regression_per_fold.csv` |
| **Random Forest** | `feature_importance_random_forest_summary.csv` | `feature_importance_random_forest_per_fold.csv` |
| **XGBoost** | `feature_importance_xgboost_summary.csv` | `feature_importance_xgboost_per_fold.csv` |
| **All Combined** | `feature_importance_data.pkl` / `.json` | — |

### Performance Data

| File | Format | Contents |
|------|--------|----------|
| `per_class_performance_data.pkl` / `.json` | Pickle/JSON | Precision, recall, F1 per class for all models |
| `incorrect_predictions_summary.csv` | CSV | All misclassifications with details |
| `results_summary_binary.txt` | Text | Overall summary with statistics |

---

## 📊 Reproduction Examples

### 1. ROC Curves - All Folds

```python
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load ROC data
with open('result_isotonic_fatigue/final_fatigue_prediction/chaos/roc_curve_data.pkl', 'rb') as f:
    roc_data = pickle.load(f)

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('ROC Curves - All 5 Folds', fontsize=16, y=0.995)

classifiers = list(roc_data.keys())
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for idx, clf_name in enumerate(classifiers):
    ax = axes[idx // 2, idx % 2]
    
    # Plot each fold
    for fold_data in roc_data[clf_name]['folds']:
        fold_num = fold_data['fold']
        ax.plot(fold_data['fpr'], fold_data['tpr'], lw=2, alpha=0.7,
               color=colors[fold_num-1],
               label=f'Fold {fold_num} (AUC = {fold_data["auc"]:.4f})')
    
    # Calculate and plot mean
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(mean_fpr)
    
    for fold_data in roc_data[clf_name]['folds']:
        fpr = np.array(fold_data['fpr'])
        tpr = np.array(fold_data['tpr'])
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
    
    mean_tpr /= len(roc_data[clf_name]['folds'])
    
    ax.plot(mean_fpr, mean_tpr, color='black', lw=3, linestyle='--',
           label=f'Mean (AUC = {roc_data[clf_name]["mean_auc"]:.4f} ± {roc_data[clf_name]["std_auc"]:.4f})')
    ax.plot([0, 1], [0, 1], 'gray', lw=2, linestyle=':', alpha=0.5)
    
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(clf_name, fontweight='bold')
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('reproduced_roc_curves.png', dpi=300)
plt.close()
```

### 2. Feature Importance - Logistic Regression

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load feature importance
lr_summary = pd.read_csv('result_isotonic_fatigue/final_fatigue_prediction/chaos/feature_importance_logistic_regression_summary.csv')

# Plot top 20 features
fig, ax = plt.subplots(figsize=(12, 10))
top_20 = lr_summary.head(20)

y_pos = np.arange(len(top_20))
colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(top_20)))

ax.barh(y_pos, top_20['Mean_AbsCoefficient'].values,
        xerr=top_20['Std_AbsCoefficient'].values,
        color=colors, alpha=0.8, capsize=5)

ax.set_yticks(y_pos)
ax.set_yticklabels(top_20['Feature'].values, fontsize=9)
ax.set_xlabel('Feature Importance (|Coefficient|)', fontsize=12)
ax.set_title('Logistic Regression - Top 20 Features', fontsize=14)
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('reproduced_feature_importance_lr.png', dpi=300)
plt.close()
```

### 3. Per-Class Performance

```python
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load per-class data
with open('result_isotonic_fatigue/final_fatigue_prediction/chaos/per_class_performance_data.pkl', 'rb') as f:
    per_class_data = pickle.load(f)

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Per-Class Performance Metrics', fontsize=16)

classifiers = list(per_class_data.keys())
class_labels = ['Non-Fatigued', 'Fatigued']

for class_idx, class_label in enumerate(class_labels):
    ax = axes[class_idx]
    
    precision_vals = [per_class_data[clf][class_label]['mean_precision'] for clf in classifiers]
    recall_vals = [per_class_data[clf][class_label]['mean_recall'] for clf in classifiers]
    f1_vals = [per_class_data[clf][class_label]['mean_f1'] for clf in classifiers]
    
    x = np.arange(len(classifiers))
    width = 0.25
    
    bars1 = ax.bar(x - width, precision_vals, width, label='Precision', color='steelblue')
    bars2 = ax.bar(x, recall_vals, width, label='Recall', color='orange')
    bars3 = ax.bar(x + width, f1_vals, width, label='F1-Score', color='green')
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Classifier', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title(f'{class_label} Performance', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([clf.split()[0] for clf in classifiers], rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('reproduced_per_class_performance.png', dpi=300)
plt.close()
```

### 4. Confusion Matrix

```python
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load complete results
with open('result_isotonic_fatigue/final_fatigue_prediction/chaos/all_results_complete.pkl', 'rb') as f:
    all_results = pickle.load(f)

# Choose classifier
clf_name = 'Logistic Regression'
y_true = all_results[clf_name]['y_true']
y_pred = all_results[clf_name]['y_pred']

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Fatigued', 'Fatigued'],
            yticklabels=['Non-Fatigued', 'Fatigued'])
ax.set_title(f'Confusion Matrix - {clf_name}', fontsize=14)
ax.set_ylabel('True Label', fontsize=12)
ax.set_xlabel('Predicted Label', fontsize=12)

plt.tight_layout()
plt.savefig('reproduced_confusion_matrix.png', dpi=300)
plt.close()
```

---

## 💡 Common Tasks

### Task 1: Compare Model Performance

```python
import pickle

with open('result_isotonic_fatigue/final_fatigue_prediction/chaos/roc_curve_data.pkl', 'rb') as f:
    roc_data = pickle.load(f)

print("Model Performance (Mean AUC ± Std):")
print("-" * 50)
for model in roc_data.keys():
    mean_auc = roc_data[model]['mean_auc']
    std_auc = roc_data[model]['std_auc']
    print(f"{model:25s}: {mean_auc:.4f} ± {std_auc:.4f}")
```

### Task 2: Get Top N Features

```python
import pandas as pd

# Logistic Regression
lr_features = pd.read_csv('result_isotonic_fatigue/final_fatigue_prediction/chaos/feature_importance_logistic_regression_summary.csv')
print("Top 10 Features (Logistic Regression):")
print(lr_features.head(10)[['Feature', 'Mean_AbsCoefficient']])

# Random Forest
rf_features = pd.read_csv('result_isotonic_fatigue/final_fatigue_prediction/chaos/feature_importance_random_forest_summary.csv')
print("\nTop 10 Features (Random Forest):")
print(rf_features.head(10)[['Feature', 'Mean_Importance']])
```

### Task 3: Analyze Misclassifications

```python
import pandas as pd

# Load errors
errors = pd.read_csv('result_isotonic_fatigue/final_fatigue_prediction/chaos/incorrect_predictions_summary.csv')

# Errors per model
print("Errors per model:")
print(errors.groupby('model').size())

# Subjects with most errors
print("\nSubjects with most errors:")
print(errors.groupby('subject_date').size().sort_values(ascending=False).head())

# Average confidence of wrong predictions
errors['confidence'] = errors.apply(
    lambda row: row['proba_class_1'] if row['predicted_label'] == 1 else row['proba_class_0'],
    axis=1
)
print(f"\nMean confidence of incorrect predictions: {errors['confidence'].mean():.4f}")
```

### Task 4: Check Feature Stability Across Folds

```python
import pandas as pd
import numpy as np

# Load per-fold data
per_fold = pd.read_csv('result_isotonic_fatigue/final_fatigue_prediction/chaos/feature_importance_logistic_regression_per_fold.csv')

# Load summary to get top features
summary = pd.read_csv('result_isotonic_fatigue/final_fatigue_prediction/chaos/feature_importance_logistic_regression_summary.csv')
top_5_features = summary.head(5)['Feature'].tolist()

print("Feature Stability (Coefficient of Variation):")
print("-" * 50)
for feature in top_5_features:
    mean_val = per_fold[feature].mean()
    std_val = per_fold[feature].std()
    cv = std_val / mean_val if mean_val != 0 else 0
    print(f"{feature:40s}: CV = {cv:.4f}")
```

### Task 5: Find Best Performing Fold

```python
import pickle
import numpy as np

with open('result_isotonic_fatigue/final_fatigue_prediction/chaos/all_results_complete.pkl', 'rb') as f:
    all_results = pickle.load(f)

clf_name = 'Logistic Regression'
best_fold_idx = np.argmax(all_results[clf_name]['auc'])
best_fold = best_fold_idx + 1

print(f"Best fold for {clf_name}: Fold {best_fold}")
print(f"  Accuracy:  {all_results[clf_name]['accuracy'][best_fold_idx]:.4f}")
print(f"  Precision: {all_results[clf_name]['precision'][best_fold_idx]:.4f}")
print(f"  Recall:    {all_results[clf_name]['recall'][best_fold_idx]:.4f}")
print(f"  F1-Score:  {all_results[clf_name]['f1'][best_fold_idx]:.4f}")
print(f"  AUC:       {all_results[clf_name]['auc'][best_fold_idx]:.4f}")
```

---

## 📖 File Formats Reference

### ROC Curve Data (roc_curve_data.pkl)

```python
{
    'Logistic Regression': {
        'folds': [
            {
                'fold': 1,
                'fpr': [0.0, 0.02, 0.05, ..., 1.0],
                'tpr': [0.0, 0.45, 0.78, ..., 1.0],
                'auc': 0.8532,
                'y_true': [0, 1, 0, 1, ...],
                'y_pred': [0, 1, 0, 0, ...]
            },
            # ... folds 2-5
        ],
        'mean_auc': 0.8234,
        'std_auc': 0.0312
    },
    # ... other classifiers
}
```

### Feature Importance Summary (CSV)

```csv
Feature,Mean_AbsCoefficient,Std_AbsCoefficient
traditional_rms,0.4523,0.0234
traditional_mnf,0.3821,0.0456
freq_low_high_ratio,0.2910,0.0321
```

### Per-Class Performance (per_class_performance_data.pkl)

```python
{
    'Logistic Regression': {
        'Non-Fatigued': {
            'precision': [0.85, 0.87, 0.83, 0.86, 0.84],
            'recall': [0.90, 0.88, 0.89, 0.91, 0.87],
            'f1-score': [0.87, 0.87, 0.86, 0.88, 0.85],
            'mean_precision': 0.85,
            'mean_recall': 0.89,
            'mean_f1': 0.87
        },
        'Fatigued': { ... }
    },
    # ... other classifiers
}
```

### Incorrect Predictions (CSV)

```csv
model,fold,subject_date,starting_time,true_label,predicted_label,sample_idx,proba_class_0,proba_class_1
Logistic Regression,1,1010_freddy,12.34,0,1,45,0.42,0.58
```

---

## 🔧 Customization

### Change Output Folder

Edit line 50 in `classify_cc_fatigue_1111.py`:
```python
output_folder = f"result_isotonic_fatigue\\final_fatigue_prediction\\YOUR_NAME"
```

### Change Number of Folds

Edit line 49:
```python
N_FOLDS_STRATIFIED = 5  # Change to your desired number
```

### Change Time Window

Edit line 54:
```python
time_crop = [0, 4000]  # [start_ms, end_ms]
```

---

## 🐛 Troubleshooting

### Issue: FileNotFoundError
**Cause**: Data files don't exist  
**Solution**: Run `classify_cc_fatigue_1111.py` first to generate the files

### Issue: Memory Error
**Cause**: Large pickle files consuming too much memory  
**Solution**: Use CSV files instead of pickle, or load data in chunks

```python
# Instead of loading all at once
import pandas as pd
features_df = pd.read_csv('features_df.csv', chunksize=1000)
for chunk in features_df:
    # Process chunk
    pass
```

### Issue: Different Plot Appearance
**Cause**: Different matplotlib/seaborn versions  
**Solution**: Check versions and install requirements

```bash
pip install matplotlib==3.7.1 seaborn==0.12.2
```

### Issue: Pickle Compatibility Error
**Cause**: Python version mismatch  
**Solution**: Try loading with encoding parameter

```python
with open('file.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')
```

---

## 📊 Quick One-Liners

```python
# Print all AUC scores
import pickle; roc=pickle.load(open('roc_curve_data.pkl','rb')); [print(f"{m}: {roc[m]['mean_auc']:.4f}") for m in roc]

# Top feature
import pandas as pd; print(pd.read_csv('feature_importance_logistic_regression_summary.csv').iloc[0]['Feature'])

# Best model
import pickle; roc=pickle.load(open('roc_curve_data.pkl','rb')); print(max(roc.items(), key=lambda x: x[1]['mean_auc'])[0])

# Total errors
import pandas as pd; print(len(pd.read_csv('incorrect_predictions_summary.csv')))
```

---

## 💾 Data Management

### Files to Keep for Complete Reproduction
- ✅ `all_results_complete.pkl` (complete results)
- ✅ `features_df.pkl` (all features)
- ✅ `reproduction_data.pkl` (metadata)
- ✅ All CSV files (easy analysis)

### Files for Quick Analysis
- ✅ `roc_curve_data.json` (readable ROC data)
- ✅ Feature importance CSVs (feature rankings)
- ✅ `per_class_performance_data.json` (readable metrics)
- ✅ `incorrect_predictions_summary.csv` (error analysis)

### Backup Strategy

```bash
# Create compressed archive
tar -czf fatigue_results_$(date +%Y%m%d).tar.gz result_isotonic_fatigue/final_fatigue_prediction/chaos/

# Or zip on Windows
powershell Compress-Archive -Path result_isotonic_fatigue\final_fatigue_prediction\chaos\* -DestinationPath fatigue_results.zip
```

---

## 📝 Summary

### What Gets Saved
1. **ROC/AUC data** for all models and folds
2. **Feature importance** for LR, RF, and XGBoost
3. **Per-class metrics** for precision, recall, F1
4. **All features** extracted from raw data
5. **Misclassification details** for error analysis
6. **Complete results** for full reproduction

### How to Use
1. Run main script to generate data
2. Use provided code snippets to reproduce plots
3. Analyze CSV files for detailed insights
4. Load pickle files for complete access

### Key Advantages
- ✅ No need to re-run expensive computations
- ✅ Easy to share results with collaborators
- ✅ Multiple formats (pickle, CSV, JSON) for flexibility
- ✅ Complete reproduction capability
- ✅ Human-readable summaries

---

## 📞 Quick Reference

| What You Need | File to Load |
|---------------|--------------|
| **ROC plots** | `roc_curve_data.pkl` |
| **Feature importance** | `feature_importance_*_summary.csv` |
| **Model comparison** | `roc_curve_data.pkl` or `all_results_complete.pkl` |
| **Error analysis** | `incorrect_predictions_summary.csv` |
| **All features** | `features_df.csv` |
| **Class metrics** | `per_class_performance_data.pkl` |

---

Last Updated: 2024-11-13  
Script: `classify_cc_fatigue_1111.py`  
Example Script: `reproduce_plots_example.py`

