"""
Script to load saved predictions and reproduce AUC plot
This script loads the predictions saved by classify_fatigue_kfold_radar_PR_1111Transfer.py
and allows you to reproduce the ROC/AUC analysis.
"""

import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
import os

# Configuration
output_folder = "results_isometric_fatigue\\radar_chaos_kfold_lastTry"

def load_predictions(output_folder):
    """Load the saved predictions from pickle or JSON"""
    
    # Try loading from pickle first (more efficient)
    pickle_file = f'{output_folder}/best_classifier_predictions_per_fold.pkl'
    if os.path.exists(pickle_file):
        print(f"Loading predictions from: {pickle_file}")
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        return data
    
    # Fallback to JSON
    json_file = f'{output_folder}/best_classifier_predictions_per_fold.json'
    if os.path.exists(json_file):
        print(f"Loading predictions from: {json_file}")
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
    
    raise FileNotFoundError(f"No predictions file found in {output_folder}")

def load_overall_results(output_folder):
    """Load the overall aggregated results"""
    
    overall_file = f'{output_folder}/best_classifier_overall_results.pkl'
    if os.path.exists(overall_file):
        print(f"Loading overall results from: {overall_file}")
        with open(overall_file, 'rb') as f:
            data = pickle.load(f)
        return data
    
    raise FileNotFoundError(f"No overall results file found in {output_folder}")

def plot_roc_curve_overall(y_true, y_pred_proba, classifier_name, save_path=None):
    """Plot overall ROC curve combining all folds"""
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'{classifier_name} (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve - {classifier_name}\n(Reproduced from Saved Predictions)', fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig, ax, roc_auc

def plot_roc_curves_per_fold(fold_data, classifier_name, save_path=None):
    """Plot ROC curves for each fold separately"""
    
    n_folds = len(fold_data)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    aucs = []
    for fold in fold_data:
        y_true = np.array(fold['y_true'])
        y_pred_proba = np.array(fold['y_pred_proba'])
        fold_num = fold['fold_number']
        
        if len(np.unique(y_true)) < 2:
            print(f"Warning: Fold {fold_num} has only one class, skipping ROC curve")
            continue
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        fold_auc = auc(fpr, tpr)
        aucs.append(fold_auc)
        
        ax.plot(fpr, tpr, lw=2, alpha=0.7,
                label=f'Fold {fold_num} (AUC = {fold_auc:.4f})')
    
    # Plot random classifier
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random', alpha=0.5)
    
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curves Per Fold - {classifier_name}\n' + 
                f'Mean AUC = {mean_auc:.4f} ± {std_auc:.4f}', fontsize=14)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig, ax, mean_auc, std_auc

def print_summary(data):
    """Print summary of the loaded data"""
    
    print("\n" + "="*80)
    print("LOADED PREDICTIONS SUMMARY")
    print("="*80)
    print(f"Classifier: {data['classifier_name']}")
    print(f"Number of folds: {data['n_folds']}")
    print(f"Number of features: {len(data['feature_names'])}")
    
    print("\n" + "-"*80)
    print("PER-FOLD INFORMATION:")
    print("-"*80)
    
    for fold in data['folds']:
        fold_num = fold['fold_number']
        n_samples = len(fold['y_true'])
        test_subjects = ', '.join([s.capitalize() for s in fold['test_subjects']])
        metrics = fold['metrics']
        
        print(f"\nFold {fold_num}:")
        print(f"  Test subjects: {test_subjects}")
        print(f"  Number of samples: {n_samples}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  AUC:       {metrics['auc']:.4f}")
    
    print("\n" + "="*80)

def load_feature_importance(output_folder):
    """Load the saved feature importance data"""
    
    # Try loading from pickle first
    pickle_file = f'{output_folder}/best_classifier_feature_importance.pkl'
    if os.path.exists(pickle_file):
        print(f"Loading feature importance from: {pickle_file}")
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        return data
    
    # Fallback to JSON
    json_file = f'{output_folder}/best_classifier_feature_importance.json'
    if os.path.exists(json_file):
        print(f"Loading feature importance from: {json_file}")
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
    
    raise FileNotFoundError(f"No feature importance file found in {output_folder}")

def plot_feature_importance(feature_importance_data, top_n=20, save_path=None):
    """Reproduce the feature importance plot with error bars"""
    
    classifier_name = feature_importance_data['classifier_name']
    n_folds = feature_importance_data['n_folds']
    
    # Get top N features
    feature_names = np.array(feature_importance_data['feature_names'])
    importance_mean = np.array(feature_importance_data['importance_mean'])
    importance_std = np.array(feature_importance_data['importance_std'])
    
    # Sort by mean importance and get top N
    sorted_indices = importance_mean.argsort()[-top_n:][::-1]
    top_features = feature_names[sorted_indices]
    top_mean = importance_mean[sorted_indices]
    top_std = importance_std[sorted_indices]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    y_pos = np.arange(len(top_features))
    
    # Set color based on classifier
    if classifier_name == 'Logistic Regression':
        color = 'green'
    elif classifier_name == 'Random Forest':
        color = 'steelblue'
    elif classifier_name == 'XGBoost':
        color = 'orange'
    else:
        color = 'gray'
    
    ax.barh(y_pos, top_mean, xerr=top_std,
            color=color, alpha=0.7, capsize=5, ecolor='black')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features, fontsize=10)
    ax.set_xlabel('Feature Importance (mean ± std)', fontsize=12)
    ax.set_title(f'{classifier_name} - Top {top_n} Features\n(Averaged across {n_folds} folds)', 
                fontsize=14)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig, ax

def plot_feature_importance_heatmap(feature_importance_data, top_n=5, save_path=None):
    """Reproduce the feature importance heatmap across folds"""
    
    classifier_name = feature_importance_data['classifier_name']
    n_folds = feature_importance_data['n_folds']
    
    # Get data
    feature_names = np.array(feature_importance_data['feature_names'])
    importance_matrix = np.array(feature_importance_data['importance_per_fold'])  # Shape: (n_folds, n_features)
    importance_mean = importance_matrix.mean(axis=0)
    
    # Get top N features
    top_indices = importance_mean.argsort()[-top_n:][::-1]
    top_features = feature_names[top_indices]
    heatmap_data = importance_matrix[:, top_indices].T  # Shape: (top_n, n_folds)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(n_folds))
    ax.set_xticklabels([f'Fold {i+1}' for i in range(n_folds)], fontsize=10)
    ax.set_yticks(np.arange(top_n))
    ax.set_yticklabels(top_features, fontsize=9)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Feature Importance', fontsize=11)
    
    ax.set_title(f'{classifier_name} - Feature Importance Across Folds\n(Top {top_n} Features)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig, ax

def print_feature_importance_summary(feature_importance_data, top_n=20):
    """Print summary of feature importance"""
    
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE SUMMARY")
    print("="*80)
    print(f"Classifier: {feature_importance_data['classifier_name']}")
    print(f"Number of folds: {feature_importance_data['n_folds']}")
    print(f"Total features: {len(feature_importance_data['feature_names'])}")
    
    print(f"\n{'-'*80}")
    print(f"TOP {top_n} MOST IMPORTANT FEATURES:")
    print('-'*80)
    print(f"{'Rank':<6} {'Feature':<45} {'Mean':<12} {'Std':<12} {'CV':<10}")
    print('-'*80)
    
    for i, feat_data in enumerate(feature_importance_data['top_20_features'][:top_n], 1):
        print(f"{i:<6} {feat_data['feature_name']:<45} "
              f"{feat_data['mean_importance']:<12.6f} "
              f"{feat_data['std_importance']:<12.6f} "
              f"{feat_data['cv']:<10.3f}")
    
    print("="*80)

def generate_readme(data, overall_data, output_folder):
    """Generate a comprehensive README file"""
    
    readme_file = f'{output_folder}/RESULTS_README.md'
    
    with open(readme_file, 'w') as f:
        f.write("# Fatigue Classification Results\n\n")
        
        # Overview
        f.write("## Overview\n\n")
        f.write(f"**Classifier:** {data['classifier_name']}\n\n")
        f.write(f"**Cross-Validation:** {data['n_folds']}-Fold Group K-Fold (by subject)\n\n")
        f.write(f"**Number of Features:** {len(data['feature_names'])}\n\n")
        
        # Overall Performance
        f.write("## Overall Performance\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        metrics = overall_data['overall_metrics']
        f.write(f"| Accuracy | {metrics['accuracy']:.4f} |\n")
        f.write(f"| Precision | {metrics['precision']:.4f} |\n")
        f.write(f"| Recall | {metrics['recall']:.4f} |\n")
        f.write(f"| F1-Score | {metrics['f1']:.4f} |\n")
        f.write(f"| AUC | {metrics['auc']:.4f} |\n\n")
        
        # Per-Fold Results
        f.write("## Per-Fold Results\n\n")
        f.write("| Fold | Test Subjects | Samples | Accuracy | Precision | Recall | F1 | AUC |\n")
        f.write("|------|---------------|---------|----------|-----------|--------|-------|-----|\n")
        
        for fold in data['folds']:
            fold_num = fold['fold_number']
            n_samples = len(fold['y_true'])
            test_subjects = ', '.join([s.capitalize() for s in fold['test_subjects']])
            m = fold['metrics']
            
            f.write(f"| {fold_num} | {test_subjects} | {n_samples} | "
                   f"{m['accuracy']:.4f} | {m['precision']:.4f} | "
                   f"{m['recall']:.4f} | {m['f1']:.4f} | {m['auc']:.4f} |\n")
        
        # Feature Information
        f.write("\n## Features Used\n\n")
        f.write(f"Total: {len(data['feature_names'])} features\n\n")
        f.write("```\n")
        for i, feat in enumerate(data['feature_names'], 1):
            f.write(f"{i:2d}. {feat}\n")
        f.write("```\n\n")
        
        # Files Generated
        f.write("## Generated Files\n\n")
        f.write("### Predictions Data\n")
        f.write("- `best_classifier_predictions_per_fold.pkl` - Per-fold predictions (pickle format)\n")
        f.write("- `best_classifier_predictions_per_fold.json` - Per-fold predictions (JSON format)\n")
        f.write("- `best_classifier_all_predictions.csv` - All predictions in tabular format\n")
        f.write("- `best_classifier_overall_results.pkl` - Overall aggregated results\n\n")
        
        f.write("### Visualizations\n")
        f.write("- `groupkfold_5fold_roc_curve.png` - Overall ROC curve\n")
        f.write("- `groupkfold_5fold_confusion_matrix.png` - Confusion matrix\n")
        f.write("- `groupkfold_5fold_per_fold_performance.png` - Performance metrics per fold\n")
        f.write("- `groupkfold_5fold_average_performance.png` - Average performance comparison\n")
        f.write("- `feature_importance_*.png` - Feature importance visualizations\n\n")
        
        # Reproducing Results
        f.write("## Reproducing Results\n\n")
        f.write("To reproduce the ROC/AUC plot from saved predictions:\n\n")
        f.write("```python\n")
        f.write("import reproduce_auc_plot\n\n")
        f.write("# Load predictions\n")
        f.write(f"data = reproduce_auc_plot.load_predictions('{output_folder}')\n\n")
        f.write("# Plot overall ROC curve\n")
        f.write("overall_data = reproduce_auc_plot.load_overall_results('{output_folder}')\n")
        f.write("fig, ax, auc = reproduce_auc_plot.plot_roc_curve_overall(\n")
        f.write("    overall_data['y_true_all'],\n")
        f.write("    overall_data['y_pred_proba_all'],\n")
        f.write("    data['classifier_name']\n")
        f.write(")\n")
        f.write("```\n\n")
        
        # Class Distribution
        f.write("## Class Distribution\n\n")
        all_true_labels = []
        for fold in data['folds']:
            all_true_labels.extend(fold['y_true'])
        all_true_labels = np.array(all_true_labels)
        
        n_no_fatigue = np.sum(all_true_labels == 0)
        n_fatigue = np.sum(all_true_labels == 1)
        
        f.write(f"- **No Fatigue (Class 0):** {n_no_fatigue} samples\n")
        f.write(f"- **Fatigue (Class 1):** {n_fatigue} samples\n")
        f.write(f"- **Total:** {len(all_true_labels)} samples\n\n")
    
    print(f"\n✓ Generated README: {readme_file}")
    return readme_file

# Main execution
if __name__ == "__main__":
    # Load predictions
    data = load_predictions(output_folder)
    overall_data = load_overall_results(output_folder)
    
    # Print summary
    print_summary(data)
    
    # Plot overall ROC curve
    print("\n" + "="*80)
    print("GENERATING ROC CURVES")
    print("="*80)
    
    fig1, ax1, overall_auc = plot_roc_curve_overall(
        np.array(overall_data['y_true_all']),
        np.array(overall_data['y_pred_proba_all']),
        data['classifier_name'],
        save_path=f'{output_folder}/reproduced_roc_curve_overall.png'
    )
    
    # Plot per-fold ROC curves
    fig2, ax2, mean_auc, std_auc = plot_roc_curves_per_fold(
        data['folds'],
        data['classifier_name'],
        save_path=f'{output_folder}/reproduced_roc_curves_per_fold.png'
    )
    
    print(f"\nOverall AUC: {overall_auc:.4f}")
    print(f"Mean AUC across folds: {mean_auc:.4f} ± {std_auc:.4f}")
    
    # Load and plot feature importance
    try:
        print("\n" + "="*80)
        print("GENERATING FEATURE IMPORTANCE PLOTS")
        print("="*80)
        
        feature_importance_data = load_feature_importance(output_folder)
        
        # Print summary
        print_feature_importance_summary(feature_importance_data, top_n=20)
        
        # Plot feature importance with error bars (top 20)
        fig3, ax3 = plot_feature_importance(
            feature_importance_data,
            top_n=20,
            save_path=f'{output_folder}/reproduced_feature_importance_top20.png'
        )
        
        # Plot feature importance heatmap (top 5)
        fig4, ax4 = plot_feature_importance_heatmap(
            feature_importance_data,
            top_n=5,
            save_path=f'{output_folder}/reproduced_feature_importance_heatmap.png'
        )
        
    except FileNotFoundError as e:
        print(f"\nWarning: {e}")
        print("Skipping feature importance plots.")
    
    # Generate README
    print("\n" + "="*80)
    print("GENERATING README")
    print("="*80)
    readme_path = generate_readme(data, overall_data, output_folder)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nAll reproduced files saved to: {output_folder}")
    print("\nYou can now:")
    print("  1. View the reproduced ROC curves")
    print("  2. View the reproduced feature importance plots")
    print("  3. Read the comprehensive README.md")
    print("  4. Load predictions from CSV for further analysis")
    
    plt.show()

