import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import interp

def load_roc_data(json_path):
    """Load ROC curve data from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def calculate_mean_roc_curve(folds_data):
    """Calculate mean ROC curve from fold data"""
    # Collect all fpr and tpr from folds
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for fold in folds_data:
        fpr = np.array(fold['fpr'])
        tpr = np.array(fold['tpr'])
        aucs.append(fold['auc'])

        # Interpolate to common fpr points
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0  # Ensure it starts at 0
        tprs.append(interp_tpr)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0  # Ensure it ends at 1

    std_tpr = np.std(tprs, axis=0)

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    return mean_fpr, mean_tpr, std_tpr, mean_auc, std_auc

def plot_comparison(init_data, chaos_data, title, save_path):
    """Plot comparison of init and chaos ROC curves"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get Logistic Regression data
    init_lr = init_data['Logistic Regression']
    chaos_lr = chaos_data['Logistic Regression']

    # Calculate mean ROC curves
    init_fpr, init_tpr, init_std, init_auc, init_auc_std = calculate_mean_roc_curve(init_lr['folds'])
    chaos_fpr, chaos_tpr, chaos_std, chaos_auc, chaos_auc_std = calculate_mean_roc_curve(chaos_lr['folds'])

    # Plot init (radar features)
    ax.plot(init_fpr, init_tpr,
            color='#2E86AB', linewidth=2.5,
            label=f'Init Features (AUC = {init_auc:.3f} ± {init_auc_std:.3f})')
    ax.fill_between(init_fpr,
                     np.maximum(init_tpr - init_std, 0),
                     np.minimum(init_tpr + init_std, 1),
                     color='#2E86AB', alpha=0.2)

    # Plot chaos
    ax.plot(chaos_fpr, chaos_tpr,
            color='#A23B72', linewidth=2.5,
            label=f'Chaos Features (AUC = {chaos_auc:.3f} ± {chaos_auc_std:.3f})')
    ax.fill_between(chaos_fpr,
                     np.maximum(chaos_tpr - chaos_std, 0),
                     np.minimum(chaos_tpr + chaos_std, 1),
                     color='#A23B72', alpha=0.2)

    # Plot diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier', alpha=0.5)

    # Formatting
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_aspect('equal')

    # Add text box with additional info
    textstr = f'Classifier: Logistic Regression\n5-Fold Cross-Validation'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Saved: {save_path}')
    plt.close()

def main():
    # Define paths
    base_path = 'evaluation_result_isotonic_fatigue'

    # Load data for 1 rep
    init_1rep_path = f'{base_path}/init_final_1repsBeforeFatigue/roc_curve_data.json'
    chaos_1rep_path = f'{base_path}/chaos_final_1repsBeforeFatigue/roc_curve_data.json'

    # Load data for 3 reps
    init_3rep_path = f'{base_path}/init_final_3repsBeforeFatigue/roc_curve_data.json'
    chaos_3rep_path = f'{base_path}/chaos_final_3repsBeforeFatigue/roc_curve_data.json'

    # Load all data
    print('Loading data...')
    init_1rep_data = load_roc_data(init_1rep_path)
    chaos_1rep_data = load_roc_data(chaos_1rep_path)
    init_3rep_data = load_roc_data(init_3rep_path)
    chaos_3rep_data = load_roc_data(chaos_3rep_path)

    # Create plots
    print('Creating plots...')

    # Plot 1: 1 rep comparison
    plot_comparison(
        init_1rep_data,
        chaos_1rep_data,
        'ROC Curve Comparison - Logistic Regression (1 Rep Before Fatigue)',
        f'{base_path}/logistic_regression_auc_1rep_comparison.png'
    )

    # Plot 2: 3 reps comparison
    plot_comparison(
        init_3rep_data,
        chaos_3rep_data,
        'ROC Curve Comparison - Logistic Regression (3 Reps Before Fatigue)',
        f'{base_path}/logistic_regression_auc_3reps_comparison.png'
    )

    print('\nPlotting complete!')
    print(f'\nSummary Statistics:')
    print(f'\n1 Rep Before Fatigue:')
    print(f'  Init Features AUC: {init_1rep_data["Logistic Regression"]["mean_auc"]:.3f} ± {init_1rep_data["Logistic Regression"]["std_auc"]:.3f}')
    print(f'  Chaos Features AUC: {chaos_1rep_data["Logistic Regression"]["mean_auc"]:.3f} ± {chaos_1rep_data["Logistic Regression"]["std_auc"]:.3f}')

    print(f'\n3 Reps Before Fatigue:')
    print(f'  Init Features AUC: {init_3rep_data["Logistic Regression"]["mean_auc"]:.3f} ± {init_3rep_data["Logistic Regression"]["std_auc"]:.3f}')
    print(f'  Chaos Features AUC: {chaos_3rep_data["Logistic Regression"]["mean_auc"]:.3f} ± {chaos_3rep_data["Logistic Regression"]["std_auc"]:.3f}')

if __name__ == '__main__':
    main()
