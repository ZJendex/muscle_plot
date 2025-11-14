import json
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# ==================== Configuration ====================
# Input directories to compare
INPUT_DIR_1 = "evaluation_result_isometric_force\\radar_IQ_10_20_corrected"  # Method 1
INPUT_DIR_2 = "evaluation_result_isometric_force\\imu z axis filter"  # Method 2

# Labels for the two methods
METHOD_1_LABEL = "Radar"
METHOD_2_LABEL = "IMU"

# Output directory
OUTPUT_DIR = "evaluation_result_isometric_force\\output"

SEGMENT = 400
OVERLAP = 0
TRAIN_EPOCH = 100
FINE_TUNE_EPOCH = 20

# MVIC_100 values for converting to kg
MVIC_100 = {
    "1010_freddy": 19,
    "1017_freddy": 16,
    "1024_freddy": 17.2,
    "1010_yibo": 15.7,
    "1018_yibo": 14.2,
    "1105_yibo": 16.7,
    "1011_haoyu": 15,
    "1019_haoyu": 16,
    "1025_haoyu": 20,
    "1012_ari": 17.5,
    "1018_ari": 21.5,
    "1012_aditi": 14.5,
    "1019_aditi": 16,
    "1024_aditi": 16.3,
    "1010_jerry": 18,
    "1017_jerry": 15,
    "1025_jerry": 15,
    "1013_jiangyifei": 15,
    "1021_jiangyifei": 14.2,
    "1025_jiangyifei": 17,
    "1013_weifan": 11,
    "1023_weifan": 17.3,
    "1030_weifan": 17.8,
    "1015_david": 15,
    "1021_david": 14.5,
    "1025_david": 15.3,
    "1012_xiwen": 11,
    "1018_xiwen": 11.5,
    "1105_xiwen": 10.4,
    "1029_bert": 28,
    "1027_haozhe": 17.4,
    "1028_jiasi": 14.5,
    "1027_qiushi": 14,
    "1027_yifeng": 26.5,
    "1030_george": 14,
    "1030_seungjoo": 16,
    "1031_yang": 15,
    "1101_anindya": 38,
    "1102_anupam": 16.2,
    "1102_pragna": 14.5,
    "1102_veronica": 14.5,
    "1103_junru": 23,
    "1104_siqi": 11,
}


def process_evaluation_file(json_file, mvic_100_value=None):
    """Process a single evaluation JSON file and extract RMSE metrics."""
    
    with open(json_file, "r") as f:
        data = json.load(f)
    
    # Extract segment-level predictions and true values
    segments = data["segments"]
    if data["test_person"] == "1027_yifeng":
        segments = segments[:366]  # Special handling for this subject
    
    # These values are already in MVIC percentage
    y_true_mvic_pct = np.array([seg["true_force_value"] for seg in segments])
    y_pred_mvic_pct = np.array([seg["predicted_force_value"] for seg in segments])
    
    # Take every 5 consecutive segments as a group and calculate the average
    y_true_mvic_pct_group = []
    y_pred_mvic_pct_group = []
    for i in range(0, len(y_true_mvic_pct), 5):
        y_true_mvic_pct_group.append(np.mean(y_true_mvic_pct[i:i+3]))
        y_pred_mvic_pct_group.append(np.mean(y_pred_mvic_pct[i:i+3]))
    
    y_true_mvic_pct = np.array(y_true_mvic_pct_group)
    y_pred_mvic_pct = np.array(y_pred_mvic_pct_group)
    
    # Calculate RMSE in MVIC percentage
    squared_errors_mvic_pct = (y_true_mvic_pct - y_pred_mvic_pct) ** 2
    rmse_mvic_pct = np.sqrt(np.mean(squared_errors_mvic_pct))
    
    # Calculate MAE in MVIC percentage
    errors_mvic_pct = np.abs(y_true_mvic_pct - y_pred_mvic_pct)
    mae_mvic_pct = np.mean(errors_mvic_pct)
    
    return {
        'subject': data['test_person'],
        'rmse_mvic_pct': rmse_mvic_pct,
        'mae_mvic_pct': mae_mvic_pct,
        'mvic_100': mvic_100_value,
        'num_predictions': len(y_true_mvic_pct),
        'errors_mvic_pct': errors_mvic_pct,  # Individual errors for std calculation
    }


def load_results_from_directory(input_dir):
    """Load all evaluation results from a directory."""
    search_pattern = os.path.join(input_dir, "*_evaluation_results.json")
    json_files = glob.glob(search_pattern)
    
    if not json_files:
        print(f"Warning: No evaluation JSON files found in '{input_dir}'!")
        return []
    
    print(f"Loading from {input_dir}: Found {len(json_files)} files")
    
    all_results = []
    for json_file in sorted(json_files):
        try:
            filename = os.path.basename(json_file)
            subject_name = filename.replace("_evaluation_results.json", "")
            mvic_100_value = MVIC_100.get(subject_name, None)
            
            result = process_evaluation_file(json_file, mvic_100_value)
            all_results.append(result)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    return all_results


def group_by_subject(results):
    """Group results by subject name (without date)."""
    subject_groups = {}
    
    for result in results:
        full_name = result['subject']
        # Extract person name (after date)
        person_name = full_name.split('_')[1] if '_' in full_name else full_name
        
        if person_name not in subject_groups:
            subject_groups[person_name] = []
        
        subject_groups[person_name].append(result)
    
    return subject_groups


def plot_comparison(groups_1, groups_2, method_1_label, method_2_label, config_suffix, filter_multi_session=False):
    """Create comparison plot with side-by-side bars.
    
    Args:
        groups_1: Dictionary of subject groups for method 1
        groups_2: Dictionary of subject groups for method 2
        method_1_label: Label for method 1
        method_2_label: Label for method 2
        config_suffix: Configuration string for filename
        filter_multi_session: If True, only include subjects with 2+ sessions in both methods
    """
    
    # Find common subjects
    common_subjects = sorted(set(groups_1.keys()) & set(groups_2.keys()))
    
    if not common_subjects:
        print("No common subjects found between the two directories!")
        return
    
    print(f"\nFound {len(common_subjects)} common subjects")
    
    # Calculate statistics for each subject
    subjects = []
    method1_means = []
    method1_stds = []
    method1_counts = []
    method2_means = []
    method2_stds = []
    method2_counts = []
    
    for subject_name in common_subjects:
        # Method 1 statistics - collect all individual errors across all sessions
        all_errors_1 = []
        for r in groups_1[subject_name]:
            all_errors_1.extend(r['errors_mvic_pct'])
        all_errors_1 = np.array(all_errors_1)
        mean_1 = np.mean(all_errors_1)  # Mean of all individual errors (same as overall MAE)
        std_1 = np.std(all_errors_1, ddof=1)  # Std of all individual errors
        count_1 = len(groups_1[subject_name])
        
        # Method 2 statistics - collect all individual errors across all sessions
        all_errors_2 = []
        for r in groups_2[subject_name]:
            all_errors_2.extend(r['errors_mvic_pct'])
        all_errors_2 = np.array(all_errors_2)
        mean_2 = np.mean(all_errors_2)  # Mean of all individual errors
        std_2 = np.std(all_errors_2, ddof=1)  # Std of all individual errors
        count_2 = len(groups_2[subject_name])
        
        # Filter: Skip if filtering is enabled and subject has single session in either method
        if filter_multi_session and (count_1 == 1 or count_2 == 1):
            continue
        
        subjects.append(subject_name)
        method1_means.append(mean_1)
        method1_stds.append(std_1)
        method1_counts.append(count_1)
        method2_means.append(mean_2)
        method2_stds.append(std_2)
        method2_counts.append(count_2)
        
        # Print with session counts
        print(f"  {subject_name} (n={count_1}/{count_2}): {method_1_label}={mean_1:.2f}±{std_1:.2f}%, "
              f"{method_2_label}={mean_2:.2f}±{std_2:.2f}%")
    
    # Check if there are any subjects to plot
    if not subjects:
        print("No subjects to plot after filtering!")
        return
    
    # Calculate overall means
    overall_mean_1 = np.mean(method1_means)
    overall_mean_2 = np.mean(method2_means)
    overall_std_1 = np.std(method1_means)
    overall_std_2 = np.std(method2_means)
    
    # Count subjects with single sessions
    single_session_count_1 = sum(1 for c in method1_counts if c == 1)
    single_session_count_2 = sum(1 for c in method2_counts if c == 1)
    
    print(f"\n📊 Statistics calculated from individual prediction errors (y_true - y_pred)")
    if not filter_multi_session:
        if single_session_count_1 > 0 or single_session_count_2 > 0:
            print(f"  Note: Some subjects have only 1 session:")
            if single_session_count_1 > 0:
                print(f"    {method_1_label}: {single_session_count_1} subjects")
            if single_session_count_2 > 0:
                print(f"    {method_2_label}: {single_session_count_2} subjects")
    else:
        print(f"  ✓ Filtered: Only subjects with 2+ sessions in both methods")
    
    # ==================== Create Comparison Plot ====================
    print("\nCreating comparison plot...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Prepare data
    x = np.arange(len(subjects))
    width = 0.35  # Width of bars
    
    # Colors similar to the reference image
    color_1 = '#4A6FA5'  # Dark blue-gray for method 1
    color_2 = '#D99D6A'  # Orange-tan for method 2
    
    # Create bars
    bars1 = ax.bar(x - width/2, method1_means, width, yerr=method1_stds,
                   label=f'{method_1_label}', color=color_1, alpha=0.9,
                   edgecolor='black', linewidth=0.8,
                   error_kw={'linewidth': 1.5, 'elinewidth': 1.5, 'capsize': 3, 
                            'capthick': 1.5, 'ecolor': 'black'})
    
    bars2 = ax.bar(x + width/2, method2_means, width, yerr=method2_stds,
                   label=f'{method_2_label}', color=color_2, alpha=0.9,
                   edgecolor='black', linewidth=0.8,
                   error_kw={'linewidth': 1.5, 'elinewidth': 1.5, 'capsize': 3,
                            'capthick': 1.5, 'ecolor': 'black'})
    
    # Add horizontal lines for means
    ax.axhline(y=overall_mean_1, color=color_1, linestyle='--', linewidth=2, 
              alpha=0.7, zorder=2)
    ax.axhline(y=overall_mean_2, color=color_2, linestyle='--', linewidth=2,
              alpha=0.7, zorder=2)
    
    # Formatting
    ax.set_xlabel('Subject', fontsize=13, fontweight='bold')
    ax.set_ylabel('MAE (%MVIC)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(range(1, len(subjects)+1), fontsize=10)
    ax.set_ylim(0, max(max(method1_means), max(method2_means)) + 
                max(max(method1_stds), max(method2_stds)) + 5)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', axis='y', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    
    # Legend with means
    legend_labels = [
        f'{method_1_label}',
        f'{method_2_label}',
        f'{method_1_label} Mean: {overall_mean_1:.1f}%',
        f'{method_2_label} Mean: {overall_mean_2:.1f}%'
    ]
    
    # Create custom legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Patch(facecolor=color_1, edgecolor='black', label=f'{method_1_label}'),
        Patch(facecolor=color_2, edgecolor='black', label=f'{method_2_label}'),
        Line2D([0], [0], color=color_1, linestyle='--', linewidth=2, 
               label=f'{method_1_label} Mean: {overall_mean_1:.1f}%'),
        Line2D([0], [0], color=color_2, linestyle='--', linewidth=2,
               label=f'{method_2_label} Mean: {overall_mean_2:.1f}%')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, 
             framealpha=0.95, edgecolor='black')
    
    plt.tight_layout()
    filename = os.path.join(OUTPUT_DIR, f"comparison_{method_1_label}_vs_{method_2_label}{config_suffix}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved comparison plot to: {filename}")
    plt.close()
    
    # ==================== Save statistics to text file ====================
    txt_filename = os.path.join(OUTPUT_DIR, f"comparison_{method_1_label}_vs_{method_2_label}_data{config_suffix}.txt")
    with open(txt_filename, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write(f"COMPARISON: {method_1_label} vs {method_2_label}\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"Configuration:\n")
        f.write(f"  Segment Length: {SEGMENT}\n")
        f.write(f"  Overlap: {OVERLAP}\n")
        f.write(f"  Train Epochs: {TRAIN_EPOCH}\n")
        f.write(f"  Fine-tune Epochs: {FINE_TUNE_EPOCH}\n\n")
        
        f.write(f"Method 1: {method_1_label} ({INPUT_DIR_1})\n")
        f.write(f"Method 2: {method_2_label} ({INPUT_DIR_2})\n\n")
        
        f.write("="*100 + "\n")
        f.write("OVERALL STATISTICS\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"Number of common subjects: {len(common_subjects)}\n\n")
        
        f.write(f"{method_1_label}:\n")
        f.write(f"  Mean MAE: {overall_mean_1:.4f}%\n")
        f.write(f"  Std MAE: {overall_std_1:.4f}%\n")
        f.write(f"  Min MAE: {min(method1_means):.4f}%\n")
        f.write(f"  Max MAE: {max(method1_means):.4f}%\n\n")
        
        f.write(f"{method_2_label}:\n")
        f.write(f"  Mean MAE: {overall_mean_2:.4f}%\n")
        f.write(f"  Std MAE: {overall_std_2:.4f}%\n")
        f.write(f"  Min MAE: {min(method2_means):.4f}%\n")
        f.write(f"  Max MAE: {max(method2_means):.4f}%\n\n")
        
        f.write(f"Difference ({method_2_label} - {method_1_label}):\n")
        f.write(f"  Mean difference: {overall_mean_2 - overall_mean_1:.4f}%\n")
        f.write(f"  Relative change: {((overall_mean_2 - overall_mean_1) / overall_mean_1 * 100):.2f}%\n\n")
        
        f.write("="*100 + "\n")
        f.write("PER-SUBJECT RESULTS\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"Note: Statistics calculated from individual prediction errors (|y_true - y_pred|)\n")
        f.write(f"  - Mean = Average absolute error across all predictions from all sessions\n")
        f.write(f"  - Std = Standard deviation of absolute errors\n\n")
        
        f.write(f"{'#':<4} {'Subject':<15} {method_1_label + ' MAE (%)':<25} "
               f"{method_2_label + ' MAE (%)':<25} {'Sessions':<15} {'Diff (%)':<12}\n")
        f.write("-"*120 + "\n")
        
        for i, (subj, m1_mean, m1_std, m1_cnt, m2_mean, m2_std, m2_cnt) in enumerate(
            zip(subjects, method1_means, method1_stds, method1_counts, 
                method2_means, method2_stds, method2_counts), 1):
            diff = m2_mean - m1_mean
            sessions_str = f"{m1_cnt}/{m2_cnt}"
            f.write(f"{i:<4} {subj:<15} {m1_mean:>8.4f}±{m1_std:<8.4f}  "
                   f"{m2_mean:>8.4f}±{m2_std:<8.4f}  {sessions_str:<15} {diff:>+8.4f}\n")
        
        f.write("\n" + "="*100 + "\n")
    
    print(f"Saved comparison data to: {txt_filename}")
    
    # Print summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY - MAE (Mean Absolute Error)")
    print("="*80)
    print(f"{method_1_label}: {overall_mean_1:.2f} ± {overall_std_1:.2f}% (mean ± std across subjects)")
    print(f"{method_2_label}: {overall_mean_2:.2f} ± {overall_std_2:.2f}% (mean ± std across subjects)")
    print(f"Difference: {overall_mean_2 - overall_mean_1:+.2f}% ({((overall_mean_2 - overall_mean_1) / overall_mean_1 * 100):+.2f}%)")
    print("="*80)


if __name__ == "__main__":
    # Ensure directories exist
    if not os.path.exists(INPUT_DIR_1):
        print(f"Error: Input directory 1 '{INPUT_DIR_1}' does not exist!")
        exit(1)
    
    if not os.path.exists(INPUT_DIR_2):
        print(f"Error: Input directory 2 '{INPUT_DIR_2}' does not exist!")
        exit(1)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")
    
    print("="*80)
    print("LOADING RESULTS FROM TWO METHODS")
    print("="*80)
    
    # Load results from both directories
    results_1 = load_results_from_directory(INPUT_DIR_1)
    results_2 = load_results_from_directory(INPUT_DIR_2)
    
    if not results_1:
        print(f"Error: No results loaded from {INPUT_DIR_1}")
        exit(1)
    
    if not results_2:
        print(f"Error: No results loaded from {INPUT_DIR_2}")
        exit(1)
    
    # Group by subject
    groups_1 = group_by_subject(results_1)
    groups_2 = group_by_subject(results_2)
    
    print(f"\nMethod 1 ({METHOD_1_LABEL}): {len(groups_1)} unique subjects")
    print(f"Method 2 ({METHOD_2_LABEL}): {len(groups_2)} unique subjects")
    
    # Create configuration suffix
    config_suffix = f"_seg{SEGMENT}_overlap{OVERLAP}_ep{TRAIN_EPOCH}_ft{FINE_TUNE_EPOCH}"
    
    # Generate comparison plot
    plot_comparison(groups_1, groups_2, METHOD_1_LABEL, METHOD_2_LABEL, config_suffix)
    
    print(f"\n{'='*80}")
    print(f"Comparison complete!")
    print(f"{'='*80}")

