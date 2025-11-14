import json
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# ==================== Configuration ====================
INPUT_DIR = "evaluation_result_isometric_force\\radar_IQ_10_20_corrected"
OUTPUT_DIR = os.path.join(os.path.dirname(INPUT_DIR), "output")

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
    
    # Calculate RMSE in kg if MVIC_100 value is provided
    rmse_kg = None
    if mvic_100_value is not None and mvic_100_value > 0:
        y_true_kg = (y_true_mvic_pct / 100) * mvic_100_value
        y_pred_kg = (y_pred_mvic_pct / 100) * mvic_100_value
        squared_errors_kg = (y_true_kg - y_pred_kg) ** 2
        rmse_kg = np.sqrt(np.mean(squared_errors_kg))
    
    return {
        'subject': data['test_person'],
        'rmse_mvic_pct': rmse_mvic_pct,
        'rmse_kg': rmse_kg,
        'mvic_100': mvic_100_value,
        'num_predictions': len(y_true_mvic_pct),
    }


def plot_rmse_multi_session(all_results, config_suffix):
    """Plot RMSE for subjects with at least 2 sessions."""
    
    # Group results by subject name
    subject_groups = {}
    for result in all_results:
        # Extract subject name (short name without date)
        subject_name = result['subject'].split('_')[1] if '_' in result['subject'] else result['subject']
        
        if subject_name not in subject_groups:
            subject_groups[subject_name] = []
        
        subject_groups[subject_name].append(result)
    
    # Filter to only subjects with at least 2 sessions
    multi_session_subjects = {k: v for k, v in subject_groups.items() if len(v) >= 2}
    
    if not multi_session_subjects:
        print("No subjects found with 2+ sessions!")
        return
    
    print(f"\nFound {len(multi_session_subjects)} subjects with 2+ sessions:")
    for subject_name, sessions in sorted(multi_session_subjects.items()):
        print(f"  - {subject_name}: {len(sessions)} sessions")
    
    # ==================== MVIC % RMSE Plot ====================
    print("\nCreating RMSE bar plot (MVIC %) for multi-session subjects...")
    
    subjects = []
    rmse_means = []
    rmse_stds = []
    session_counts = []
    
    for subject_name in sorted(multi_session_subjects.keys()):
        sessions = multi_session_subjects[subject_name]
        rmse_list = [s['rmse_mvic_pct'] for s in sessions]
        
        subjects.append(subject_name)
        rmse_means.append(np.mean(rmse_list))
        rmse_stds.append(np.std(rmse_list, ddof=1))  # Sample std
        session_counts.append(len(sessions))
        
        print(f"  {subject_name}: {len(sessions)} sessions, RMSE = {np.mean(rmse_list):.4f} ± {rmse_stds[-1]:.4f} %MVIC")
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x_pos = np.arange(len(subjects))
    bars = ax.bar(x_pos, rmse_means, yerr=rmse_stds, 
                   capsize=6, alpha=0.85, color='#3498DB', 
                   edgecolor='black', linewidth=1.5, 
                   error_kw={'linewidth': 2.5, 'ecolor': '#E74C3C', 'capthick': 2.5})
    
    # Add value labels on top of bars
    for i, (mean, std, count) in enumerate(zip(rmse_means, rmse_stds, session_counts)):
        ax.text(i, mean + std + 0.3, f'{mean:.2f}%', 
               ha='center', va='bottom', fontsize=10, fontweight='bold')
        # Add session count below x-axis label
        ax.text(i, -0.5, f'(n={count})', 
               ha='center', va='top', fontsize=9, style='italic', color='gray')
    
    # Add average line
    overall_mean = np.mean(rmse_means)
    ax.axhline(y=overall_mean, color='red', linestyle='--', linewidth=2.5, 
              label=f'Average RMSE: {overall_mean:.2f}%', zorder=5)
    
    # Formatting
    ax.set_xlabel('Subject', fontsize=15, fontweight='bold')
    ax.set_ylabel('RMSE (%MVIC)', fontsize=15, fontweight='bold')
    ax.set_title(f'RMSE per Multi-Session Subject (%MVIC)\nSubjects with ≥2 Sessions\n(Segment: {SEGMENT}, Overlap: {OVERLAP}, Train Epochs: {TRAIN_EPOCH}, Fine-tune: {FINE_TUNE_EPOCH})', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(subjects, rotation=45, ha='right', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.legend(loc='upper right', fontsize=12, framealpha=0.95)
    
    # Adjust y-axis limits to accommodate session count labels
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min - 2, y_max)
    
    # Add statistics box
    textstr = f'Statistics (Multi-Session Subjects):\n'
    textstr += f'Number of subjects: {len(subjects)}\n'
    textstr += f'Mean RMSE: {overall_mean:.4f}%\n'
    textstr += f'Std RMSE: {np.std(rmse_means):.4f}%\n'
    textstr += f'Min RMSE: {min(rmse_means):.4f}%\n'
    textstr += f'Max RMSE: {max(rmse_means):.4f}%\n'
    textstr += f'Total sessions: {sum(session_counts)}'
    
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.9)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    filename = os.path.join(OUTPUT_DIR, f"rmse_multi_session_mvic{config_suffix}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved MVIC RMSE plot to: {filename}")
    plt.close()
    
    # ==================== kg RMSE Plot (if available) ====================
    # Check which multi-session subjects have kg data
    subjects_with_kg = []
    rmse_means_kg = []
    rmse_stds_kg = []
    session_counts_kg = []
    
    for subject_name in sorted(multi_session_subjects.keys()):
        sessions = multi_session_subjects[subject_name]
        rmse_kg_list = [s['rmse_kg'] for s in sessions if s['rmse_kg'] is not None]
        
        if len(rmse_kg_list) >= 2:  # Only include if at least 2 sessions have kg data
            subjects_with_kg.append(subject_name)
            rmse_means_kg.append(np.mean(rmse_kg_list))
            rmse_stds_kg.append(np.std(rmse_kg_list, ddof=1))
            session_counts_kg.append(len(rmse_kg_list))
    
    if subjects_with_kg:
        print(f"\nCreating RMSE bar plot (kg) for {len(subjects_with_kg)} multi-session subjects with kg data...")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x_pos = np.arange(len(subjects_with_kg))
        bars = ax.bar(x_pos, rmse_means_kg, yerr=rmse_stds_kg, 
                       capsize=6, alpha=0.85, color='#E67E22', 
                       edgecolor='black', linewidth=1.5, 
                       error_kw={'linewidth': 2.5, 'ecolor': '#16A085', 'capthick': 2.5})
        
        # Add value labels on top of bars
        for i, (mean, std, count) in enumerate(zip(rmse_means_kg, rmse_stds_kg, session_counts_kg)):
            ax.text(i, mean + std + 0.05, f'{mean:.2f}', 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
            # Add session count below x-axis label
            ax.text(i, -0.1, f'(n={count})', 
                   ha='center', va='top', fontsize=9, style='italic', color='gray')
        
        # Add average line
        overall_mean_kg = np.mean(rmse_means_kg)
        ax.axhline(y=overall_mean_kg, color='red', linestyle='--', linewidth=2.5, 
                  label=f'Average RMSE: {overall_mean_kg:.2f} kg', zorder=5)
        
        # Formatting
        ax.set_xlabel('Subject', fontsize=15, fontweight='bold')
        ax.set_ylabel('RMSE (kg)', fontsize=15, fontweight='bold')
        ax.set_title(f'RMSE per Multi-Session Subject (kg)\nSubjects with ≥2 Sessions\n(Segment: {SEGMENT}, Overlap: {OVERLAP}, Train Epochs: {TRAIN_EPOCH}, Fine-tune: {FINE_TUNE_EPOCH})', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(subjects_with_kg, rotation=45, ha='right', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.legend(loc='upper right', fontsize=12, framealpha=0.95)
        
        # Adjust y-axis limits to accommodate session count labels
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min - 0.2, y_max)
        
        # Add statistics box
        textstr_kg = f'Statistics (Multi-Session Subjects):\n'
        textstr_kg += f'Number of subjects: {len(subjects_with_kg)}\n'
        textstr_kg += f'Mean RMSE: {overall_mean_kg:.4f} kg\n'
        textstr_kg += f'Std RMSE: {np.std(rmse_means_kg):.4f} kg\n'
        textstr_kg += f'Min RMSE: {min(rmse_means_kg):.4f} kg\n'
        textstr_kg += f'Max RMSE: {max(rmse_means_kg):.4f} kg\n'
        textstr_kg += f'Total sessions: {sum(session_counts_kg)}'
        
        props_kg = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
        ax.text(0.02, 0.98, textstr_kg, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=props_kg, family='monospace')
        
        plt.tight_layout()
        filename_kg = os.path.join(OUTPUT_DIR, f"rmse_multi_session_kg{config_suffix}.png")
        plt.savefig(filename_kg, dpi=300, bbox_inches='tight')
        print(f"Saved kg RMSE plot to: {filename_kg}")
        plt.close()
    else:
        print("\nNo multi-session subjects with kg data available.")


if __name__ == "__main__":
    # Ensure directories exist
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' does not exist!")
        exit(1)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")
    
    # Find all evaluation JSON files
    search_pattern = os.path.join(INPUT_DIR, "*_evaluation_results.json")
    json_files = glob.glob(search_pattern)
    
    if not json_files:
        print(f"No evaluation JSON files found in '{INPUT_DIR}'!")
        exit(1)
    
    print(f"Found {len(json_files)} evaluation files in '{INPUT_DIR}'")
    
    # Process all files
    all_results = []
    for json_file in sorted(json_files):
        try:
            filename = os.path.basename(json_file)
            subject_name = filename.replace("_evaluation_results.json", "")
            mvic_100_value = MVIC_100.get(subject_name, None)
            
            result = process_evaluation_file(json_file, mvic_100_value)
            all_results.append(result)
            print(f"Processed: {subject_name} (RMSE: {result['rmse_mvic_pct']:.4f}%)")
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_results:
        print("\nNo files were successfully processed!")
        exit(1)
    
    # Create configuration suffix
    config_suffix = f"_seg{SEGMENT}_overlap{OVERLAP}_ep{TRAIN_EPOCH}_ft{FINE_TUNE_EPOCH}"
    
    # Generate plots for multi-session subjects
    plot_rmse_multi_session(all_results, config_suffix)
    
    print(f"\n{'='*80}")
    print(f"All done! Processed {len(all_results)} files.")
    print(f"{'='*80}")

