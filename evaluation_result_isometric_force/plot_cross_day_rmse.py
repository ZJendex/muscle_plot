import json
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from datetime import datetime

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


def parse_date_from_filename(filename):
    """Parse date from filename format: MMDD_name_evaluation_results.json"""
    try:
        # Remove the _evaluation_results.json suffix
        name_part = filename.replace("_evaluation_results.json", "")
        # Extract MMDD (first 4 characters)
        date_str = name_part.split('_')[0]
        # Parse as MMDD format (assuming year 2024)
        month = int(date_str[:2])
        day = int(date_str[2:4])
        # Create a datetime object (using 2024 as year)
        return datetime(2024, month, day)
    except:
        return None


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


def plot_cross_day_rmse(all_results, config_suffix):
    """Plot accumulated cross-day RMSE for subjects with multiple sessions."""
    
    # Group results by subject name and sort by date
    subject_sessions = {}
    
    for result in all_results:
        full_name = result['subject']
        # Extract person name (after date)
        person_name = full_name.split('_')[1] if '_' in full_name else full_name
        date_str = full_name.split('_')[0]
        
        # Parse date
        date = parse_date_from_filename(full_name + "_evaluation_results.json")
        
        if date is None:
            continue
        
        if person_name not in subject_sessions:
            subject_sessions[person_name] = []
        
        subject_sessions[person_name].append({
            'date': date,
            'date_str': date_str,
            'full_name': full_name,
            'rmse_mvic': result['rmse_mvic_pct'],
            'rmse_kg': result['rmse_kg'],
            'mvic_100': result['mvic_100']
        })
    
    # Sort sessions by date for each person and filter to those with 2+ sessions
    multi_session_subjects = {}
    for person_name, sessions in subject_sessions.items():
        if len(sessions) >= 2:
            # Sort by date
            sessions_sorted = sorted(sessions, key=lambda x: x['date'])
            multi_session_subjects[person_name] = sessions_sorted
    
    if not multi_session_subjects:
        print("No subjects found with 2+ sessions!")
        return
    
    print(f"\nFound {len(multi_session_subjects)} subjects with 2+ sessions:")
    for person_name, sessions in sorted(multi_session_subjects.items()):
        dates_str = " -> ".join([s['date'].strftime('%m/%d') for s in sessions])
        print(f"  {person_name}: {len(sessions)} sessions ({dates_str})")
    
    # ==================== Plot 1: RMSE progression for each subject (MVIC %) ====================
    print("\nCreating cross-day RMSE progression plot (MVIC %)...")
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Use different colors for each subject
    colors = plt.cm.tab20(np.linspace(0, 1, len(multi_session_subjects)))
    
    all_week1_rmse = []
    all_week2_rmse = []
    all_week3_rmse = []
    
    x_offset = 0
    x_positions = {}
    x_labels = []
    x_ticks = []
    
    for idx, (person_name, sessions) in enumerate(sorted(multi_session_subjects.items())):
        color = colors[idx]
        
        # Extract RMSE values for each week
        week_rmse = [s['rmse_mvic'] for s in sessions]
        week_dates = [s['date'].strftime('%m/%d') for s in sessions]
        
        # Determine x positions for this subject
        num_weeks = len(week_rmse)
        x_pos = np.arange(x_offset, x_offset + num_weeks)
        x_positions[person_name] = x_pos
        
        # Plot line connecting the weeks
        ax.plot(x_pos, week_rmse, marker='o', markersize=10, linewidth=2.5, 
               color=color, label=person_name, alpha=0.8)
        
        # Add error bars (show individual points, no aggregation yet)
        for i, (pos, rmse, date) in enumerate(zip(x_pos, week_rmse, week_dates)):
            ax.scatter(pos, rmse, s=150, color=color, edgecolors='black', 
                      linewidth=1.5, zorder=10, alpha=0.9)
            # Add date labels on points
            ax.text(pos, rmse + 0.3, date, ha='center', va='bottom', 
                   fontsize=8, rotation=0, color=color, fontweight='bold')
        
        # Add x-axis labels
        for i, date in enumerate(week_dates):
            x_labels.append(f"Week {i+1}")
            x_ticks.append(x_pos[i])
        
        # Collect RMSE values for statistics
        if len(week_rmse) >= 1:
            all_week1_rmse.append(week_rmse[0])
        if len(week_rmse) >= 2:
            all_week2_rmse.append(week_rmse[1])
        if len(week_rmse) >= 3:
            all_week3_rmse.append(week_rmse[2])
        
        # Update offset for next subject
        x_offset += num_weeks + 0.5  # Add small gap between subjects
    
    # Formatting
    ax.set_xlabel('Cross-Day Sessions (Sorted by Date)', fontsize=15, fontweight='bold')
    ax.set_ylabel('RMSE (%MVIC)', fontsize=15, fontweight='bold')
    ax.set_title(f'Cross-Day RMSE Progression per Subject (%MVIC)\n(Segment: {SEGMENT}, Overlap: {OVERLAP}, Train Epochs: {TRAIN_EPOCH}, Fine-tune: {FINE_TUNE_EPOCH})', 
                fontsize=17, fontweight='bold', pad=20)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.legend(loc='upper left', fontsize=10, ncol=2, framealpha=0.95, 
             bbox_to_anchor=(0.02, 0.98))
    
    # Add statistics box
    textstr = f'Cross-Day Statistics:\n'
    textstr += f'Subjects: {len(multi_session_subjects)}\n'
    if all_week1_rmse:
        textstr += f'Week 1 Avg: {np.mean(all_week1_rmse):.2f}% (n={len(all_week1_rmse)})\n'
    if all_week2_rmse:
        textstr += f'Week 2 Avg: {np.mean(all_week2_rmse):.2f}% (n={len(all_week2_rmse)})\n'
    if all_week3_rmse:
        textstr += f'Week 3 Avg: {np.mean(all_week3_rmse):.2f}% (n={len(all_week3_rmse)})'
    
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.9)
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', horizontalalignment='right', 
           bbox=props, family='monospace')
    
    plt.tight_layout()
    filename = os.path.join(OUTPUT_DIR, f"cross_day_rmse_progression_mvic{config_suffix}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved cross-day RMSE progression plot to: {filename}")
    plt.close()
    
    # ==================== Plot 2: Accumulated bar plot with error bars (MVIC %) ====================
    print("\nCreating accumulated cross-day RMSE bar plot (MVIC %)...")
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Calculate mean and std for each week across all subjects
    weeks_data = []
    week_labels = []
    
    if all_week1_rmse:
        weeks_data.append({
            'mean': np.mean(all_week1_rmse),
            'std': np.std(all_week1_rmse, ddof=1) if len(all_week1_rmse) > 1 else 0,
            'count': len(all_week1_rmse)
        })
        week_labels.append('Week 1')
    
    if all_week2_rmse:
        weeks_data.append({
            'mean': np.mean(all_week2_rmse),
            'std': np.std(all_week2_rmse, ddof=1) if len(all_week2_rmse) > 1 else 0,
            'count': len(all_week2_rmse)
        })
        week_labels.append('Week 2')
    
    if all_week3_rmse:
        weeks_data.append({
            'mean': np.mean(all_week3_rmse),
            'std': np.std(all_week3_rmse, ddof=1) if len(all_week3_rmse) > 1 else 0,
            'count': len(all_week3_rmse)
        })
        week_labels.append('Week 3')
    
    # Plot bars with professional style
    x_pos = np.arange(len(weeks_data))
    means = [w['mean'] for w in weeks_data]
    stds = [w['std'] for w in weeks_data]
    counts = [w['count'] for w in weeks_data]
    
    # Use consistent light blue/gray color like the reference image
    bar_color = '#8BA8C8'  # Light blue-gray
    
    bars = ax.bar(x_pos, means, yerr=stds, 
                   capsize=5, alpha=1.0, 
                   color=bar_color,
                   edgecolor='none', linewidth=0, 
                   error_kw={'linewidth': 1.5, 'ecolor': 'black', 'capthick': 1.5})
    
    # Formatting - clean and professional
    ax.set_xlabel('Week', fontsize=12)
    ax.set_ylabel('RMSE (%MVIC)', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(week_labels, fontsize=11)
    
    # Set y-axis to start at 0
    ax.set_ylim(0, max(means) + max(stds) + 5)
    
    # Remove grid and spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    
    # Clean tick parameters
    ax.tick_params(axis='both', which='major', labelsize=11, length=5, width=1)
    
    plt.tight_layout()
    filename_bar = os.path.join(OUTPUT_DIR, f"cross_day_rmse_accumulated_mvic{config_suffix}.png")
    plt.savefig(filename_bar, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved accumulated RMSE bar plot to: {filename_bar}")
    
    # Print statistics to console instead
    print(f"\nAccumulated Statistics:")
    print(f"  Subjects with multi-sessions: {len(multi_session_subjects)}")
    for i, (label, data) in enumerate(zip(week_labels, weeks_data)):
        print(f"  {label}: {data['mean']:.2f}±{data['std']:.2f}% (n={data['count']})")
    
    plt.close()
    
    # ==================== Plot 3: Individual subject comparison (grouped bars) ====================
    print("\nCreating individual subject cross-day comparison (MVIC %)...")
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Prepare data for grouped bar chart
    subjects_sorted = sorted(multi_session_subjects.keys())
    week1_values = []
    week2_values = []
    week3_values = []
    
    for person_name in subjects_sorted:
        sessions = multi_session_subjects[person_name]
        week_rmse = [s['rmse_mvic'] for s in sessions]
        
        week1_values.append(week_rmse[0] if len(week_rmse) >= 1 else 0)
        week2_values.append(week_rmse[1] if len(week_rmse) >= 2 else 0)
        week3_values.append(week_rmse[2] if len(week_rmse) >= 3 else 0)
    
    x = np.arange(len(subjects_sorted))
    width = 0.25
    
    # Plot bars for each week
    bars1 = ax.bar(x - width, week1_values, width, label='Week 1', 
                   color='#3498DB', alpha=0.85, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x, week2_values, width, label='Week 2', 
                   color='#E67E22', alpha=0.85, edgecolor='black', linewidth=1)
    
    # Only add Week 3 if there's data
    if any(v > 0 for v in week3_values):
        bars3 = ax.bar(x + width, week3_values, width, label='Week 3', 
                       color='#2ECC71', alpha=0.85, edgecolor='black', linewidth=1)
    
    # Formatting
    ax.set_xlabel('Subject', fontsize=15, fontweight='bold')
    ax.set_ylabel('RMSE (%MVIC)', fontsize=15, fontweight='bold')
    ax.set_title(f'Cross-Day RMSE Comparison per Subject (%MVIC)\n(Segment: {SEGMENT}, Overlap: {OVERLAP}, Train Epochs: {TRAIN_EPOCH}, Fine-tune: {FINE_TUNE_EPOCH})', 
                fontsize=17, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(subjects_sorted, rotation=45, ha='right', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.legend(loc='upper right', fontsize=12, framealpha=0.95)
    
    plt.tight_layout()
    filename_comp = os.path.join(OUTPUT_DIR, f"cross_day_rmse_comparison_mvic{config_suffix}.png")
    plt.savefig(filename_comp, dpi=300, bbox_inches='tight')
    print(f"Saved individual comparison plot to: {filename_comp}")
    plt.close()


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
    
    # Generate cross-day RMSE plots
    plot_cross_day_rmse(all_results, config_suffix)
    
    print(f"\n{'='*80}")
    print(f"All done! Processed {len(all_results)} files.")
    print(f"{'='*80}")

