import json
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import csv

# ==================== Configuration ====================
INPUT_DIR = "evaluation_result_isometric_force\\radar_IQ_10_20_corrected"
OUTPUT_DIR = os.path.join(os.path.dirname(INPUT_DIR), "output")
BMI_CSV_FILE = os.path.join(OUTPUT_DIR, "subject_bmi_data.csv")

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


def load_bmi_data(csv_file):
    """Load BMI data from CSV file."""
    bmi_dict = {}
    
    if not os.path.exists(csv_file):
        print(f"Warning: BMI CSV file not found: {csv_file}")
        return bmi_dict
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['Name'].lower()
            bmi_dict[name] = {
                'bmi': float(row['BMI']),
                'category': row['Category'],
                'height_cm': float(row['Height_cm']),
                'weight_kg': float(row['Weight_kg'])
            }
    
    return bmi_dict


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


def plot_rmse_by_bmi_category(all_results, bmi_dict, config_suffix):
    """Plot RMSE grouped by BMI category."""
    
    # Group RMSE values by BMI category
    bmi_category_rmse = {}
    subjects_by_category = {}
    
    for result in all_results:
        full_name = result['subject']
        # Extract person name (after date)
        person_name = full_name.split('_')[1] if '_' in full_name else full_name
        person_name_lower = person_name.lower()
        
        # Get BMI category for this person
        if person_name_lower in bmi_dict:
            original_category = bmi_dict[person_name_lower]['category']
            rmse = result['rmse_mvic_pct']
            
            # Merge categories: Underweight + Normal weight, Overweight + Obese
            if original_category in ['Underweight', 'Normal weight']:
                category = 'Normal/\nUnderweight'
            elif original_category in ['Overweight', 'Obese']:
                category = 'Overweight/\nObese'
            else:
                category = original_category
            
            if category not in bmi_category_rmse:
                bmi_category_rmse[category] = []
                subjects_by_category[category] = set()
            
            bmi_category_rmse[category].append(rmse)
            subjects_by_category[category].add(person_name)
        else:
            print(f"Warning: No BMI data found for {person_name}")
    
    if not bmi_category_rmse:
        print("No data to plot!")
        return
    
    # Calculate statistics for each category
    categories_data = []
    for category in sorted(bmi_category_rmse.keys()):
        rmse_values = bmi_category_rmse[category]
        categories_data.append({
            'category': category,
            'mean': np.mean(rmse_values),
            'std': np.std(rmse_values, ddof=1) if len(rmse_values) > 1 else 0,
            'count': len(rmse_values),
            'subjects': len(subjects_by_category[category])
        })
    
    # Sort categories in a logical order
    category_order = ['Normal/\nUnderweight', 'Overweight/\nObese']
    categories_data.sort(key=lambda x: category_order.index(x['category']) if x['category'] in category_order else 999)
    
    print(f"\nRMSE by BMI Category (Merged):")
    for data in categories_data:
        print(f"  {data['category'].replace(chr(10), '/')}: {data['mean']:.2f}±{data['std']:.2f}% "
              f"(n={data['count']} sessions, {data['subjects']} subjects)")
    
    # ==================== Create Bar Plot ====================
    print("\nCreating RMSE by BMI category bar plot...")
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Prepare data
    x_pos = np.arange(len(categories_data))
    means = [d['mean'] for d in categories_data]
    stds = [d['std'] for d in categories_data]
    labels = [d['category'] for d in categories_data]
    counts = [d['count'] for d in categories_data]
    subjects = [d['subjects'] for d in categories_data]
    
    # Use consistent light blue/gray color like the reference image
    bar_color = '#8BA8C8'
    
    bars = ax.bar(x_pos, means, yerr=stds, 
                   capsize=5, alpha=1.0, 
                   color=bar_color,
                   edgecolor='none', linewidth=0, 
                   error_kw={'linewidth': 1.5, 'ecolor': 'black', 'capthick': 1.5})
    
    # Add statistics on each bar
    for i, (mean, std, count, subj) in enumerate(zip(means, stds, counts, subjects)):
        # Add mean value above the bar
        ax.text(i, mean + std + 0.5, f'{mean:.2f}', 
               ha='center', va='bottom', fontsize=11, fontweight='bold')
        # Add n (sessions) and subjects count on the bar
        ax.text(i, mean / 2, f'n={count}\n({subj} subj)', 
               ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    
    # Formatting - clean and professional
    ax.set_xlabel('BMI Category', fontsize=12)
    ax.set_ylabel('RMSE (%MVIC)', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=11)
    
    # Set y-axis to start at 0
    ax.set_ylim(0, max(means) + max(stds) + 3)
    
    # Remove grid and spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    
    # Clean tick parameters
    ax.tick_params(axis='both', which='major', labelsize=11, length=5, width=1)
    
    plt.tight_layout()
    filename = os.path.join(OUTPUT_DIR, f"rmse_by_bmi_category{config_suffix}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved RMSE by BMI category plot to: {filename}")
    plt.close()
    
    # ==================== Save statistics to text file ====================
    txt_filename = os.path.join(OUTPUT_DIR, f"rmse_by_bmi_category_data{config_suffix}.txt")
    with open(txt_filename, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("RMSE BY BMI CATEGORY (MERGED) - MVIC Percentage (%)\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"Configuration:\n")
        f.write(f"  Segment Length: {SEGMENT}\n")
        f.write(f"  Overlap: {OVERLAP}\n")
        f.write(f"  Train Epochs: {TRAIN_EPOCH}\n")
        f.write(f"  Fine-tune Epochs: {FINE_TUNE_EPOCH}\n\n")
        
        f.write("Category Merging:\n")
        f.write("  - Normal/Underweight = Underweight + Normal weight\n")
        f.write("  - Overweight/Obese = Overweight + Obese\n\n")
        
        f.write("="*100 + "\n")
        f.write("SUMMARY BY MERGED BMI CATEGORY\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"{'BMI Category':<25} {'Mean RMSE (%)':<15} {'Std RMSE (%)':<15} {'Sessions':<12} {'Subjects':<12}\n")
        f.write("-"*100 + "\n")
        for data in categories_data:
            cat_label = data['category'].replace('\n', '/')
            f.write(f"{cat_label:<25} "
                   f"{data['mean']:<15.4f} "
                   f"{data['std']:<15.4f} "
                   f"{data['count']:<12} "
                   f"{data['subjects']:<12}\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write("DETAILED SUBJECT DATA BY MERGED BMI CATEGORY\n")
        f.write("="*100 + "\n\n")
        
        for category in sorted(bmi_category_rmse.keys(), 
                              key=lambda x: category_order.index(x) if x in category_order else 999):
            cat_label = category.replace('\n', '/')
            f.write(f"\n{cat_label}:\n")
            f.write(f"  Number of subjects: {len(subjects_by_category[category])}\n")
            f.write(f"  Subjects: {', '.join(sorted(subjects_by_category[category]))}\n")
            f.write(f"  Number of sessions: {len(bmi_category_rmse[category])}\n")
            f.write(f"  RMSE values: {', '.join([f'{v:.4f}' for v in sorted(bmi_category_rmse[category])])}\n")
            f.write(f"  Mean: {np.mean(bmi_category_rmse[category]):.4f}%\n")
            f.write(f"  Std: {np.std(bmi_category_rmse[category], ddof=1) if len(bmi_category_rmse[category]) > 1 else 0:.4f}%\n")
            f.write(f"  Min: {min(bmi_category_rmse[category]):.4f}%\n")
            f.write(f"  Max: {max(bmi_category_rmse[category]):.4f}%\n")
        
        f.write("\n" + "="*100 + "\n")
    
    print(f"Saved BMI category data to: {txt_filename}")


if __name__ == "__main__":
    # Load BMI data
    print("Loading BMI data...")
    bmi_dict = load_bmi_data(BMI_CSV_FILE)
    
    if not bmi_dict:
        print("Error: Could not load BMI data. Please run calculate_subject_bmi.py first.")
        exit(1)
    
    print(f"Loaded BMI data for {len(bmi_dict)} subjects")
    
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
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_results:
        print("\nNo files were successfully processed!")
        exit(1)
    
    print(f"Processed {len(all_results)} evaluation files")
    
    # Create configuration suffix
    config_suffix = f"_seg{SEGMENT}_overlap{OVERLAP}_ep{TRAIN_EPOCH}_ft{FINE_TUNE_EPOCH}"
    
    # Generate plots
    plot_rmse_by_bmi_category(all_results, bmi_dict, config_suffix)
    
    print(f"\n{'='*80}")
    print(f"All done!")
    print(f"{'='*80}")

