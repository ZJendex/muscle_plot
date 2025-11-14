import json
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# ==================== 配置区域 ====================

INPUT_DIR = "evaluation_result_isometric_force\\imu z axis filter"  # 当前目录，可以修改为其他路径，例如: "./server" 或 "D:/path/to/json/files"

# get the dir from the input_dir
OUTPUT_DIR = os.path.join(os.path.dirname(INPUT_DIR), "output")

# =================================================
SEGMENT = 400
OVERLAP = 0
TRAIN_EPOCH = 100
FINE_TUNE_EPOCH = 20


def process_evaluation_file(json_file, mvic_100_value=None):
    """Process a single evaluation JSON file for regression task."""
    
    print(f"\n{'='*80}")
    print(f"Processing: {json_file}")
    print(f"{'='*80}")
    
    # --- Load JSON ---
    with open(json_file, "r") as f:
        data = json.load(f)
    
    # Extract configuration parameters (segment_length, overlap, epochs)
    segment_length = SEGMENT
    overlap = OVERLAP
    num_epochs = TRAIN_EPOCH
    fine_tune_epochs = FINE_TUNE_EPOCH

    # Extract segment-level predictions and true values
    # Note: true_force_value and predicted_force_value are already in MVIC percentage
    segments = data["segments"]
    if data["test_person"] == "1027_yifeng":
        # only take the first 366 segments
        segments = segments[:366]
    
    # These values are already in MVIC percentage
    y_true_mvic_pct = np.array([seg["true_force_value"] for seg in segments])
    y_pred_mvic_pct = np.array([seg["predicted_force_value"] for seg in segments])
    print(f"shape of y_true_mvic_pct: {y_true_mvic_pct.shape}")
    print(f"shape of y_pred_mvic_pct: {y_pred_mvic_pct.shape}")

    # take every 5 consecutive segments as a group and calculate the average of the group
    y_true_mvic_pct_group = []
    y_pred_mvic_pct_group = []
    for i in range(0, len(y_true_mvic_pct), 5):
        y_true_mvic_pct_group.append(np.mean(y_true_mvic_pct[i:i+3]))
        y_pred_mvic_pct_group.append(np.mean(y_pred_mvic_pct[i:i+3]))
    y_true_mvic_pct_group = np.array(y_true_mvic_pct_group)
    y_pred_mvic_pct_group = np.array(y_pred_mvic_pct_group)
    print(f"shape of y_true_mvic_pct_group: {y_true_mvic_pct_group.shape}")
    print(f"shape of y_pred_mvic_pct_group: {y_pred_mvic_pct_group.shape}")

    y_true_mvic_pct = y_true_mvic_pct_group
    y_pred_mvic_pct = y_pred_mvic_pct_group
    
    # Calculate errors in MVIC percentage
    errors_mvic_pct = np.abs(y_true_mvic_pct - y_pred_mvic_pct)
    squared_errors_mvic_pct = (y_true_mvic_pct - y_pred_mvic_pct) ** 2
    
    # Metrics in MVIC percentage
    mae_mvic_pct = np.mean(errors_mvic_pct)
    rmse_mvic_pct = np.sqrt(np.mean(squared_errors_mvic_pct))
    r2 = data.get("r2_score", 0.0)
    correlation = data.get("correlation", 0.0)
    
    # Convert to kg if MVIC_100 value is provided
    y_true = None
    y_pred = None
    errors = None
    mae = None
    rmse = None
    
    if mvic_100_value is not None and mvic_100_value > 0:
        # Convert MVIC percentage to kg
        y_true = (y_true_mvic_pct / 100) * mvic_100_value
        y_pred = (y_pred_mvic_pct / 100) * mvic_100_value
        
        # Calculate errors in kg
        errors = np.abs(y_true - y_pred)
        squared_errors = (y_true - y_pred) ** 2
        
        # Metrics in kg
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(squared_errors))
    
    # Print metrics
    print(f"\n===== Regression Metrics (MVIC %) =====")
    print(f"MAE (%MVIC): {mae_mvic_pct:.4f}%")
    print(f"RMSE (%MVIC): {rmse_mvic_pct:.4f}%")
    print(f"R² Score: {r2:.4f}")
    print(f"Correlation: {correlation:.4f}")
    print(f"Number of predictions: {len(y_true_mvic_pct)}")
    
    if mvic_100_value is not None:
        print(f"\n===== Regression Metrics (kg) =====")
        print(f"MVIC_100: {mvic_100_value} kg")
        print(f"MAE (kg): {mae:.4f}")
        print(f"RMSE (kg): {rmse:.4f}")
    
    return {
        'subject': data['test_person'],
        'mae': mae,  # in kg (if mvic_100 provided)
        'rmse': rmse,  # in kg (if mvic_100 provided)
        'r2_score': r2,
        'correlation': correlation,
        'num_predictions': len(y_true_mvic_pct),
        'errors': errors,  # Individual errors for CDF (kg, if mvic_100 provided)
        'y_true': y_true,  # in kg (if mvic_100 provided)
        'y_pred': y_pred,  # in kg (if mvic_100 provided)
        # MVIC-related metrics (primary)
        'mvic_100': mvic_100_value,
        'errors_mvic_pct': errors_mvic_pct,  # Individual errors for CDF (MVIC %)
        'mae_mvic_pct': mae_mvic_pct,  # MAE in MVIC %
        'rmse_mvic_pct': rmse_mvic_pct,  # RMSE in MVIC %
        'y_true_mvic_pct': y_true_mvic_pct,  # in MVIC %
        'y_pred_mvic_pct': y_pred_mvic_pct,  # in MVIC %
        # Configuration parameters
        'segment_length': segment_length,
        'overlap': overlap,
        'num_epochs': num_epochs,
        'fine_tune_epochs': fine_tune_epochs,
    }

if __name__ == "__main__":
    MVIC_100 = {
        "1010_freddy":19,
        "1017_freddy":16,
        "1024_freddy":17.2,
        "1010_yibo":15.7,
        "1018_yibo":14.2,
        "1105_yibo":16.7,
        "1011_haoyu":15,
        "1019_haoyu":16,
        "1025_haoyu":20,
        "1012_ari":17.5,
        "1018_ari":21.5,
        "1012_aditi":14.5,
        "1019_aditi":16,
        "1024_aditi":16.3,
        "1010_jerry":18,
        "1017_jerry":15,
        "1025_jerry":15,
        "1013_jiangyifei":15,
        "1021_jiangyifei":14.2,
        "1025_jiangyifei":17,
        "1013_weifan":11,
        "1023_weifan":17.3,
        "1030_weifan":17.8,
        "1015_david":15,
        "1021_david":14.5,
        "1025_david":15.3,
        "1012_xiwen":11,
        "1018_xiwen":11.5,
        "1105_xiwen":10.4,
        "1029_bert":28,
        "1027_haozhe":17.4,
        "1028_jiasi":14.5,
        "1027_qiushi":14,
        "1027_yifeng":26.5,
        "1030_george":14,
        "1030_seungjoo":16,
        "1031_yang":15,
        "1101_anindya":38,
        "1102_anupam":16.2,
        "1102_pragna":14.5,
        "1102_veronica":14.5,
        "1103_junru":23,
        "1104_siqi":11,
    }
    # 确保输入目录存在
    if not os.path.exists(INPUT_DIR):
        print(f"错误: 输入文件夹 '{INPUT_DIR}' 不存在!")
        exit(1)
    
    # 创建输出目录（如果不存在）
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已创建输出文件夹: {OUTPUT_DIR}")
    
    # 在指定目录中查找所有评估JSON文件
    search_pattern = os.path.join(INPUT_DIR, "*_evaluation_results.json")
    json_files = glob.glob(search_pattern)
    
    if not json_files:
        print(f"在 '{INPUT_DIR}' 中未找到评估JSON文件!")
        print(f"查找模式: *_evaluation_results.json")
        exit(1)
    
    print(f"在 '{INPUT_DIR}' 中找到 {len(json_files)} 个评估文件:")
    for f in json_files:
        print(f"  - {os.path.basename(f)}")
    
    # Process all files and collect results
    all_results = []
    for json_file in sorted(json_files):
        try:
            # Extract subject name from filename
            filename = os.path.basename(json_file)
            subject_name = filename.replace("_evaluation_results.json", "")
            
            # Get MVIC_100 value for this subject
            mvic_100_value = MVIC_100.get(subject_name, None)
            
            if mvic_100_value is None:
                print(f"Warning: No MVIC_100 value found for {subject_name}, using None")
            
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
    
    # Extract segment_length and overlap from first result (should be same for all)
    segment_length = SEGMENT
    overlap = OVERLAP
    num_epochs = TRAIN_EPOCH
    fine_tune_epochs = FINE_TUNE_EPOCH
    print(f"\nConfiguration: Segment Length = {segment_length}, Overlap = {overlap}, Train Epochs = {num_epochs}, Fine-tune Epochs = {fine_tune_epochs}")
    
    # Create filename suffix with configuration
    config_suffix = f"_seg{segment_length}_overlap{overlap}_ep{num_epochs}_ft{fine_tune_epochs}"
    
    # Print summary table - MVIC Percentage (Primary, shown first)
    print("\n" + "="*130)
    print("SUMMARY OF ALL SUBJECTS - MVIC Percentage (%) [PRIMARY METRICS]")
    print("="*130)
    print(f"{'Subject':<25} {'MVIC_100 (kg)':<15} {'MAE (%)':<12} {'RMSE (%)':<12} {'R²':<12} {'Correlation':<12} {'Segments':<10}")
    print("-"*130)
    for result in all_results:
        mvic_str = f"{result['mvic_100']:.2f}" if result['mvic_100'] is not None else "N/A"
        print(f"{result['subject']:<25} "
              f"{mvic_str:<15} "
              f"{result['mae_mvic_pct']:<12.4f} "
              f"{result['rmse_mvic_pct']:<12.4f} "
              f"{result['r2_score']:<12.4f} "
              f"{result['correlation']:<12.4f} "
              f"{result['num_predictions']:<10}")
    
    # Calculate MVIC percentage statistics
    avg_mae_mvic = np.mean([r['mae_mvic_pct'] for r in all_results])
    avg_rmse_mvic = np.mean([r['rmse_mvic_pct'] for r in all_results])
    avg_r2 = np.mean([r['r2_score'] for r in all_results])
    avg_corr = np.mean([r['correlation'] for r in all_results])
    
    std_mae_mvic = np.std([r['mae_mvic_pct'] for r in all_results])
    std_rmse_mvic = np.std([r['rmse_mvic_pct'] for r in all_results])
    std_r2 = np.std([r['r2_score'] for r in all_results])
    std_corr = np.std([r['correlation'] for r in all_results])
    
    print("-"*130)
    print(f"{'AVERAGE':<25} "
          f"{'':<15} "
          f"{avg_mae_mvic:<12.4f} "
          f"{avg_rmse_mvic:<12.4f} "
          f"{avg_r2:<12.4f} "
          f"{avg_corr:<12.4f}")
    print(f"{'STD':<25} "
          f"{'':<15} "
          f"{std_mae_mvic:<12.4f} "
          f"{std_rmse_mvic:<12.4f} "
          f"{std_r2:<12.4f} "
          f"{std_corr:<12.4f}")
    print("="*130)
    
    # Print summary table - kg (Secondary, only for subjects with MVIC_100 data)
    results_with_mvic = [r for r in all_results if r['mvic_100'] is not None and r['mae'] is not None]
    
    if results_with_mvic:
        print("\n" + "="*130)
        print("SUMMARY OF ALL SUBJECTS - kg [SECONDARY METRICS - For Reference]")
        print("="*130)
        print(f"{'Subject':<25} {'MAE (kg)':<12} {'RMSE (kg)':<12} {'R²':<12} {'Correlation':<12} {'Segments':<10}")
        print("-"*130)
        for result in results_with_mvic:
            print(f"{result['subject']:<25} "
                  f"{result['mae']:<12.4f} "
                  f"{result['rmse']:<12.4f} "
                  f"{result['r2_score']:<12.4f} "
                  f"{result['correlation']:<12.4f} "
                  f"{result['num_predictions']:<10}")
        
        # Calculate kg statistics
        avg_mae = np.mean([r['mae'] for r in results_with_mvic])
        avg_rmse = np.mean([r['rmse'] for r in results_with_mvic])
        
        std_mae = np.std([r['mae'] for r in results_with_mvic])
        std_rmse = np.std([r['rmse'] for r in results_with_mvic])
        
        print("-"*130)
        print(f"{'AVERAGE':<25} "
              f"{avg_mae:<12.4f} "
              f"{avg_rmse:<12.4f} "
              f"{avg_r2:<12.4f} "
              f"{avg_corr:<12.4f}")
        print(f"{'STD':<25} "
              f"{std_mae:<12.4f} "
              f"{std_rmse:<12.4f} "
              f"{std_r2:<12.4f} "
              f"{std_corr:<12.4f}")
        print("="*130)
    else:
        print("\n⚠️  No MVIC_100 data available for kg calculations")
    
    # Save detailed summary to file
    summary_filename = os.path.join(OUTPUT_DIR, f"summary_all_subjects{config_suffix}.txt")
    with open(summary_filename, 'w', encoding='utf-8') as f:
        f.write("="*130 + "\n")
        f.write("REGRESSION EVALUATION SUMMARY - ALL SUBJECTS\n")
        f.write("="*130 + "\n\n")
        
        # MVIC Percentage Results (Primary)
        f.write("INDIVIDUAL SUBJECT RESULTS - MVIC Percentage (%) [PRIMARY METRICS]\n")
        f.write("-"*130 + "\n")
        f.write(f"{'Subject':<25} {'MVIC_100 (kg)':<15} {'MAE (%)':<12} {'RMSE (%)':<12} {'R²':<12} {'Correlation':<12} {'Segments':<10}\n")
        f.write("-"*130 + "\n")
        for result in all_results:
            mvic_str = f"{result['mvic_100']:.2f}" if result['mvic_100'] is not None else "N/A"
            f.write(f"{result['subject']:<25} "
                   f"{mvic_str:<15} "
                   f"{result['mae_mvic_pct']:<12.4f} "
                   f"{result['rmse_mvic_pct']:<12.4f} "
                   f"{result['r2_score']:<12.4f} "
                   f"{result['correlation']:<12.4f} "
                   f"{result['num_predictions']:<10}\n")
        
        f.write("\n" + "="*130 + "\n")
        f.write("SUMMARY STATISTICS - MVIC Percentage (%)\n")
        f.write("="*130 + "\n\n")
        
        f.write(f"{'Metric':<30} {'Mean':<15} {'Std':<15}\n")
        f.write("-"*60 + "\n")
        f.write(f"{'MAE (%MVIC)':<30} {avg_mae_mvic:<15.4f} {std_mae_mvic:<15.4f}\n")
        f.write(f"{'RMSE (%MVIC)':<30} {avg_rmse_mvic:<15.4f} {std_rmse_mvic:<15.4f}\n")
        f.write(f"{'R² Score':<30} {avg_r2:<15.4f} {std_r2:<15.4f}\n")
        f.write(f"{'Correlation':<30} {avg_corr:<15.4f} {std_corr:<15.4f}\n")
        
        # kg Results (Secondary, only if available)
        if results_with_mvic:
            f.write("\n\n" + "="*130 + "\n")
            f.write("INDIVIDUAL SUBJECT RESULTS - kg [SECONDARY METRICS - For Reference]\n")
            f.write("-"*130 + "\n")
            f.write(f"{'Subject':<25} {'MAE (kg)':<12} {'RMSE (kg)':<12} {'R²':<12} {'Correlation':<12} {'Segments':<10}\n")
            f.write("-"*130 + "\n")
            for result in results_with_mvic:
                f.write(f"{result['subject']:<25} "
                       f"{result['mae']:<12.4f} "
                       f"{result['rmse']:<12.4f} "
                       f"{result['r2_score']:<12.4f} "
                       f"{result['correlation']:<12.4f} "
                       f"{result['num_predictions']:<10}\n")
            
            f.write("\n" + "="*130 + "\n")
            f.write("SUMMARY STATISTICS - kg\n")
            f.write("="*130 + "\n\n")
            
            f.write(f"{'Metric':<30} {'Mean':<15} {'Std':<15}\n")
            f.write("-"*60 + "\n")
            f.write(f"{'MAE (kg)':<30} {avg_mae:<15.4f} {std_mae:<15.4f}\n")
            f.write(f"{'RMSE (kg)':<30} {avg_rmse:<15.4f} {std_rmse:<15.4f}\n")
            f.write(f"{'R² Score':<30} {avg_r2:<15.4f} {std_r2:<15.4f}\n")
            f.write(f"{'Correlation':<30} {avg_corr:<15.4f} {std_corr:<15.4f}\n")
        
        f.write("\n" + "="*130 + "\n")
    
    print(f"\nSummary saved to: {summary_filename}")
    
    # ==================== Create MVIC Percentage CDF Plots (PRIMARY) ====================
    print("\nCreating CDF plots for MVIC percentage errors (PRIMARY METRICS)...")
    
    # Collect all MVIC percentage errors from all subjects
    all_errors_mvic = []
    for result in all_results:
        all_errors_mvic.extend(result['errors_mvic_pct'])
    
    all_errors_mvic = np.array(all_errors_mvic)
    
    # Create CDF plot with individual subjects (MVIC %)
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Sort errors for CDF
    sorted_errors_mvic = np.sort(all_errors_mvic)
    cdf_mvic = np.arange(1, len(sorted_errors_mvic) + 1) / len(sorted_errors_mvic)
    
    # Plot combined CDF (thick line)
    ax.plot(sorted_errors_mvic, cdf_mvic, linewidth=3, color='#FF6B35', 
           label='All Subjects Combined', zorder=10)
    
    # Add individual subject CDFs with lighter colors
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_results)))
    for idx, result in enumerate(all_results):
        errors = result['errors_mvic_pct']
        sorted_e = np.sort(errors)
        cdf_e = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
        subject_name = result['subject'].split('_')[1] if '_' in result['subject'] else result['subject']
        ax.plot(sorted_e, cdf_e, linewidth=1.5, alpha=0.5, color=colors[idx], 
               label=f"{subject_name}")
    
    # Add reference lines for cumulative probabilities
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, label='50th percentile')
    ax.axhline(y=0.9, color='gray', linestyle=':', linewidth=1.5, alpha=0.6, label='90th percentile')
    ax.axhline(y=0.95, color='gray', linestyle='-.', linewidth=1.5, alpha=0.6, label='95th percentile')
    
    # Calculate and display key percentiles
    p50_mvic = np.percentile(all_errors_mvic, 50)
    p90_mvic = np.percentile(all_errors_mvic, 90)
    p95_mvic = np.percentile(all_errors_mvic, 95)
    
    ax.axvline(x=p50_mvic, color='blue', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.axvline(x=p90_mvic, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.axvline(x=p95_mvic, color='darkred', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Labels and formatting
    ax.set_xlabel('Absolute Error (%MVIC)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=13, fontweight='bold')
    ax.set_title(f'CDF of Prediction Errors (%MVIC) Across All Subjects\n(Segment: {segment_length}, Overlap: {overlap}, Train Epochs: {num_epochs}, Fine-tune: {fine_tune_epochs})', 
                fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=9, ncol=2, framealpha=0.95)
    
    # Add text box with statistics
    textstr_mvic = f'Statistics (%MVIC):\n'
    textstr_mvic += f'Mean MAE: {np.mean(all_errors_mvic):.4f}%\n'
    textstr_mvic += f'Median (50th): {p50_mvic:.4f}%\n'
    textstr_mvic += f'90th percentile: {p90_mvic:.4f}%\n'
    textstr_mvic += f'95th percentile: {p95_mvic:.4f}%\n'
    textstr_mvic += f'Total predictions: {len(all_errors_mvic)}'
    
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.85)
    ax.text(0.05, 0.95, textstr_mvic, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    cdf_mvic_filename = os.path.join(OUTPUT_DIR, f"error_cdf_mvic_all_subjects{config_suffix}.png")
    plt.savefig(cdf_mvic_filename, dpi=300, bbox_inches='tight')
    print(f"Saved MVIC CDF plot (with individual subjects) to: {cdf_mvic_filename}")
    plt.close()
    
    # Create simplified version with just the combined CDF (MVIC %)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sorted_errors_mvic, cdf_mvic, linewidth=3.5, color='#FF6B35', label='All Subjects Combined')
    
    # Add reference lines
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
    ax.axhline(y=0.9, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
    ax.axhline(y=0.95, color='gray', linestyle='-.', linewidth=1.5, alpha=0.6)
    ax.axvline(x=p50_mvic, color='blue', linestyle='--', linewidth=2, alpha=0.6, label=f'Median: {p50_mvic:.4f}%')
    ax.axvline(x=p90_mvic, color='red', linestyle='--', linewidth=2, alpha=0.6, label=f'90th: {p90_mvic:.4f}%')
    ax.axvline(x=p95_mvic, color='darkred', linestyle='--', linewidth=2, alpha=0.6, label=f'95th: {p95_mvic:.4f}%')
    
    ax.set_xlabel('Absolute Error (%MVIC)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=13, fontweight='bold')
    ax.set_title(f'CDF of Prediction Errors (%MVIC) - All Subjects Combined\n(Segment: {segment_length}, Overlap: {overlap}, Train Epochs: {num_epochs}, Fine-tune: {fine_tune_epochs})', 
                fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    
    # Add text box
    ax.text(0.05, 0.95, textstr_mvic, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    cdf_mvic_simple_filename = os.path.join(OUTPUT_DIR, f"error_cdf_mvic_combined{config_suffix}.png")
    plt.savefig(cdf_mvic_simple_filename, dpi=300, bbox_inches='tight')
    print(f"Saved MVIC CDF plot (combined only) to: {cdf_mvic_simple_filename}")
    plt.close()
    
    # ==================== Create kg CDF Plots (SECONDARY - For Reference) ====================
    # Only create kg plots for subjects with MVIC_100 data
    if results_with_mvic:
        print("\nCreating CDF plots for kg errors (SECONDARY - For Reference)...")
    # ==================== Create kg CDF Plots (SECONDARY - For Reference) ====================
    # Only create kg plots for subjects with MVIC_100 data
    if results_with_mvic:
        print("\nCreating CDF plots for kg errors (SECONDARY - For Reference)...")
        
        # Collect all errors in kg from subjects with MVIC_100 data
        all_errors = []
        for result in results_with_mvic:
            all_errors.extend(result['errors'])
        
        all_errors = np.array(all_errors)
        
        # Create CDF plot with individual subjects
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Sort errors for CDF
        sorted_errors = np.sort(all_errors)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        
        # Plot combined CDF (thick line)
        ax.plot(sorted_errors, cdf, linewidth=3, color='#FF6B35', 
               label='All Subjects Combined', zorder=10)
        
        # Add individual subject CDFs with lighter colors
        colors = plt.cm.tab20(np.linspace(0, 1, len(results_with_mvic)))
        for idx, result in enumerate(results_with_mvic):
            errors = result['errors']
            sorted_e = np.sort(errors)
            cdf_e = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
            subject_name = result['subject'].split('_')[1] if '_' in result['subject'] else result['subject']
            ax.plot(sorted_e, cdf_e, linewidth=1.5, alpha=0.5, color=colors[idx], 
                   label=f"{subject_name}")
        
        # Add reference lines for cumulative probabilities
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, label='50th percentile')
        ax.axhline(y=0.9, color='gray', linestyle=':', linewidth=1.5, alpha=0.6, label='90th percentile')
        ax.axhline(y=0.95, color='gray', linestyle='-.', linewidth=1.5, alpha=0.6, label='95th percentile')
        
        # Calculate and display key percentiles
        p50 = np.percentile(all_errors, 50)
        p90 = np.percentile(all_errors, 90)
        p95 = np.percentile(all_errors, 95)
        
        ax.axvline(x=p50, color='blue', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.axvline(x=p90, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.axvline(x=p95, color='darkred', linestyle='--', linewidth=1.5, alpha=0.5)
        
        # Labels and formatting
        ax.set_xlabel('Absolute Error (kg)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Cumulative Probability', fontsize=13, fontweight='bold')
        ax.set_title(f'CDF of Prediction Errors (kg) Across All Subjects [For Reference]\n(Segment: {segment_length}, Overlap: {overlap}, Train Epochs: {num_epochs}, Fine-tune: {fine_tune_epochs})', 
                    fontsize=15, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='lower right', fontsize=9, ncol=2, framealpha=0.95)
        
        # Add text box with statistics
        textstr = f'Statistics (kg):\n'
        textstr += f'Mean MAE: {np.mean(all_errors):.4f}\n'
        textstr += f'Median (50th): {p50:.4f}\n'
        textstr += f'90th percentile: {p90:.4f}\n'
        textstr += f'95th percentile: {p95:.4f}\n'
        textstr += f'Total predictions: {len(all_errors)}'
        
        props_kg = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=props_kg, family='monospace')
        
        plt.tight_layout()
        cdf_filename = os.path.join(OUTPUT_DIR, f"error_cdf_kg_all_subjects{config_suffix}.png")
        plt.savefig(cdf_filename, dpi=300, bbox_inches='tight')
        print(f"Saved kg CDF plot (with individual subjects) to: {cdf_filename}")
        plt.close()
        
        # Create simplified version with just the combined CDF
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sorted_errors, cdf, linewidth=3.5, color='#FF6B35', label='All Subjects Combined')
        
        # Add reference lines
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
        ax.axhline(y=0.9, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
        ax.axhline(y=0.95, color='gray', linestyle='-.', linewidth=1.5, alpha=0.6)
        ax.axvline(x=p50, color='blue', linestyle='--', linewidth=2, alpha=0.6, label=f'Median: {p50:.4f}')
        ax.axvline(x=p90, color='red', linestyle='--', linewidth=2, alpha=0.6, label=f'90th: {p90:.4f}')
        ax.axvline(x=p95, color='darkred', linestyle='--', linewidth=2, alpha=0.6, label=f'95th: {p95:.4f}')
        
        ax.set_xlabel('Absolute Error (kg)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Cumulative Probability', fontsize=13, fontweight='bold')
        ax.set_title(f'CDF of Prediction Errors (kg) - All Subjects Combined [For Reference]\n(Segment: {segment_length}, Overlap: {overlap}, Train Epochs: {num_epochs}, Fine-tune: {fine_tune_epochs})', 
                    fontsize=15, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
        
        # Add text box
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=props_kg, family='monospace')
        
        plt.tight_layout()
        cdf_simple_filename = os.path.join(OUTPUT_DIR, f"error_cdf_kg_combined{config_suffix}.png")
        plt.savefig(cdf_simple_filename, dpi=300, bbox_inches='tight')
        print(f"Saved kg CDF plot (combined only) to: {cdf_simple_filename}")
        plt.close()
        
        cdf_filename_exists = True
        cdf_simple_filename_exists = True
    else:
        cdf_filename_exists = False
        cdf_simple_filename_exists = False
        cdf_filename = None
        cdf_simple_filename = None
    
    # ==================== Create MAE Bar Plot with Error Bars (MVIC %) ====================
    print("\nCreating MAE bar plot with error bars (MVIC %)...")
    
    # Group results by subject name (e.g., all "freddy" sessions together)
    subject_groups_mvic = {}
    for result in all_results:
        # Extract subject name (short name without date)
        subject_name = result['subject'].split('_')[1] if '_' in result['subject'] else result['subject']
        
        if subject_name not in subject_groups_mvic:
            subject_groups_mvic[subject_name] = []
        
        subject_groups_mvic[subject_name].append({
            'mae': result['mae_mvic_pct'],
            'full_name': result['subject']
        })
    
    # Calculate mean and std for each subject across all their sessions
    subjects = []
    mae_values_mvic = []
    mae_std_values_mvic = []
    
    for subject_name in sorted(subject_groups_mvic.keys()):
        sessions = subject_groups_mvic[subject_name]
        mae_list = [s['mae'] for s in sessions]
        
        subjects.append(subject_name)
        mae_values_mvic.append(np.mean(mae_list))
        
        # If multiple sessions, calculate std; otherwise std = 0
        if len(mae_list) > 1:
            mae_std_values_mvic.append(np.std(mae_list, ddof=1))  # Sample std
        else:
            mae_std_values_mvic.append(0.0)
        
        print(f"  {subject_name}: {len(sessions)} session(s), MAE = {np.mean(mae_list):.4f} ± {mae_std_values_mvic[-1]:.4f} %MVIC")
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x_pos = np.arange(len(subjects))
    bars = ax.bar(x_pos, mae_values_mvic, yerr=mae_std_values_mvic, 
                   capsize=5, alpha=0.8, color='#4A90E2', 
                   edgecolor='black', linewidth=1.2, 
                   error_kw={'linewidth': 2, 'ecolor': '#FF6B35'})
    
    # Add value labels on top of bars
    for i, (mae, std) in enumerate(zip(mae_values_mvic, mae_std_values_mvic)):
        ax.text(i, mae + std + 0.5, f'{mae:.2f}', 
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add average line
    ax.axhline(y=avg_mae_mvic, color='red', linestyle='--', linewidth=2.5, 
              label=f'Average MAE: {avg_mae_mvic:.2f}%', zorder=5)
    
    # Formatting
    ax.set_xlabel('Subject', fontsize=14, fontweight='bold')
    ax.set_ylabel('MAE (%MVIC)', fontsize=14, fontweight='bold')
    ax.set_title(f'MAE per Subject with Standard Deviation (%MVIC)\n(Segment: {segment_length}, Overlap: {overlap}, Train Epochs: {num_epochs}, Fine-tune: {fine_tune_epochs})', 
                fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(subjects, rotation=45, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.legend(loc='upper right', fontsize=12, framealpha=0.95)
    
    # Add statistics box
    textstr_bar = f'Statistics:\n'
    textstr_bar += f'Mean MAE: {avg_mae_mvic:.4f}%\n'
    textstr_bar += f'Std MAE: {std_mae_mvic:.4f}%\n'
    textstr_bar += f'Min MAE: {min(mae_values_mvic):.4f}%\n'
    textstr_bar += f'Max MAE: {max(mae_values_mvic):.4f}%\n'
    textstr_bar += f'Subjects: {len(subjects)}'
    
    props_bar = dict(boxstyle='round', facecolor='lightgreen', alpha=0.85)
    ax.text(0.02, 0.98, textstr_bar, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=props_bar, family='monospace')
    
    plt.tight_layout()
    mae_bar_filename = os.path.join(OUTPUT_DIR, f"mae_per_subject_with_errorbar_mvic{config_suffix}.png")
    plt.savefig(mae_bar_filename, dpi=300, bbox_inches='tight')
    print(f"Saved MAE bar plot (MVIC %) to: {mae_bar_filename}")
    plt.close()
    
    # ==================== Create MAE Bar Plot with Error Bars (kg) - if available ====================
    if results_with_mvic:
        print("\nCreating MAE bar plot with error bars (kg)...")
        
        # Group results by subject name (e.g., all "freddy" sessions together)
        subject_groups_kg = {}
        for result in results_with_mvic:
            # Extract subject name (short name without date)
            subject_name = result['subject'].split('_')[1] if '_' in result['subject'] else result['subject']
            
            if subject_name not in subject_groups_kg:
                subject_groups_kg[subject_name] = []
            
            subject_groups_kg[subject_name].append({
                'mae': result['mae'],
                'full_name': result['subject']
            })
        
        # Calculate mean and std for each subject across all their sessions
        subjects_kg = []
        mae_values_kg = []
        mae_std_values_kg = []
        
        for subject_name in sorted(subject_groups_kg.keys()):
            sessions = subject_groups_kg[subject_name]
            mae_list = [s['mae'] for s in sessions]
            
            subjects_kg.append(subject_name)
            mae_values_kg.append(np.mean(mae_list))
            
            # If multiple sessions, calculate std; otherwise std = 0
            if len(mae_list) > 1:
                mae_std_values_kg.append(np.std(mae_list, ddof=1))  # Sample std
            else:
                mae_std_values_kg.append(0.0)
            
            print(f"  {subject_name}: {len(sessions)} session(s), MAE = {np.mean(mae_list):.4f} ± {mae_std_values_kg[-1]:.4f} kg")
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(16, 8))
        
        x_pos = np.arange(len(subjects_kg))
        bars = ax.bar(x_pos, mae_values_kg, yerr=mae_std_values_kg, 
                       capsize=5, alpha=0.8, color='#9B59B6', 
                       edgecolor='black', linewidth=1.2, 
                       error_kw={'linewidth': 2, 'ecolor': '#E74C3C'})
        
        # Add value labels on top of bars
        for i, (mae, std) in enumerate(zip(mae_values_kg, mae_std_values_kg)):
            ax.text(i, mae + std + 0.05, f'{mae:.2f}', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add average line
        ax.axhline(y=avg_mae, color='red', linestyle='--', linewidth=2.5, 
                  label=f'Average MAE: {avg_mae:.2f} kg', zorder=5)
        
        # Formatting
        ax.set_xlabel('Subject', fontsize=14, fontweight='bold')
        ax.set_ylabel('MAE (kg)', fontsize=14, fontweight='bold')
        ax.set_title(f'MAE per Subject with Standard Deviation (kg) [For Reference]\n(Segment: {segment_length}, Overlap: {overlap}, Train Epochs: {num_epochs}, Fine-tune: {fine_tune_epochs})', 
                    fontsize=16, fontweight='bold', pad=15)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(subjects_kg, rotation=45, ha='right', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.legend(loc='upper right', fontsize=12, framealpha=0.95)
        
        # Add statistics box
        textstr_bar_kg = f'Statistics:\n'
        textstr_bar_kg += f'Mean MAE: {avg_mae:.4f} kg\n'
        textstr_bar_kg += f'Std MAE: {std_mae:.4f} kg\n'
        textstr_bar_kg += f'Min MAE: {min(mae_values_kg):.4f} kg\n'
        textstr_bar_kg += f'Max MAE: {max(mae_values_kg):.4f} kg\n'
        textstr_bar_kg += f'Subjects: {len(subjects_kg)}'
        
        props_bar_kg = dict(boxstyle='round', facecolor='lightyellow', alpha=0.85)
        ax.text(0.02, 0.98, textstr_bar_kg, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=props_bar_kg, family='monospace')
        
        plt.tight_layout()
        mae_bar_kg_filename = os.path.join(OUTPUT_DIR, f"mae_per_subject_with_errorbar_kg{config_suffix}.png")
        plt.savefig(mae_bar_kg_filename, dpi=300, bbox_inches='tight')
        print(f"Saved MAE bar plot (kg) to: {mae_bar_kg_filename}")
        plt.close()
    
    # ==================== Create RMSE Bar Plot with Error Bars (MVIC %) ====================
    print("\nCreating RMSE bar plot with error bars (MVIC %)...")
    
    # Group results by subject name (e.g., all "freddy" sessions together)
    subject_groups_rmse_mvic = {}
    for result in all_results:
        # Extract subject name (short name without date)
        subject_name = result['subject'].split('_')[1] if '_' in result['subject'] else result['subject']
        
        if subject_name not in subject_groups_rmse_mvic:
            subject_groups_rmse_mvic[subject_name] = []
        
        subject_groups_rmse_mvic[subject_name].append({
            'rmse': result['rmse_mvic_pct'],
            'full_name': result['subject']
        })
    
    # Calculate mean and std for each subject across all their sessions
    subjects_rmse = []
    rmse_values_mvic = []
    rmse_std_values_mvic = []
    
    for subject_name in sorted(subject_groups_rmse_mvic.keys()):
        sessions = subject_groups_rmse_mvic[subject_name]
        rmse_list = [s['rmse'] for s in sessions]
        
        subjects_rmse.append(subject_name)
        rmse_values_mvic.append(np.mean(rmse_list))
        
        # If multiple sessions, calculate std; otherwise std = 0
        if len(rmse_list) > 1:
            rmse_std_values_mvic.append(np.std(rmse_list, ddof=1))  # Sample std
        else:
            rmse_std_values_mvic.append(0.0)
        
        print(f"  {subject_name}: {len(sessions)} session(s), RMSE = {np.mean(rmse_list):.4f} ± {rmse_std_values_mvic[-1]:.4f} %MVIC")
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x_pos = np.arange(len(subjects_rmse))
    bars = ax.bar(x_pos, rmse_values_mvic, yerr=rmse_std_values_mvic, 
                   capsize=5, alpha=0.8, color='#2ECC71', 
                   edgecolor='black', linewidth=1.2, 
                   error_kw={'linewidth': 2, 'ecolor': '#E67E22'})
    
    # Add value labels on top of bars
    for i, (rmse, std) in enumerate(zip(rmse_values_mvic, rmse_std_values_mvic)):
        ax.text(i, rmse + std + 0.5, f'{rmse:.2f}', 
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add average line
    ax.axhline(y=avg_rmse_mvic, color='red', linestyle='--', linewidth=2.5, 
              label=f'Average RMSE: {avg_rmse_mvic:.2f}%', zorder=5)
    
    # Formatting
    ax.set_xlabel('Subject', fontsize=14, fontweight='bold')
    ax.set_ylabel('RMSE (%MVIC)', fontsize=14, fontweight='bold')
    ax.set_title(f'RMSE per Subject with Standard Deviation (%MVIC)\n(Segment: {segment_length}, Overlap: {overlap}, Train Epochs: {num_epochs}, Fine-tune: {fine_tune_epochs})', 
                fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(subjects_rmse, rotation=45, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.legend(loc='upper right', fontsize=12, framealpha=0.95)
    
    # Add statistics box
    textstr_rmse_bar = f'Statistics:\n'
    textstr_rmse_bar += f'Mean RMSE: {avg_rmse_mvic:.4f}%\n'
    textstr_rmse_bar += f'Std RMSE: {std_rmse_mvic:.4f}%\n'
    textstr_rmse_bar += f'Min RMSE: {min(rmse_values_mvic):.4f}%\n'
    textstr_rmse_bar += f'Max RMSE: {max(rmse_values_mvic):.4f}%\n'
    textstr_rmse_bar += f'Subjects: {len(subjects_rmse)}'
    
    props_rmse_bar = dict(boxstyle='round', facecolor='lightcoral', alpha=0.85)
    ax.text(0.02, 0.98, textstr_rmse_bar, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=props_rmse_bar, family='monospace')
    
    plt.tight_layout()
    rmse_bar_filename = os.path.join(OUTPUT_DIR, f"rmse_per_subject_with_errorbar_mvic{config_suffix}.png")
    plt.savefig(rmse_bar_filename, dpi=300, bbox_inches='tight')
    print(f"Saved RMSE bar plot (MVIC %) to: {rmse_bar_filename}")
    plt.close()
    
    # ==================== Create RMSE Bar Plot with Error Bars (kg) - if available ====================
    if results_with_mvic:
        print("\nCreating RMSE bar plot with error bars (kg)...")
        
        # Group results by subject name (e.g., all "freddy" sessions together)
        subject_groups_rmse_kg = {}
        for result in results_with_mvic:
            # Extract subject name (short name without date)
            subject_name = result['subject'].split('_')[1] if '_' in result['subject'] else result['subject']
            
            if subject_name not in subject_groups_rmse_kg:
                subject_groups_rmse_kg[subject_name] = []
            
            subject_groups_rmse_kg[subject_name].append({
                'rmse': result['rmse'],
                'full_name': result['subject']
            })
        
        # Calculate mean and std for each subject across all their sessions
        subjects_rmse_kg = []
        rmse_values_kg = []
        rmse_std_values_kg = []
        
        for subject_name in sorted(subject_groups_rmse_kg.keys()):
            sessions = subject_groups_rmse_kg[subject_name]
            rmse_list = [s['rmse'] for s in sessions]
            
            subjects_rmse_kg.append(subject_name)
            rmse_values_kg.append(np.mean(rmse_list))
            
            # If multiple sessions, calculate std; otherwise std = 0
            if len(rmse_list) > 1:
                rmse_std_values_kg.append(np.std(rmse_list, ddof=1))  # Sample std
            else:
                rmse_std_values_kg.append(0.0)
            
            print(f"  {subject_name}: {len(sessions)} session(s), RMSE = {np.mean(rmse_list):.4f} ± {rmse_std_values_kg[-1]:.4f} kg")
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(16, 8))
        
        x_pos = np.arange(len(subjects_rmse_kg))
        bars = ax.bar(x_pos, rmse_values_kg, yerr=rmse_std_values_kg, 
                       capsize=5, alpha=0.8, color='#E67E22', 
                       edgecolor='black', linewidth=1.2, 
                       error_kw={'linewidth': 2, 'ecolor': '#16A085'})
        
        # Add value labels on top of bars
        for i, (rmse, std) in enumerate(zip(rmse_values_kg, rmse_std_values_kg)):
            ax.text(i, rmse + std + 0.05, f'{rmse:.2f}', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add average line
        ax.axhline(y=avg_rmse, color='red', linestyle='--', linewidth=2.5, 
                  label=f'Average RMSE: {avg_rmse:.2f} kg', zorder=5)
        
        # Formatting
        ax.set_xlabel('Subject', fontsize=14, fontweight='bold')
        ax.set_ylabel('RMSE (kg)', fontsize=14, fontweight='bold')
        ax.set_title(f'RMSE per Subject with Standard Deviation (kg) [For Reference]\n(Segment: {segment_length}, Overlap: {overlap}, Train Epochs: {num_epochs}, Fine-tune: {fine_tune_epochs})', 
                    fontsize=16, fontweight='bold', pad=15)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(subjects_rmse_kg, rotation=45, ha='right', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.legend(loc='upper right', fontsize=12, framealpha=0.95)
        
        # Add statistics box
        textstr_rmse_bar_kg = f'Statistics:\n'
        textstr_rmse_bar_kg += f'Mean RMSE: {avg_rmse:.4f} kg\n'
        textstr_rmse_bar_kg += f'Std RMSE: {std_rmse:.4f} kg\n'
        textstr_rmse_bar_kg += f'Min RMSE: {min(rmse_values_kg):.4f} kg\n'
        textstr_rmse_bar_kg += f'Max RMSE: {max(rmse_values_kg):.4f} kg\n'
        textstr_rmse_bar_kg += f'Subjects: {len(subjects_rmse_kg)}'
        
        props_rmse_bar_kg = dict(boxstyle='round', facecolor='lightsteelblue', alpha=0.85)
        ax.text(0.02, 0.98, textstr_rmse_bar_kg, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=props_rmse_bar_kg, family='monospace')
        
        plt.tight_layout()
        rmse_bar_kg_filename = os.path.join(OUTPUT_DIR, f"rmse_per_subject_with_errorbar_kg{config_suffix}.png")
        plt.savefig(rmse_bar_kg_filename, dpi=300, bbox_inches='tight')
        print(f"Saved RMSE bar plot (kg) to: {rmse_bar_kg_filename}")
        plt.close()
    
    print(f"\n{'='*100}")
    print(f"All done! Processed {len(all_results)} files successfully.")
    print(f"Generated files:")
    print(f"  - {summary_filename}")
    print(f"  - {cdf_mvic_filename}")
    print(f"  - {cdf_mvic_simple_filename}")
    print(f"  - {mae_bar_filename}")
    print(f"  - {rmse_bar_filename}")
    if cdf_filename_exists:
        print(f"  - {cdf_filename}")
        print(f"  - {cdf_simple_filename}")
        print(f"  - {mae_bar_kg_filename}")
        print(f"  - {rmse_bar_kg_filename}")
    print(f"{'='*100}")


