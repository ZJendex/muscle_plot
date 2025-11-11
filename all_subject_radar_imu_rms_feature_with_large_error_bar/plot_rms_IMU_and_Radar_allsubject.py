import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import sys
from scipy.signal import detrend, butter, filtfilt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import analyzeMuscleData, extract_motion_peaks

def get_phase_displacement(complex_signal):
    """
    Convert unwrapped phase to radial displacement:
        disp = phase * c / (4*pi*Fc)
    """
    c = 3e8
    phase = np.unwrap(np.angle(complex_signal))
    phase = detrend(phase)
    disp = phase * c / (4 * np.pi * 77e9)
    return disp


def bandpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='bandpass', analog=False)
    return filtfilt(b, a, data)

def second_derivative(x, h):
    """
    Seven-point stencil approximation of the second derivative for better stability:
        y[i] = (4*x[i] + (x[i+1]+x[i-1]) - 2*(x[i+2]+x[i-2]) - (x[i+3]+x[i-3])) / (16*h*h)
    """
    y = np.zeros_like(x)
    for i in range(3, len(x) - 3):
        y[i] = (4*x[i] + (x[i+1]+x[i-1]) - 2*(x[i+2]+x[i-2]) - (x[i+3]+x[i-3])) / (16*h*h)
    return y

def process_single_subject(pkl_file, t_window=[14, 17]):
    """
    Process a single subject's force dataset and extract RMS features.
    
    Returns:
        rms_radar: list of RMS values for radar
        rms_imu: list of RMS values for IMU
        subject_name: name of the subject
    """
    print(f"Processing: {os.path.basename(pkl_file)}")
    
    try:
        with open(pkl_file, "rb") as f:
            dataset_imu = pickle.load(f)    
            dataset_radar = pickle.load(f)  
            dataset_loadcell = pickle.load(f)
            file_names = pickle.load(f) 
    except Exception as e:
        print(f"  Error loading {pkl_file}: {e}")
        return None, None, None

    # Extract subject name from filename
    subject_name = os.path.basename(pkl_file).split('_force_')[0]

    # pick the Z, Y, X axis for IMU
    dataset_imu_z = [dataset_imu[i] for i in range(0, len(dataset_imu), 3)]
    dataset_imu_y = [dataset_imu[i] for i in range(1, len(dataset_imu), 3)]
    dataset_imu_x = [dataset_imu[i] for i in range(2, len(dataset_imu), 3)]
    dataset_radar_middle = [dataset_radar[i] for i in range(4, len(dataset_radar), 9)]

    # Feature collection
    feas_imu = []
    feas_radar = []

    # -- data preparation --
    for i, (fn, (t_rd, v_rd), (t_imu, v_imu), (t_imu_y, v_imu_y), (t_imu_x, v_imu_x), (t_lc, v_lc)) in enumerate(zip(file_names, dataset_radar_middle, dataset_imu_z, dataset_imu_y, dataset_imu_x, dataset_loadcell)):
        try:
            fs_imu = len(t_imu)/(t_imu[-1] - t_imu[0])
            fs_radar = len(t_rd)/(t_rd[-1] - t_rd[0])
            
            # crop the radar and imu datasets at same time range
            mask = (t_imu >= t_window[0]) & (t_imu <= t_window[1])
            t_imu = t_imu[mask]
            v_imu = v_imu[mask]
            
            mask = (t_rd >= t_window[0]) & (t_rd <= t_window[1])
            t_rd = t_rd[mask]
            v_rd = v_rd[mask]
            v_rd = get_phase_displacement(v_rd)
            v_rd = second_derivative(v_rd, 1.0 / 1000)
            
            # --- detrend before analysis ---
            v_imu_detrended = detrend(v_imu)
            v_rd_detrended = detrend(v_rd)
            
            # --- filter before analysis ---
            v_imu_filtered = bandpass_filter(v_imu_detrended, [5, 100], fs_imu)
            v_rd_filtered = bandpass_filter(v_rd_detrended, [5, 100], fs_radar)
            
            # --- extract motion peaks from radar---
            t_envelope, f_envelope, p_db_envelope, outliers, energy_envelope_norm = extract_motion_peaks(v_rd_filtered, fs_radar, 128)
            
            # --- extract features ---
            fea_imu = analyzeMuscleData(v_imu_filtered, t_imu, [], fs_imu, if_chaos=False)
            fea_radar = analyzeMuscleData(v_rd_filtered, t_rd, outliers, fs_radar, if_chaos=False)
            
            # --- save data ---
            feas_imu.append(fea_imu)
            feas_radar.append(fea_radar)
        except Exception as e:
            print(f"  Error processing data point {i} in {subject_name}: {e}")
            continue

    # Extract RMS values
    rms_radar = [feat["rms"] for feat in feas_radar]
    rms_imu = [feat["rms"] for feat in feas_imu]
    
    print(f"  Extracted {len(rms_radar)} radar RMS values and {len(rms_imu)} IMU RMS values")
    
    return rms_radar, rms_imu, subject_name


def group_data_with_error(data, group_size=3):
    """Group data into chunks and calculate mean and std for error bars"""
    n_groups = len(data) // group_size
    # Trim data to be divisible by group_size
    data_trimmed = np.array(data[:n_groups * group_size])
    # Reshape into groups
    data_grouped = data_trimmed.reshape(n_groups, group_size)
    # Calculate mean and std for each group
    means = np.mean(data_grouped, axis=1)
    stds = np.std(data_grouped, axis=1)
    return means, stds, n_groups


def main():
    # Configuration
    force_dataset_path = "datasets/FORCE_datasets"
    output_folder = "datasets/PLOTTING/results_all_subjects"
    t_window = [14, 17]  # Time window for cropping
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all pickle files
    pkl_files = glob.glob(os.path.join(force_dataset_path, "*_force_*.pkl"))
    pkl_files = [f for f in pkl_files if not f.endswith('.zip')]
    
    print(f"Found {len(pkl_files)} force dataset files")
    
    # Storage for all subjects
    all_subjects_data = {}
    
    # Process each subject
    for pkl_file in pkl_files:
        rms_radar, rms_imu, subject_name = process_single_subject(pkl_file, t_window)
        
        if rms_radar is not None and rms_imu is not None:
            all_subjects_data[subject_name] = {
                'rms_radar': rms_radar,
                'rms_imu': rms_imu
            }
    
    print(f"\nSuccessfully processed {len(all_subjects_data)} subjects")
    
    # Save processed data
    with open(os.path.join(output_folder, "all_subjects_rms_data.pkl"), "wb") as f:
        pickle.dump(all_subjects_data, f)
    print(f"Saved processed data to {output_folder}/all_subjects_rms_data.pkl")
    
    # ==========================
    # Normalize RMS by max of each subject
    # ==========================
    all_subjects_data_normalized = {}
    
    for subject_name, data in all_subjects_data.items():
        rms_radar = np.array(data['rms_radar'])
        rms_imu = np.array(data['rms_imu'])
        
        # Find max for each sensor for this subject
        max_radar = np.max(rms_radar) if len(rms_radar) > 0 and np.max(rms_radar) > 0 else 1.0
        max_imu = np.max(rms_imu) if len(rms_imu) > 0 and np.max(rms_imu) > 0 else 1.0
        
        # Normalize by max
        rms_radar_normalized = rms_radar / max_radar
        rms_imu_normalized = rms_imu / max_imu
        
        all_subjects_data_normalized[subject_name] = {
            'rms_radar': rms_radar_normalized.tolist(),
            'rms_imu': rms_imu_normalized.tolist(),
            'max_radar': max_radar,
            'max_imu': max_imu
        }
        
        print(f"{subject_name}: Radar max={max_radar:.4f}, IMU max={max_imu:.4f}")
    
    # Save normalized data
    with open(os.path.join(output_folder, "all_subjects_rms_data_normalized.pkl"), "wb") as f:
        pickle.dump(all_subjects_data_normalized, f)
    print(f"\nSaved normalized data to {output_folder}/all_subjects_rms_data_normalized.pkl")
    
    # ==========================
    # Plot 1: Individual subjects with error bars (NORMALIZED)
    # ==========================
    
    # Prepare data for plotting
    subject_names = list(all_subjects_data_normalized.keys())
    
    # Radar RMS with error bars
    fig_radar, ax_radar = plt.subplots(figsize=(16, 8))
    
    for idx, subject_name in enumerate(subject_names):
        rms_radar = all_subjects_data_normalized[subject_name]['rms_radar']
        
        # Group every 3 data points
        rms_mean, rms_std, n_groups = group_data_with_error(rms_radar, 3)
        
        # Plot with offset for visibility
        x_positions = np.arange(n_groups) + idx * 0.05
        ax_radar.errorbar(x_positions, rms_mean, yerr=rms_std, 
                         fmt='-o', capsize=3, capthick=1, linewidth=1.5, 
                         markersize=4, alpha=0.7, label=subject_name)
    
    # Set x-axis to show MVIC percentages
    mvic_levels = [10, 20, 30, 40, 50, 60, 70, 80][:n_groups]
    ax_radar.set_xticks(np.arange(n_groups))
    ax_radar.set_xticklabels(mvic_levels)
    ax_radar.set_xlabel("%MVIC", fontsize=12)
    ax_radar.set_ylabel("Normalized RMS", fontsize=12)
    ax_radar.set_title("Radar RMS across All Subjects (Normalized)", fontsize=14, fontweight='bold')
    ax_radar.grid(True, alpha=0.3)
    ax_radar.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(f"{output_folder}/radar_rms_all_subjects_normalized.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_folder}/radar_rms_all_subjects_normalized.png")
    
    # IMU RMS with error bars
    fig_imu, ax_imu = plt.subplots(figsize=(16, 8))
    
    for idx, subject_name in enumerate(subject_names):
        rms_imu = all_subjects_data_normalized[subject_name]['rms_imu']
        
        # Group every 3 data points
        rms_mean, rms_std, n_groups = group_data_with_error(rms_imu, 3)
        
        # Plot with offset for visibility
        x_positions = np.arange(n_groups) + idx * 0.05
        ax_imu.errorbar(x_positions, rms_mean, yerr=rms_std, 
                       fmt='-o', capsize=3, capthick=1, linewidth=1.5, 
                       markersize=4, alpha=0.7, label=subject_name)
    
    # Set x-axis to show MVIC percentages
    mvic_levels = [10, 20, 30, 40, 50, 60, 70, 80][:n_groups]
    ax_imu.set_xticks(np.arange(n_groups))
    ax_imu.set_xticklabels(mvic_levels)
    ax_imu.set_xlabel("%MVIC", fontsize=12)
    ax_imu.set_ylabel("Normalized RMS", fontsize=12)
    ax_imu.set_title("IMU RMS across All Subjects (Normalized)", fontsize=14, fontweight='bold')
    ax_imu.grid(True, alpha=0.3)
    ax_imu.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(f"{output_folder}/imu_rms_all_subjects_normalized.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_folder}/imu_rms_all_subjects_normalized.png")
    
    # ==========================
    # Plot 2: Aggregated across all subjects (NORMALIZED)
    # ==========================
    
    # Aggregate all RMS values at each MVIC level
    max_groups = max([len(data['rms_radar']) // 3 for data in all_subjects_data_normalized.values()])
    
    radar_rms_aggregated = [[] for _ in range(max_groups)]
    imu_rms_aggregated = [[] for _ in range(max_groups)]
    
    for subject_name, data in all_subjects_data_normalized.items():
        rms_radar = data['rms_radar']
        rms_imu = data['rms_imu']
        
        # Group every 3 data points
        rms_radar_mean, _, n_groups_radar = group_data_with_error(rms_radar, 3)
        rms_imu_mean, _, n_groups_imu = group_data_with_error(rms_imu, 3)
        
        # Add to aggregated lists
        for i, val in enumerate(rms_radar_mean):
            if i < max_groups:
                radar_rms_aggregated[i].append(val)
        
        for i, val in enumerate(rms_imu_mean):
            if i < max_groups:
                imu_rms_aggregated[i].append(val)
    
    # Calculate mean and std across all subjects for each MVIC level
    radar_means = [np.mean(vals) if len(vals) > 0 else 0 for vals in radar_rms_aggregated]
    radar_stds = [np.std(vals) if len(vals) > 0 else 0 for vals in radar_rms_aggregated]
    
    imu_means = [np.mean(vals) if len(vals) > 0 else 0 for vals in imu_rms_aggregated]
    imu_stds = [np.std(vals) if len(vals) > 0 else 0 for vals in imu_rms_aggregated]
    
    # Plot aggregated data
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # Radar aggregated
    x_positions = np.arange(len(radar_means))
    mvic_levels = [10, 20, 30, 40, 50, 60, 70, 80][:len(radar_means)]
    axes[0].errorbar(x_positions, radar_means, yerr=radar_stds,
                     fmt='-o', capsize=5, capthick=2, linewidth=2.5, 
                     markersize=8, color='blue', ecolor='lightblue')
    axes[0].set_ylabel("Normalized RMS", fontsize=14)
    axes[0].set_xlabel("%MVIC", fontsize=12)
    axes[0].set_title(f"Radar RMS value )", 
                     fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels(mvic_levels)
    
    # IMU aggregated
    x_positions = np.arange(len(imu_means))
    mvic_levels = [10, 20, 30, 40, 50, 60, 70, 80][:len(imu_means)]
    axes[1].errorbar(x_positions, imu_means, yerr=imu_stds,
                     fmt='-o', capsize=5, capthick=2, linewidth=2.5, 
                     markersize=8, color='green', ecolor='lightgreen')
    axes[1].set_ylabel("Normalized RMS", fontsize=14)
    axes[1].set_xlabel("%MVIC", fontsize=12)
    axes[1].set_title(f"IMU RMS", 
                     fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(mvic_levels)
    
    plt.tight_layout()
    plt.savefig(f"{output_folder}/rms_aggregated_all_subjects_normalized.png", dpi=300)
    print(f"Saved: {output_folder}/rms_aggregated_all_subjects_normalized.png")
    
    # ==========================
    # Plot 3: Combined Radar and IMU on same plot (NORMALIZED)
    # ==========================
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x_positions = np.arange(len(radar_means))
    mvic_levels = [10, 20, 30, 40, 50, 60, 70, 80][:len(radar_means)]
    
    ax.errorbar(x_positions - 0.1, radar_means, yerr=radar_stds,
                fmt='-o', capsize=5, capthick=2, linewidth=2.5, 
                markersize=8, color='blue', ecolor='lightblue', label='Radar')
    
    ax.errorbar(x_positions + 0.1, imu_means, yerr=imu_stds,
                fmt='-s', capsize=5, capthick=2, linewidth=2.5, 
                markersize=8, color='green', ecolor='lightgreen', label='IMU')
    
    ax.set_ylabel("Normalized RMS", fontsize=14)
    ax.set_xlabel("%MVIC", fontsize=12)
    ax.set_title(f"Radar vs IMU RMS Comparison (Normalized, n={len(all_subjects_data_normalized)} subjects)", 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(mvic_levels)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_folder}/rms_comparison_radar_vs_imu_normalized.png", dpi=300)
    print(f"Saved: {output_folder}/rms_comparison_radar_vs_imu_normalized.png")
    
    print("\n=== Processing Complete ===")
    print(f"Total subjects processed: {len(all_subjects_data)}")
    print(f"Output folder: {output_folder}")


if __name__ == "__main__":
    main()

