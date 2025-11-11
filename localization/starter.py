import os
import pickle
import sys
from util import cfar_2d
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, root_dir)
from read_bin_radar import RadarReader
import matplotlib.pyplot as plt
import numpy as np
from utils.chaos import compute_recurrence_plot, raq_measures, compute_ami_tau
from util import chunk_data
from utils.signal import stft_spectrogram
time_take = [4,5,6,7,8,9,10,11,12,13,14]
name_data = "1017_jerry"
folder_name = "1017_jerry_ccMVIC70_r1"
file_name = [f for f in os.listdir(f"datasets\isotonic_analysis\datasets_ia\{folder_name}") if "data_dict_radar_type_RAW_" in f and ".pkl" in f]

print("================================================")
print(f"name_data: {name_data}")
print(f"folder_name: {folder_name}")
print("================================================")
print(f"file_name: {file_name}")
print("================================================")

import re
# Extract both values in one go
matches = re.search(r'range_(\d+)_angle_(\d+)', file_name[0])
if matches:
    range_value = int(matches.group(1))
    angle_value = int(matches.group(2))
    print(f"Range: {range_value}, Angle: {angle_value}")


Radar_Data_Path = f"data\\final_data\\{name_data}\\{folder_name}"
# read the .bin file name that contains the string "Radar_capture"
radar_files = [f for f in os.listdir(Radar_Data_Path) if "Radar_capture" in f]
# get the absolute path of the frirst radar_file
radar_file_path = os.path.join(Radar_Data_Path, radar_files[0])
# read the radar_file_path
reader_rd = RadarReader(radar_file_path)
reader_rd.load_data()

# plot the radar heaetmap where teh range from 5-20
plt.figure(figsize=(10, 8))
at_time = 5 # seconds
radar_heatmap_at_time, _ = reader_rd._create_3d_slice(reader_rd.processed_frames, reader_rd.acfg.rangeResolution, reader_rd.acfg.distLimit)
radar_heatmap_at_time = radar_heatmap_at_time[int(time_take[0]*reader_rd.acfg.fs):int(time_take[-1]*reader_rd.acfg.fs),5:20,:]
print(np.shape(radar_heatmap_at_time))
radar_heatmap_at_time = np.sum(radar_heatmap_at_time, axis=0)/10000
plt.imshow(radar_heatmap_at_time, aspect="auto", origin="lower",
            extent=[0, reader_rd.acfg.AZIM_FFT, 5, 20])
plt.xlabel("Angle bins"); plt.ylabel("Range bins")
plt.title("Radar Heatmap")
plt.colorbar(label="dB")
plt.savefig(f"datasets\\isotonic_analysis\\DRAW_Results\\{folder_name}\\heatmap_{name_data}_{range_value}_{angle_value}_5-20.png")
plt.close()
print("  Saved heatmap plot")



# # for each location, plot and save the phase signal
# for (range_bin, angle_bin) in locations:
#     data = reader_rd._extract_phase_signal(range_bin, angle_bin)
#     radar_v = data["acceleration_filtered"]
#     time_axis = np.arange(len(radar_v)) / reader_rd.acfg.fs
#     # plot the radar_v, radar_v stft and ENTR
#     fig, axes = plt.subplots(4, 1, figsize=(10, 8*4))
#     axes[0].plot(time_axis, radar_v)
#     axes[0].set_title("Radar Phase 2nd Derivative")
#     # plot the stft of the radar_v
#     t, f, p_db, Sxx = stft_spectrogram(radar_v, 1000, 256)
#     axes[1].pcolormesh(t, f, p_db)
#     axes[1].set_title("Radar Phase 2nd Derivative STFT")
#     axes[1].set_ylim(0, 100)
#     # plot the ENTR
#     # chunk the radar_v into 1 second chunks
#     radar_v_chunked, _ = chunk_data(radar_v,time_axis,1000)
#     entr_values = []
#     det_values = []
#     for radar_v_chunk in radar_v_chunked:
#         m = 3
#         tau, _, _ = compute_ami_tau(radar_v_chunk, 1000)
#         rp_matrix, _ = compute_recurrence_plot(radar_v_chunk, m, tau, 10)
#         entr_value = raq_measures(rp_matrix)['ENTR']
#         det_value = raq_measures(rp_matrix)['DET']
#         entr_values.append(entr_value)
#         det_values.append(det_value)
#     entr_values = np.array(entr_values)
#     det_values = np.array(det_values)
#     axes[2].plot(entr_values)
#     axes[2].set_title("ENTR")
#     axes[2].set_ylim(1.5, 5)
#     axes[3].plot(det_values)
#     axes[3].set_title("DET")
#     axes[3].set_ylim(0.5, 1)
#     plt.savefig(f"datasets\\isotonic_analysis\\DRAW_Results\\{folder_name}\\phase_signal_{name_data}_{range_value}_{angle_value}_{range_bin}_{angle_bin}.png")
#     plt.close()
#     print("  Saved phase signal plot")


# Plot all locations in one figure with 4 columns
locations = [(11, 8), (39, 2), (16, 11), (11, 6)]

fig, axes = plt.subplots(4, 4, figsize=(24, 16))
plt.subplots_adjust(hspace=0.3, wspace=0.3)

for col_idx, (range_bin, angle_bin) in enumerate(locations):
    data = reader_rd._extract_phase_signal(range_bin, angle_bin)
    radar_v = data["acceleration_filtered"]
    time_axis = np.arange(len(radar_v)) / reader_rd.acfg.fs

    # crop the time to 5 to 25 seconds
    radar_v = radar_v[int(5*reader_rd.acfg.fs):int(25*reader_rd.acfg.fs)]
    time_axis = time_axis[int(5*reader_rd.acfg.fs):int(25*reader_rd.acfg.fs)]

    # set time start at zero
    time_axis = time_axis - time_axis[0]
    
    # Row 0: Radar Phase 2nd Derivative
    axes[0, col_idx].plot(time_axis, radar_v, linewidth=0.8)
    axes[0, col_idx].set_ylim(-100, 100)
    axes[0, col_idx].set_title(f"Bin ({range_bin}, {angle_bin})", fontsize=11, fontweight='bold')
    axes[0, col_idx].set_xlabel("Time (s)", fontsize=9)
    if col_idx == 0:
        axes[0, col_idx].set_ylabel("Acceleration (m/s²)", fontsize=10, fontweight='bold')

    # remove the box and ticks
    axes[0, col_idx].spines['top'].set_visible(False)
    axes[0, col_idx].spines['right'].set_visible(False)
    axes[0, col_idx].spines['bottom'].set_visible(False)
    axes[0, col_idx].spines['left'].set_visible(False)
    axes[0, col_idx].tick_params(top=False, right=False, bottom=False, left=False)
    # remove all the labels
    axes[0, col_idx].set_xticklabels([])
    axes[0, col_idx].set_yticklabels([])
    
    # Row 1: STFT
    t, f, p_db, Sxx = stft_spectrogram(radar_v, 1000, 256)
    im = axes[1, col_idx].pcolormesh(t, f, p_db, shading='auto')
    axes[1, col_idx].set_ylim(0, 100)
    axes[1, col_idx].set_xlabel("Time (s)", fontsize=9)
    if col_idx == 0:
        axes[1, col_idx].set_ylabel("Frequency (Hz)", fontsize=10, fontweight='bold')

    # remove the box and ticks
    axes[1, col_idx].spines['top'].set_visible(False)
    axes[1, col_idx].spines['right'].set_visible(False)
    axes[1, col_idx].spines['bottom'].set_visible(False)
    axes[1, col_idx].spines['left'].set_visible(False)
    axes[1, col_idx].tick_params(top=False, right=False, bottom=False, left=False)
    # remove all the labels
    axes[1, col_idx].set_xticklabels([])
    axes[1, col_idx].set_yticklabels([])
    # Row 2: ENTR
    radar_v_chunked, _ = chunk_data(radar_v, time_axis, 1000)
    entr_values = []
    det_values = []
    for radar_v_chunk in radar_v_chunked:
        m = 3
        tau, _, _ = compute_ami_tau(radar_v_chunk, 1000)
        rp_matrix, _ = compute_recurrence_plot(radar_v_chunk, m, tau, 10)
        entr_value = raq_measures(rp_matrix)['ENTR']
        det_value = raq_measures(rp_matrix)['DET']
        entr_values.append(entr_value)
        det_values.append(det_value)
    
    entr_values = np.array(entr_values)
    det_values = np.array(det_values)
    
    axes[2, col_idx].plot(entr_values, linewidth=3)
    axes[2, col_idx].set_ylim(1.5, 5)
    axes[2, col_idx].set_xlabel("Chunk Index", fontsize=9)
    # print the std of ENTR as title
    axes[2, col_idx].set_title(f"ENTR STD: {np.std(entr_values)}", fontsize=10, fontweight='bold')
    if col_idx == 0:
        axes[2, col_idx].set_ylabel("ENTR", fontsize=10, fontweight='bold')

    # remove the box and ticks
    axes[2, col_idx].spines['top'].set_visible(False)
    axes[2, col_idx].spines['right'].set_visible(False)
    axes[2, col_idx].spines['bottom'].set_visible(False)
    axes[2, col_idx].spines['left'].set_visible(False)
    axes[2, col_idx].tick_params(top=False, right=False, bottom=False, left=False)
    # remove all the labels
    axes[2, col_idx].set_xticklabels([])
    axes[2, col_idx].set_yticklabels([])
    # Row 3: DET
    axes[3, col_idx].plot(det_values, linewidth=1)
    axes[3, col_idx].set_ylim(0.5, 1)
    axes[3, col_idx].set_xlabel("Chunk Index", fontsize=9)
    if col_idx == 0:
        axes[3, col_idx].set_ylabel("DET", fontsize=10, fontweight='bold')

    # remove the box and ticks
    axes[3, col_idx].spines['top'].set_visible(False)
    axes[3, col_idx].spines['right'].set_visible(False)
    axes[3, col_idx].spines['bottom'].set_visible(False)
    axes[3, col_idx].spines['left'].set_visible(False)
    axes[3, col_idx].tick_params(top=False, right=False, bottom=False, left=False)

    # remove all the labels
    axes[3, col_idx].set_xticklabels([])
    axes[3, col_idx].set_yticklabels([])

# Add row titles on the left
axes[0, 0].text(-0.3, 0.5, 'Phase Signal', transform=axes[0, 0].transAxes,
                fontsize=13, fontweight='bold', va='center', rotation=90)
axes[1, 0].text(-0.3, 0.5, 'STFT', transform=axes[1, 0].transAxes,
                fontsize=13, fontweight='bold', va='center', rotation=90)
axes[2, 0].text(-0.3, 0.5, 'Entropy', transform=axes[2, 0].transAxes,
                fontsize=13, fontweight='bold', va='center', rotation=90)
axes[3, 0].text(-0.3, 0.5, 'Determinism', transform=axes[3, 0].transAxes,
                fontsize=13, fontweight='bold', va='center', rotation=90)

fig.suptitle(f'Multi-Location Analysis - {name_data} ({range_value}, {angle_value})', 
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig(f"datasets\\isotonic_analysis\\DRAW_Results\\{folder_name}\\phase_signal_all_{name_data}_{range_value}_{angle_value}.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved combined phase signal plot")

