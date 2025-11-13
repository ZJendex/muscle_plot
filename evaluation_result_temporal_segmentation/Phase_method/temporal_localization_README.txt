================================================================================
TEMPORAL LOCALIZATION DATA - README
================================================================================

FILE: temporal_localization_data.pkl

This pickle file contains comprehensive data from the temporal localization analysis
of muscle fatigue detection using radar signals and camera groundtruth.

================================================================================
DATA STRUCTURE
================================================================================

1. temporal_locations (list of dicts)
   Description: Detected peak and foot locations for each subject
   Structure:
     - subject_date: str, subject identifier (e.g., '1029_bert')
     - folder_index: int, index of the folder used
     - foot_locations: list of float, time points where foot of peak detected (seconds)
     - peak_indices: array, indices of detected peaks in the feature array

2. force_reps_datasets (list of dicts)
   Description: Force repetition data segments for analysis
   Structure:
     - subject_date: str, subject identifier
     - folder_index: int, folder index
     - data: array, radar data window
     - time: array, time stamps for data window
     - peak_idx: float, time of peak in seconds
     - left_idx, right_idx: int, indices for data window
     - full_data: array, complete radar data
     - full_time: array, complete time series
     - radar_data_envelope: array, envelope of radar data

3. cameraFused_reps_datasets (list of dicts)
   Description: Camera-based groundtruth repetition time windows
   Structure:
     - subject_date: str, subject identifier
     - starting_time: float, start time of the repetition (seconds)
     - fatigue_yes_label: int, 0=non-fatigue, 1=fatigue
     - data: array, radar data for this time window
     - time: array, time stamps for this window

4. selected_radar_features (list of dicts)
   Description: Extracted radar features for each subject
   Structure:
     - subject_date: str, subject identifier
     - folder_index: int, folder index
     - det_radar: array, Determinism feature from recurrence quantification
     - lam_radar: array, Laminarity feature
     - tt_radar: array, Trapping Time feature
     - rms_radar: array, Root Mean Square feature
     - MPF_radar: array, Mean Power Frequency feature
     - feature_t: array, time stamps for features

5. camera_angle_groundtruth (list of dicts)
   Description: Camera-based elbow angle groundtruth data
   Structure:
     - subject_date: str, subject identifier
     - folder_index: int, folder index
     - phases: list of tuples (phase_type, start_time, end_time)
         phase_type: str, 'up' or 'down'
         start_time, end_time: float, in seconds
     - df: pandas.DataFrame with columns ['timestamp', 'angle_deg', ...]

6. evaluation_results (dict)
   Description: Statistical evaluation of temporal localization accuracy
   Structure:
     - fatigue: dict with keys:
         count: total number of fatigue samples
         hits: number of successful detections
         hit_rate_pct: percentage of successful detections
         abs_time_diff: list of absolute time differences (seconds)
         time_diff: list of signed time differences (seconds)
         mean_abs_time_diff, std_abs_time_diff, median_abs_time_diff: statistics
     - non_fatigue: same structure as fatigue
     - combined: same structure, combining fatigue and non-fatigue

7. parameters (dict)
   Description: Processing parameters used in the analysis
   Structure:
     - time_length: float, length of time window in seconds
     - dir_name: str, output directory name
     - output_name: str, base name for output files
     - date_persons: list of str, subject identifiers processed
     - error_paths: list of str, paths that encountered errors

================================================================================
USAGE EXAMPLE
================================================================================

import pickle
import numpy as np

# Load the data
with open('temporal_localization_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Access evaluation results
print(f"Overall hit rate: {data['evaluation_results']['combined']['hit_rate_pct']:.2f}%")

# Access temporal locations for a specific subject
for temp_loc in data['temporal_locations']:
    if temp_loc['subject_date'] == '1029_bert':
        print(f"Foot locations: {temp_loc['foot_locations']}")

# Access radar features
for features in data['selected_radar_features']:
    det = features['det_radar']
    t = features['feature_t']
    # Plot or analyze features

================================================================================
METADATA
================================================================================

Generated on: 2025-11-13 02:34:08
Number of subjects processed: 15
Number of subjects with errors: 0
Total fatigue samples: 59
Total non-fatigue samples: 97
Total samples: 156
Time window length: 4 seconds

================================================================================
NOTES
================================================================================

- All time values are in seconds
- Foot locations with value -1 indicate failed detection
- Hit rate is calculated as: (number of hits / total count) * 100
- Time difference is calculated as: abs(camera_starting_time - foot_location)
- A 'hit' means a foot location was detected within the time window
- Features are detrended and normalized before analysis

================================================================================
END OF README
================================================================================
