"""
Temporal feature extraction for pose sequences.

Replicates extract_temporal_features() from LookAtMeProtoApp/src/shot_detection/shot_predictor.py
"""

import numpy as np
from typing import Optional, Any, List, Tuple


def get_feature_names(num_features: int = 27) -> List[str]:
    """
    Generate descriptive names for all temporal features.
    
    Parameters
    ----------
    num_features : int
        Number of input features per frame (default: 27)
        
    Returns
    -------
    list
        List of feature names in the same order as extract_temporal_features() output
    """
    feature_names = []
    
    # Base feature names (27 features: 24 body-relative + 3 absolute)
    base_feature_names = [
        'left_shoulder_x_body_rel', 'left_shoulder_y_body_rel',
        'right_shoulder_x_body_rel', 'right_shoulder_y_body_rel',
        'left_elbow_x_body_rel', 'left_elbow_y_body_rel',
        'right_elbow_x_body_rel', 'right_elbow_y_body_rel',
        'left_wrist_x_body_rel', 'left_wrist_y_body_rel',
        'right_wrist_x_body_rel', 'right_wrist_y_body_rel',
        'left_hip_x_body_rel', 'left_hip_y_body_rel',
        'right_hip_x_body_rel', 'right_hip_y_body_rel',
        'left_knee_x_body_rel', 'left_knee_y_body_rel',
        'right_knee_x_body_rel', 'right_knee_y_body_rel',
        'left_ankle_x_body_rel', 'left_ankle_y_body_rel',
        'right_ankle_x_body_rel', 'right_ankle_y_body_rel',
        'hip_y_abs', 'hip_x_abs', 'shoulder_center_y_abs'
    ]
    
    # Ensure we have the right number
    if len(base_feature_names) != num_features:
        # Fallback: use generic names
        base_feature_names = [f'feature_{i}' for i in range(num_features)]
    
    # 1. Statistical features per keypoint (5 stats × num_features)
    for feat_name in base_feature_names:
        feature_names.append(f'{feat_name}_mean')
        feature_names.append(f'{feat_name}_std')
        feature_names.append(f'{feat_name}_min')
        feature_names.append(f'{feat_name}_max')
        feature_names.append(f'{feat_name}_median')
    
    # 2. Velocity features (3 stats × num_features)
    for feat_name in base_feature_names:
        feature_names.append(f'{feat_name}_velocity_mean')
        feature_names.append(f'{feat_name}_velocity_std')
        feature_names.append(f'{feat_name}_velocity_max_abs')
    
    # 3. Acceleration features (3 stats × num_features)
    for feat_name in base_feature_names:
        feature_names.append(f'{feat_name}_acceleration_mean')
        feature_names.append(f'{feat_name}_acceleration_std')
        feature_names.append(f'{feat_name}_acceleration_max_abs')
    
    # 4. Temporal correlation
    feature_names.append('temporal_correlation_first_second_half')
    
    # 5. Range of motion
    feature_names.append('range_of_motion_total_distance')
    
    # 6. Start positions (num_features)
    for feat_name in base_feature_names:
        feature_names.append(f'{feat_name}_start')
    
    # 7. End positions (num_features)
    for feat_name in base_feature_names:
        feature_names.append(f'{feat_name}_end')
    
    return feature_names


def extract_temporal_features(sequences, profiler: Optional[Any] = None):
    """
    Extract temporal features from sequences (same as analyze_misclassifications.py).
    
    Parameters
    ----------
    sequences : np.ndarray
        Array of shape (N, T, D) where N=num_sequences, T=30, D=num_features
    profiler : Profiler, optional
        Profiler instance (for compatibility, not used here)
        
    Returns
    -------
    np.ndarray
        Feature array of shape (N, num_extracted_features)
    """
    N, T, D = sequences.shape
    features = []
    
    for seq in sequences:
        seq_features = []
        
        # 1. Statistical features per keypoint
        for d in range(D):
            keypoint_values = seq[:, d]
            seq_features.extend([
                np.mean(keypoint_values),
                np.std(keypoint_values),
                np.min(keypoint_values),
                np.max(keypoint_values),
                np.median(keypoint_values),
            ])
        
        # 2. Velocity features
        velocity = np.diff(seq, axis=0)
        for d in range(D):
            vel_values = velocity[:, d]
            seq_features.extend([
                np.mean(vel_values),
                np.std(vel_values),
                np.max(np.abs(vel_values)),
            ])
        
        # 3. Acceleration features
        if T > 2:
            acceleration = np.diff(velocity, axis=0)
            for d in range(D):
                accel_values = acceleration[:, d]
                seq_features.extend([
                    np.mean(accel_values),
                    np.std(accel_values),
                    np.max(np.abs(accel_values)),
                ])
        else:
            seq_features.extend([0.0] * (D * 3))
        
        # 4. Temporal correlation
        mid_point = T // 2
        first_half = seq[:mid_point].flatten()
        second_half = seq[mid_point:].flatten()
        if len(first_half) == len(second_half):
            correlation = np.corrcoef(first_half, second_half)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            seq_features.append(correlation)
        else:
            seq_features.append(0.0)
        
        # 5. Range of motion
        total_distance = np.sum(np.sqrt(np.sum(np.diff(seq, axis=0)**2, axis=1)))
        seq_features.append(total_distance)
        
        # 6. Start and end positions
        seq_features.extend(seq[0, :].tolist())
        seq_features.extend(seq[-1, :].tolist())
        
        features.append(seq_features)
    
    return np.array(features)
