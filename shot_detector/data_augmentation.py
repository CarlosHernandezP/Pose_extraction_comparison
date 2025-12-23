"""
Data augmentation for pose sequences to increase dataset size.

Applies various augmentation techniques to 30-frame pose sequences.
"""

import numpy as np
from typing import List, Dict, Optional

try:
    from scipy.interpolate import interp1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available. Temporal warping will use numpy interpolation.")


def mirror_sequence(sequence: np.ndarray) -> np.ndarray:
    """
    Horizontally mirrors a pose sequence by flipping left/right keypoints.
    
    For body-relative coordinates, this means:
    - Swapping left/right keypoint pairs
    - Negating x-coordinates (body-relative)
    
    Parameters
    ----------
    sequence : np.ndarray
        Array of shape (30, 27) - 30 frames, 27 features
        Features order: 12 keypoints × 2 (x, y) = 24, then 3 absolute features
        left_shoulder(0,1), right_shoulder(2,3), left_elbow(4,5), right_elbow(6,7),
        left_wrist(8,9), right_wrist(10,11), left_hip(12,13), right_hip(14,15),
        left_knee(16,17), right_knee(18,19), left_ankle(20,21), right_ankle(22,23),
        hip_y_abs(24), hip_x_abs(25), shoulder_center_y_abs(26)
        
    Returns
    -------
    np.ndarray
        Mirrored sequence of same shape
    """
    mirrored = sequence.copy()
    
    # Swap left/right keypoint pairs (each pair is 2 consecutive indices)
    # left_shoulder (0,1) <-> right_shoulder (2,3)
    # left_elbow (4,5) <-> right_elbow (6,7)
    # left_wrist (8,9) <-> right_wrist (10,11)
    # left_hip (12,13) <-> right_hip (14,15)
    # left_knee (16,17) <-> right_knee (18,19)
    # left_ankle (20,21) <-> right_ankle (22,23)
    
    swap_groups = [
        (0, 2),   # left_shoulder <-> right_shoulder (x coordinates)
        (1, 3),   # left_shoulder <-> right_shoulder (y coordinates)
        (4, 6),   # left_elbow <-> right_elbow (x)
        (5, 7),   # left_elbow <-> right_elbow (y)
        (8, 10),  # left_wrist <-> right_wrist (x)
        (9, 11),  # left_wrist <-> right_wrist (y)
        (12, 14), # left_hip <-> right_hip (x)
        (13, 15), # left_hip <-> right_hip (y)
        (16, 18), # left_knee <-> right_knee (x)
        (17, 19), # left_knee <-> right_knee (y)
        (20, 22), # left_ankle <-> right_ankle (x)
        (21, 23), # left_ankle <-> right_ankle (y)
    ]
    
    # Swap left/right keypoints
    for idx1, idx2 in swap_groups:
        temp = mirrored[:, idx1].copy()
        mirrored[:, idx1] = mirrored[:, idx2]
        mirrored[:, idx2] = temp
    
    # Negate all x-coordinates (body-relative features)
    # x coordinates are at even indices: 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22
    x_indices = list(range(0, 24, 2))  # All x coordinates of body-relative features
    for idx in x_indices:
        mirrored[:, idx] = -mirrored[:, idx]
    
    # For absolute features: negate hip_x_abs (index 25)
    # hip_y_abs (24) and shoulder_center_y_abs (26) stay the same
    mirrored[:, 25] = -mirrored[:, 25]
    
    return mirrored


def temporal_warp(sequence: np.ndarray, speed_factor: float) -> np.ndarray:
    """
    Temporally warps a sequence by speeding up or slowing down.
    
    Parameters
    ----------
    sequence : np.ndarray
        Array of shape (30, 27)
    speed_factor : float
        Speed factor: >1.0 speeds up, <1.0 slows down
        Typical values: 0.9, 1.0, 1.1
        
    Returns
    -------
    np.ndarray
        Warped sequence of shape (30, 27)
    """
    T, D = sequence.shape
    original_indices = np.arange(T)
    
    # Calculate new temporal positions
    new_length = int(T / speed_factor)
    new_indices = np.linspace(0, T - 1, new_length)
    
    # Interpolate each feature dimension
    warped = np.zeros((T, D))
    for d in range(D):
        if HAS_SCIPY:
            interp_func = interp1d(original_indices, sequence[:, d], 
                                  kind='linear', fill_value='extrapolate')
            warped_values = interp_func(new_indices)
        else:
            # Fallback to numpy interpolation
            warped_values = np.interp(new_indices, original_indices, sequence[:, d])
        
        # Resample back to T frames
        if len(warped_values) >= T:
            # Take evenly spaced samples
            sample_indices = np.linspace(0, len(warped_values) - 1, T).astype(int)
            warped[:, d] = warped_values[sample_indices]
        else:
            # Pad with last value
            warped[:len(warped_values), d] = warped_values
            warped[len(warped_values):, d] = warped_values[-1]
    
    return warped


def add_gaussian_noise(sequence: np.ndarray, noise_std: float = 0.015) -> np.ndarray:
    """
    Adds Gaussian noise to keypoint coordinates.
    
    Parameters
    ----------
    sequence : np.ndarray
        Array of shape (30, 27)
    noise_std : float
        Standard deviation of noise (default: 0.015 = 1.5% of typical values)
        
    Returns
    -------
    np.ndarray
        Noisy sequence of same shape
    """
    noisy = sequence.copy()
    noise = np.random.normal(0, noise_std, sequence.shape)
    noisy = noisy + noise
    return noisy


def frame_dropout(sequence: np.ndarray, dropout_rate: float = 0.1) -> np.ndarray:
    """
    Randomly drops frames and interpolates to maintain sequence length.
    
    Parameters
    ----------
    sequence : np.ndarray
        Array of shape (30, 27)
    dropout_rate : float
        Fraction of frames to drop (default: 0.1 = 10%)
        
    Returns
    -------
    np.ndarray
        Sequence with dropped frames interpolated
    """
    T, D = sequence.shape
    num_drop = int(T * dropout_rate)
    
    if num_drop == 0:
        return sequence.copy()
    
    # Randomly select frames to drop
    drop_indices = np.random.choice(T, size=num_drop, replace=False)
    keep_indices = np.setdiff1d(np.arange(T), drop_indices)
    
    # Interpolate dropped frames
    result = sequence.copy()
    for d in range(D):
        if len(keep_indices) > 1:
            interp_func = interp1d(keep_indices, sequence[keep_indices, d],
                                  kind='linear', fill_value='extrapolate')
            result[:, d] = interp_func(np.arange(T))
        else:
            # If too many dropped, forward fill
            result[:, d] = sequence[keep_indices[0], d]
    
    return result


def reverse_sequence(sequence: np.ndarray) -> np.ndarray:
    """
    Reverses the temporal order of a sequence.
    
    Note: May not be semantically valid for shot detection, use with caution.
    
    Parameters
    ----------
    sequence : np.ndarray
        Array of shape (30, 27)
        
    Returns
    -------
    np.ndarray
        Reversed sequence
    """
    return sequence[::-1, :].copy()


def augment_sequence(sequence: np.ndarray, config: Optional[Dict] = None) -> List[np.ndarray]:
    """
    Applies data augmentation to a sequence based on configuration.
    
    Parameters
    ----------
    sequence : np.ndarray
        Array of shape (30, 27) - original sequence
    config : dict, optional
        Configuration dictionary specifying which augmentations to apply:
        {
            'mirror': bool,           # Horizontal mirroring
            'temporal_warp': bool,    # Temporal warping (fast/slow)
            'gaussian_noise': bool,   # Add Gaussian noise
            'frame_dropout': bool,    # Random frame dropout
            'reverse': bool,          # Reverse sequence (use with caution)
            'noise_std': float,       # Noise std for gaussian_noise (default: 0.015)
            'dropout_rate': float,    # Dropout rate for frame_dropout (default: 0.1)
            'warp_factors': list,     # Speed factors for temporal_warp (default: [0.9, 1.1])
        }
        If None, uses default config with mirror and temporal_warp enabled.
        
    Returns
    -------
    list
        List of augmented sequences (including original as first element)
    """
    if config is None:
        config = {
            'mirror': True,
            'temporal_warp': True,
            'gaussian_noise': True,
            'frame_dropout': False,
            'reverse': False
        }
    
    augmented_sequences = [sequence.copy()]  # Always include original
    
    # Horizontal mirroring
    if config.get('mirror', False):
        augmented_sequences.append(mirror_sequence(sequence))
    
    # Temporal warping
    if config.get('temporal_warp', False):
        warp_factors = config.get('warp_factors', [0.9, 1.1])
        for factor in warp_factors:
            augmented_sequences.append(temporal_warp(sequence, factor))
    
    # Gaussian noise
    if config.get('gaussian_noise', False):
        noise_std = config.get('noise_std', 0.015)
        augmented_sequences.append(add_gaussian_noise(sequence, noise_std))
    
    # Frame dropout
    if config.get('frame_dropout', False):
        dropout_rate = config.get('dropout_rate', 0.1)
        augmented_sequences.append(frame_dropout(sequence, dropout_rate))
    
    # Sequence reversal (use with caution)
    if config.get('reverse', False):
        augmented_sequences.append(reverse_sequence(sequence))
    
    return augmented_sequences


def augment_dataset(sequences: np.ndarray, labels: np.ndarray, 
                   config: Optional[Dict] = None) -> tuple:
    """
    Augments an entire dataset of sequences.
    
    Parameters
    ----------
    sequences : np.ndarray
        Array of shape (N, 30, 27) - N sequences
    labels : np.ndarray
        Array of shape (N,) - labels for each sequence
    config : dict, optional
        Augmentation configuration (see augment_sequence())
        
    Returns
    -------
    tuple
        (augmented_sequences, augmented_labels) where:
        - augmented_sequences: (M, 30, 27) where M >= N
        - augmented_labels: (M,) corresponding labels
    """
    augmented_sequences_list = []
    augmented_labels_list = []
    
    for i in range(len(sequences)):
        seq = sequences[i]
        label = labels[i]
        
        # Get augmented versions
        aug_seqs = augment_sequence(seq, config)
        
        # Add all augmented sequences with same label
        for aug_seq in aug_seqs:
            augmented_sequences_list.append(aug_seq)
            augmented_labels_list.append(label)
    
    return np.array(augmented_sequences_list), np.array(augmented_labels_list)
