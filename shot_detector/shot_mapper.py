"""
Flexible shot type mapping system for aggregating shot types into classes.

Supports hierarchical mapping and easy extension for future aggregations.
"""

from typing import Dict, List, Optional, Set
import os
import glob
import pandas as pd


# Default mapping for 4-class system: forehand, backhand, serve, idle
DEFAULT_SHOT_MAPPING = {
    'forehand': [
        'forehand',
        'forehand_volley',
        'forehand_wall_exit',
        'forehand_contrapared',
        'flat_smash',
        'topspin_smash',
        'vibora',
        'bandeja',
        'bajada'
    ],
    'backhand': [
        'backhand',
        'backhand_volley',
        'backhand_wall_exit'
    ],
    'serve': [
        'serve'
    ],
    'idle': [
        'idle'
    ]
}


def map_shot_to_class(shot_type: str, mapping: Optional[Dict[str, List[str]]] = None) -> Optional[str]:
    """
    Maps a single shot type to an aggregated class.
    
    Parameters
    ----------
    shot_type : str
        The shot type to map (e.g., 'forehand_volley', 'serve')
    mapping : dict, optional
        Mapping dictionary. If None, uses DEFAULT_SHOT_MAPPING.
        
    Returns
    -------
    str or None
        The aggregated class name, or None if shot_type is not found in mapping.
    """
    if mapping is None:
        mapping = DEFAULT_SHOT_MAPPING
    
    # Filter out invalid shot types
    if shot_type is None or str(shot_type).strip().lower() == 'shot':
        return None
    
    shot_type = str(shot_type).strip().lower()
    
    # Search for shot_type in mapping
    for class_name, shot_list in mapping.items():
        if shot_type in [s.lower() for s in shot_list]:
            return class_name
    
    return None


def get_all_shot_types(csv_dir: str) -> Set[str]:
    """
    Discovers all unique shot types from CSV files in a directory.
    
    Parameters
    ----------
    csv_dir : str
        Directory containing shot CSV files (from /home/daniele/shots_csvs/)
        
    Returns
    -------
    set
        Set of unique shot type strings
    """
    shot_types = set()
    
    csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'Shot' in df.columns:
                unique_shots = df['Shot'].unique()
                for shot in unique_shots:
                    if pd.notna(shot) and str(shot).strip().lower() != 'shot':
                        shot_types.add(str(shot).strip())
        except Exception as e:
            print(f"Warning: Could not read {csv_file}: {e}")
            continue
    
    return shot_types


def get_all_shot_types_from_pose_files(data_dir: str) -> Set[str]:
    """
    Discovers all unique shot types from pose CSV filenames.
    
    Parameters
    ----------
    data_dir : str
        Directory containing pose CSV files (e.g., shot_detector/data/)
        
    Returns
    -------
    set
        Set of unique shot type strings extracted from filenames
    """
    shot_types = set()
    
    pattern = os.path.join(data_dir, '*_pose.csv')
    pose_files = glob.glob(pattern)
    
    for pose_file in pose_files:
        # Extract shot type from filename: {video}_{frame}_{shot_type}_{player}_pose.csv
        basename = os.path.basename(pose_file)
        parts = basename.replace('_pose.csv', '').split('_')
        
        # Try to find shot type (usually before player position: left/right/top/bottom)
        player_positions = {'left', 'right', 'top', 'bottom'}
        shot_type_parts = []
        
        for i, part in enumerate(parts):
            if part.lower() in player_positions:
                # Shot type is everything before this part
                if i > 0:
                    # Find where the shot type starts (after frame number)
                    # Frame number is usually numeric, shot type comes after
                    shot_type_parts = parts[2:i]  # Skip video name and frame
                    break
        
        if shot_type_parts:
            shot_type = '_'.join(shot_type_parts)
            if shot_type and shot_type.lower() != 'shot':
                shot_types.add(shot_type)
    
    return shot_types


def create_mapping_from_config(config_dict: Dict) -> Dict[str, List[str]]:
    """
    Creates mapping from configuration dictionary.
    
    Parameters
    ----------
    config_dict : dict
        Configuration dictionary with 'shot_mapping' key
        
    Returns
    -------
    dict
        Mapping dictionary compatible with map_shot_to_class()
    """
    if 'shot_mapping' in config_dict:
        return config_dict['shot_mapping']
    return config_dict


def validate_mapping(mapping: Dict[str, List[str]], discovered_shots: Set[str]) -> Dict[str, List[str]]:
    """
    Validates that all discovered shots are mapped and reports unmapped shots.
    
    Parameters
    ----------
    mapping : dict
        Mapping dictionary
    discovered_shots : set
        Set of discovered shot types
        
    Returns
    -------
    dict
        Dictionary with 'mapped', 'unmapped', and 'warnings' keys
    """
    all_mapped_shots = set()
    for shot_list in mapping.values():
        all_mapped_shots.update([s.lower() for s in shot_list])
    
    discovered_lower = {s.lower() for s in discovered_shots}
    unmapped = discovered_lower - all_mapped_shots
    
    result = {
        'mapped': len(discovered_shots) - len(unmapped),
        'unmapped': list(unmapped),
        'warnings': []
    }
    
    if unmapped:
        result['warnings'].append(
            f"Found {len(unmapped)} unmapped shot types: {', '.join(unmapped)}"
        )
    
    return result


def extract_shot_type_from_filename(filename: str) -> Optional[str]:
    """
    Extracts shot type from pose CSV filename.
    
    Parameters
    ----------
    filename : str
        Filename like: 
        - {video}_{frame}_{shot_type}_{player}_pose.csv (active player)
        - {video}_{frame}_idle_{player}_pose.csv (idle player)
        Example: 15-11-2025-15-57_rpi-BO-0001_156_serve_left_pose.csv
        Example: 15-11-2025-15-57_rpi-BO-0001_156_idle_left_pose.csv
        
    Returns
    -------
    str or None
        Extracted shot type, or 'idle' for idle player files, or None if not found
    """
    basename = os.path.basename(filename)
    if not basename.endswith('_pose.csv'):
        return None
    
    # Remove extension
    name = basename.replace('_pose.csv', '')
    parts = name.split('_')
    
    # Check for idle files: {video}_{frame}_idle_{player}
    if 'idle' in [p.lower() for p in parts]:
        # Find idle index
        idle_idx = -1
        for i, part in enumerate(parts):
            if part.lower() == 'idle':
                idle_idx = i
                break
        
        if idle_idx != -1:
            # Verify there's a player position after idle
            player_positions = {'left', 'right', 'top', 'bottom'}
            if idle_idx + 1 < len(parts) and parts[idle_idx + 1].lower() in player_positions:
                return 'idle'
    
    # Find player position (left, right, top, bottom)
    player_positions = {'left', 'right', 'top', 'bottom'}
    player_idx = -1
    
    for i, part in enumerate(parts):
        if part.lower() in player_positions:
            player_idx = i
            break
    
    if player_idx == -1:
        return None
    
    # Shot type is the part just before the player position
    # Format: ..._frame_{shot_type}_{player}
    # We need to find the frame number (usually numeric) and take what's after it
    if player_idx < 1:
        return None
    
    # Try to find frame number - it's usually a numeric string
    # Look backwards from player position
    shot_type_idx = player_idx - 1
    
    # Check if the part before player is numeric (frame number)
    # If so, shot type might be compound (e.g., forehand_volley)
    # Actually, the format is: ..._frame_{shot_type}_{player}
    # So shot_type is parts[player_idx - 1] (single word) or could be compound
    
    # Handle compound shot types: look for the last numeric part before player
    # Everything between the last numeric part and player is the shot type
    frame_idx = -1
    for i in range(player_idx - 1, -1, -1):
        try:
            int(parts[i])  # Check if numeric
            frame_idx = i
            break
        except ValueError:
            continue
    
    if frame_idx == -1:
        # No frame number found, assume shot type is just before player
        shot_type = parts[player_idx - 1]
    else:
        # Shot type is everything between frame and player
        if player_idx - frame_idx > 1:
            shot_type_parts = parts[frame_idx + 1:player_idx]
            shot_type = '_'.join(shot_type_parts)
        else:
            shot_type = parts[player_idx - 1]
    
    if shot_type and shot_type.lower() != 'shot':
        return shot_type
    
    return None
