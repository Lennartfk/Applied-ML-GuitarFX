import os
import glob
import numpy as np
from typing import List, Tuple

effect_labels = {
    "11": "No Effect",
    "12": "No Effect",
    "21": "Feedback Delay",
    "22": "Slapback Delay",
    "23": "Reverb",
    "31": "Chorus",
    "32": "Flanger",
    "33": "Phaser",
    "34": "Tremolo",
    "35": "Vibrato",
    "41": "Distortion",
    "42": "Overdrive",
}

effect_to_index = {
    "Feedback Delay": 0,
    "Slapback Delay": 1,
    "Reverb": 2,
    "Chorus": 3,
    "Flanger": 4,
    "Phaser": 5,
    "Tremolo": 6,
    "Vibrato": 7,
    "Distortion": 8,
    "Overdrive": 9,
    "No Effect": 10
}

def get_wav_files(dataset_paths, max_files=None, skip_alternate=True):
    all_files = []
    for path in dataset_paths:
        subfolders = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        for subfolder in subfolders:
            found_files = glob.glob(os.path.join(subfolder, "**", "*.wav"), recursive=True)
            print(f"Found {len(found_files)} WAV files in subfolder: {subfolder}")
            all_files.extend(found_files)

    if skip_alternate:
        # Skip every other file
        all_files = all_files[::2]

    if max_files:
        return all_files[:max_files]
    return all_files

def extract_multilabel_from_filename(filename):
    parts = filename.split("-")
    effect_codes = []

    if len(parts) >= 4:
        primary_code = parts[2][1:3]
        effect_codes.append(primary_code)

    if len(parts) > 4:
        extra_code_block = parts[4]
        extra_codes = [extra_code_block[i:i+2] for i in range(0, len(extra_code_block), 2)]
        effect_codes.extend(extra_codes)

    label_vector = np.zeros(len(effect_to_index), dtype=np.float32)
    for code in effect_codes:
        effect_name = effect_labels.get(code)
        if effect_name and effect_name in effect_to_index:
            label_vector[effect_to_index[effect_name]] = 1.0

    return label_vector

def replace_all_trailing_ones_with_zeros(X: np.ndarray, padding_threshold: int = 5):
    """
    Detect samples with trailing ones in the last `padding_threshold` frames and replace
    all trailing ones from the end of each sample with zeros, after user confirmation.

    Args:
        X: np.ndarray, shape (num_samples, freq_bins, time_frames)
        padding_threshold: int, number of trailing frames to check for ones.

    Returns:
        X_new: np.ndarray with trailing ones replaced by zeros (if confirmed).
        num_modified: number of samples modified.
    """
    X_new = X.copy()
    candidates = []

    for i in range(len(X_new)):
        trailing_frames = X_new[i, :, -padding_threshold:]
        if np.all(trailing_frames == 1):
            candidates.append(i)

    num_candidates = len(candidates)
    print(f"Found {num_candidates} samples with trailing ones in the last {padding_threshold} frames.")

    if num_candidates == 0:
        print("No samples need modification.")
        return X_new, 0

    answer = input("Do you want to replace trailing ones with zeros in these samples? (yes/no): ").strip().lower()
    if answer not in ['yes', 'y']:
        print("No modifications applied.")
        return X_new, 0

    num_modified = 0
    for i in candidates:
        time_frames = X_new.shape[2]
        last_one_idx = time_frames - 1

        # Move backward while all freq bins at this time frame == 1
        while last_one_idx >= 0 and np.all(X_new[i, :, last_one_idx] == 1):
            last_one_idx -= 1
        
        start_replace = last_one_idx + 1
        X_new[i, :, start_replace:] = 0
        num_modified += 1

    print(f"Modified {num_modified} samples by replacing trailing ones with zeros.")
    return X_new, num_modified


def delete_samples_with_trailing_ones(X: np.ndarray, y: np.ndarray, padding_threshold: int = 5):
    """
    Remove samples (and their corresponding labels) with trailing ones in the last `padding_threshold` frames.

    Args:
        X: np.ndarray, shape (num_samples, freq_bins, time_frames)
        y: np.ndarray, shape (num_samples, num_classes)
        padding_threshold: int, number of trailing frames to check for ones.

    Returns:
        X_new: np.ndarray with offending samples removed.
        y_new: np.ndarray with corresponding labels removed.
        num_deleted: number of samples removed.
    """
    keep_indices = []

    for i in range(len(X)):
        trailing_frames = X[i, :, -padding_threshold:]
        if not np.all(trailing_frames == 1):
            keep_indices.append(i)

    num_deleted = len(X) - len(keep_indices)
    X_new = X[keep_indices]
    y_new = y[keep_indices]

    print(f"Deleted {num_deleted} samples with trailing ones in the last {padding_threshold} frames.")

    return X_new, y_new, num_deleted


