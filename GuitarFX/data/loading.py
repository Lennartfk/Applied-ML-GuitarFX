import os
import glob
from typing import Union, List

import numpy as np

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


def get_wav_files(dataset_paths: Union[str, List[str]], max_files: int = None) -> List[str]:
    """
    Returns:
        List[str]: List of wav files from the dataset path
    """
    all_files = []
    for path in dataset_paths:
        found_files = glob.glob(os.path.join(path, "**", "*.wav"), recursive=True)
        all_files.extend(found_files)
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
