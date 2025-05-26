import os
import glob


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


def get_wav_files(dataset_paths, max_files=None):
    """Return list of .wav file paths across all dataset paths, optionally capped."""
    all_files = []
    for path in dataset_paths:
        all_files.extend(glob.glob(os.path.join(path, "*", "*.wav")))
    if max_files:
        return all_files[:max_files]
    return all_files


def extract_label_from_filename(filename):
    """Extract label string from filename using the effect code."""
    effect_code = filename.split("-")[2][1:3]
    return effect_labels.get(effect_code, "Unknown")