import librosa
import librosa.display
import numpy as np
from ..data.preprocessing import PreProcessing
from ..io.file_io import extract_multilabel_from_filename, get_wav_files
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Tuple
import gc
import uuid
import json


class CNNFeatureExtractor(PreProcessing):
    """Extract 2d mel spectrogram features for CNN input."""

    def __init__(self, dataset_paths, n_mels=128, hop_length=512, cache_dir=None):
        super().__init__(dataset_paths)
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.cache_dir = cache_dir
        self.class_names = [
            "Feedback Delay", "Slapback Delay", "Reverb", "Chorus", "Flanger",
            "Phaser", "Tremolo", "Vibrato", "Distortion", "Overdrive", "No Effect"
        ]

    def _extract_mel(self, y, sr):
        """Extract mel spectrogram with fixed width (128 time frames)."""
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=self.n_mels, hop_length=self.hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        if mel_spec_db.shape[1] < 128:
            pad_width = 128 - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :128]

        mel_spec_db -= mel_spec_db.min()
        mel_spec_db /= mel_spec_db.max() + 1e-8

        return mel_spec_db

    def _execute_mel_spectrograms(self, max_samples_per_classifier=None):
        """Extract 2D mel spectrograms for each file in dataset with memory error handling."""

        label_names = []
        X = []
        y = []

        wav_files = get_wav_files(self.dataset_paths, max_files=max_samples_per_classifier)

        for wav_file_path in tqdm(wav_files, desc="Processing mel spectrograms"):
            try:
                file_name = os.path.basename(wav_file_path)
                effect_label = extract_multilabel_from_filename(file_name)

                signal, sr = self._signal_processing(wav_file_path, augment=True)
                mel_spec = self._extract_mel(signal, sr)

                X.append(mel_spec)
                y.append(effect_label)

                del signal 
                gc.collect()

            except MemoryError:
                print(f"MemoryError: Skipping file {wav_file_path} due to insufficient memory.")
                continue  # Skip this file and continue with the next one

        X = np.array(X)
        y = np.array(y)

        label_names = self.class_names

        return X, y, label_names

    def _save_features(self, X, y, label_names, filename="features.npz"):
        os.makedirs("data", exist_ok=True)
        path = os.path.join("data", filename)
        np.savez(path, X=X, y=y, label_names=label_names)
        print(f"Features saved to {path}")

    def _load_features(self, filename="features.npz"):
        path = os.path.join("data", filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} does not exist.")
        data = np.load(path, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        label_names = data["label_names"].tolist()
        print(f"Features loaded from {path}")
        print(label_names)
        return X, y, label_names

    def get_cnn_features(self, max_samples_per_classifier=None, read_file=True, filename="features.npz"):
        """
        Return CNN features either by loading from cache or by extracting and saving.

        Args:
            max_samples_per_classifier (int | None): max samples per class to process.
            read_file (bool): if True, try to load cached features; else extract fresh.
            filename (str): filename for caching features (.npz).

        Returns:
            X (np.ndarray): feature array (num_samples, n_mels, 128)
            y (np.ndarray): labels array
            label_names (list): list of label strings
        """
        if read_file:
            try:
                return self._load_features(filename)
            except FileNotFoundError:
                print("Cache not found, extracting features...")

        X, y, label_names = self._execute_mel_spectrograms(max_samples_per_classifier=max_samples_per_classifier)
        self._save_features(X, y, label_names, filename)
        return X, y, label_names

    def extract_mel_for_prediction(self, bytes):
        """Extract a single mel spectrogram ready for CNN prediction."""
        signal, sr = self.signal_processing_bytes(bytes)
        mel_spec = self._extract_mel(signal, sr)
        mel_spec = np.expand_dims(mel_spec, axis=-1)
        mel_spec = np.expand_dims(mel_spec, axis=0)

        return mel_spec
    
    def plot_mel(self, mel_spec):
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spec, x_axis='time', y_axis='mel', sr=44100)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        plt.tight_layout()
        plt.show()

    def get_audio_paths_and_labels(self, max_samples_per_classifier=None) -> Tuple[List[str], np.ndarray, List[str]]:
        wav_files = get_wav_files(self.dataset_paths, max_files=max_samples_per_classifier)

        file_paths = []
        labels = []

        for wav_path in wav_files:
            file_name = os.path.basename(wav_path)
            label = extract_multilabel_from_filename(file_name)

            file_paths.append(wav_path)
            labels.append(label)

        return file_paths, np.array(labels), self.class_names

    def extract_features_from_audio(self, audio_list: List[np.ndarray]) -> np.ndarray:
        features = []
        for y in tqdm(audio_list, desc="Extracting mel features"):
            mel_spec = self._extract_mel(y, sr=44100)
            features.append(mel_spec)
        return np.array(features)
