from GuitarFX.data.preprocessing import PreProcessing
from GuitarFX.io.file_io import get_wav_files, extract_multilabel_from_filename

import os
from typing import List, Tuple

import librosa
import numpy as np
from tqdm import tqdm


class FeatureExtractor(PreProcessing):
    """Handle the feature extraction from the audio dataset."""

    def __init__(self, dataset_paths: str | List[str]) -> None:
        """
        Initialize the feature extractor for the competetive baseline support
        vector machine.

        Inputs:
            dataset_paths (str | List[str]): Path(s) to dataset.
        """
        self.dataset_paths = dataset_paths

    def extract_mean_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        These are the features we use for the competetive baseline model,
        which is support vector machine with radial basis function (RBF)
        kernel. Can also be used for other model that can only one-dimensional
        data.
        """
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        chromatogram = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        room_mean_square = librosa.feature.rms(y=y)

        features = np.concatenate(
            [
                np.mean(mfcc, axis=1),
                np.mean(chromatogram, axis=1),
                np.mean(spectral_contrast, axis=1),
                np.mean(zero_crossing_rate, axis=1),
                np.mean(room_mean_square, axis=1),
            ]
        )

        return features

    def execute_mean_features(self, max_samples_per_classifier: int = None) \
            -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Extract the mean features for each file in dataset.

        Inputs:
                max_samples_per_classifier (int): Limit the sample amount per
                classifier to limit the amount of time spent for
                pre-processing.

        Returns:

        """
        feature_names = [
            "mfcc_1",
            "mfcc_2",
            "mfcc_3",
            "mfcc_4",
            "mfcc_5",
            "mfcc_6",
            "mfcc_7",
            "mfcc_8",
            "mfcc_9",
            "mfcc_10",
            "mfcc_11",
            "mfcc_12",
            "mfcc_13",
            "mfcc_14",
            "mfcc_15",
            "mfcc_16",
            "mfcc_17",
            "mfcc_18",
            "mfcc_19",
            "mfcc_20",
            "chroma_1",
            "chroma_2",
            "chroma_3",
            "chroma_4",
            "chroma_5",
            "chroma_6",
            "chroma_7",
            "chroma_8",
            "chroma_9",
            "chroma_10",
            "chroma_11",
            "chroma_12",
            "spectral_contrast_1",
            "spectral_contrast_2",
            "spectral_contrast_3",
            "spectral_contrast_4",
            "spectral_contrast_5",
            "spectral_contrast_6",
            "spectral_contrast_7",
            "zero_crossing_rate",
            "rms",
        ]
        label_names = []

        X = []
        y = []

        wav_files = get_wav_files(
            self.dataset_paths,
            max_files=max_samples_per_classifier
        )

        for wav_file_path in tqdm(wav_files, desc="Processing audiofiles"):
            file_name = os.path.basename(wav_file_path)
            effect_label = extract_multilabel_from_filename(file_name)
            label_names.append(effect_label)

            signal, sr = self._signal_processing(wav_file_path)
            features = self.extract_mean_features(signal, sr)

            X.append(features)
            y.append(effect_label)

        X = np.array(X)
        y = np.array(y)

        return X, y, feature_names, label_names
