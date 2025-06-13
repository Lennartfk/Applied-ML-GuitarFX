from ..data.preprocessing import PreProcessing
from ..io.file_io import extract_multilabel_from_filename, get_wav_files

import os
from typing import Union, List, Tuple, Optional

import gc
import librosa
import librosa.display
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class CNNFeatureExtractor(PreProcessing):
    """Extract 2d mel spectrogram features for CNN input."""

    def __init__(
            self,
            dataset_paths: Union[str, List[str]],
            n_mels: int = 128,
            hop_length: int = 512
    ) -> None:
        """
        Initialize the feature extractor for the multi-effect convolutional
        neural neural network. This is mainly for extracting
        dB-mel spectograms.

        Args:
            dataset_paths (str | List[str]): Path(s) to dataset.
            n_mels (int): Number of mel frequency bands.
            hop_length (int): Hop length for STFT.
        """
        super().__init__(dataset_paths)

        self.n_mels = n_mels
        self.hop_length = hop_length
        self.class_names = [
            "Feedback Delay", "Slapback Delay", "Reverb", "Chorus", "Flanger",
            "Phaser", "Tremolo", "Vibrato", "Distortion", "Overdrive",
            "No Effect"
        ]

    def _extract_mel(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract a normalized mel spectrogram with a fixed width of 128 time
        frames.

        Args:
            y (np.ndarray): 1D audio signal.
            sr (int): Sampling rate of the audio signal.

        Returns:
            np.ndarray: Normalized dB-scale mel spectrogram
            (shape: [n_mels, 128]).
        """
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

    def _execute_mel_spectrograms(
            self,
            max_samples_per_classifier: int = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract 2D mel spectrograms for each file in dataset with memory
        error handling.

        Args:
            max_samples_per_classifier (int): The amount of samples to process
            for each class.

        Returns:
            Tuple[np.ndarray, np.ndarray, List[str]]: 
        """
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
                print(f"MemoryError: Skipping file {wav_file_path} due" +
                      "to insufficient memory.")
                continue

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

    def get_cnn_features(
        self,
        output_dir: str = "features",
        split_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
        label_names: Optional[List[str]] = None,
        augment_train: bool = True
    ) -> Tuple[np.memmap, np.memmap, np.memmap, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Extract or load CNN features into memmap format (.dat) and return them along with labels.

        Args:
            output_dir (str): Folder where .dat files and labels will be stored.
            split_data (tuple or None): Tuple of (X_train_fp, X_val_fp, X_test_fp, y_train, y_val, y_test).
                If None, raises error.
            label_names (list or None): List of label class names. Required if split_data is provided.
            augment_train (bool): Whether to augment training features.

        Returns:
            X_train, X_val, X_test (np.memmap): Memmap arrays for features.
            y_train, y_val, y_test (np.ndarray): Corresponding labels.
            label_names (List[str]): Label class names.
        """
        if split_data is None or label_names is None:
            raise ValueError("split_data and label_names must be provided")

        os.makedirs(output_dir, exist_ok=True)

        # Unpack split data
        X_train_fp, X_val_fp, X_test_fp, y_train, y_val, y_test = split_data

        # Define memmap paths
        train_memmap_path = os.path.join(output_dir, "train_features.dat")
        val_memmap_path = os.path.join(output_dir, "val_features.dat")
        test_memmap_path = os.path.join(output_dir, "test_features.dat")
        label_path = os.path.join(output_dir, "feature_labels.npz")

        feature_shape = (128, 128)
        dtype = np.float32

        def save_features(filepaths, destination, augment):
            audio_list = self.process_filepaths(filepaths, augment=augment)
            shape = (len(audio_list), *feature_shape)
            memmap = np.memmap(destination, dtype=dtype, mode='w+', shape=shape)
            for i, audio in enumerate(tqdm(audio_list, desc=f"Extracting to {os.path.basename(destination)}")):
                memmap[i] = self._extract_mel(audio, sr=44100)
            memmap.flush()

        # Extract or load features
        if not os.path.exists(train_memmap_path):
            save_features(X_train_fp, train_memmap_path, augment=augment_train)
        if not os.path.exists(val_memmap_path):
            save_features(X_val_fp, val_memmap_path, augment=False)
        if not os.path.exists(test_memmap_path):
            save_features(X_test_fp, test_memmap_path, augment=False)

        # Save labels
        np.savez(label_path, y_train=y_train, y_val=y_val, y_test=y_test, label_names=label_names)

        # Load memmap arrays
        X_train = np.memmap(train_memmap_path, dtype=dtype, mode='r', shape=(len(X_train_fp), *feature_shape))
        X_val = np.memmap(val_memmap_path, dtype=dtype, mode='r', shape=(len(X_val_fp), *feature_shape))
        X_test = np.memmap(test_memmap_path, dtype=dtype, mode='r', shape=(len(X_test_fp), *feature_shape))

        return X_train, X_val, X_test, y_train, y_val, y_test, label_names

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
