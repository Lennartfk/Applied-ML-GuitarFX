import librosa
import numpy as np
from GuitarFX.data.preprocessing import PreProcessing
from GuitarFX.data.loading import extract_label_from_filename, get_wav_files
import os
from tqdm import tqdm

class CNNFeatureExtractor(PreProcessing):
    """Extract 2d mel spectrogram features for CNN input."""

    def __init__(self, dataset_paths, n_mels=128, hop_length=512, cache_dir=None):
        super().__init__(dataset_paths)
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.cache_dir = cache_dir

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

        return mel_spec_db


    def _execute_mel_spectrograms(self, max_samples_per_classifier=None):
        """Extract 2D mel spectrograms for each file in dataset."""

        label_names = []
        X = []
        y = []

        wav_files = get_wav_files(self.dataset_paths, max_files=max_samples_per_classifier)

        for wav_file_path in tqdm(wav_files, desc="Processing mel spectrograms"):
            file_name = os.path.basename(wav_file_path)
            effect_label = extract_label_from_filename(file_name)
            label_names.append(effect_label)

            signal, sr = self.signal_processing(wav_file_path)
            mel_spec = self._extract_mel(signal, sr)

            X.append(mel_spec)
            y.append(effect_label)

        X = np.array(X)
        y = np.array(y)

        return X, y, label_names

    def save_features(self, X, y, label_names, filename="features.npz"):
        os.makedirs("data", exist_ok=True)
        path = os.path.join("data", filename)
        np.savez(path, X=X, y=y, label_names=label_names)
        print(f"Features saved to {path}")


    def load_features(self, filename="features.npz"):
        path = os.path.join("data", filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} does not exist.")
        data = np.load(path, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        label_names = data["label_names"].tolist() 
        print(f"Features loaded from {path}")
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
                return self.load_features(filename)
            except FileNotFoundError:
                print("Cache not found, extracting features...")

        X, y, label_names = self._execute_mel_spectrograms(max_samples_per_classifier=max_samples_per_classifier)
        self.save_features(X, y, label_names, filename)
        return X, y, label_names