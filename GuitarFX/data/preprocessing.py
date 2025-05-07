from typing import List
import numpy as np
import librosa
from sklearn.model_selection import KFold, train_test_split


class PreProcessing:
    def __init__(self, dataset_path: List[str] | str) -> None:
        self.dataset_path = dataset_path

    def signal_processing(file_path):
        """
        Pre-process the audio signal.
        """
        # Get the audio signal
        y, sr = librosa.load(file_path)

        # Resample at 44.1kHz
        y_resampled, sr = librosa.resample(y, orig_sr=sr, target_sr=44100)

        # Switch any potential multi channel audio signal to a mono channel
        y_mono = librosa.to_mono(y_resampled)

        # Trim leading and trailing silence
        y_trimmed = librosa.effects.trim(y_mono)

        return y_trimmed, sr

    def extract_mean_features(y, sr):
        """
        These are the features we use for the competetive baseline model,
        which is support vector machine with radial basis function (RBF)
        kernel. Can also be used for other model that can only one-dimensional
        data.
        """
        mffcs = librosa.feature.mfcc(y, sr=sr, n_mfcc=20)
        chromatogram = librosa.feature.chroma_stft(y, sr=sr)
        spectrogram = librosa.feature.spectral_centroid(y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y, sr=sr)
        root_mean_squared_error = librosa.feature.rms(y, sr=sr)

        return {
            "mffcs": np.mean(mffcs, axis=1),
            "zero_crossing_rate": np.mean(zero_crossing_rate, axis=1),
            "chromatogram": np.mean(chromatogram, axis=1),
            "spectrogram": np.mean(spectrogram, axis=1),
            "root_mean_squared_error": np.mean(root_mean_squared_error, axis=1),
        }

    def extract_signal_representations(y, sr):
        """
        Features for the convolutional neural network (CNN)
        """
        pass

    def data_splitting(features, labels):
        """
        Split the data in a training split, data split and test split for
        k-fold cross-valiation.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=0.1,
            train_size=0.9,
            random_state=42
        )

        kf = KFold(n_splits=5)
        folds = kf.split(X_train)

        return X_train, X_test, y_train, y_test, folds

    def data_augmentation(data):
        """
        """
        pass

    def data_saving(data):
        """
        """
        pass

    def data_cleaning(data):
        """
        """
        pass

    def execute():
        """
        Execute the pre-processing pipeline
        """
        features = []
        labels = []
