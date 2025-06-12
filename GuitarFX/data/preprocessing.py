from typing import List, Union, Tuple
import librosa
from sklearn.model_selection import KFold, train_test_split
import io
import numpy as np
import soundfile as sf
import scipy.signal
import random
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from tqdm import tqdm
import gc


class PreProcessing:
    """
    This class is a pre-processing pipeline is speficically made for the
    pre-processing of guitar effect classification. However, this can be
    extended to other audio-related classification tasks.
    """

    def __init__(self, dataset_paths: Union[List[str], str]) -> None:
        """
        Inputs:
            dataset_paths (List[str | str]): List of paths or a singular path
            refering to where the audio dataset(s) are stored. It is presumed
            that the path to the dataset contains dictonaries that have the
            classifier as its name and the children of those dictonaries
            contain the audio files being part of that classification group.
        """
        if dataset_paths is not None:
            self.dataset_paths = list(dataset_paths)

    def bandpass_filter(self, y, sr, low=80, high=5000):
        sos = scipy.signal.butter(10, [low,high], btype='band', fs=sr, output='sos')
        return scipy.signal.sosfilt(sos,y)

    def rms_normalize(self, y):
        rms = np.sqrt(np.mean(y**2))
        return y / (rms + 1e-6)

    def trim(self, y, top_db=25):
        # Only trim leading silence, keep trailing tails
        intervals = librosa.effects.split(y, top_db=top_db, frame_length=1024, hop_length=256)
        if len(intervals) == 0:
            return y  # no non-silent region found
        start = intervals[0][0]
        return y[start:] 

    def is_clipped(self, y, threshold=0.98):
        return np.any(np.abs(y) >= threshold)

    def augment_audio(self, y, sr):
        # Randomly apply one augmentation
        choice = random.choice(['noise', 'stretch', 'none'])

        if choice == 'noise':
            noise = np.random.normal(0, 0.005, y.shape)
            y = y + noise
        elif choice == 'stretch':
            rate = random.uniform(0.8, 1.2)
            y = librosa.effects.time_stretch(y, rate=rate)
            if len(y) > sr * 5:
                y = y[:sr * 5]
        return y

    def _signal_processing(self, file_path, augment=True):
        y, sr = librosa.load(file_path, sr=44100, mono=True)
        y = self.bandpass_filter(y, sr)
        y = self.rms_normalize(y)
        if self.is_clipped(y):
            y *= 0.8  
        y = self.trim(y)
        if augment:
            y = self.augment_audio(y, sr)

        return y, sr

    def signal_processing_bytes(self, bytes_data, augment=True):
        data, sr = sf.read(io.BytesIO(bytes_data))
        y = librosa.to_mono(data.T)
        y = librosa.resample(y, orig_sr=sr, target_sr=44100)
        y = self.bandpass_filter(y, 44100)
        y = self.rms_normalize(y)
        if self.is_clipped(y):
            y *= 0.8
        y = self.trim(y)
        if augment:
            y = self.augment_audio(y, 44100)
        return y, 44100

    def data_splitting(self, features, labels):
        """
        Split the data in a training split, data split and test split for
        k-fold cross-valiation.
        """
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            features, labels, test_size=0.1, train_size=0.9, random_state=23
        )

        kf = KFold(n_splits=5)
        folds = list(kf.split(X_train_val))

        return X_train_val, X_test, y_train_val, y_test, folds

    def iterative_stratified_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.25,
        seed: int = 23
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split dataset into train/val/test sets with iterative multilabel stratification.

        Parameters:
            test_size: fraction for test set
            val_size: fraction of train_val set to use as val

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        print("Using iterative stratified test/val/train split...")
        msss = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=seed
        )
        for train_val_idx, test_idx in msss.split(X, y):
            X_train_val, X_test = X[train_val_idx], X[test_idx]
            y_train_val, y_test = y[train_val_idx], y[test_idx]

        msss_val = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=val_size, random_state=seed
        )
        for train_idx, val_idx in msss_val.split(X_train_val, y_train_val):
            X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
            y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

        return X_train, X_val, X_test, y_train, y_val, y_test

    def subsample_iterative_stratification(self, X, y, fraction=0.5, seed=23):
        splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=1 - fraction, random_state=seed)
        for sample_idx, _ in splitter.split(X, y):
            return X[sample_idx], y[sample_idx]

    def process_filepaths(self, file_paths: List[str], augment: bool) -> List[np.ndarray]:
        """
        Load audio files from file_paths applying augmentation if specified,
        returning a list of processed audio arrays.
        """
        processed_audio = []
        for fp in tqdm(file_paths, desc=f"{'Augmenting and processing' if augment else 'Processing'} audio files"):
            y, sr = self._signal_processing(fp, augment=augment)
            processed_audio.append(y)
        return processed_audio
