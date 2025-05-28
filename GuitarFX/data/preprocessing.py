from typing import List
import librosa
from sklearn.model_selection import KFold, train_test_split
import io
import soundfile as sf

class PreProcessing:
    """
    This class is a pre-processing pipeline is speficically made for the
    pre-processing of guitar effect classification. However, this can be
    extended to other audio-related classification tasks.
    """

    def __init__(self, dataset_paths: List[str] | str) -> None:
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

    def signal_processing(self, file_path):
        """
        Pre-process the audio signal.
        """
        # Get the audio signal, resample at 44.1kHz and switch any potential
        # multi channel audio signal to a mono channel
        y, sr = librosa.load(file_path, sr=44100, mono=True)

        # Trim leading and trailing silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=30)

        return y_trimmed, sr

    def signal_processing_bytes(self, bytes):
        # Get the audio signal at (samples, channel)
        data, sr = sf.read(io.BytesIO(bytes))

        # Convert to mono channel
        y = librosa.to_mono(data.T)

        # resample at 44.1kHz
        new_sr = 44100
        y = librosa.resample(y, orig_sr=sr, target_sr=new_sr)

        # Trim leading and trailing silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=30)

        return y_trimmed, new_sr

    def extract_signal_representations(self, y, sr):
        """
        Features for the convolutional neural network (CNN)
        """
        pass

    def data_splitting(self, features, labels):
        """
        Split the data in a training split, data split and test split for
        k-fold cross-valiation.
        """
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            features, labels, test_size=0.1, train_size=0.9, random_state=42
        )

        kf = KFold(n_splits=5)
        folds = list(kf.split(X_train_val))

        return X_train_val, X_test, y_train_val, y_test, folds

    def data_augmentation(self, data):
        """ """
        pass

    def data_saving(self, dataframe):
        """ """
        pass

    def data_cleaning(self, data):
        """ """
        pass
