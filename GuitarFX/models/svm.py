from GuitarFX.features.baseline_features import FeatureExtractor

from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd


def get_features(
    dataset_paths,
    read_csv=True,
    csv_path="guitar_monophon_mean_features.csv",
    max_samples_per_classifier=None,
):
    """
    Load features from CSV if available, otherwise extract and save them.

    Args:
        dataset_paths (list): Paths to dataset directories.
        read_csv (bool): If True, load features from CSV. If False, extract
        them.
        csv_path (str): Path to the CSV file to load/save.
        max_samples_per_classifier (int): Optional limit for sample count per
        class.

    Returns:
        tuple: (X, y, feature_names, label_names)
    """
    if read_csv:
        df = pd.read_csv(csv_path)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        feature_names = list(df.columns[:-1])
        label_names = sorted(set(y))
    else:
        extractor = FeatureExtractor(dataset_paths)
        X, y, feature_names, label_names = extractor.execute_mean_features(
            max_samples_per_classifier
        )
        df = pd.DataFrame(X, columns=feature_names)
        df["label"] = y
        df.to_csv(csv_path, index=False)

    return X, y, feature_names, label_names


class CustomSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, kernel="rbf", gamma="scale"):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma

    def fit(self, X, y):
        self.model_ = SVC(C=self.C, kernel=self.kernel, gamma=self.gamma,
                          probability=True)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def save(self):
        pass

    def load(self):
        pass
