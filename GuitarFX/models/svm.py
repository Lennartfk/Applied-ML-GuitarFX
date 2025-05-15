from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
from GuitarFX.features.baseline_features import FeatureExtractor


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
        read_csv (bool): If True, load features from CSV. If False, extract them.
        csv_path (str): Path to the CSV file to load/save.
        max_samples_per_classifier (int): Optional limit for sample count per class.

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


def train_svm(X_train, y_train, X_test, y_test, kernel="rbf", C=1, gamma="scale"):
    """
    Trains SVM classifier on training data and evaluate on test data.

    Parameters:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training labels.
        X_test (numpy.ndarray): Test features.
        y_test (numpy.ndarray): Test labels.
        kernel (str): SVM kernel type (default is 'rbf').
        C (float): Regularization parameter.
        gamma (str): Kernel coefficient.

    Returns:
        float: Test set accuracy.
    """
    # Create the model
    model = SVC(kernel=kernel, C=C, gamma=gamma)

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    test_acc = accuracy_score(y_test, y_pred)

    return test_acc
