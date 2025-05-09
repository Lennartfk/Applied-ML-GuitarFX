from GuitarFX.data.preprocessing import PreProcessing
from GuitarFX.models.svm import train_svm, get_features
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import GridSearchCV
import numpy as np


if __name__ == '__main__':
    """
    Run the competitive baseline SVM model using mean audio features.
    """
    dataset_paths = [
        r'datasets\IDMT-SMT-AUDIO-EFFECTS\Gitarre monophon',
        r'datasets\IDMT-SMT-AUDIO-EFFECTS\Gitarre monophon2'
    ]
    
    pre_processing = PreProcessing(dataset_paths)

    # Set read_csv=False if you want to re-extract features
    X, y, feature_names, label_names = get_features(
        dataset_paths=pre_processing.dataset_paths,
        read_csv=True,  # Change to False to extract and save again
        csv_path="guitar_monophon_mean_features.csv"
    )

    X_train_val, X_test, y_train_val, y_test, folds = pre_processing.data_splitting(X, y)

    scaler = StandardScaler()
    X_train_val = scaler.fit_transform(X_train_val)
    X_test = scaler.transform(X_test)

    test_acc = train_svm(
        X_train=X_train_val,
        y_train=y_train_val,
        X_test=X_test,
        y_test=y_test,
        kernel='rbf',
        C=100,
        gamma=0.01
    )

    print(f"Test set accuracy: {test_acc:.4f}")
