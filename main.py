from GuitarFX.data.preprocessing import PreProcessing
from GuitarFX.models.svm import CustomSVM, get_features
from GuitarFX.io.model_io import save_svm_model
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

if __name__ == "__main__":
    """
    Run the competitive baseline SVM model using mean audio features
    for multi-label guitar effects classification.
    """
    base_path = (
        r"C:\Users\lenna\Documents\RUG\Jaar 2\Periode 2b"
        r"\Applied Machine Learning"
        r"\Project (AML)\Datasets\IDMT-SMT-AUDIO-EFFECTS"
        r"\IDMT-SMT-AUDIO-EFFECTS"
        r"\IDMT-SMT-AUDIO-EFFECTS"
    )
    dataset_paths = [
        fr"{base_path}\Gitarre monophon",
        fr"{base_path}\Gitarre monophon2",
        fr"{base_path}\Gitarre polyphon",
        fr"{base_path}\Gitarre polyphon2",
    ]

    # Initialize pre-processing object
    pre_processing = PreProcessing(dataset_paths)

    # Extract or load features
    X, y, feature_names, label_names = get_features(
        dataset_paths=pre_processing.dataset_paths,
        read_csv=True,  # Set to False to force re-extraction
        csv_path="data/svm_features_multi.csv"
    )

    print("y:", y)
    print("type of y:", type(y))

    # Multi-label encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Train-test split
    X_train_val, X_test, y_train_val, y_test, folds = pre_processing.data_splitting(  # noqa E501
        X, y_encoded
    )

    # Standardization
    scaler = StandardScaler()
    X_train_val = scaler.fit_transform(X_train_val)
    X_test = scaler.transform(X_test)

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
accuracies = []

for train_index, test_index in tqdm(skf.split(X, y), total=n_splits,
                                    desc="CV folds"):
    X_train_val, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train_val, y_test = y.iloc[train_index], y.iloc[test_index]

    unique_classes = np.unique(y_train_val)
    if len(unique_classes) < 2:
        print("Skipping fold due to insufficient classes in training:"
              f"{unique_classes}")
        continue

    scaler = StandardScaler()
    X_train_val = scaler.fit_transform(X_train_val)
    X_test = scaler.transform(X_test)

    svm = CustomSVM(C=100, kernel="rbf", gamma=0.01)
    svm = svm.fit(X_train_val, y_train_val)

    y_pred_proba = svm.predict_proba(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    accuracy = np.mean(y_pred == y_test)
    accuracies.append(accuracy)

mean_acc = np.mean(accuracies)
std_err_acc = np.std(accuracies) / np.sqrt(len(accuracies))

print(f"Accuracy: {mean_acc:.4f} ± {std_err_acc:.4f} (standard error)")

save_svm_model(svm, scaler, label_encoder, path="models")
