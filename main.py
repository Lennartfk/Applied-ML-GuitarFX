from GuitarFX.data.preprocessing import PreProcessing
from GuitarFX.models.svm import CustomSVM, get_features
from GuitarFX.metrics.metrics import ModelMetrics

from sklearn.preprocessing import StandardScaler, LabelEncoder

if __name__ == "__main__":
    """
    Run the competitive baseline SVM model using mean audio features.
    """
    dataset_paths = [
        (
            r"C:\Users\lenna\Documents\RUG\Jaar 2\Periode 2b\Applied Machine Learning"
            r"\Project (AML)\Datasets\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS"
            r"\IDMT-SMT-AUDIO-EFFECTS\Gitarre monophon\Gitarre monophon\Samples"
        ),
        (
            r"C:\Users\lenna\Documents\RUG\Jaar 2\Periode 2b\Applied Machine Learning"
            r"\Project (AML)\Datasets\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS"
            r"\IDMT-SMT-AUDIO-EFFECTS\Gitarre monophon2\Gitarre monophon2\Samples"
        ),
    ]

    pre_processing = PreProcessing(dataset_paths)

    # Set read_csv=False if you want to re-extract features
    X, y, feature_names, label_names = get_features(
        dataset_paths=pre_processing.dataset_paths,
        read_csv=True,  # Change to False to extract and save again
        csv_path="data/guitar_monophon_mean_features.csv",
    )

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit(y)

    X_train_val, X_test, y_train_val, y_test, folds = pre_processing.data_splitting(
        X, y
    )

    scaler = StandardScaler()
    X_train_val = scaler.fit_transform(X_train_val)
    X_test = scaler.transform(X_test)

    y_train_val_encoded = label_encoder.transform(y_train_val)
    y_test_encoded = label_encoder.transform(y_test)

    # The hyper-parameters were decided by the use of a grid search using the
    # training data for fitting the parameters and validation for evaluating
    # each combination.
    svm = CustomSVM(C=100, kernel="rbf", gamma=0.01)
    svm = svm.fit(X_train_val, y_train_val_encoded)
    y_pred_proba = svm.predict_proba(X_test)

    svm_metrics = ModelMetrics(
        y_pred=y_pred_proba,
        y_actual=y_test_encoded,
        label_encoder=label_encoder
    )
    svm_metrics.report_all_results()
