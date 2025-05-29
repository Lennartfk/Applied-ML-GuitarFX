import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import Counter

from GuitarFX.features.cnn_features import CNNFeatureExtractor
from GuitarFX.models.Guitar2dCNN import GuitarEffectCNN
from GuitarFX.metrics.metrics import ModelMetrics

def most_common_class_accuracy(y_val):
    class_counts = Counter(y_val)
    most_common_class, count = class_counts.most_common(1)[0]
    percentage = count / len(y_val) * 100
    print(f"Most common class: '{most_common_class}' with {count} samples out of {len(y_val)}")
    print(f"Baseline accuracy by always predicting '{most_common_class}': {percentage:.2f}%")
    return percentage

def main():
    dataset_paths = [
        r"C:\Users\lenna\Documents\RUG\Jaar 2\Periode 2b\Applied Machine Learning\Project (AML)\Datasets\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS\Gitarre monophon",
        r"C:\Users\lenna\Documents\RUG\Jaar 2\Periode 2b\Applied Machine Learning\Project (AML)\Datasets\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS\Gitarre monophon2",
        r"C:\Users\lenna\Documents\RUG\Jaar 2\Periode 2b\Applied Machine Learning\Project (AML)\Datasets\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS\Gitarre polyphon",
        r"C:\Users\lenna\Documents\RUG\Jaar 2\Periode 2b\Applied Machine Learning\Project (AML)\Datasets\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS\Gitarre polyphon2"
    ]

    # Feature extraction
    extractor = CNNFeatureExtractor(dataset_paths)
    X, y, label_names = extractor.get_cnn_features(read_file=True, filename="cnn_scaled.npz")  # load cached if exists

    # Encode string labels to integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X = X[..., np.newaxis]

    # Split into train and test sets
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
    )

    print("Classes:", label_encoder.classes_)
    num_classes = len(label_encoder.classes_)


    # Build model (needed to load weights later)
    model = GuitarEffectCNN(num_classes=num_classes)

    # Comment out to skip retraining
    # history = model.train(
    #     train_dataset=(X_train, y_train),
    #     val_dataset=(X_val, y_val),
    #     epochs=30,
    #     learning_rate=0.0001
    # )

    # Load previously trained model weights
    print("\nLoading trained model from disk...")
    model.model = load_model("models/guitar_effect_cnn.h5")
    print("Model loaded.")

    # Predict on test set
    print("\nPredicting on test set...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Calculate test accuracy
    test_accuracy = np.mean(y_pred == y_test) * 100
    print(f"Test accuracy: {test_accuracy:.2f}%")

    # Report model metrics
    metrics = ModelMetrics(
        y_pred=y_pred_probs,
        y_actual=y_test,
        label_encoder=label_encoder,
        train_accuracy=None,  # no history since no training now
        val_accuracy=None,
        train_loss=None,
        val_loss=None
    )
    metrics.report_all_results()

if __name__ == "__main__":
    main()


