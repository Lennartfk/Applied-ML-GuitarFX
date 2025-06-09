import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from kerastuner.engine.hyperparameters import HyperParameters
import argparse
import os
import pickle

from GuitarFX.features.cnn_features import CNNFeatureExtractor
from GuitarFX.models.Guitar2dCNN import GuitarEffectCNN
from GuitarFX.metrics.metrics import ModelMetrics


def sample_balanced_dataset(X, y, fraction=0.5, seed=42):
    np.random.seed(seed)
    total_samples = int(len(X) * fraction)

    indices = np.arange(len(X))
    np.random.shuffle(indices)

    selected_indices = []
    class_totals = y.sum(axis=0)
    class_counts = np.zeros_like(class_totals)

    for idx in indices:
        if len(selected_indices) >= total_samples:
            break

        sample_labels = y[idx]
        if sample_labels.sum() == 0:
            continue

        if np.all(class_counts + sample_labels <= class_totals * fraction * 1.2):
            selected_indices.append(idx)
            class_counts += sample_labels

    return X[selected_indices], y[selected_indices]

def print_class_distribution(y, label_names):
    class_counts = y.sum(axis=0)
    total_samples = len(y)
    print(f"Total samples: {total_samples}")
    for i, label in enumerate(label_names):
        count = int(class_counts[i])
        percent = 100 * count / total_samples
        print(f"{label}: {count} samples ({percent:.2f}%)")

def main(tune=False):
    dataset_paths = [
        r"C:\Users\lenna\Documents\RUG\Jaar 2\Periode 2b\Applied Machine Learning\Project (AML)\Datasets\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS\Gitarre monophon",
        r"C:\Users\lenna\Documents\RUG\Jaar 2\Periode 2b\Applied Machine Learning\Project (AML)\Datasets\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS\Gitarre monophon2",
        r"C:\Users\lenna\Documents\RUG\Jaar 2\Periode 2b\Applied Machine Learning\Project (AML)\Datasets\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS\Gitarre polyphon",
        r"C:\Users\lenna\Documents\RUG\Jaar 2\Periode 2b\Applied Machine Learning\Project (AML)\Datasets\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS\Gitarre polyphon2"
    ]

    extractor = CNNFeatureExtractor(dataset_paths)

    data = np.load("data/cnn_multi.npz")
    X, y = data["X"], data["y"]

    label_names = data["label_names"].tolist()

    # Sample 50% balanced dataset
    X_sampled, y_sampled = sample_balanced_dataset(X, y, fraction=0.5, seed=23)
    print_class_distribution(y_sampled, label_names)

    X, y = X_sampled, y_sampled
    X = X[..., np.newaxis]

    print(f"Extracted features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    # Split data
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=23)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=23)

    num_classes = y.shape[1]
    model = GuitarEffectCNN(num_classes=num_classes, label_smoothing=0.1)

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    if tune:
        print("Starting hyperparameter tuning...")
        model.setup_tuner(max_epochs=20, directory='kt_dir', project_name='guitarfx_tuning')

        # Use 20% balanced subset for tuning to speed things up
        X_train_small, y_train_small = sample_balanced_dataset(X_train, y_train, fraction=0.2, seed=23)

        model.search(
            train_dataset=(X_train_small, y_train_small),
            val_dataset=(X_val, y_val),
            epochs=15,
            batch_size=64,
            callbacks=[early_stop]
        )

    else:
        print("Training model without hyperparameter tuning...")

        # with open("models/cnn_hyper_tuner.pkl", "rb") as f:
        #     best_hp_values = pickle.load(f)

        # hp = HyperParameters()
        # for k, v in best_hp_values.items():
        #     hp.values[k] = v

        # model.model = model.build_model()
        # model.train(
        #     train_dataset=(X_train, y_train),
        #     val_dataset=(X_val, y_val),
        #     epochs=30,
        #     batch_size=32,
        #     lr=0.0001,
        #     callbacks=[early_stop]
        # )
        print("hi")

    # Save model and history
    # os.makedirs("models", exist_ok=True)
    # model.save("models/cnn_test.h5")

    model.load("models/cnn_test.h5")

    # Predict and evaluate
    y_pred_probs = model.predict(X_test)
    y_pred = model.predict_classes(X_test, threshold=0.5)
    history = model.get_training_history()

    train_auc = history.get("auc")
    val_auc = history.get("val_auc")
    train_loss = history.get("loss")
    val_loss = history.get("val_loss")

    metrics = ModelMetrics(
        y_pred=y_pred,
        y_actual=y_test,
        label_names=label_names,
        threshold=0.5,
        train_auc=train_auc,
        val_auc=val_auc,
        train_loss=train_loss,
        val_loss=val_loss,
    )

    metrics.report_all_results()
    metrics.classification_metrics_report_single_vs_multi()

    idx = np.random.randint(0, len(X_test))
    pred_labels = [label_names[i] for i, val in enumerate(y_pred[idx]) if val == 1]
    true_labels = [label_names[i] for i, val in enumerate(y_test[idx]) if val == 1]

    print("\nSample Prediction:")
    print(f"Predicted: {pred_labels}")
    print(f"Actual:    {true_labels}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GuitarEffectCNN with optional hyperparameter tuning.")
    parser.add_argument('--tune', action='store_true')
    args = parser.parse_args()
    main(tune=args.tune)