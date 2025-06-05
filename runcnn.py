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

def sample_balanced_subset(X, y, fraction=0.2, seed=23):
    np.random.seed(seed)
    num_classes = y.shape[1]
    total_samples = int(len(X) * fraction)
    samples_per_class = total_samples // num_classes

    indices = []
    for i in range(num_classes):
        class_indices = np.where(y[:, i] == 1)[0]
        chosen = np.random.choice(class_indices, samples_per_class, replace=False)
        indices.extend(chosen)

    np.random.shuffle(indices)
    return X[indices], y[indices]

def main(tune=False):
    dataset_paths = [
        r"C:\Users\lenna\Documents\RUG\Jaar 2\Periode 2b\Applied Machine Learning\Project (AML)\Datasets\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS\Gitarre monophon",
        r"C:\Users\lenna\Documents\RUG\Jaar 2\Periode 2b\Applied Machine Learning\Project (AML)\Datasets\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS\Gitarre monophon2",
        r"C:\Users\lenna\Documents\RUG\Jaar 2\Periode 2b\Applied Machine Learning\Project (AML)\Datasets\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS\Gitarre polyphon",
        r"C:\Users\lenna\Documents\RUG\Jaar 2\Periode 2b\Applied Machine Learning\Project (AML)\Datasets\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS\Gitarre polyphon2"
    ]

    extractor = CNNFeatureExtractor(dataset_paths)
    X, y, label_names = extractor.get_cnn_features(read_file=True, filename="cnn_scaled_onehot.npz")
    X = X[..., np.newaxis]

    print(f"Extracted features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    # Split data
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=23)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=23)

    num_classes = y.shape[1]
    model = GuitarEffectCNN(num_classes=num_classes, label_smoothing=0.1)

    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, verbose=1)

    if tune:
        print("Starting hyperparameter tuning...")
        model.setup_tuner(max_epochs=20, directory='kt_dir', project_name='guitarfx_tuning')

        # Use 20% balanced subset for tuning to speed things up
        X_train_small, y_train_small = sample_balanced_subset(X_train, y_train, fraction=0.2, seed=23)

        model.search(
            train_dataset=(X_train_small, y_train_small),
            val_dataset=(X_val, y_val),
            epochs=15,
            batch_size=64,
            callbacks=[early_stop]
        )

    else:
        print("Training model without hyperparameter tuning...")

        with open("models/cnn_hyper_tuner.pkl", "rb") as f:
            best_hp_values = pickle.load(f)

        hp = HyperParameters()
        for k, v in best_hp_values.items():
            hp.values[k] = v

        model.model = model.build_model(hp)
        model.train(
            train_dataset=(X_train, y_train),
            val_dataset=(X_val, y_val),
            epochs=30,
            batch_size=32,
            lr=0.0001,
            callbacks=[early_stop]
        )

    # Save model and history
    os.makedirs("models", exist_ok=True)
    model.save("models/cnn_hyper_full.h5")

    # Predict and evaluate
    y_pred_probs = model.predict(X_test)
    history = model.get_training_history()

    train_acc = history.get("accuracy") or history.get("acc")
    val_acc = history.get("val_accuracy") or history.get("val_acc")
    train_loss = history.get("loss")
    val_loss = history.get("val_loss")

    metrics = ModelMetrics(
        y_pred=y_pred_probs,
        y_actual=y_test,
        label_names=label_names,
        threshold=0.5,
        train_accuracy=train_acc,
        val_accuracy=val_acc,
        train_loss=train_loss,
        val_loss=val_loss,
    )
    metrics.report_all_results()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GuitarEffectCNN with optional hyperparameter tuning.")
    parser.add_argument('--tune', action='store_true')
    args = parser.parse_args()
    main(tune=args.tune)
