import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os

from GuitarFX.features.cnn_features import CNNFeatureExtractor
from GuitarFX.models.Guitar2dCNN import GuitarEffectCNN
from GuitarFX.metrics.metrics import ModelMetrics

def main():
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

    # Train/test split (no stratify for multilabel)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=23)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=23)

    num_classes = y.shape[1]

    model = GuitarEffectCNN(num_classes=num_classes, label_smoothing=0.1)

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    # Train model
    model.train(
        train_dataset=(X_train, y_train),
        val_dataset=(X_val, y_val),
        epochs=30,
        learning_rate=0.001,
        batch_size=64,
        callbacks=[early_stop]
    )

    # Save model and history
    model.save("models/cnn_onehot_fixedlr.h5")

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
    main()