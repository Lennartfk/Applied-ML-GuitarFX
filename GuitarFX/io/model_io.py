import os
from typing import List

import numpy as np
import joblib
import pickle
import tensorflow as tf


def save_svm_model(model, scaler, label_encoder, path):
    """
    Save SVM model, scaler, and label encoder.

    Args:
        model: Trained SVM model object.
        scaler: Scaler object used for feature normalization.
        label_encoder: Label encoder object for target labels.
        path (str): File path to save the model. Scaler and label encoder
            are saved to related paths by replacing "model" in this path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    joblib.dump(scaler, path.replace("model", "scaler"))
    joblib.dump(label_encoder, path.replace("model", "label_encoder"))
    print("SVM model, scaler, and label encoder saved to" +
          f"{os.path.dirname(path)}")


def load_svm_model(path):
    model = joblib.load(path)
    scaler = joblib.load(path.replace("model", "scaler"))
    label_encoder = joblib.load(path.replace("model", "label_encoder"))
    return model, scaler, label_encoder


def save_cnn_model(cnn_obj, model_path):
    """
    cnn_obj: instance of GuitarEffectCNN class
    Saves the keras model + history + tuner best HP
    """
    cnn_obj.model.save(model_path)
    print(f"CNN model saved to {model_path}")

    # Save history
    history_path = os.path.splitext(model_path)[0] + "_history.pkl"
    with open(history_path, "wb") as f:
        pickle.dump(cnn_obj.history.history if cnn_obj.history else None, f)
    print(f"Training history saved to {history_path}")

    # Save tuner info if any
    tuner_path = os.path.splitext(model_path)[0] + "_tuner.pkl"
    with open(tuner_path, "wb") as f:
        pickle.dump(cnn_obj.best_hp.values if cnn_obj.best_hp else None, f)
    print(f"Tuner hyperparameters saved to {tuner_path}")


def load_cnn_model(model_path):
    model = tf.keras.models.load_model(model_path, compile=False)
    history_path = os.path.splitext(model_path)[0] + "_history.pkl"
    tuner_path = os.path.splitext(model_path)[0] + "_tuner.pkl"

    history = None
    best_hp = None

    if os.path.exists(history_path):
        with open(history_path, "rb") as f:
            history = pickle.load(f)
    if os.path.exists(tuner_path):
        with open(tuner_path, "rb") as f:
            best_hp = pickle.load(f)

    return model, history, best_hp


def print_class_distribution(y: np.ndarray, label_names: List[str]) -> None:
    """
    Print the distribution of each class in the multilabel dataset.
    """
    class_counts = y.sum(axis=0)
    total_samples = len(y)
    print(f"Total samples: {total_samples}")
    for i, label in enumerate(label_names):
        count = int(class_counts[i])
        percent = 100 * count / total_samples
        print(f"{label}: {count} samples ({percent:.2f}%)")


def load_hyperparameters(pickle_path):
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Trained tuner not found at: {pickle_path}")

    with open(pickle_path, "rb") as f:
        return pickle.load(f)
