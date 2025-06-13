import os
import sys
import argparse
import pickle
import logging
from typing import List, Optional, Tuple
import librosa.display
import matplotlib.pyplot as plt
from collections import Counter

import numpy as np
from kerastuner.engine.hyperparameters import HyperParameters

from GuitarFX.io.model_io import print_class_distribution, load_hyperparameters, load_cnn_model
from GuitarFX.io.file_io import replace_all_trailing_ones_with_zeros, delete_samples_with_trailing_ones
from GuitarFX.data.preprocessing import PreProcessing
from GuitarFX.features.cnn_features import CNNFeatureExtractor
from GuitarFX.models.Guitar2dCNN import GuitarEffectCNN
from GuitarFX.metrics.metrics import ModelMetrics
from GuitarFX.io.model_io import save_cnn_model

# Constants
DEFAULT_DATASET_PATHS = [
    r"C:\Users\lenna\Documents\RUG\Jaar 2\Periode 2b\Applied Machine Learning\Project (AML)\Datasets\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS\Gitarre monophon",
    r"C:\Users\lenna\Documents\RUG\Jaar 2\Periode 2b\Applied Machine Learning\Project (AML)\Datasets\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS\Gitarre monophon2",
    r"C:\Users\lenna\Documents\RUG\Jaar 2\Periode 2b\Applied Machine Learning\Project (AML)\Datasets\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS\Gitarre polyphon",
    r"C:\Users\lenna\Documents\RUG\Jaar 2\Periode 2b\Applied Machine Learning\Project (AML)\Datasets\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS\Gitarre polyphon2",
    r"C:\Users\lenna\Documents\Coding_Projects\multi_effect",

]

FEATURES_DIR = r"C:\Users\lenna\Documents\RUG\Jaar 2\Periode 2b\Applied Machine Learning\Project (AML)\Applied-ML-GuitarFX\data"
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "augmented_new.h5")
TUNER_PICKLE = os.path.join(MODEL_DIR, "multi_tuned_tuner.pkl")
SPLIT_FILE = os.path.join(FEATURES_DIR, "split_data.npz")

def setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    logging.basicConfig(level=level, format='[%(levelname)s] %(message)s')

def verify_dataset_paths(paths: List[str]) -> List[str]:
    return [p for p in paths if os.path.exists(p)]

def shuffle_dataset(X, y, seed=None):
    if seed is not None:
        np.random.seed(seed)
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]

def train_and_evaluate(
    model: GuitarEffectCNN,
    preprocessing: PreProcessing,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: List[str],
    tune: bool = False,
    use_kfold: bool = False
):
    if use_kfold:
        logging.info("Starting K-Fold training...")
        X = np.concatenate([X_train, X_val, X_test])
        y = np.concatenate([y_train, y_val, y_test])
        model.kfold_training(X, y, label_names, n_splits=5)
        return

    if tune:
        logging.info("Starting hyperparameter tuning...")
        model.setup_tuner(max_epochs=20, directory='kt_dir', project_name='augmented_tuning')
        model.search(
            train_dataset=(X_train, y_train),
            val_dataset=(X_val, y_val),
            epochs=15,
            batch_size=64,
            callbacks=[]
        )
    else:
        logging.info("Training model...")
        model.train(
            train_dataset=(X_train, y_train),
            val_dataset=(X_val, y_val),
            epochs=30,
            batch_size=32,
        )

    os.makedirs(MODEL_DIR, exist_ok=True)
    save_cnn_model(model, MODEL_FILE)
    logging.info(f"Model saved to {MODEL_FILE}")

    y_pred = model.predict_classes(X_test, threshold=0.5)
    history = model.get_training_history()

    metrics = ModelMetrics(
        y_pred=y_pred,
        y_actual=y_test,
        label_names=label_names,
        threshold=0.5,
        train_auc=history.get("auc"),
        val_auc=history.get("val_auc"),
        train_loss=history.get("loss"),
        val_loss=history.get("val_loss"),
    )
    metrics.report_all_results()
    metrics.classification_metrics_report_single_vs_multi()

def main(tune: bool = False, use_kfold: bool = False, dataset_paths: Optional[List[str]] = None, load_model: Optional[str] = None, history_path: Optional[str] = None) -> None:
    dataset_paths = verify_dataset_paths(dataset_paths or DEFAULT_DATASET_PATHS)

    preprocessing = PreProcessing(dataset_paths)
    extractor = CNNFeatureExtractor(dataset_paths)

    if os.path.exists(SPLIT_FILE):
        logging.info("Loading cached split...")
        split_data = np.load(SPLIT_FILE, allow_pickle=True)
        X_train_fp, X_val_fp, X_test_fp = split_data["X_train_fp"], split_data["X_val_fp"], split_data["X_test_fp"]
        y_train, y_val, y_test = split_data["y_train"], split_data["y_val"], split_data["y_test"]
        label_names = split_data["label_names"].tolist()
    else:
        logging.info("Performing stratified split...")
        file_paths, labels, label_names = extractor.get_audio_paths_and_labels()
        X_train_fp, X_val_fp, X_test_fp, y_train, y_val, y_test = preprocessing.iterative_stratified_split(
            np.array(file_paths), labels, test_size=0.2, val_size=0.25, seed=23
        )
        
        train_indices = preprocessing.subsample_iterative_stratification(X_train_fp, y_train, fraction=0.7, seed=23)
        X_train_fp = X_train_fp[train_indices]
        y_train = y_train[train_indices]

        os.makedirs(FEATURES_DIR, exist_ok=True)
        np.savez_compressed(
            SPLIT_FILE,
            X_train_fp=X_train_fp,
            X_val_fp=X_val_fp,
            X_test_fp=X_test_fp,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            label_names=np.array(label_names, dtype=object)
        )

    X_train, X_val, X_test, y_train, y_val, y_test, label_names = extractor.get_cnn_features(
        output_dir=FEATURES_DIR,
        split_data=(X_train_fp, X_val_fp, X_test_fp, y_train, y_val, y_test),
        label_names=label_names,
        augment_train=True
    )

    # Clean + shuffle
    X_train, y_train, _ = delete_samples_with_trailing_ones(X_train, y_train)
    X_val, y_val, _ = delete_samples_with_trailing_ones(X_val, y_val)
    X_test, y_test, _ = delete_samples_with_trailing_ones(X_test, y_test)

    X_train, y_train = shuffle_dataset(X_train, y_train, seed=23)
    X_val, y_val = shuffle_dataset(X_val, y_val, seed=23)
    X_test, y_test = shuffle_dataset(X_test, y_test, seed=23)

    model = GuitarEffectCNN(input_shape=(128, 128, 1), num_classes=len(label_names))

    if load_model:
        model, history, _ = load_cnn_model(load_model)
        y_pred_probs = model.predict(X_test[..., np.newaxis])
        y_pred = (y_pred_probs > 0.5).astype(int)
        metrics = ModelMetrics(
            y_pred=y_pred,
            y_pred_probs=y_pred_probs,
            y_actual=y_test,
            label_names=label_names,
            threshold=0.5,
            train_auc=history.get('auc'),
            val_auc=history.get('val_auc'),
            train_loss=history.get('loss'),
            val_loss=history.get('val_loss'),
        )
        metrics.report_all_results()
        metrics.classification_metrics_report_single_vs_multi()
        return
    
    logging.info(f"Training with tune={tune}, use_kfold={use_kfold}...")
    train_and_evaluate(
        model=model,
        preprocessing=preprocessing,
        X_train=X_train[..., np.newaxis],
        y_train=y_train,
        X_val=X_val[..., np.newaxis],
        y_val=y_val,
        X_test=X_test[..., np.newaxis],
        y_test=y_test,
        label_names=label_names,
        tune=tune,
        use_kfold=use_kfold
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GuitarEffectCNN")
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--kfold', action='store_true')
    parser.add_argument('--dataset_paths', nargs='*', default=None)
    parser.add_argument('-v', '--verbose', action='count', default=0)
    parser.add_argument('--load_model', type=str)
    parser.add_argument('--history', type=str)
    args = parser.parse_args()

    setup_logging(args.verbose)
    main(tune=args.tune, use_kfold=args.kfold, dataset_paths=args.dataset_paths, load_model=args.load_model, history_path=args.history)
