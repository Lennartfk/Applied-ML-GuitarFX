import os
import glob
import numpy as np
from GuitarFX.features.cnn_features import CNNFeatureExtractor
from GuitarFX.models.Guitar2dCNN import GuitarEffectCNN
from GuitarFX.data.loading import extract_multilabel_from_filename
from GuitarFX.metrics.metrics import ModelMetrics 

def get_wav_files_recursive(folder_path):
    pattern = os.path.join(folder_path, "**", "*.wav")
    return glob.glob(pattern, recursive=True)

def get_features_and_labels_from_files(file_list):
    extractor = CNNFeatureExtractor([])
    X_list = []
    y_list = []
    for filepath in file_list:
        signal, sr = extractor.signal_processing(filepath)
        mel = extractor._extract_mel(signal, sr)
        X_list.append(mel)
        y_list.append(extract_multilabel_from_filename(os.path.basename(filepath)))
    X = np.array(X_list)[..., np.newaxis]  # Add channel dim
    y = np.array(y_list)
    label_names = extractor.class_names
    return X, y, label_names

def main():
    folder = r"C:\Users\lenna\Documents\Coding_Projects\multi_effect"
    wav_files = get_wav_files_recursive(folder)
    print(f"Found {len(wav_files)} wav files.")

    X, y_true, label_names = get_features_and_labels_from_files(wav_files)

    model = GuitarEffectCNN(num_classes=len(label_names))
    model.load("models/cnn_hyper_full.h5")

    y_pred_probs = model.predict(X)

    threshold = 0.5
    y_pred = (y_pred_probs >= threshold).astype(int)
    metrics = ModelMetrics(
        y_pred=y_pred_probs,
        y_actual=y_true,
        label_names=label_names,
        threshold=0.5,
    )

    metrics.report_all_results()
    
    for i, label in enumerate(label_names):
        if label in ["Distortion", "Reverb", "Overdrive"]:
            print(f"{label}:")
            print("True positives:", np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1)))
            print("False negatives:", np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 0)))
            print("False positives:", np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1)))
            print()


if __name__ == "__main__":
    main()
