import librosa
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import glob
from sklearn.neighbors import KNeighborsClassifier

from tkinter import *
from tkinter.ttk import *
import time
import threading

# Define the mapping
effect_labels = { 
    '11': 'No Effect',
    '12': 'No Effect, Amp Simulation',
    '21': 'Feedback Delay',
    '22': 'Slapback Delay',
    '23': 'Reverb',
    '31': 'Chorus',
    '32': 'Flanger',
    '33': 'Phaser',
    '34': 'Tremolo',
    '35': 'Vibrato',
    '41': 'Distortion',
    '42': 'Overdrive'
}

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.mean(chroma, axis=1),
        np.mean(spec_contrast, axis=1),
        np.mean(zcr, axis=1),
        np.mean(rms, axis=1)
    ])
    return features

# Load dataset
X = []
y = []

folder_path = r'C:\Users\daan3\OneDrive\Documenten\repos\Applied-ML-GuitarFX\datasets\IDMT-SMT-AUDIO-EFFECTS\Gitarre monophon\Samples'  # Adjust the path accordingly

wav_files = glob.glob(os.path.join(folder_path, '**', '*.wav'), recursive=True)

n = len(wav_files)
t = 0
for file in wav_files:
    features = extract_features(file)
    X.append(features)

    file_name = os.path.basename(file)
    effect_code = file_name.split('-')[2][1:3]
    effect_label = effect_labels[effect_code]
    y.append(effect_label)
    t += 1
    print(f"Processed ({t}/{n})")

X = np.array(X)
y = np.array(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

svm_model = SVC(kernel='rbf', C=10, gamma='scale')
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"SVM (RBF) Accuracy: {acc * 100:.2f}%")