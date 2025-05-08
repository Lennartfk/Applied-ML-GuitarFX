import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import os
import librosa
import glob
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

from tkinter import *
from tkinter.ttk import *
import time
import threading

class ProgressWindow:
    def __init__(self, master, n_files: int):
        self.n_files = n_files
        self.current_n_file = 0

        self.percent = StringVar()
        self.text = StringVar()

        self.bar = Progressbar(master, orient=HORIZONTAL, length=300, mode='determinate')
        self.bar.pack(pady=10)

        Label(master, textvariable=self.percent).pack()
        Label(master, textvariable=self.text).pack()

        progress = (self.current_n_file / self.n_files) * 100
        self.bar['value'] = progress
        self.percent.set(f"{int(progress)}%")
        self.text.set(f"Processing: Starting... ({self.current_n_file} / {self.n_files})")
        window.update_idletasks()

    def update(self, file_name):
        self.current_n_file += 1
        progress = (self.current_n_file / self.n_files) * 100
        self.bar['value'] = progress
        self.percent.set(f"{int(progress)}%")
        self.text.set(f"Processing: {file_name}. ({self.current_n_file} / {self.n_files})")
        window.update_idletasks()

folder_path = r'C:\Users\daan3\OneDrive\Documenten\repos\Applied-ML-GuitarFX\datasets\IDMT-SMT-AUDIO-EFFECTS\Gitarre monophon\Samples'  # Adjust the path accordingly

wav_files = glob.glob(os.path.join(folder_path, '**', '*.wav'), recursive=True)

features = []
labels = []
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

def process_files():
    progress = ProgressWindow(window, len(wav_files))
    for file in wav_files:
        y, sr = librosa.load(file, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs = mfccs.T
        features.append(mfccs)

        file_name = os.path.basename(file)
        effect_code = file_name.split('-')[2][1:3]
        effect_label = effect_labels[effect_code]
        labels.append(effect_label)

        progress.update(file_name)
    
    window.after(500, window.destroy)

def start():
    threading.Thread(target=process_files).start()

window = Tk()
window.title("File Processing Progress")
start()
window.mainloop()

# features_flat = [np.mean(f, axis=0) for f in features]

# scaler = StandardScaler()
# features_scaled = scaler.fit_transform(features_flat)

# tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
# tsne_results = tsne.fit_transform(features_scaled)

# df = pd.DataFrame()
# df['tsne-2d-one'] = tsne_results[:,0]
# df['tsne-2d-two'] = tsne_results[:,1]
# df['label'] = labels

# plt.figure(figsize=(12,8))
# sns.scatterplot(
#     x='tsne-2d-one', y='tsne-2d-two',
#     hue='label',
#     palette=sns.color_palette("hsv", len(set(labels))),
#     data=df,
#     legend="full",
#     alpha=0.7
# )
# plt.title("t-SNE Visualization of Audio Effects (MFCC Features)")
# plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Flatten features (mean + std of MFCCs)
X = np.array([np.concatenate([np.mean(f, axis=0), np.std(f, axis=0)]) for f in features])

# Step 2: Encode string labels to integers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Step 3: Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Train KNN
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

# Step 6: Evaluate
y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"KNN Accuracy: {acc:.2%}")

# Step 7: Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=label_encoder.classes_, xticks_rotation=45)
plt.title("KNN Confusion Matrix")
plt.tight_layout()
plt.show()
