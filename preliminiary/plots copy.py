import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import os
import librosa
import glob
import pandas as pd
import seaborn as sns

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
        mfccs = mfccs.T  # Shape: (n_frames, n_mfcc)
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

mfccs = np.array(features)
mfccs = np.vstack(mfccs)
print(mfccs)
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
mfccs_2d = tsne.fit_transform(mfccs)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(mfccs_2d[:, 0], mfccs_2d[:, 1], c=np.linspace(0, 1, mfccs_2d.shape[0]), cmap='viridis')
plt.title("MFCCs Visualized with t-SNE")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.colorbar(label='Time')
plt.show()


