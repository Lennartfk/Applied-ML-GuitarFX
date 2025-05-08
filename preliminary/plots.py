import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import librosa
import glob
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import scipy.io.wavfile as wav
import time

folder_path = r'C:\Users\lenna\Documents\RUG\Jaar 2\Periode 2b\Applied Machine Learning\Project (AML)\Datasets\IDMT-SMT-AUDIO-EFFECTS\IDMT-SMT-AUDIO-EFFECTS'  

wav_files = glob.glob(os.path.join(folder_path, '**', '*.wav'), recursive=True)

features = []
labels = []

effect_labels = {
    '11': 'No Effect',
    '12': 'No Effect',
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


# Feature extraction parameters
n_mels = 64
hop_length = 512
n_fft = 1024
sr_target = 22050
max_duration = 2.0
max_length = int(np.ceil((sr_target * max_duration) / hop_length))

# Feature extraction loop
for file in tqdm(wav_files, desc="Processing Audio Files"):
    y, sr = librosa.load(file, sr=sr_target, duration=max_duration)

    # Extract log-mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
    log_mel = librosa.power_to_db(mel)

    # Pad or truncate to fixed length
    if log_mel.shape[1] < max_length:
        pad_width = max_length - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant')
    else:
        log_mel = log_mel[:, :max_length]

    # Flatten for t-SNE
    features.append(log_mel.flatten())

    # Extract label
    file_name = os.path.basename(file)
    effect_code = file_name.split('-')[2][1:3]
    effect_label = effect_labels[effect_code]
    labels.append(effect_label)

# Convert and scale
features = np.array(features)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

print("Feature shape (flat):", scaled_features.shape)

# t-SNE
start_time = time.time()

tsne = TSNE(n_components=2)
tsne_result = tsne.fit_transform(scaled_features)

df_tsne = pd.DataFrame(tsne_result, columns=['comp1', 'comp2'])
df_tsne['Effect'] = labels

end_time = time.time()

elapsed_time = end_time - start_time
print(f"t-SNE took {elapsed_time:.2f} seconds to run")

# Plot t-SNE
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df_tsne, x='comp1', y='comp2', hue='Effect')
plt.title('t-SNE of Guitar Effects')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend(title='Effect')
plt.tight_layout()
plt.show()

# histogram
plt.figure(figsize=(12, 6))
sns.histplot(features[0], kde=True)
plt.title('Distribution of MFCC Features (Mean of Coefficients)')
plt.xlabel('MFCC Value')
plt.ylabel('Frequency')
plt.show()

# Correlation matrix
correlation_matrix = pd.DataFrame(scaled_features).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of MFCC Features')
plt.show()

# Class distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df_tsne, x='Effect')
plt.title('Class Distribution of Guitar Effects')
plt.xticks(rotation=90)
plt.show()


