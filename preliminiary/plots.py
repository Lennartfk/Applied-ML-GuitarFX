import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import os
import librosa
import glob
import pandas as pd
import seaborn as sns

folder_path = r'C:\Users\lenna\Documents\RUG\Jaar 2\Periode 2b\Applied Machine Learning\Project (AML)\Datasets\IDMT-SMT-AUDIO-EFFECTS\Gitarre monophon\Samples'  # Adjust the path accordingly

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

for file in wav_files:
	y, sr = librosa.load(file, sr=None)
	mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
	mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
	flattened_spectogram = mel_spectrogram_db.flatten()
	features.append(flattened_spectogram)

	file_name = os.path.basename(file)
	effect_code = file_name.split('-')[2][1:3]
	effect_label = effect_labels[effect_code]
	labels.append(effect_label)

features = np.array(features)

tsne = TSNE(n_components=2)
tsne_result = tsne.fit_transform(features)

df = pd.DataFrame(tsne_result, columns=['comp1', 'comp2'])
df['Effect'] = labels

plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='comp1', y='comp2', hue='Effect')
plt.title('t-SNE of Guitar Effects')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend(title='Effect')
plt.tight_layout()
plt.show()


