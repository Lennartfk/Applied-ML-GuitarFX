import librosa
import numpy as np    
from GuitarFX.data.preprocessing import PreProcessing
import os

class FeatureExtractor(PreProcessing):
	"""Handle the feature extraction from the audio dataset."""
	
	def __init__(self, dataset_paths):
		self.dataset_paths = dataset_paths

	def extract_mean_features(self, y, sr):
		"""
		These are the features we use for the competetive baseline model,
		which is support vector machine with radial basis function (RBF)
		kernel. Can also be used for other model that can only one-dimensional
		data.
		"""
		mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
		chromatogram = librosa.feature.chroma_stft(y=y, sr=sr)
		spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
		zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
		room_mean_square = librosa.feature.rms(y=y)

		features = np.concatenate([
			np.mean(mfcc, axis=1),
			np.mean(chromatogram, axis=1),
			np.mean(spectral_contrast, axis=1),
			np.mean(zero_crossing_rate, axis=1),
			np.mean(room_mean_square, axis=1)
		])

		# feature_names = (
		#     [f'mfcc_{i+1}' for i in range(20)]
		#     [f'chroma_{i+1}' for i in range(chromatogram.shape[0])] +
		#     [f'spectral_contrast_{i+1}' for i in range(spectral_contrast.shape[0])] +
		#     ['zero_crossing_rate'] +
		#     ['rms']
		# )

		# print(feature_names)

		return features

	def execute_mean_features(self, max_samples_per_classifier = None):
		"""
		Extract the mean features for each file in dataset

		Inputs:
			max_samples_per_classifier (int): Limit the sample amount per
			classifier to limit the amount of time spent for pre-processing.
		"""
		feature_names = ['mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7', 'mfcc_8', 'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12', 'mfcc_13', 'mfcc_14', 'mfcc_15', 'mfcc_16', 'mfcc_17', 'mfcc_18', 'mfcc_19', 'mfcc_20', 'chroma_1', 'chroma_2', 'chroma_3', 'chroma_4', 'chroma_5', 'chroma_6', 'chroma_7', 'chroma_8', 'chroma_9', 'chroma_10', 'chroma_11', 'chroma_12', 'spectral_contrast_1', 'spectral_contrast_2', 'spectral_contrast_3', 'spectral_contrast_4', 'spectral_contrast_5', 'spectral_contrast_6', 'spectral_contrast_7', 'zero_crossing_rate', 'rms']
		label_names = []

		X = []
		y = []

		total_files = 20592
		current_number_file = 0

		for dataset_path in self.dataset_paths:
			label_dicts = os.listdir(dataset_path)
			label_names.extend(label_dicts)

			for label_dict in label_dicts:
				label_dict_path = os.path.join(dataset_path, label_dict)

				for wav_file in os.listdir(label_dict_path):
					current_number_file += 1
					try:
						wav_file_path = os.path.join(label_dict_path, wav_file)

						signal, sr = self.signal_processing(wav_file_path)
						features = self.extract_mean_features(signal, sr)

						X.append(features)
						y.append(label_dict)

					except Exception as e:
						print("Couldn't load file.")

					print(f"Processed file ({current_number_file}/{total_files})")

					if max_samples_per_classifier is not None:
						if current_number_file % max_samples_per_classifier == 0:
							break

		X = np.array(X)
		y = np.array(y)

		return X, y, feature_names, label_names