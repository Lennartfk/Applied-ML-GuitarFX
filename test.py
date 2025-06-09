from GuitarFX.features.cnn_features import CNNFeatureExtractor
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow_addons as tfa
from tensorflow.keras.metrics import AUC, Precision, Recall


# Load model
model = load_model("models/cnn_test.h5", custom_objects={
    "AdamW": tfa.optimizers.AdamW,
    "AUC": AUC,
    "Precision": Precision,
    "Recall": Recall,
    })

# Load audio bytes from file
with open("C:/Users/lenna/Documents/RUG/Jaar 2/Periode 2b/Applied Machine Learning/Project (AML)/Datasets/IDMT-SMT-AUDIO-EFFECTS/IDMT-SMT-AUDIO-EFFECTS/IDMT-SMT-AUDIO-EFFECTS/Gitarre monophon/Gitarre monophon/Samples/Distortion/G73-64505-4411-37732.wav", "rb") as f:
    audio_bytes = f.read()

preprocessor = CNNFeatureExtractor(dataset_paths=None)
X = preprocessor.extract_mel_for_prediction(audio_bytes)  # shape: (1, 128, 128, 1)

preds = model.predict(X)
print("Predictions:", preds)