from ..features.cnn_features import CNNFeatureExtractor

from typing import List, Union

import numpy as np
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import AUC, Precision, Recall
import tensorflow_addons as tfa
from starlette.responses import RedirectResponse
from fastapi import FastAPI, UploadFile, HTTPException, File

model_path = "models/augmented_new.h5"
try:
    model = load_model(model_path, custom_objects={
        "AdamW": tfa.optimizers.AdamW,
        "AUC": AUC,
        "Precision": Precision,
        "Recall": Recall,
        }
    )
except OSError:
    raise RuntimeError(f"Failed to load model from {model_path}")


class EffectConfidence(BaseModel):
    """
    Schema represents a single audio effect and its associated confidence
    score.
    """
    effect: str = "Chorus"
    confidence: float = 0.3


class EffectPrediction(BaseModel):
    """Schema contains the predicted effects for a specific audio file."""
    file_name: str = "music_effect.wav"
    confidences: List[EffectConfidence]


class EffectPredictionResponse(BaseModel):
    """Schema wraps a list of effect predictions for multiple audio files."""
    predictions: List[EffectPrediction]


app = FastAPI(
    title="Guitar Effect Classifier",
    summary="An API endpoint to classify guitar effects from an audio file." +
    "Trained on IDMT-SMT-AUDIO-EFFECTS",
    description="""
# An API endpoint to access a CNN and SVM trained on IDMT-SMT-AUDIO-EFFECTS.
## Model usage
The model is trained on 2 second wav audio files.
Consequently, it is designed to receive audio files of similair length. Thus,
it is not recommended to input an audio file any longer than this duration.
The model is build to classify single guitar effects only.

## Types of guitar effects
The model is trained on the IDMT-Audio-Effects dataset, that includes audio
effect recordings played by an electric guitar. The audio effects included in
the dataset are:
Chorus, Distortion, EQ, FeedbackDelay, Flanger, NoFX, Overdrive, Phaser, Reverb
, SlapbackDelay, Tremolo and Vibrate. For any prediction-related request, the
audio file should include one of these guitar effects, otherwise the models
in our endpoints will return unusable and incorrect predictions. For more
information please refer to the following link:
https://www.idmt.fraunhofer.de/en/publications/datasets/audio_effects.html

## Limitations
The model was trained on a highly controlled dataset, which can lead to
inaccurate predictions when processing poorly recorded audio samples of
guitar effects played on a guitar. To mitigate this issue and enhance the
model's generalization, we introduced noise into the training data.
However, this does not completely eliminate the problem.
    """,
    version="1.0.0"
)


@app.get("/", description="Root endpoint that redirects to the documentation.")
async def root() -> RedirectResponse:
    """
    Root endpoint that redirects to the documentation.
    """
    return RedirectResponse(url='/docs')


@app.post(
    "/predict",
    description="""
    Guitar effect classifier endpoint.

    Accepts one or multiple audio files containing a guitar effect specified
    in the "types of guitar effects section"

    The endpoint processes the audio files by converting them to mel
    spectograms, then predicting the confidence scores for each pre-defined
    guitar effect using a 2-dimensional CNN.

    Add {'audio_files': audio_file} to send request. Add multiple bodies to
    predict for multiple audio files.
    """,
    response_model=EffectPredictionResponse,
    response_description="""
    Predicted confidence scores for each guitar effect, for every submitted
    audio file. Confidence values range from 0 (no confidence) to 1 (high
    confidence).
    """
)
async def predict_cnn(
        audio_files: Union[UploadFile, List[UploadFile]] = File(...)
):
    """
    Predict audio guitar effect in uploaded audio file(s) using a multi-effect
    CNN.

    Args:
        audio_files (Union[UploadFile, List[UploadFile]]): One or more audio
        files.

    Returns:
        EffectPredictionResponse: JSON structured API containing the
        predictions of the CNN model.
    """
    if isinstance(audio_files, UploadFile):
        audio_files = [audio_files]

    X = []
    files_names = []

    try:
        for audio_file in audio_files:
            preprocessor = CNNFeatureExtractor(dataset_paths=None)
            bytes = await audio_file.read()

            mel_spec = preprocessor.extract_mel_for_prediction(bytes)

            X.append(mel_spec)
            files_names.append(audio_file.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error processing " +
                            f"files: {e}")

    X = [np.squeeze(x) for x in X]
    X = [np.expand_dims(x, axis=-1) for x in X]
    X = np.stack(X)

    predictions = model.predict(X)

    results = []
    classes = [
        "Feedback Delay", "Slapback Delay", "Reverb", "Chorus", "Flanger",
        "Phaser", "Tremolo", "Vibrato", "Distortion", "Overdrive", "No Effect"
    ]

    for file_name, prediction in zip(files_names, predictions):
        confidences = [
            EffectConfidence(effect=effect, confidence=conf)
            for effect, conf in zip(classes, prediction)
        ]
        results.append(EffectPrediction(
            file_name=file_name,
            confidences=confidences
        ))

    return EffectPredictionResponse(predictions=results)
