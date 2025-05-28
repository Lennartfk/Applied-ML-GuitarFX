"""
deployment.py

Run "fastapi dev fastapi dev .\GuitarFX\api\api.py in your terminal to start a
local API server at 127.0.0.1:8000. Ensure that a Keras model is saved in the
models/ directory. If you have your own trained CNN model, update the
model_path variable to point to it. Ensure you have installed fastapi by
installing fastapi[standard] in your virtual environment venv or conda.

Example request I did on the /predict endpoint:
curl -X POST "http://127.0.0.1:8000/predict" -F "audio_files=@\"C:/Users/daan3/OneDrive/Documenten/repos/Applied-ML-GuitarFX/datasets/IDMT-SMT-AUDIO-EFFECTS/Gitarre monophon/Chorus/G61-40100-3311-28081.wav\""
"""
from ..features.cnn_features import CNNFeatureExtractor

from typing import List

from pydantic import BaseModel
from tensorflow.keras.models import load_model
from starlette.responses import RedirectResponse
from fastapi import FastAPI, UploadFile, HTTPException, File

model_path = "models/guitar_effect_cnn_1_epoch.h5"
try:
    model = load_model(model_path)
except OSError:
    raise RuntimeError(f"Failed to load model from {model_path}")


class EffectConfidence(BaseModel):
    effect: str = "Chorus"
    confidence: float = 0.3


class EffectPrediction(BaseModel):
    file_name: str = "music_effect.wav"
    confidences: List[EffectConfidence]


class EffectPredictionResponse(BaseModel):
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
async def root():
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

    Add {'audio_file': audio_file} to send request. Add multiple bodies to
    predict for multiple audio files.
    """,
    response_model=EffectPredictionResponse,
    response_description="""
    Predicted confidence scores for each guitar effect, for every submitted
    audio file. Confidence values range from 0 (no confidence) to 1 (high
    confidence).
    """
)
async def predict_cnn(audio_files: UploadFile | List[UploadFile] = File(...)):
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
        raise HTTPException(status_code=400, detail=f"Error processing files: {e}")

    predictions = model.predict(X)

    results = []
    classes = ['Chorus', 'Distortion', 'Feedback Delay', 'Flanger',
               'No Effect', 'Overdrive', 'Phaser', 'Reverb',
               'Slapback Delay', 'Tremolo', 'Vibrato']

    for file_name, prediction in zip(files_names, predictions):
        confidences = [EffectConfidence(effect=effect, confidence=conf) for effect, conf in zip(classes, prediction)]
        results.append(EffectPrediction(file_name=file_name, confidences=confidences))

    return EffectPredictionResponse(predictions=results)
