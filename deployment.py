from typing import List

import PIL
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from starlette.responses import RedirectResponse


class GuitarEffectConfidence(BaseModel):
    effect: str = "Chorus"
    confidence: float = 0.3


class GuitarEffectPredictions(BaseModel):
    predictions: List[GuitarEffectConfidence]


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
The model is not build to classify multiple guitar effects in one audio file.

## Types of guitar effects
The effects included in the IDMT-SMT-AUDIO-EFFECTS are:
Chorus, Distortion, EQ, FeedbackDelay, Flanger, NoFX, Overdrive, Phaser, Reverb
, SlapbackDelay, Tremolo and Vibrate. For any prediction-related request, the
audio file should include one of these guitar effects, otherwise the models
in our endpoints will return unusable and incorrect predictions.

## Limitations
The model was trained on a highly controlled dataset, which can lead to
inaccurate predictions when processing poorly recorded audio samples of
guitar effects played on a guitar. To mitigate this issue and enhance the
model's generalization, we introduced noise into the training data.
However, this does not completely eliminate the problem.
    """,
    version="alpha"
)


@app.get("/", description="Root endpoint that redirects to the documentation.")
async def root():
    return RedirectResponse(url='/docs')


@app.post("/models/svm", description="""Guitar effect classifier endpoint.
          Add {'audio_file': audio_file} to send request. Audio file should be
          a guitar effect which is played on a guitar. Returns confidences of
          each effect.""",
          response_model=GuitarEffectPredictions,
          response_description="Confidence for each guitar effect. Confidence"
          "Ranges from 0 to 1.")
async def predict_svm():
    return GuitarEffectPredictions(predictions=[GuitarEffectConfidence()])


@app.post("/models/cnn", description="""Guitar effect classifier endpoint.
          Add {'audio_file': audio_file} to send request. Audio file should be
          a guitar effect which is played on a guitar. Returns confidences of
          each effect.""",
          response_model=GuitarEffectPredictions,
          response_description="Confidence for each guitar effect. Confidence"
          "Ranges from 0 to 1.")
async def predict_cnn():
    return GuitarEffectPredictions(predictions=[GuitarEffectConfidence()])

"""
Run "fastapi dev deployment.py" in de terminal om een local server te openen.
"""
