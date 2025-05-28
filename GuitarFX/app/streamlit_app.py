import streamlit as st
import numpy as np
from GuitarFX.io.model_io import load_model
from GuitarFX.data.preprocessing import PreProcessing
from GuitarFX.models.svm import get_features

st.title("Guitar FX SVM Classifier")

# Load the model
model, scaler, label_encoder = load_model("saved_models/svm_model.pkl")

# File uploader
uploaded_file = st.file_uploader("Upload a guitar audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save file to disk if needed or read directly (optional)
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.read())

    # Process features (assumes get_features can handle one file)
    X, _, _, _ = get_features(
        dataset_paths=["temp_audio.wav"],
        read_csv=False
    )
    X_scaled = scaler.transform(X)

    # Predict
    probs = model.predict_proba(X_scaled)
    pred = np.argmax(probs, axis=1)
    pred_label = label_encoder.inverse_transform(pred)

    st.write("### Prediction:", pred_label[0])
    st.write("### Probabilities:", dict(zip(label_encoder.classes_, probs[0])))
