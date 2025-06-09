import streamlit as st
import requests
import io

API_URL = "http://127.0.0.1:8000/predict"

st.title("🎸 Guitar Effect Classifier")

st.markdown("""
Upload one or more `.wav` audio files containing guitar effects.  
Our model will predict the confidence scores for each effect.
""")

uploaded_files = st.file_uploader("Upload guitar audio files", type=["wav"], accept_multiple_files=True)

if uploaded_files:
    if st.button("Classify Audio Files"):
        files = [('audio_files', (file.name, file, 'audio/wav')) for file in uploaded_files]

        with st.spinner("Analyzing audio... this may take a few seconds"):
            try:
                response = requests.post(API_URL, files=files)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                st.error(f"API request failed: {e}")
            else:
                data = response.json()
                classes = ['Chorus', 'Distortion', 'Feedback Delay', 'Flanger',
                           'No Effect', 'Overdrive', 'Phaser', 'Reverb',
                           'Slapback Delay', 'Tremolo', 'Vibrato']

                for prediction in data["predictions"]:
                    st.subheader(f"File: {prediction['file_name']}")
                    matching_files = [f for f in uploaded_files if f.name == prediction['file_name']]
                    if matching_files:
                        audio_bytes = matching_files[0].read()
                        st.audio(audio_bytes, format='audio/wav')
                        matching_files[0].seek(0)
                    for conf in prediction["confidences"]:
                        effect = conf["effect"]
                        confidence = conf["confidence"]
                        st.progress(int(confidence * 100))
                        st.write(f"**{effect}:** {confidence:.2%}")

                st.success("Prediction done")
else:
    st.info("Upload one or more guitar audio `.wav` files to get started.")
