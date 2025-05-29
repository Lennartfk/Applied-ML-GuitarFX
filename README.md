# GuitarFX Audio Effect Classification with API

This project implements a CNN model to classify guitar audio effect using melspectrogram features. It includes a REST API to predict effects from audio features.

---

## Installation

### 1. Clone the repository

git clone https://github.com/Lennartfk/Applied-ML-GuitarFX
cd Applied-ML-GuitarFX

### 2. Install depencies
Use python 3.8+

pip install -r requirements.txt

### 3. Running the API
You can start the API server with:
fastapi dev .\GuitarFX\api\api.py

this starts a server at: 127.0.0.1:8000, and docs at: http://127.0.0.1:8000/docs