# GuitarFX Audio Effect Classification with API

This project implements a CNN model to classify guitar audio effects using melspectrogram features. It also includes a REST API to predict effects from audio features.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Lennartfk/Applied-ML-GuitarFX
cd Applied-ML-GuitarFX
```

### 2. Install dependencies

We recommend using **Python 3.9.21**. If you use `conda`, execute the following commands:

```bash
conda create -n guitarfx python=3.9.21
conda activate guitarfx
pip install -r requirements.txt
```

### 3. Running the API

Start the API server by running:

```bash
fastapi dev .\GuitarFX\api\api.py
```

This will start a server at: `http://127.0.0.1:8000`

API documentation (Swagger UI) is available at:  
`http://127.0.0.1:8000/docs`

---

You can download a sample audio file to test the model from this Google Drive link:  
[Sample Audio Files](https://drive.google.com/drive/folders/10PvtgqYn_CtyYDe_oubWQDvIqLgPumFy?usp=drive_link)
