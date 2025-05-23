import joblib
import os

def save_model(model, scaler, label_encoder, path="saved_models/custom_svm_model.joblib"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    joblib.dump(scaler, path.replace("model", "scaler"))
    joblib.dump(label_encoder, path.replace("model", "label_encoder"))
    print(f"Saved model, scaler, and label encoder to {os.path.dirname(path)}")

def load_model(path="saved_models/custom_svm_model.joblib"):
    model = joblib.load(path)
    scaler = joblib.load(path.replace("model", "scaler"))
    label_encoder = joblib.load(path.replace("model", "label_encoder"))
    return model, scaler, label_encoder
