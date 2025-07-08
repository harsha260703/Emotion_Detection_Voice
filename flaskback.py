from flask import Flask, request, jsonify
import os
import librosa
import numpy as np
import joblib
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = joblib.load("emotion_model.pkl")

# Feature extraction
def extract_features(file_path, sr=16000):
    audio, sr = librosa.load(file_path, sr=sr)
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
    return np.hstack([mfccs, chroma, mel])

# Home route
@app.route("/")
def home():
    return "Voice Emotion Detection API is running."

# Predict route
@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided."}), 400

    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "Empty file name."}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        features = extract_features(file_path).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({"emotion": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
