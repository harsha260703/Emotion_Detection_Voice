import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

emotion_map = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}

def extract_features(audio, sr=16000):
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
    return np.hstack([mfccs, chroma, mel])

def build_dataset(base_path):
    X, y = [], []
    for actor in os.listdir(base_path):
        actor_path = os.path.join(base_path, actor)
        if os.path.isdir(actor_path):
            for file in os.listdir(actor_path):
                if file.endswith(".wav"):
                    emotion_code = file.split("-")[2]
                    emotion = emotion_map.get(emotion_code)
                    if emotion:
                        path = os.path.join(actor_path, file)
                        audio, sr = librosa.load(path, sr=16000)
                        features = extract_features(audio, sr)
                        X.append(features)
                        y.append(emotion)
    return np.array(X), np.array(y)

def train():
    dataset_path = "voicedetection/datasets"
    X, y = build_dataset(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    print(classification_report(y_test, model.predict(X_test)))
    joblib.dump(model, "emotion_model.pkl")

if __name__ == "__main__":
    train()
