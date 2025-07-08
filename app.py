import streamlit as st
import numpy as np
import librosa
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import os
import joblib

# Load model
model = joblib.load("emotion_model.pkl")

# Feature extraction function
def extract_features(audio, sr=16000):
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
    return np.hstack([mfccs, chroma, mel])

# Page Title
st.title("🎙️ Voice Emotion Detection App")
st.markdown("This app can **record your voice** or accept a `.wav` file to predict the emotion using a trained model.")

# Section 1: Upload Audio File
st.header("📂 Upload a .wav file")

uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        audio_path = tmp_file.name

    audio, sr = librosa.load(audio_path, sr=16000)
    features = extract_features(audio, sr).reshape(1, -1)
    prediction = model.predict(features)
    st.success(f"🎯 Predicted Emotion: **{prediction[0].capitalize()}**")

# Section 2: Record from Microphone
st.header("🎤 Record Using Microphone")
duration = st.slider("⏱️ Select duration (seconds)", 1, 10, 3)

if st.button("🔴 Record"):
    fs = 16000
    st.info("Recording... Speak into your mic.")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    st.success("✅ Recording complete!")

    temp_audio_path = os.path.join(tempfile.gettempdir(), "recorded.wav")
    write(temp_audio_path, fs, recording)

    st.audio(temp_audio_path, format="audio/wav")
    with open(temp_audio_path, "rb") as f:
        st.download_button("⬇️ Download Recording", f, file_name="recorded.wav")

    audio, sr = librosa.load(temp_audio_path, sr=16000)
    features = extract_features(audio, sr).reshape(1, -1)
    prediction = model.predict(features)
    st.success(f"🎯 Predicted Emotion: **{prediction[0].capitalize()}**")
