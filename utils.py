# utils.py
import numpy as np
import librosa

def extract_features(audio_path, sr=16000):
    audio, sr = librosa.load(audio_path, sr=sr)
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
    return np.hstack([mfccs, chroma, mel])
