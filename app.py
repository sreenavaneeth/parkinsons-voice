import streamlit as st
import sounddevice as sd
import soundfile as sf
import librosa
import numpy as np
import joblib
import tempfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd, requests, io

st.title("🧠 Parkinson’s Voice Detector")
st.write("Record your voice and get a Parkinson’s risk estimate!")

# ---- train model from UCI dataset ----
@st.cache_resource
def train_model():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    df = pd.read_csv(io.BytesIO(requests.get(url).content))
    X = df.drop(columns=["name","status"])
    y = df["status"]
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(Xs, y)
    return model, scaler

model, scaler = train_model()

# ---- record voice ----
duration = st.slider("Recording duration (s)", 3, 10, 5)
if st.button("🎤 Record"):
    st.info("Recording... speak a clear vowel like 'aaaa'.")
    fs = 22050
    rec = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp.name, rec, fs)
    st.audio(tmp.name)
    y, sr = librosa.load(tmp.name, sr=fs)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=22)
    feat = np.hstack([np.mean(mfcc,axis=1), np.var(mfcc,axis=1)])
    feat = np.pad(feat, (0, max(0,44-len(feat))))[:44]
    feat = scaler.transform([np.tile(feat, int(np.ceil(model.n_features_in_/44)))[:model.n_features_in_]])
    prob = model.predict_proba(feat)[0,1]
    st.metric("Risk probability", f"{prob:.2f}")
    st.write("🟢 Likely healthy" if prob<0.5 else "🔴 Possible risk")
