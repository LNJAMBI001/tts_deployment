import os
import gdown
import streamlit as st
import numpy as np
import onnxruntime as ort
import pickle
from piper_phonemize import phonemize_espeak

st.set_page_config(page_title="Text to Speech App", layout="centered")
st.title("🗣️ Text to Speech with Piper TTS")

# ------------------------
# Download ONNX model
# ------------------------
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")
DRIVE_URL = "https://drive.google.com/uc?id=1-01P3elJi2U1S2yMkBXVsGiBMVwJNhzs"

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.isfile(MODEL_PATH):
    with st.spinner("Downloading model from Google Drive..."):
        gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)

# ------------------------
# Load model and tokenizer
# ------------------------
@st.cache_resource
def load_model():
    return ort.InferenceSession(MODEL_PATH)

@st.cache_resource
def load_tokenizer():
    with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "rb") as f:
        return pickle.load(f)

try:
    session = load_model()
    tokenizer = load_tokenizer()
except Exception as e:
    st.error(f"Error loading model/tokenizer: {e}")
    st.stop()

# ------------------------
# UI
# ------------------------
text = st.text_area("Enter text to synthesize:", height=150)
voice = st.selectbox("Voice", ["en-us"])  # More voices can be added

if st.button("🎤 Generate Speech") and text.strip():
    try:
        phonemes = tokenizer(text, voice=voice)
        flat_phonemes = [item for sublist in phonemes for item in sublist]
        phoneme_ids = np.array([[int(p) for p in flat_phonemes]], dtype=np.int64)

        inputs = {"phoneme_ids": phoneme_ids}
        output = session.run(None, inputs)[0]  # May need adjusting depending on model

        # Save audio to WAV (assuming float32 waveform)
        import scipy.io.wavfile as wavfile
        wav_path = os.path.join(MODEL_DIR, "output.wav")
        wavfile.write(wav_path, rate=22050, data=output.squeeze())

        st.success("✅ Audio generated!")
        st.audio(wav_path, format="audio/wav")
    except Exception as e:
        st.error(f"❌ Error during inference: {e}")
