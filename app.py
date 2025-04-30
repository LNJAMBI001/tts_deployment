import os
import gdown
import streamlit as st
import numpy as np
import onnxruntime as ort
import pickle
import scipy.io.wavfile as wavfile
from pathlib import Path

st.set_page_config(page_title="Text to Speech App", layout="centered")
st.title("🗣️ Text to Speech with Piper TTS")

# ------------------------
# Download model files
# ------------------------
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")

MODEL_DRIVE_URL = "https://drive.google.com/uc?id=1-01P3elJi2U1S2yMkBXVsGiBMVwJNhzs"
TOKENIZER_DRIVE_URL = "https://drive.google.com/file/d/185z5i6ZkmeMgk7KcAiJ5K3zzcfB2HSCr/view?usp=drive_link"  

os.makedirs(MODEL_DIR, exist_ok=True)

# Function to safely download files
def download_file(url, output_path):
    try:
        if not os.path.isfile(output_path):
            with st.spinner(f"Downloading {os.path.basename(output_path)}..."):
                gdown.download(url, output_path, quiet=False)
                return True
        return True
    except Exception as e:
        st.error(f"Error downloading file: {e}")
        return False

# Download necessary files
model_downloaded = download_file(MODEL_DRIVE_URL, MODEL_PATH)

# Simplified phonemization function as a fallback if piper-phonemize fails
def simple_phonemize(text, voice="en-us"):
    """Very basic phonemization fallback"""
    # This is a simplified placeholder. In production, implement proper phonemization
    # or ensure piper-phonemize is correctly installed
    chars = [ord(c) for c in text]
    # Mock phoneme structure expected by the model
    return [[chars]]

# ------------------------
# Load model
# ------------------------
@st.cache_resource
def load_model():
    if os.path.isfile(MODEL_PATH):
        return ort.InferenceSession(MODEL_PATH)
    else:
        st.error("Model file not found. Please check the download URL.")
        return None

session = load_model()

# ------------------------
# UI
# ------------------------
text = st.text_area("Enter text to synthesize:", height=150)
voice = st.selectbox("Voice", ["en-us"])  # More voices can be added

st.info("Note: This is a simplified version. For full functionality, ensure the tokenizer.pkl file is properly configured.")

if st.button("🎤 Generate Speech") and text.strip() and session:
    try:
        # Try to import and use piper_phonemize
        try:
            from piper_phonemize import phonemize_espeak
            phonemes = phonemize_espeak(text, voice)
        except ImportError:
            st.warning("Using simplified phonemization as piper-phonemize is not available.")
            phonemes = simple_phonemize(text, voice)
        
        # Convert phonemes to input format expected by the model
        flat_phonemes = [item for sublist in phonemes for item in sublist]
        phoneme_ids = np.array([[int(p) for p in flat_phonemes]], dtype=np.int64)
        
        # Run inference
        inputs = {"phoneme_ids": phoneme_ids}
        outputs = session.run(None, inputs)
        
        # Assuming the model outputs audio waveform as the first output
        output = outputs[0]  
        
        # Save audio to WAV
        wav_path = os.path.join(MODEL_DIR, "output.wav")
        wavfile.write(wav_path, rate=22050, data=output.squeeze())
        
        st.success("✅ Audio generated!")
        st.audio(wav_path, format="audio/wav")
    except Exception as e:
        st.error(f"❌ Error during inference: {str(e)}")
        st.error("Please check if all model files are correctly downloaded and the input format is correct.")
