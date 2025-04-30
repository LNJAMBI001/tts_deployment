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

# Fixed Google Drive URLs for direct downloading
MODEL_DRIVE_URL = "https://drive.google.com/uc?id=1-01P3elJi2U1S2yMkBXVsGiBMVwJNhzs"
# Corrected URL format for tokenizer - convert from view to download link
TOKENIZER_DRIVE_URL = "https://drive.google.com/uc?id=185z5i6ZkmeMgk7KcAiJ5K3zzcfB2HSCr"

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
tokenizer_downloaded = download_file(TOKENIZER_DRIVE_URL, TOKENIZER_PATH)

# Check if both files downloaded successfully
if not model_downloaded or not tokenizer_downloaded:
    st.error("Failed to download required files. Please check the download URLs.")
    st.stop()

# Load tokenizer
@st.cache_resource
def load_tokenizer():
    if os.path.isfile(TOKENIZER_PATH):
        try:
            with open(TOKENIZER_PATH, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Error loading tokenizer: {e}")
            return None
    else:
        st.error("Tokenizer file not found. Please check the download URL.")
        return None

# Simplified phonemization function as a fallback if piper-phonemize fails
def simple_phonemize(text, voice="en-us"):
    """Very basic phonemization fallback"""
    # This is a simplified placeholder. In production, implement proper phonemization
    # or ensure piper-phonemize is correctly installed
    tokenizer = load_tokenizer()
    if tokenizer is None:
        # Fallback if tokenizer failed to load
        chars = [ord(c) for c in text]
        return [[chars]]
    
    # Use tokenizer to convert text to phoneme IDs
    # Adjust this part based on your tokenizer's actual interface
    try:
        # This is a placeholder - replace with actual tokenizer usage
        phonemes = tokenizer.encode(text)
        return [[phonemes]]
    except:
        # Ultimate fallback
        chars = [ord(c) for c in text]
        return [[chars]]

# ------------------------
# Load model
# ------------------------
@st.cache_resource
def load_model():
    if os.path.isfile(MODEL_PATH):
        try:
            return ort.InferenceSession(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.error("Model file not found. Please check the download URL.")
        return None

session = load_model()

# ------------------------
# UI
# ------------------------
text = st.text_area("Enter text to synthesize:", height=150)
voice = st.selectbox("Voice", ["en-us"])  # More voices can be added

# Debug information
with st.expander("Debug Information"):
    st.write(f"Model file exists: {os.path.isfile(MODEL_PATH)}")
    st.write(f"Tokenizer file exists: {os.path.isfile(TOKENIZER_PATH)}")
    st.write(f"Model directory contents: {os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else 'Directory not found'}")

if st.button("🎤 Generate Speech") and text.strip():
    if not session:
        st.error("Model could not be loaded. Please check the error messages above.")
        st.stop()
        
    try:
        # Try to import and use piper_phonemize
        try:
            from piper_phonemize import phonemize_espeak
            phonemes = phonemize_espeak(text, voice)
            st.success("Using piper-phonemize for phonemization")
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
