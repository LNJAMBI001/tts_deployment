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

# ------------------------
# Very simple text-to-phoneme conversion
# ------------------------
def simple_phonemize(text, language="en-us"):
    """
    Very basic phonemization that just converts text to ASCII values.
    This is a placeholder and won't produce proper phonemes.
    """
    # Just convert each character to its ASCII value and create a flat list
    phoneme_ids = []
    for char in text:
        # Convert to int and ensure it's within a reasonable range for the model
        char_id = ord(char) % 256  # Cap at 256 to avoid large values
        phoneme_ids.append(char_id)
    
    # Add padding or special tokens if needed (model-dependent)
    # You might need to add start/end tokens depending on your model
    
    return phoneme_ids

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
text = st.text_area("Enter text to synthesize:", value="Hello, this is a test.", height=150)
voice = st.selectbox("Voice", ["en-us"])  # More voices can be added

# Debug information
with st.expander("Debug Information"):
    st.write(f"Model file exists: {os.path.isfile(MODEL_PATH)}")
    st.write(f"Tokenizer file exists: {os.path.isfile(TOKENIZER_PATH)}")
    st.write(f"Model directory contents: {os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else 'Directory not found'}")

# Show model input requirements
if session:
    with st.expander("Model Information"):
        try:
            # Get input names and shapes
            input_details = []
            for i in range(len(session.get_inputs())):
                input_detail = session.get_inputs()[i]
                input_details.append({
                    "name": input_detail.name,
                    "shape": input_detail.shape,
                    "type": input_detail.type
                })
            
            st.write("Model expects these inputs:")
            st.write(input_details)
            
            # Get output info
            output_details = []
            for i in range(len(session.get_outputs())):
                output_detail = session.get_outputs()[i]
                output_details.append({
                    "name": output_detail.name,
                    "shape": output_detail.shape,
                    "type": output_detail.type
                })
            
            st.write("Model produces these outputs:")
            st.write(output_details)
        except Exception as e:
            st.error(f"Could not get model details: {e}")

if st.button("🎤 Generate Speech") and text.strip():
    if not session:
        st.error("Model could not be loaded. Please check the error messages above.")
        st.stop()
        
    try:
        # Use the simple phonemizer since piper-phonemize is not available
        st.info("Using simplified phonemization (character-based)")
        
        # Get phoneme IDs using our simple method
        phoneme_ids = simple_phonemize(text, voice)
        
        st.write(f"Generated {len(phoneme_ids)} phoneme IDs")
        
        # Create the input tensor with proper shape for the model
        # From error message: the model expects 'input', 'input_lengths', and 'scales'
        input_tensor = np.array([phoneme_ids], dtype=np.int64)
        input_lengths = np.array([len(phoneme_ids)], dtype=np.int64)
        
        # 'scales' seems to be a 3-element array of floats, likely controlling 
        # aspects of the speech like duration, pitch, and energy
        # Using default values (1.0) which should be neutral
        scales = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        
        st.write(f"Input tensor shape: {input_tensor.shape}")
        st.write(f"Input lengths: {input_lengths}")
        st.write(f"Scales: {scales}")
        
        # Run inference with properly named inputs
        inputs = {
            "input": input_tensor,
            "input_lengths": input_lengths,
            "scales": scales
        }
        
        # Add a checkbox to show the actual input
        if st.checkbox("Show model input"):
            st.write("Model input:")
            st.write({k: v.shape for k, v in inputs.items()})
            st.write({k: v.dtype for k, v in inputs.items()})
        
        outputs = session.run(None, inputs)
        
        # Assuming the model outputs audio waveform as the first output
        output = outputs[0]
        
        st.write(f"Output shape: {output.shape}")
        
        # Check if output looks like audio (1D array of floats)
        if len(output.shape) > 1:
            output = output.squeeze()
        
        # Normalize audio if needed
        if output.max() > 1.0 or output.min() < -1.0:
            output = np.clip(output, -1.0, 1.0)
        
        # Convert to proper format for WAV if needed
        if output.dtype != np.int16:
            output = (output * 32767).astype(np.int16)
        
        # Save audio to WAV
        wav_path = os.path.join(MODEL_DIR, "output.wav")
        wavfile.write(wav_path, rate=22050, data=output)
        
        st.success("✅ Audio generated!")
        st.audio(wav_path, format="audio/wav")
    except Exception as e:
        st.error(f"❌ Error during inference: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        
        # Provide more specific troubleshooting steps
        st.error("Troubleshooting tips:")
        st.markdown("""
        1. Check that the input names match exactly what the model expects
        2. Ensure input tensor shapes match the model's requirements
        3. Verify data types (int64 for IDs, float32 for scales)
        4. Try adjusting the scale values if audio quality is poor
        """)
