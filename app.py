import os
import gdown
import streamlit as st
import numpy as np
import onnxruntime as ort
import pickle
import scipy.io.wavfile as wavfile
import subprocess
import sys

st.set_page_config(page_title="Text to Speech App", layout="centered")
st.title("🗣️ Text to Speech with Piper TTS")

# ------------------------
# Install required packages
# ------------------------
def install_requirements():
    with st.spinner("Installing required packages..."):
        # Install piper-phonemize
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "piper-phonemize"])
            st.success("Successfully installed piper-phonemize!")
            return True
        except Exception as e:
            st.error(f"Failed to install piper-phonemize: {e}")
            return False

# Check if installation button is clicked
if 'requirements_installed' not in st.session_state:
    st.session_state.requirements_installed = False

if not st.session_state.requirements_installed:
    if st.button("Install piper-phonemize"):
        st.session_state.requirements_installed = install_requirements()

# ------------------------
# Download model files
# ------------------------
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")

# Fixed Google Drive URLs for direct downloading
MODEL_DRIVE_URL = "https://drive.google.com/uc?id=1-01P3elJi2U1S2yMkBXVsGiBMVwJNhzs"
# Corrected URL format for tokenizer
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

# ------------------------
# Load tokenizer and model
# ------------------------
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
tokenizer = load_tokenizer()

# ------------------------
# Phonemization Functions
# ------------------------
def use_piper_phonemize(text, language="en-us", phonemizer_type="espeak"):
    """Use the piper-phonemize library to convert text to phonemes"""
    try:
        # Try to import piper_phonemize after installation
        from piper_phonemize import phonemize_espeak, phonemize_segments
        
        if phonemizer_type == "espeak":
            # Use espeak phonemizer
            phoneme_list = phonemize_espeak(text, language)
            st.success("✅ Used piper-phonemize with espeak")
        else:
            # Use segments phonemizer (alternative)
            phoneme_list = phonemize_segments(text, language)
            st.success("✅ Used piper-phonemize with segments")
            
        return phoneme_list
    except Exception as e:
        st.warning(f"Could not use piper-phonemize: {e}")
        return None
        
def simple_phonemize(text, language="en-us"):
    """Fallback phonemization if piper-phonemize is not available"""
    # Just convert each character to its ASCII value and create a flat list
    phoneme_ids = []
    for char in text:
        char_id = ord(char) % 256
        phoneme_ids.append(char_id)
    
    # Format it for compatibility with piper's phoneme list structure
    # The phoneme list should be a list of lists of lists
    # [[word1_phonemes], [word2_phonemes], ...]
    return [[phoneme_ids]]

# ------------------------
# UI
# ------------------------
text = st.text_area("Enter text to synthesize:", 
                   value="The quick brown fox jumps over the lazy dog.",
                   height=150)

col1, col2, col3 = st.columns(3)

with col1:
    voice = st.selectbox("Voice", ["en-us", "en-gb"])
    
with col2:
    phonemizer = st.selectbox("Phonemizer", ["piper (espeak)", "piper (segments)", "simple"])
    
with col3:
    # Speech parameters
    duration_scale = st.slider("Speech Duration", 0.5, 2.0, 1.0, 0.1)
    pitch_scale = st.slider("Pitch", 0.5, 2.0, 1.0, 0.1)
    energy_scale = st.slider("Energy", 0.5, 2.0, 1.0, 0.1)

# Debug information
with st.expander("Debug Information"):
    st.write(f"Model file exists: {os.path.isfile(MODEL_PATH)}")
    st.write(f"Tokenizer file exists: {os.path.isfile(TOKENIZER_PATH)}")
    st.write(f"Model directory contents: {os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else 'Directory not found'}")
    
    # Check if piper-phonemize is available
    try:
        import piper_phonemize
        st.success("piper-phonemize is installed!")
    except ImportError:
        st.warning("piper-phonemize is not installed. Use the install button above.")
    
    # Show tokenizer info if available
    if tokenizer:
        st.write("Tokenizer loaded successfully")
        st.write(f"Tokenizer type: {type(tokenizer)}")
        try:
            if hasattr(tokenizer, '__dict__'):
                st.write("Tokenizer attributes:")
                for key in list(tokenizer.__dict__.keys())[:10]:  # Show first 10 attributes to avoid clutter
                    st.write(f"- {key}")
        except:
            st.write("Could not inspect tokenizer attributes")

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
        # Get phonemes based on selected method
        phoneme_list = None
        
        if phonemizer.startswith("piper") and st.session_state.requirements_installed:
            phonemizer_type = "espeak" if "espeak" in phonemizer else "segments"
            phoneme_list = use_piper_phonemize(text, voice, phonemizer_type)
        
        # Fallback to simple phonemizer if piper fails or isn't selected
        if phoneme_list is None:
            st.warning("Using simplified phonemization (character-based)")
            phoneme_list = simple_phonemize(text, voice)
        
        # Flatten phoneme list (convert from list of lists of lists to a single list)
        flat_phonemes = []
        for word_phonemes in phoneme_list:
            for phoneme in word_phonemes:
                if isinstance(phoneme, list):
                    flat_phonemes.extend(phoneme)
                else:
                    flat_phonemes.append(phoneme)
        
        st.write(f"Generated {len(flat_phonemes)} phoneme IDs")
        
        # Create the input tensor with proper shape for the model
        input_tensor = np.array([flat_phonemes], dtype=np.int64)
        input_lengths = np.array([len(flat_phonemes)], dtype=np.int64)
        
        # Set scales according to UI sliders
        scales = np.array([duration_scale, pitch_scale, energy_scale], dtype=np.float32)
        
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
        if st.checkbox("Show model input details"):
            st.write("Model input:")
            st.write({k: v.shape for k, v in inputs.items()})
            st.write({k: v.dtype for k, v in inputs.items()})
            # Show a sample of the phoneme IDs
            st.write("Sample phoneme IDs (first 20):")
            st.write(flat_phonemes[:20])
        
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
        
        # Show a plot of the audio waveform
        if st.checkbox("Show audio waveform"):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(output)
            ax.set_title("Audio Waveform")
            ax.set_xlabel("Sample")
            ax.set_ylabel("Amplitude")
            st.pyplot(fig)
            
            # Show some audio statistics
            st.write(f"Audio length: {len(output)} samples ({len(output)/22050:.2f} seconds)")
            st.write(f"Audio min: {output.min()}, max: {output.max()}, mean: {output.mean():.2f}")
            
    except Exception as e:
        st.error(f"❌ Error during inference: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        
        # Provide more specific troubleshooting steps
        st.error("Troubleshooting tips:")
        st.markdown("""
        1. If you're seeing phonemization errors, try installing piper-phonemize
        2. If you're seeing shape errors, check the phoneme list structure
        3. Try adjusting the scale values if audio quality is poor
        4. Try a different input text (some characters might not be supported)
        """)
