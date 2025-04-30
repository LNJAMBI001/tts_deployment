import os
import gdown
import streamlit as st
import numpy as np
import onnxruntime as ort
import pickle
import scipy.io.wavfile as wavfile
import subprocess
import sys
import re
import logging
import importlib

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PiperTTS")

st.set_page_config(page_title="Text to Speech App", layout="centered")
st.title("🗣️ Text to Speech with Piper TTS")

# ------------------------
# Installation functions
# ------------------------
def check_if_package_installed(package_name):
    """Check if a package is installed"""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def install_requirements():
    """Install required packages with enhanced error handling"""
    with st.spinner("Installing required packages..."):
        success = False
        installation_log = []
        
        # List of packages to try installing
        packages = [
            "piper-phonemize",
            "gruut",
            "espeak-phonemize",
            "phonemizer"
        ]
        
        # First check which are already installed
        for package in packages:
            if check_if_package_installed(package.replace('-', '_')):
                installation_log.append(f"✅ {package} is already installed")
                success = True
                continue
                
        # Try to install the missing ones individually
        for package in packages:
            package_name = package.replace('-', '_')  # Python packages use underscores
            if not check_if_package_installed(package_name):
                try:
                    logger.info(f"Attempting to install {package}...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", package])
                    installation_log.append(f"✅ Successfully installed {package}")
                    success = True
                except Exception as e:
                    installation_log.append(f"❌ Failed to install {package}: {str(e)}")
        
        # If all direct installs failed, try alternative methods
        if not success:
            try:
                installation_log.append("Trying to install piper-phonemize from GitHub...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/rhasspy/piper-phonemize"])
                installation_log.append("✅ Successfully installed piper-phonemize from GitHub")
                success = True
            except Exception as e:
                installation_log.append(f"❌ Failed to install from GitHub: {str(e)}")
                
                # Try with specific build flags
                try:
                    installation_log.append("Trying with specific build flags...")
                    env = os.environ.copy()
                    env["CFLAGS"] = "-fPIC"
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/rhasspy/piper-phonemize"], env=env)
                    installation_log.append("✅ Successfully installed piper-phonemize with custom flags")
                    success = True
                except Exception as e2:
                    installation_log.append(f"❌ Failed with custom flags: {str(e2)}")
        
        # Display installation results
        st.write("### Installation Results")
        for log in installation_log:
            if log.startswith("✅"):
                st.success(log)
            elif log.startswith("❌"):
                st.error(log)
            else:
                st.info(log)
        
        return success

# Alternative text-to-phoneme function using regex patterns
def text_to_ipa(text):
    """
    Convert text to a very simplified IPA (International Phonetic Alphabet) representation.
    This is a fallback when proper phonemizers aren't available.
    """
    # Simple mapping of English phonemes - very basic approximation
    phoneme_map = {
        'a': 'æ', 'e': 'ɛ', 'i': 'ɪ', 'o': 'ɒ', 'u': 'ʌ',
        'th': 'θ', 'sh': 'ʃ', 'ch': 'tʃ', 'ng': 'ŋ',
        'ay': 'eɪ', 'ai': 'eɪ', 'ow': 'oʊ', 'ee': 'iː',
        'er': 'ɜr', 'ar': 'ɑr', 'or': 'ɔr', 'ir': 'ɪr',
    }
    
    # Apply substitutions
    text = text.lower()
    for pattern, replacement in phoneme_map.items():
        text = text.replace(pattern, replacement)
    
    # Add spaces between each character to help with tokenization
    spaced_text = ' '.join(text)
    
    try:
        # Convert to character codes
        char_codes = [ord(c) for c in spaced_text]
        
        # Format into phoneme structure expected by model
        # Split by words (assuming spaces denote word boundaries)
        words = spaced_text.split()
        phoneme_list = []
        
        if not words:  # Handle empty or only spaces
            return [[char_codes]]
        
        for word in words:
            word_codes = [ord(c) for c in word]
            if word_codes:  # Only add non-empty
                phoneme_list.append([word_codes])
        
        if not phoneme_list:  # Fallback if nothing was added
            return [[char_codes]]
        
        return phoneme_list
    except Exception as e:
        logger.error(f"Error in text_to_ipa: {e}")
        # Ultimate fallback
        return [[[ord(c) for c in text]]]

# Check if installation button is clicked
if 'requirements_installed' not in st.session_state:
    st.session_state.requirements_installed = False
    
    # Auto-check which packages are installed
    st.session_state.available_phonemizers = {
        "piper_phonemize": check_if_package_installed("piper_phonemize"),
        "espeak_phonemize": check_if_package_installed("espeak_phonemize"),
        "gruut": check_if_package_installed("gruut"),
        "phonemizer": check_if_package_installed("phonemizer")
    }
    
    if any(st.session_state.available_phonemizers.values()):
        st.session_state.requirements_installed = True

if not st.session_state.requirements_installed:
    if st.button("Install Phonemization Tools"):
        st.session_state.requirements_installed = install_requirements()
        
        # Update available phonemizers
        st.session_state.available_phonemizers = {
            "piper_phonemize": check_if_package_installed("piper_phonemize"),
            "espeak_phonemize": check_if_package_installed("espeak_phonemize"),
            "gruut": check_if_package_installed("gruut"),
            "phonemizer": check_if_package_installed("phonemizer")
        }

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

try:
    session = load_model()
    tokenizer = load_tokenizer()
except Exception as e:
    st.error(f"Error loading model or tokenizer: {e}")
    st.error("Please reload the application and try again.")
    st.stop()

# ------------------------
# Enhanced phonemization
# ------------------------
def phonemize_with_piper(text, language="en-us"):
    """Try to use piper-phonemize"""
    try:
        from piper_phonemize import phonemize_espeak
        return phonemize_espeak(text, language)
    except Exception as e:
        logger.warning(f"Error with piper-phonemize: {e}")
        return None

def phonemize_with_espeak(text, language="en-us"):
    """Try to use espeak-phonemize"""
    try:
        from espeak_phonemize import phonemize
        phonemes = phonemize(text, language)
        # Convert to format expected by the model
        phoneme_list = []
        for word in phonemes.split():
            phoneme_list.append([[ord(c) for c in word]])
        return phoneme_list
    except Exception as e:
        logger.warning(f"Error with espeak-phonemize: {e}")
        return None

def phonemize_with_gruut(text, language="en-us"):
    """Try to use gruut"""
    try:
        import gruut
        phonemizer = gruut.get_phonemizer(language)
        words = phonemizer.phonemize(text)
        phoneme_list = []
        for word in words:
            phoneme_list.append([[ord(c) for c in word.phonemes]])
        return phoneme_list
    except Exception as e:
        logger.warning(f"Error with gruut: {e}")
        return None

def phonemize_with_phonemizer(text, language="en-us"):
    """Try to use phonemizer"""
    try:
        from phonemizer.backend import EspeakBackend
        from phonemizer.phonemize import phonemize
        
        # Map language codes
        lang_map = {
            "en-us": "en-us",
            "en-gb": "en-gb"
        }
        
        espeak_lang = lang_map.get(language, "en-us")
        backend = EspeakBackend(language=espeak_lang)
        
        # Phonemize with espeak
        phonemes = phonemize(
            text, 
            backend=backend,
            strip=True,
            preserve_punctuation=True,
            with_stress=True
        )
        
        # Convert to format expected by model
        phoneme_list = []
        for word in phonemes.split():
            phoneme_list.append([[ord(c) for c in word]])
        
        return phoneme_list
    except Exception as e:
        logger.warning(f"Error with phonemizer: {e}")
        return None

def try_all_phonemizers(text, language="en-us"):
    """Try all available phonemizers in order of preference"""
    results = []
    
    # Check which phonemizers are available
    available = st.session_state.available_phonemizers
    
    # Try piper-phonemize first (best option)
    if available.get("piper_phonemize", False):
        result = phonemize_with_piper(text, language)
        if result:
            logger.info("Used piper-phonemize successfully")
            return result, "piper-phonemize"
    
    # Try espeak-phonemize
    if available.get("espeak_phonemize", False):
        result = phonemize_with_espeak(text, language)
        if result:
            logger.info("Used espeak-phonemize successfully")
            return result, "espeak-phonemize"
    
    # Try gruut
    if available.get("gruut", False):
        result = phonemize_with_gruut(text, language)
        if result:
            logger.info("Used gruut successfully")
            return result, "gruut"
    
    # Try phonemizer
    if available.get("phonemizer", False):
        result = phonemize_with_phonemizer(text, language)
        if result:
            logger.info("Used phonemizer successfully")
            return result, "phonemizer"
    
    # Fallback: Use text_to_ipa
    result = text_to_ipa(text)
    logger.info("Used IPA approximation fallback")
    return result, "IPA approximation"

# ------------------------
# UI
# ------------------------
text = st.text_area("Enter text to synthesize:", 
                   value="The quick brown fox jumps over the lazy dog.",
                   height=150)

col1, col2 = st.columns(2)

with col1:
    voice = st.selectbox("Voice", ["en-us", "en-gb"])
    
with col2:
    # Only show phonemization options that are available
    phonemizer_options = ["Auto-detect best available"]
    
    available = getattr(st.session_state, "available_phonemizers", {})
    if available.get("piper_phonemize", False):
        phonemizer_options.append("piper-phonemize")
    if available.get("espeak_phonemize", False):
        phonemizer_options.append("espeak-phonemize")
    if available.get("gruut", False):
        phonemizer_options.append("gruut")
    if available.get("phonemizer", False):
        phonemizer_options.append("phonemizer")
        
    # Always add fallbacks
    phonemizer_options.extend(["IPA Approximation", "Simple ASCII"])
    
    phonemizer_method = st.selectbox("Phonemization Method", phonemizer_options)

# Speech parameters
st.subheader("Speech Parameters")
col1, col2, col3 = st.columns(3)

with col1:
    duration_scale = st.slider("Duration", 0.5, 2.0, 1.0, 0.1, 
                               help="Controls speech speed. Higher values make speech slower.")
with col2:
    pitch_scale = st.slider("Pitch", 0.5, 2.0, 1.0, 0.1,
                           help="Controls voice pitch. Higher values make voice higher pitched.")
with col3:
    energy_scale = st.slider("Energy", 0.5, 2.0, 1.0, 0.1,
                            help="Controls volume and emphasis. Higher values make speech louder.")

# Debug information
with st.expander("Debug Information"):
    st.write(f"Model file exists: {os.path.isfile(MODEL_PATH)}")
    st.write(f"Tokenizer file exists: {os.path.isfile(TOKENIZER_PATH)}")
    st.write(f"Model directory contents: {os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else 'Directory not found'}")
    
    # Check for phonemization tools
    st.subheader("Available Phonemization Tools")
    
    available_phonemizers = st.session_state.get("available_phonemizers", {})
    for phonemizer, is_available in available_phonemizers.items():
        if is_available:
            st.success(f"✅ {phonemizer} is installed")
        else:
            st.warning(f"❌ {phonemizer} is not installed")
    
    # Show tokenizer info if available
    if tokenizer:
        st.subheader("Tokenizer Information")
        st.write("Tokenizer loaded successfully")
        st.write(f"Tokenizer type: {type(tokenizer)}")
        try:
            if hasattr(tokenizer, '__dict__'):
                st.write("Tokenizer attributes:")
                for key in list(tokenizer.__dict__.keys())[:10]:  # Show first 10 attributes
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
        used_method = None
        
        if phonemizer_method == "Auto-detect best available":
            phoneme_list, used_method = try_all_phonemizers(text, voice)
            st.info(f"Using {used_method} for phonemization")
        elif phonemizer_method == "piper-phonemize" and st.session_state.available_phonemizers.get("piper_phonemize", False):
            phoneme_list = phonemize_with_piper(text, voice)
            used_method = "piper-phonemize"
        elif phonemizer_method == "espeak-phonemize" and st.session_state.available_phonemizers.get("espeak_phonemize", False):
            phoneme_list = phonemize_with_espeak(text, voice)
            used_method = "espeak-phonemize"
        elif phonemizer_method == "gruut" and st.session_state.available_phonemizers.get("gruut", False):
            phoneme_list = phonemize_with_gruut(text, voice)
            used_method = "gruut"
        elif phonemizer_method == "phonemizer" and st.session_state.available_phonemizers.get("phonemizer", False):
            phoneme_list = phonemize_with_phonemizer(text, voice)
            used_method = "phonemizer"
        elif phonemizer_method == "IPA Approximation":
            phoneme_list = text_to_ipa(text)
            used_method = "IPA approximation"
        else:
            # Simple ASCII - simplest fallback
            phoneme_list = [[[ord(c) for c in text]]]
            used_method = "Simple ASCII"
        
        if phoneme_list is None:
            st.warning("Selected phonemizer failed. Falling back to IPA approximation.")
            phoneme_list = text_to_ipa(text)
            used_method = "IPA approximation (fallback)"
            
        st.success(f"Successfully phonemized text using {used_method}")
        
        # Flatten phoneme list (convert from list of lists of lists to a single list)
        flat_phonemes = []
        for word_phonemes in phoneme_list:
            for phoneme in word_phonemes:
                if isinstance(phoneme, list):
                    flat_phonemes.extend(phoneme)
                else:
                    flat_phonemes.append(phoneme)
        
        # Make sure we have phonemes
        if not flat_phonemes:
            st.error("No phonemes were generated. Please try a different text or phonemization method.")
            st.stop()
            
        st.write(f"Generated {len(flat_phonemes)} phoneme IDs")
        
        # Create the input tensor with proper shape for the model
        input_tensor = np.array([flat_phonemes], dtype=np.int64)
        input_lengths = np.array([len(flat_phonemes)], dtype=np.int64)
        
        # Set scales according to UI sliders
        scales = np.array([duration_scale, pitch_scale, energy_scale], dtype=np.float32)
        
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
        
        with st.spinner("Generating audio..."):
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
        
        # Show audio duration
        duration_seconds = len(output) / 22050
        st.info(f"Audio length: {duration_seconds:.2f} seconds")
        
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
            st.write(f"Audio min: {output.min()}, max: {output.max()}, mean: {output.mean():.2f}")
            
    except Exception as e:
        st.error(f"❌ Error during inference: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        
        # Provide specific troubleshooting tips
        st.error("Troubleshooting tips:")
        st.markdown("""
        1. Try using a different phonemization method
        2. Try a different text input (some characters might not be supported)
        3. Increase the duration scale to get longer audio
        4. Check if the text is too long for the model
        """)
