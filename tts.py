import streamlit as st
import onnxruntime as ort
import numpy as np
import torch
import torchaudio
from io import BytesIO
from piper_phonemize import phonemize_espeak, phoneme_ids_espeak
import os
import requests

# Google Drive file ID and download URL
GDRIVE_FILE_ID = "1-01P3elJi2U1S2yMkBXVsGiBMVwJNhzs"
MODEL_PATH = "exported_model.onnx"
MODEL_URL = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"

# Download the model if it doesn't exist
def download_model():
    if not os.path.exists(MODEL_PATH):
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

@st.cache_resource
def load_model(model_path):
    download_model()
    return ort.InferenceSession(model_path)

session = load_model(MODEL_PATH)

def synthesize(session, phoneme_ids_tensor: torch.Tensor):
    input_lengths = torch.tensor([phoneme_ids_tensor.size(1)], dtype=torch.int64)
    scales = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)  # Fixed shape

    inputs = {
        "input": phoneme_ids_tensor.numpy(),
        "input_lengths": input_lengths.numpy(),
        "scales": scales.numpy()
    }

    outputs = session.run(None, inputs)
    audio = outputs[0].squeeze()
    audio = audio / np.abs(audio).max()
    audio_pcm = np.int16(audio * 32767)
    return 22050, audio_pcm

st.title("🗣️ Piper TTS Demo")
text_input = st.text_area("Enter text to synthesize:")

if st.button("Synthesize"):
    if text_input.strip():
        st.info("🔊 Generating speech...")
        try:
            voice = "en-us"
            phonemes = phonemize_espeak(text_input, voice)[0]
            ids = phoneme_ids_espeak(phonemes)
            phoneme_ids_tensor = torch.tensor([ids], dtype=torch.int64)
            sr, audio_data = synthesize(session, phoneme_ids_tensor)

            buffer = BytesIO()
            torchaudio.save(buffer, torch.tensor(audio_data).unsqueeze(0), sr, format="wav")
            buffer.seek(0)
            st.audio(buffer.read(), format="audio/wav")

        except Exception as e:
            st.error(f"❌ Error during synthesis: {e}")
    else:
        st.warning("⚠️ Please enter text to synthesize.")
