# hin_vosk.py

import os
import json
import streamlit as st
from vosk import Model, KaldiRecognizer
import wave
import numpy as np
from googletrans import Translator
import requests
import zipfile
import io
from audio_recorder_streamlit import audio_recorder

# -----------------------------
# Configuration and Setup
# -----------------------------

# Define the Vosk model URL and local path
MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-hi-0.22.zip"
MODEL_PATH = "vosk-model-small-hi-0.22"

# -----------------------------
# Session State Initialization
# -----------------------------

if 'transcript_hindi' not in st.session_state:
    st.session_state.transcript_hindi = ""
if 'transcript_english' not in st.session_state:
    st.session_state.transcript_english = ""

# -----------------------------
# Helper Functions
# -----------------------------

@st.cache_resource(show_spinner=False)
def download_vosk_model(model_url, model_path):
    """
    Downloads and extracts the Vosk Hindi model if it's not already present.
    """
    if not os.path.exists(model_path):
        st.info("Downloading Vosk Hindi model. This may take a few minutes...")
        response = requests.get(model_url, stream=True)
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall()
            st.success("Vosk Hindi model downloaded and extracted.")
        else:
            st.error("Failed to download the Vosk model.")
            st.stop()

@st.cache_resource
def load_vosk_model(model_path):
    """
    Loads the Vosk Hindi model.
    """
    model = Model(model_path)
    return model

@st.cache_resource
def load_translator():
    """
    Initializes the Google Translator.
    """
    translator = Translator()
    return translator

def translate_text(text, translator):
    """
    Translates Hindi text to English using Google Translate.
    """
    if not text.strip():
        return ""
    translation = translator.translate(text, src='hi', dest='en')
    return translation.text

def transcribe_audio(audio_bytes, model, translator):
    """
    Transcribes the recorded audio and translates it.
    """
    # Save the audio bytes to a temporary WAV file
    temp_wav_path = "temp.wav"
    with open(temp_wav_path, "wb") as f:
        f.write(audio_bytes)

    try:
        wf = wave.open(temp_wav_path, "rb")
    except wave.Error:
        st.error("Unsupported audio format. Please ensure the recording is in WAV format.")
        return

    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in [8000, 16000, 32000, 44100, 48000]:
        st.error("Audio must be WAV format mono PCM with a supported sample rate (8000, 16000, 32000, 44100, 48000 Hz).")
        wf.close()
        os.remove(temp_wav_path)
        return

    rec = KaldiRecognizer(model, wf.getframerate())
    transcript_hindi = ""
    transcript_english = ""

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = rec.Result()
            result_dict = json.loads(result)
            text_hindi = result_dict.get("text", "")
            if text_hindi:
                transcript_hindi += " " + text_hindi
                translation = translate_text(text_hindi, translator)
                transcript_english += " " + translation

    final_result = rec.FinalResult()
    result_dict = json.loads(final_result)
    text_hindi = result_dict.get("text", "")
    if text_hindi:
        transcript_hindi += " " + text_hindi
        translation = translate_text(text_hindi, translator)
        transcript_english += " " + translation

    wf.close()
    os.remove(temp_wav_path)
    return transcript_hindi.strip(), transcript_english.strip()

# -----------------------------
# Model and Translator Loading
# -----------------------------

# Download and extract the Vosk model if necessary
download_vosk_model(MODEL_URL, MODEL_PATH)

# Load the Vosk model and translator
model = load_vosk_model(MODEL_PATH)
translator = load_translator()

# -----------------------------
# Streamlit App UI
# -----------------------------

st.title("🗣️ Hindi Speech-to-Text Converter")
st.markdown("""
    This application converts your Hindi speech from a recorded audio input to English text using Vosk for speech recognition and Google Translate for translation.

    **Instructions:**
    1. Click on **Start Recording** to begin.
    2. Speak clearly in Hindi.
    3. Click on **Stop Recording** to end the session.
    4. The Hindi transcription and its English translation will appear below.
""")

# -----------------------------
# Audio Recorder
# -----------------------------

recorded_audio = audio_recorder(key="voice_input_initial")

if recorded_audio:
    st.success("Recording complete.")
    with st.spinner("Transcribing and translating..."):
        transcript_hindi, transcript_english = transcribe_audio(recorded_audio, model, translator)
    # Display results
    st.header("Transcription Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hindi Text")
        st.text_area("Hindi", transcript_hindi, height=300)
    
    with col2:
        st.subheader("English Translation")
        st.text_area("English", transcript_english, height=300)

# -----------------------------
# Footer
# -----------------------------

st.markdown("""
    ---
    © 2024 Hindi Speech-to-Text App using Vosk and Streamlit
""")
