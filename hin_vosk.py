# hin_vosk.py

import os
import json
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings, AudioProcessorBase
from vosk import Model, KaldiRecognizer
import numpy as np
from googletrans import Translator
import requests
import zipfile
import io

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
if 'recording' not in st.session_state:
    st.session_state.recording = False

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

# -----------------------------
# Custom Audio Processor
# -----------------------------

class VoskAudioProcessor(AudioProcessorBase):
    def __init__(self, model, translator):
        self.model = model
        self.rec = KaldiRecognizer(self.model, 16000)
        self.translator = translator

    def recv(self, frame):
        data = frame.to_ndarray().astype(np.int16).tobytes()
        if self.rec.AcceptWaveform(data):
            result = self.rec.Result()
            result_dict = json.loads(result)
            text_hindi = result_dict.get("text", "")
            if text_hindi:
                st.session_state.transcript_hindi += " " + text_hindi
                translation = translate_text(text_hindi, self.translator)
                st.session_state.transcript_english += " " + translation
        return frame

# -----------------------------
# Model and Translator Loading
# -----------------------------

# Download and extract the Vosk model if necessary
download_vosk_model(MODEL_URL, MODEL_PATH)

# Load the Vosk model and translator
model = load_vosk_model(MODEL_PATH)
translator = load_translator()

# -----------------------------
# WebRTC Configuration
# -----------------------------

RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

MEDIA_STREAM_CONSTRAINTS = {
    "audio": True,
    "video": False
}

# -----------------------------
# Streamlit App UI
# -----------------------------

st.title("🗣️ Hindi Speech-to-Text Converter")
st.markdown("""
    This application converts your Hindi speech to English text in real-time using Vosk for speech recognition and Google Translate for translation.

    **Instructions:**
    1. Click on **Start Recording** to begin.
    2. Speak clearly in Hindi.
    3. Your Hindi speech and its English translation will appear below.
    4. Click on **Stop Recording** to end the session.
""")

# -----------------------------
# Sidebar Controls
# -----------------------------

st.sidebar.header("Control Panel")
start = st.sidebar.button("Start Recording")
stop = st.sidebar.button("Stop Recording")
reset = st.sidebar.button("Reset Transcriptions")

# Handle Start and Stop buttons
if start:
    st.session_state.recording = True
    st.session_state.transcript_hindi = ""
    st.session_state.transcript_english = ""
    st.sidebar.write("Recording... Speak into your microphone.")

if stop:
    st.session_state.recording = False
    st.sidebar.write("Recording stopped.")

# -----------------------------
# Display Transcriptions
# -----------------------------

st.header("Transcription Results")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Hindi Text")
    st.text_area("Hindi", st.session_state.transcript_hindi, height=300)

with col2:
    st.subheader("English Translation")
    st.text_area("English", st.session_state.transcript_english, height=300)

# -----------------------------
# Initialize WebRTC Streamer
# -----------------------------

if st.session_state.recording:
    webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints=MEDIA_STREAM_CONSTRAINTS,
        audio_processor_factory=lambda: VoskAudioProcessor(model, translator),
        async_processing=True,
    )

# -----------------------------
# Handle Reset Button
# -----------------------------

if reset:
    st.session_state.transcript_hindi = ""
    st.session_state.transcript_english = ""
    st.sidebar.write("Transcriptions have been reset.")

# -----------------------------
# Footer
# -----------------------------

st.markdown("""
    ---
    © 2024 Hindi Speech-to-Text App using Vosk and Streamlit
""")
