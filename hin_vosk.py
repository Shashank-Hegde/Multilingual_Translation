# hin_vosk.py

import os
import json
import streamlit as st
from vosk import Model, KaldiRecognizer
import wave
from googletrans import Translator
import requests
import zipfile
import io
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, ClientSettings

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
if 'audio_recorded' not in st.session_state:
    st.session_state.audio_recorded = False

# -----------------------------
# Helper Classes and Functions
# -----------------------------

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        super().__init__()
        self.audio_data = bytes()

    def recv(self, frame):
        self.audio_data += frame.to_bytes()
        return frame

def save_audio_file(audio_bytes, file_extension="wav"):
    """
    Saves the audio bytes to a file with the specified extension.

    Args:
        audio_bytes (bytes): The audio data.
        file_extension (str): The extension of the audio file.

    Returns:
        str: The filename if saved successfully, else None.
    """
    try:
        # Generate a unique filename
        file_name = f"recorded_audio.{file_extension}"
        with open(file_name, "wb") as f:
            f.write(audio_bytes)
        return file_name
    except Exception as e:
        st.error(f"Error saving audio file: {e}")
        return None

def transcribe_audio(file_path, model):
    """
    Transcribes the audio file using Vosk.

    Args:
        file_path (str): Path to the audio file.
        model (Model): Loaded Vosk model.

    Returns:
        str: Transcribed Hindi text if successful, else None.
    """
    try:
        wf = wave.open(file_path, "rb")
    except wave.Error:
        st.error("Unsupported audio format. Please ensure the recording is in WAV format.")
        return None

    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in [8000, 16000, 32000, 44100, 48000]:
        st.error("Audio must be WAV format mono PCM with a supported sample rate (8000, 16000, 32000, 44100, 48000 Hz).")
        wf.close()
        return None

    rec = KaldiRecognizer(model, wf.getframerate())
    transcript = ""

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = rec.Result()
            result_dict = json.loads(result)
            text = result_dict.get("text", "")
            transcript += " " + text

    final_result = rec.FinalResult()
    result_dict = json.loads(final_result)
    text = result_dict.get("text", "")
    transcript += " " + text

    wf.close()
    return transcript.strip()

def translate_to_english(text):
    """
    Translates Hindi text to English using Google Translate.

    Args:
        text (str): Hindi text.

    Returns:
        str: Translated English text.
    """
    try:
        translator = Translator()
        translation = translator.translate(text, src='hi', dest='en')
        return translation.text
    except Exception as e:
        st.error(f"Translation failed: {e}")
        return ""

def download_vosk_model(model_url, model_path):
    """
    Downloads and extracts the Vosk Hindi model if it's not already present.

    Args:
        model_url (str): URL to download the model.
        model_path (str): Local path to extract the model.
    """
    if not os.path.exists(model_path):
        st.info("Downloading Vosk Hindi model. This may take a few minutes...")
        try:
            response = requests.get(model_url, stream=True)
            if response.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    z.extractall()
                st.success("Vosk Hindi model downloaded and extracted.")
            else:
                st.error("Failed to download the Vosk model.")
                st.stop()
        except Exception as e:
            st.error(f"An error occurred while downloading the model: {e}")
            st.stop()

def load_vosk_model(model_path):
    """
    Loads the Vosk Hindi model.

    Args:
        model_path (str): Path to the extracted Vosk model.

    Returns:
        Model: Loaded Vosk model.
    """
    try:
        model = Model(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load Vosk model: {e}")
        st.stop()

# -----------------------------
# Model Loading
# -----------------------------

# Download and extract the Vosk model if necessary
download_vosk_model(MODEL_URL, MODEL_PATH)

# Load the Vosk model
model = load_vosk_model(MODEL_PATH)

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
# Audio Recorder using streamlit-webrtc
# -----------------------------

def audio_recorder_webrtc():
    webrtc_ctx = webrtc_streamer(
        key="voice-input",
        audio_processor_factory=AudioProcessor,
        client_settings=ClientSettings(
            media_stream_constraints={"audio": True, "video": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        ),
        async_processing=True,
    )

    if webrtc_ctx.audio_processor and webrtc_ctx.state.playing:
        audio_bytes = webrtc_ctx.audio_processor.audio_data
        if audio_bytes and not st.session_state.get('audio_recorded'):
            st.session_state.audio_recorded = True
            # Save the audio file
            file_name = save_audio_file(audio_bytes, "wav")
            if file_name:
                st.success("Audio recorded and saved successfully!")
                st.info("Transcribing your audio... Please wait.")
                # Transcribe the audio
                transcribed_text = transcribe_audio(file_name, model)
                if transcribed_text:
                    # Translate the transcribed text to English
                    translated_text = translate_to_english(transcribed_text)
                    # Display the results
                    st.subheader("📝 Transcribed Text (English):")
                    st.write(translated_text)
                else:
                    st.error("Failed to transcribe the audio.")
            else:
                st.error("Failed to save the audio file.")
    else:
        st.write("Please record your symptoms using the microphone button above.")

# Embed the audio recorder
audio_recorder_webrtc()

# -----------------------------
# Optional: Fallback Text Input
# -----------------------------

st.markdown("---")
st.write("**Alternatively, you can type your symptoms below:**")
user_input = st.text_area("Enter your symptoms here...")
if st.button("Submit Symptoms"):
    if user_input.strip() == "":
        st.warning("Please enter your symptoms.")
    else:
        # Translate to English if necessary
        translated_input = translate_to_english(user_input)
        st.subheader("📝 Your Input (English):")
        st.write(translated_input)
