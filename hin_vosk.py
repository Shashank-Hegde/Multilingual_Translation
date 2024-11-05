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

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'symptoms_processed' not in st.session_state:
    st.session_state.symptoms_processed = False
if 'followup_question' not in st.session_state:
    st.session_state.followup_question = ""
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1  # Initial step
if 'followup_count' not in st.session_state:
    st.session_state.followup_count = 0
if 'matched_symptoms' not in st.session_state:
    st.session_state.matched_symptoms = set()

# -----------------------------
# Helper Functions
# -----------------------------

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

def transcribe_audio(file_path):
    """
    Transcribes the audio file using Vosk.

    Args:
        file_path (str): Path to the audio file.

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

def correct_spelling(text):
    """
    Corrects the spelling of the translated text.

    Args:
        text (str): Translated English text.

    Returns:
        str: Corrected English text.
    """
    # Implement spell correction logic here
    # For now, return the text as is
    return text

def extract_and_prepare_questions(conversation_history):
    """
    Extracts and prepares follow-up questions based on conversation history.

    Args:
        conversation_history (list): List of conversation history.

    Returns:
        str: Prepared follow-up questions.
    """
    # Implement question extraction logic here
    # For now, return a placeholder question
    return "Can you describe your symptoms in more detail?"

def translate_to_hindi(text):
    """
    Translates English text to Hindi using Google Translate.

    Args:
        text (str): English text.

    Returns:
        str: Translated Hindi text.
    """
    try:
        translator = Translator()
        translation = translator.translate(text, src='en', dest='hi')
        return translation.text
    except Exception as e:
        st.error(f"Translation failed: {e}")
        return ""

def generate_audio(text, lang='hi'):
    """
    Generates audio from text using Google Translate's TTS.

    Args:
        text (str): Text to convert to speech.
        lang (str): Language code.

    Returns:
        bytes: Audio data in bytes if successful, else None.
    """
    try:
        from gtts import gTTS
        import base64

        tts = gTTS(text=text, lang=lang)
        audio_file = "question_audio.mp3"
        tts.save(audio_file)
        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
        os.remove(audio_file)
        return audio_bytes
    except Exception as e:
        st.error(f"Audio generation failed: {e}")
        return None

def embed_audio_autoplay(audio_bytes):
    """
    Embeds and auto-plays the audio in the Streamlit app.

    Args:
        audio_bytes (bytes): Audio data in bytes.
    """
    try:
        import base64
        audio_base64 = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            Your browser does not support the audio element.
        </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Failed to embed audio: {e}")

def handle_yes_no_response(question, response):
    """
    Handles yes/no responses to add or remove symptoms.

    Args:
        question (str): The question asked.
        response (str): The user's response.
    """
    response_lower = response.lower()
    if 'yes' in response_lower:
        # Logic to add symptoms
        st.session_state.matched_symptoms.add('new_symptom')  # Replace with actual logic
        st.success("Added new symptom based on your response.")
    elif 'no' in response_lower:
        # Logic to remove symptoms
        st.session_state.matched_symptoms.discard('irrelevant_symptom')  # Replace with actual logic
        st.warning("Removed irrelevant symptom based on your response.")
    else:
        st.info("Please respond with 'Yes' or 'No'.")

def extract_symptoms(text):
    """
    Extracts symptoms from the user's response.

    Args:
        text (str): User's response in English.

    Returns:
        set: A set of detected symptoms.
    """
    # Implement symptom extraction logic here
    # For now, return an empty set
    return set()

# -----------------------------
# Model and Translator Loading
# -----------------------------

@st.cache_resource(show_spinner=False)
def download_vosk_model(model_url, model_path):
    """
    Downloads and extracts the Vosk Hindi model if it's not already present.
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

@st.cache_resource
def load_vosk_model(model_path):
    """
    Loads the Vosk Hindi model.
    """
    try:
        model = Model(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load Vosk model: {e}")
        st.stop()

@st.cache_resource
def load_translator():
    """
    Initializes the Google Translator.
    """
    try:
        translator = Translator()
        return translator
    except Exception as e:
        st.error(f"Failed to initialize translator: {e}")
        st.stop()

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
    1. Click on **Click to record** to begin.
    2. Speak clearly in Hindi.
    3. Click on **Stop Recording** to end the session.
    4. The Hindi transcription and its English translation will appear below.
""")

# -----------------------------
# Audio Recorder
# -----------------------------

# Simplified audio_recorder call with only supported parameters
recorded_audio = audio_recorder(
    key="voice_input_initial",
    text="Click to record"
)

if recorded_audio and not st.session_state.get('symptoms_processed'):
    st.audio(recorded_audio, format="audio/wav")
    file_name = save_audio_file(recorded_audio, "wav")
    if file_name:
        st.success("Audio recorded and saved successfully!")
        st.info("Transcribing your audio... Please wait.")
        transcribed_text = transcribe_audio(file_name)
        if transcribed_text:
            # Translate to English if necessary
            translated_text = translate_to_english(transcribed_text)
            # Correct spelling in the translated text
            corrected_text = correct_spelling(translated_text)
            st.subheader("📝 Transcribed Text (English):")
            st.write(corrected_text)
            st.session_state.conversation_history.append({
                'user': corrected_text
            })
            st.session_state.followup_question = extract_and_prepare_questions(st.session_state.conversation_history)
            st.session_state.current_step = 2  # Proceed to follow-up questions
            st.session_state.symptoms_processed = True
            st.experimental_rerun()
        else:
            st.error("Failed to transcribe the audio.")
    else:
        st.error("Failed to save the audio file.")
else:
    st.write("Please record your symptoms using the microphone button above.")

# Optionally, provide a fallback text input
st.write("**Alternatively, you can type your symptoms below:**")
user_input = st.text_area("Enter your symptoms here...")
if st.button("Submit Symptoms"):
    if user_input.strip() == "":
        st.warning("Please enter your symptoms.")
    else:
        # Translate to English if necessary
        translated_input = translate_to_english(user_input)
        # Correct spelling in the translated text
        corrected_input = correct_spelling(translated_input)
        st.subheader("📝 Your Input:")
        st.write(corrected_input)
        st.session_state.conversation_history.append({
            'user': corrected_input
        })
        st.session_state.followup_question = extract_and_prepare_questions(st.session_state.conversation_history)
        st.session_state.current_step = 2  # Proceed to follow-up questions
        st.session_state.symptoms_processed = True
        st.experimental_rerun()

# -----------------------------
# Follow-up Questions
# -----------------------------

if st.session_state.current_step == 2:
    if st.session_state.followup_count >= 5:
        st.info("You have reached the maximum number of follow-up questions.")
        st.session_state.current_step = 3  # Proceed to report
        st.experimental_rerun()

    if not st.session_state.followup_question:
        st.info("No follow-up questions required based on your inputs.")
        st.session_state.current_step = 3  # Proceed to report
        st.experimental_rerun()

    current_question = st.session_state.followup_question
    question_number = st.session_state.followup_count + 1

    # Display the question
    st.subheader(f"🔍 Follow-Up Question {question_number}:")
    st.write(f"**English:** {current_question}")

    # Generate the question audio in Hindi
    if not st.session_state.get(f'question_{st.session_state.followup_count}_played'):
        # Translate the question to Hindi for audio playback
        translated_question = translate_to_hindi(current_question)
        question_audio = generate_audio(translated_question, lang='hi')
        if question_audio:
            # Embed and autoplay the audio
            embed_audio_autoplay(question_audio)
            st.session_state[f'question_{st.session_state.followup_count}_played'] = True
        else:
            st.error("Failed to generate question audio.")

    # Record the user's answer
    st.write("**Please record your answer using the microphone button below:**")
    response_audio_bytes = audio_recorder(key=f"voice_input_followup_{st.session_state.followup_count}")
    if response_audio_bytes and not st.session_state.get(f'answer_{st.session_state.followup_count}_processed'):
        st.audio(response_audio_bytes, format="audio/wav")
        response_file_name = save_audio_file(response_audio_bytes, "wav")
        if response_file_name:
            st.success("Audio recorded and saved successfully!")
            st.info("Transcribing your audio... Please wait.")
            response_transcribed = transcribe_audio(response_file_name)
            if response_transcribed:
                # Translate to English if necessary
                translated_response = translate_to_english(response_transcribed)
                # Correct spelling in the translated text
                corrected_response = correct_spelling(translated_response)
                st.subheader(f"📝 Your Answer to Question {question_number} (English):")
                st.write(corrected_response)
                # Handle yes/no responses to add/remove symptoms
                handle_yes_no_response(current_question, corrected_response)
                # Extract any new symptoms from the response
                extracted_new_symptoms = extract_symptoms(corrected_response)
                if extracted_new_symptoms:
                    st.session_state.matched_symptoms.update(extracted_new_symptoms)
                    st.success(f"New symptom(s) detected and added: {', '.join(extracted_new_symptoms)}")
                st.session_state.conversation_history.append({
                    'followup_question_en': current_question,
                    'response': corrected_response
                })
                st.session_state.followup_question = extract_and_prepare_questions(st.session_state.conversation_history)
                st.session_state[f'answer_{st.session_state.followup_count}_processed'] = True
                st.session_state.followup_count += 1  # Increment the follow-up question counter
                st.experimental_rerun()
            else:
                st.error("Failed to transcribe your audio response.")
        else:
            st.error("Failed to save the audio file.")
    else:
        st.write("Please record your answer using the microphone button above.")

# -----------------------------
# Report (Placeholder)
# -----------------------------

if st.session_state.current_step == 3:
    st.header("📄 Your Symptom Report")
    st.write("Here is a summary of your reported symptoms and follow-up responses.")
    # Implement report generation logic here
