import streamlit as st 
import torchaudio
from google import genai
from google.genai import types
import soundfile as sf
from kokoro import KPipeline
import re
import os
from dotenv import load_dotenv
from transformers import pipeline

@st.cache_resource
def load_asr_model():
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny",
        generate_kwargs={"language": "en", "task": "transcribe"}
    )

# Get sentence from Gemini
def generate_sentence():
    response = client.models.generate_content(
        model="gemini-2.0-flash", 
        config=types.GenerateContentConfig(
            system_instruction="You are an English language tutor.",
            temperature=1.0),
        contents="Generate one A2-level English sentence suitable for speaking practice."
    )
    return response.text

# Load environment
load_dotenv()
gemini_api = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=gemini_api)

st.set_page_config(page_title="English Speaking Practice", layout="centered")
st.title("🗣️ English Speaking Practice")
st.markdown("Practice your English pronunciation by listening and repeating.")


def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text) 
    text = re.sub(r"\s+", " ", text) 
    return text


def analyze_pronunciation(audio_input, sentence):
    with st.spinner("🔍 Analyzing your pronunciation..."):
        waveform, sample_rate = torchaudio.load(audio_input)

        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.squeeze().numpy()

        # Transcribe with Whisper
        asr_pipe = load_asr_model()
        transcription = asr_pipe(waveform)["text"]

    st.success("✅ Done!")

    st.write("📝 **Your Transcription:**")
    st.markdown(f"### \"{transcription}\"")

    # Feedback
    if clean_text(sentence) == clean_text(transcription):
        st.success("💯 Great job! Your pronunciation is clear.")
    else:
        st.warning("🤔 Almost there! Try saying it more clearly.")
    
    return True


# Init only once
if "audio_input" not in st.session_state:
    st.session_state.audio_input = None

if "sentence" not in st.session_state:
    with st.spinner("Generating practice sentence..."):
        st.session_state.sentence = generate_sentence()

sentence = st.session_state.sentence


# Button trigger
if st.button("🔁 New Sentence"):
    with st.spinner("Generating practice sentence..."):
        st.session_state.sentence = generate_sentence()

sentence = st.session_state.sentence

st.divider()

# Generate speech with Kokoro
kokoro_pipe = KPipeline(lang_code='a')

generator = kokoro_pipe(
    sentence, voice='af_heart',
    speed=1
)
_, _, audio = next(generator)
audio_path = "output.wav"
sf.write(audio_path, audio, 24000)

st.subheader("🔊 Listen to the Pronunciation")
with open(audio_path, "rb") as audio_file:
    st.audio(audio_file.read(), format="audio/wav")

st.divider()
with st.form(key="Submit Audio"):
    st.subheader("🎤 Record Your Pronunciation")
    audio_input = st.audio_input("Record a voice message")
    submit_button = st.form_submit_button(label="✅ Submit Audio for Analysis")

    if submit_button:
        if audio_input:
            analyze_pronunciation(audio_input=audio_input, sentence=sentence)
        else:
            st.warning("⚠️ Please record your voice first before submitting.")

