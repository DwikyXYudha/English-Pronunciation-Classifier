import streamlit as st 
import torchaudio
from google import genai
from google.genai import types
import soundfile as sf
from scipy.stats import skew, kurtosis
from kokoro import KPipeline
import re
import os
import numpy as np
from dotenv import load_dotenv
from transformers import pipeline
import joblib
import pyloudnorm as pyln
import polars as pl
from torchaudio.transforms import MFCC
import torch

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
            temperature=0.5),
        contents="Generate one A2-level English sentence suitable for speaking practice."
    )
    return response.text

def decode_label(label):
    
    if label == 4:
        return "Excellent"
    elif label == 3:
        return "Good" 
    elif label == 2:
        return "Understandable"
    elif label == 1:
        return "Poor"
    else:
        return "Extremely poor"


# Load model and encoder
best_model = joblib.load("models/self-trained/GradientBoostingClassifier.pkl")
label_encoder = joblib.load("files/label_encoder.pkl")

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

def extract_statistical_features(mfcc):
    if isinstance(mfcc, torch.Tensor):
        mfcc = mfcc.numpy()

    features = []
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))
    features.extend(np.min(mfcc, axis=1))
    features.extend(np.max(mfcc, axis=1))
    features.extend(np.median(mfcc, axis=1))
    features.extend(skew(mfcc, axis=1))
    features.extend(kurtosis(mfcc, axis=1))
    delta = np.diff(mfcc, axis=1)
    features.extend(np.mean(delta, axis=1))
    features.extend(np.std(delta, axis=1))
    return np.array(features)


def analyze_pronunciation(audio_input, sentence):
    with st.spinner("🔍 Analyzing your pronunciation..."):
        n_mfcc = 13
        waveform, sample_rate = torchaudio.load(audio_input)

        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.squeeze().numpy()

        # measure the loudness first 
        meter = pyln.Meter(sample_rate) # create BS.1770 meter
        loudness = meter.integrated_loudness(waveform)

        # loudness normalize audio to -23 dB LUFS
        norm_audio  = pyln.normalize.loudness(waveform, loudness, -23.0)
        norm_audio_tensor = torch.tensor(norm_audio, dtype=torch.float32)

        # Extract MFCC 
        mfcc_transform = MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False}
        )
        mfcc = mfcc_transform(norm_audio_tensor)

        features = extract_statistical_features(mfcc).reshape(1, -1)

        # Predict
        prediction = best_model.predict(features)[0]
        category = label_encoder.inverse_transform([prediction])[0]
        label = decode_label(category)

        st.success(f"✅ Your Pronunciation: **{label}**")

        # Transcribe with Whisper
        asr_pipe = load_asr_model()
        transcription = asr_pipe(waveform)["text"]

    st.write("📝 **Your Transcription:**")
    st.markdown(f"### \"{transcription}\"")
    
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

