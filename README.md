!pip install gradio

import torch
import torchaudio
import librosa
import numpy as np
import gradio as gr
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Processor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import os
from typing import Tuple, Dict

# Configuration
SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Audio Processing ---
def load_audio(audio_file: str) -> Tuple[np.ndarray, int]:
    waveform, sample_rate = librosa.load(audio_file, sr=SAMPLE_RATE)
    return waveform, sample_rate

# --- 2. Emotion Recognition Module ---
def load_emotion_model():
    model = Wav2Vec2ForSequenceClassification.from_pretrained("audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim").to(DEVICE)
    processor = Wav2Vec2Processor.from_pretrained("audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim")
    return model, processor

emotion_model, emotion_processor = load_emotion_model()

def predict_emotion(waveform: np.ndarray) -> str:
    inputs = emotion_processor(waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
    predicted_id = torch.argmax(logits, dim=-1).item()
    emotions = ["Neutral", "Happy", "Sad", "Angry", "Fearful", "Disgusted", "Surprised"]
    return emotions[predicted_id]

# --- 3. Speech-to-Text (STT) Module ---
def load_stt_model():
    stt_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    stt_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(DEVICE)
    return stt_model, stt_processor

stt_model, stt_processor = load_stt_model()

def transcribe_audio(waveform: np.ndarray) -> str:
    input_features = stt_processor(waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features.to(DEVICE)
    with torch.no_grad():
        predicted_ids = stt_model.generate(input_features)
    transcription = stt_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

# --- 4. Sarcasm Detection Module ---
def load_sarcasm_model():
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-irony")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-irony").to(DEVICE)
    return model, tokenizer

sarcasm_model, sarcasm_tokenizer = load_sarcasm_model()

def detect_sarcasm(text: str) -> Tuple[str, float]:
    inputs = sarcasm_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        logits = sarcasm_model(**inputs).logits
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
    prediction = torch.argmax(probabilities, axis=0).item()
    return "Sarcastic" if prediction == 1 else "Not Sarcastic", probabilities[1]

# --- 5. Basic AI Response Module ---
def get_basic_ai_response(emotion: str, sarcasm_result: str) -> str:
    responses = {
        "Neutral": "Okay. How can I help you?",
        "Happy": "That's great to hear!",
        "Sad": "I'm here for you.",
        "Angry": "Please calm down.",
        "Fearful": "I understand your concern.",
        "Disgusted": "That's not good.",
        "Surprised": "Wow, that's unexpected!",
    }
    response = responses.get(emotion, "I'm not sure how to respond.")
    if sarcasm_result == "Sarcastic":
        response = "I detect sarcasm. " + response
    return response

# --- 6. Main Analysis Function ---
def analyze_audio(audio_file: str) -> Dict[str, str]:
    waveform, _ = load_audio(audio_file)
    emotion = predict_emotion(waveform)
    transcription = transcribe_audio(waveform)
    sarcasm_result, sarcasm_prob = detect_sarcasm(transcription)
    ai_response = get_basic_ai_response(emotion, sarcasm_result)

    return {
        "Emotion": emotion,
        "Transcription": transcription,
        "Sarcasm": f"{sarcasm_result} (Probability: {sarcasm_prob:.2f})",
        "AI Response": ai_response
    }

# --- 7. Gradio Interface ---
if __name__ == "__main__":
    ui = gr.Interface(
        fn=analyze_audio,
        inputs=gr.Audio(type="filepath", label="Record Audio"),
        outputs=gr.Textbox(label="Analysis Results"),
        title="SER System with Improved Models",
        description="This system detects emotion, transcribes speech, identifies sarcasm, and provides a response.",
    )
    ui.launch()
