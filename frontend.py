import streamlit as st
import pandas as pd
import numpy as np

st.title("Transcripción y Análisis de Audio :mic:")

# Subir el archivo
uploaded_file = st.file_uploader("Sube tu archivo de audio", type=["wav", "mp3"])

# Asegurar que se haya subido correctamente
if uploaded_file is None:
    raise ValueError("No file provided")

# Transcripccion (audio a texto)
transcription = "Output of Transcription Functionality" #transcribe_audio(uploaded_file)
st.write("Transcripción del audio:")
st.write(transcription)

tone = "Output of Tone Analyzer" #analyze_tone(uploaded_file)

st.write("Análisis de tono:")
if tone == "enfado":
    st.image("emojis/angry.png")
elif tone == "neutral":
    st.image("emojis/neutral.png")
elif tone == "alegría":
    st.image("emojis/happy.png")
else:
    st.write("Tono no reconocido")


text_analysis = "Output of Text Analyzer" #analyze_text(transcription)
st.write("Análisis del texto:")
st.write(text_analysis)
