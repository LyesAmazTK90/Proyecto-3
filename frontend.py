import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
        page_title="Tone Analyzer",
        page_icon=":loud_sound:",
        layout="wide",
    )

st.title("Transcripción y Análisis de Audio :microphone: :notes: :musical_note:")

# Subir el archivo
st.subheader("Sube tu archivo de audio WAV o MP3:")
uploaded_file = st.file_uploader("Sube tu archivo de audio", type=["wav", "mp3"], label_visibility='hidden')

# Asegurar que se haya subido correctamente
if uploaded_file is not None:

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
