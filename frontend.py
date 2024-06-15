import streamlit as st
from src.voice_to_text import transcribe_audio
from src.streamlit_helpers import get_processor, get_model

st.set_page_config(
        page_title="Tone Analyzer",
        page_icon=":loud_sound:",
        layout="wide",
    )

st.title("Transcripción y Análisis de Audio :microphone: :notes: :musical_note:")

# Subir el archivo
st.subheader("Sube tu archivo de audio WAV o MP3:")
uploaded_file = st.file_uploader("Sube tu archivo de audio", type=["wav", "mp3"], label_visibility='hidden')

# Crear columnas para separar contenido y componentes
c1, c2, c3 = st.columns(3)

# Mensaje temporal mientras usuario sube archivo
no_result_message = "Suba un archivo de audio WAV o MP3 para ver resultados"

with c1:
    # Subtitulo
    st.subheader("Transcripción del audio:")

    # Creacion de texto temporal
    placeholder1 = st.empty()
    with placeholder1.container():
        st.write(no_result_message)

with c2:
    # Subtitulo
    st.subheader("Análisis de tono:")

    # Creacion de texto temporal
    placeholder2 = st.empty()
    with placeholder2.container():
        st.write(no_result_message)

with c3:
    # Subtitulo
    c3.subheader("Análisis del texto:")

    # Creacion de texto temporal
    placeholder3 = st.empty()
    with placeholder3.container():
        st.write(no_result_message)

# Asegurar que se haya subido correctamente
if uploaded_file is not None:

    # Cargar modelos y procesadores
    processor = get_processor()
    model = get_model()

    with c1:
        # Borrar placeholder
        placeholder1.empty()

        # Transcripcion
        transcription = transcribe_audio(uploaded_file, processor, model)
        c1.write(transcription)

    with c2:
        # Borrar placeholder
        placeholder2.empty()

        # Analisis de tono
        tone = "Output of Tone Analyzer" #analyze_tone(uploaded_file)

        if tone == "enfado":
            st.image("emojis/angry.png")
        elif tone == "neutral":
            st.image("emojis/neutral.png")
        elif tone == "alegría":
            st.image("emojis/happy.png")
        else:
            st.write("Tono no reconocido")

    with c3:
        # Borrar placeholder
        placeholder3.empty()
        
        # Analisis de texto
        text_analysis = "Output of Text Analyzer" #analyze_text(transcription)
        st.write(text_analysis)
