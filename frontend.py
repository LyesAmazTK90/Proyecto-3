import streamlit as st
from src.audio import transcribe_audio, analyze_tone, analyze_text
from src.streamlit_helpers import   get_processor,\
                                    get_transcription_model,\
                                    get_voice_sentiment_model,\
                                    get_text_sentiment_model,\
                                    get_tokenizer

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

    with c1:
        # Borrar placeholder
        placeholder1.empty()

        # Cargar modelos y procesadores
        processor = get_processor()
        transcription_model = get_transcription_model()

        # Transcripcion
        transcription = transcribe_audio(uploaded_file, processor, transcription_model)
        
        c1.write(transcription)

    with c2:
        # Borrar placeholder
        placeholder2.empty()

        # Cargar modelo
        voice_sentiment_model = get_voice_sentiment_model('saved_models/voice_tone_model.json', "saved_models/Emotion_Voice_Detection_Model_test2.h5")

        # Analisis de tono
        tone = analyze_tone(uploaded_file, voice_sentiment_model)

        st.write("Tono:", tone)

        img_width = 150
        emotion = tone.split("_")[1]

        if emotion == "angry":
            st.image("emojis/angry.png", width=img_width)
        elif emotion == "disgust":
            st.image("emojis/disgust.jpeg", width=img_width)
        elif emotion == "fear":
            st.image("emojis/fear.jpeg", width=img_width)
        elif emotion == "happy":
            st.image("emojis/happy.jpeg", width=img_width)
        elif emotion == "neutral":
            st.image("emojis/neutral.png", width=img_width)
        elif emotion == "sad":
            st.image("emojis/sad.png", width=img_width)
        elif emotion == "surprised":
            st.image("emojis/surpirsed.png", width=img_width)

    with c3:
        # Borrar placeholder
        placeholder3.empty()
        
        tokenizer = get_tokenizer('saved_models/tokenizer.pkl')
        text_sentiment_model = get_text_sentiment_model('saved_models/sentiment_model.sav')

        # Analisis de texto
        num_value, text_sentiment = analyze_text(transcription, tokenizer, text_sentiment_model)

        st.write(f"Valor numerico: {num_value}")
        st.write(text_sentiment)
