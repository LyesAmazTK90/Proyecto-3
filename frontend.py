import io
import streamlit as st
from src.audio import transcribe_audio, analyze_tone, analyze_text
from src.streamlit_helpers import get_processor, \
    get_transcription_model, \
    get_voice_sentiment_model, \
    get_text_sentiment_model, \
    get_tokenizer, \
    get_combined_sentiment
from audio_recorder_streamlit import audio_recorder
import matplotlib.pyplot as plt
from matplotlib import cm
import librosa
import librosa.display
import numpy as np
import base64

st.set_page_config(
    page_title="Tone Analyzer",
    page_icon=":loud_sound:",
    layout="wide",
)

st.title("Audio Transcription and Analysis :microphone: :notes: :musical_note:")

# Crear columnas para separar contenido y componentes
c1, c2, c3 = st.columns([3,4,3], gap='small')

# Mensaje temporal mientras usuario sube archivo
no_result_message = "Record or upload a WAV or MP3 audio file to see results"

with c1:
    # Subir el archivo
    st.subheader("Upload your WAV or MP3 audio file:")

    # Creacion de texto temporal
    placeholder1 = st.empty()
    with placeholder1.container():
        st.write(no_result_message)
        
    uploaded_file = st.file_uploader("Upload your audio file", type=["wav", "mp3"], label_visibility='hidden')
    
    # Que el usuario pueda escuchar lo que subió:
    if uploaded_file:
        st.audio(uploaded_file, format="audio/wav")
        # uploaded_file = io.BytesIO(uploaded_file)
    
    st.divider()
    
    # O graba tu propio audio
    st.subheader("Or record your own audio:")
    
    # Creacion de texto temporal
    placeholder2 = st.empty()
    with placeholder2.container():
        st.write(no_result_message)
        
    # Utilizando audio_recorder_streamlit
    audio_bytes = audio_recorder(text="Click to record 5 seconds", recording_color="red", neutral_color="white", icon_name="microphone-lines", icon_size="3x", energy_threshold=(-1.0, 1.0), pause_threshold=5.0,)

    # Que el usuario pueda escuchar lo que grabó:
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        recording = io.BytesIO(audio_bytes)
    
with c2:
    # Creacion de texto temporal
    placeholder3 = st.empty()
    with placeholder3.container():
        #Graficas
        st.subheader("Audio Graphics:")
        st.write(no_result_message)

with c3:
    # Creacion de texto temporal
    placeholder4 = st.empty()
    with placeholder4.container():
        # Subtitulo
        st.subheader("Tone Analysis:")
        st.write(no_result_message)
        st.divider()
        
    # Creacion de texto temporal
    placeholder5 = st.empty()
    with placeholder5.container():
        # Subtitulo
        st.subheader("Audio Transcription:")
        st.write(no_result_message)
        st.divider()

    # Creacion de texto temporal
    placeholder6 = st.empty()
    with placeholder6.container():
        # Subtitulo
        st.subheader("Text Analysis:")
        st.write(no_result_message)
        st.divider()

    # Creacion de texto temporal
    placeholder7 = st.empty()
    with placeholder7.container():
        # Subtitulo
        st.subheader("Combined Analysis:")
        st.write(no_result_message)

# Cargar modelos y dependencias
voice_sentiment_model = get_voice_sentiment_model(
    'saved_models/voice_tone_model.json', 
    "saved_models/Emotion_Voice_Detection_Model_test2.h5")
processor = get_processor()
transcription_model = get_transcription_model()
tokenizer = get_tokenizer('saved_models/tokenizer.pkl')
text_sentiment_model = get_text_sentiment_model('saved_models/sentiment_model.sav')

if uploaded_file:
    
    with c2:
        # Borrar placeholder
        placeholder3.empty()
        st.subheader("Audio Graphics:")
        
        #Se carga el audio:
        data, sampling_rate = librosa.load(uploaded_file)
        
        plt.rc('font', size=7)

        #Se crea el plot de la gráfica:
        fig1 = plt.figure(figsize=(7, 3))
        librosa.display.waveshow(data, sr=sampling_rate, axis='s')
        st.pyplot(fig1)
        
        c2.write("")
        
        hop_length=128

        # Compute the mel-spectrogram:
        mel_spectrogram = librosa.feature.melspectrogram(y=data, sr=sampling_rate)

        # Compute the Mel-frequency cepstral coefficients (MFCCs)
        mfccs = librosa.feature.mfcc(S=librosa.power_to_db(mel_spectrogram), sr=sampling_rate)

        # Plot the mel-spectrogram
        fig2 = plt.figure(figsize=(7, 3))
        librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), sr=sampling_rate, hop_length=hop_length, y_axis="mel", x_axis="time", cmap=cm.jet)
        plt.colorbar(format="%+2.0f dB")
        plt.title("Mel-spectrogram")
        plt.tight_layout()
        st.pyplot(fig2)
        
        
    with c3:
        # Borrar placeholder
        placeholder4.empty()
        # Subtitulo
        st.subheader("Tone Analysis:")
        
        # Analisis de tono
        tone = analyze_tone(uploaded_file, voice_sentiment_model)
    
        formatted_tone = tone.replace("_", " ").title()

        st.markdown(f"<h3 style='text-align: center;'>{formatted_tone}</h3>", unsafe_allow_html=True)

        img_width = 150
        emotion = tone.split("_")[1]

        if emotion == "angry":
            image_path = "emojis/angry.png"
        elif emotion == "disgust":
            image_path = "emojis/disgust.png"
        elif emotion == "fear":
            image_path = "emojis/fear.png"
        elif emotion == "happy":
            image_path = "emojis/happy.png"
        elif emotion == "neutral":
            image_path = "emojis/neutral.png"
        elif emotion == "sad":
            image_path = "emojis/sad.png"
        elif emotion == "surprised":
            image_path = "emojis/surprised.png"
        
        # Función para cargar imagen en base64
        def load_image(image_path):
            with open(image_path, "rb") as f:
                image_data = f.read()
            encoded_image = base64.b64encode(image_data).decode()
            return encoded_image

        # Obtener la imagen en base64
        encoded_image = load_image(image_path)

        # Agrega la imagen centrada en la tercera columna con tamaño 150px
        st.markdown(f"""<div style="display: flex; justify-content: center;">
                    <img src="data:image/png;base64,{encoded_image}" style="width: 150px; height: auto;">
                    </div>""",unsafe_allow_html=True)
        
        st.divider()

        # Borrar placeholder
        placeholder5.empty()
        # Subtitulo
        st.subheader("Audio Transcription:")

        # Transcripcion
        transcription = transcribe_audio(uploaded_file, processor, transcription_model)

        st.markdown(f'<span style="font-size: 18px; text-align: center; display: block"><i>{transcription}</i></span>', unsafe_allow_html=True)

        st.divider()

        # Borrar placeholder
        placeholder6.empty()
        # Subtitulo
        st.subheader("Text Analysis:")

        # Analisis de texto
        text_sentiment = analyze_text(transcription, tokenizer, text_sentiment_model)

        st.markdown(f"<h3 style='text-align: center;'>{text_sentiment}</h3>", unsafe_allow_html=True)

        # Borrar placeholder
        placeholder6.empty()
        # Subtitulo
        st.subheader("Combined Analysis:")

        comb_result = get_combined_sentiment(emotion, text_sentiment)

        st.markdown(f"<h3 style='text-align: center;'>{text_sentiment}</h3>", unsafe_allow_html=True)
