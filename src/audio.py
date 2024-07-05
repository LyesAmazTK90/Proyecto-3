import torch
import librosa
import pandas as pd
import numpy as np
from io import BufferedIOBase
from tensorflow.keras.preprocessing.sequence import pad_sequences


def reset_buffer(func):
    def wrapper(*args, **kwargs):
        # Call the original function
        result = func(*args, **kwargs)
        
        # Reset the buffer of any file-like arguments
        for arg in args:
            if isinstance(arg, BufferedIOBase):
                arg.seek(0)
        
        # Also check in kwargs
        for arg in kwargs.values():
            if isinstance(arg, BufferedIOBase):
                arg.seek(0)
        
        return result
    return wrapper


@reset_buffer
def transcribe_audio(file, processor, model):
    audio, sr = librosa.load(file, sr=16000)
    input_values = processor(audio, sampling_rate=sr, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    transcription = transcription.lower().capitalize()
    return transcription 


@reset_buffer
def analyze_tone(file, model):
    # Cargar el archivo de audio subido
    X, sample_rate = librosa.load(file, res_type='kaiser_fast', duration=2.5, sr=22050*2, offset=0.5)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)

    # Transformar para conformar al formato del CNN
    df = pd.DataFrame(mfccs, columns=['feature']).T

    df = df.fillna(0)

    X = np.array(df)

    X_cnn = np.expand_dims(X, axis=2)

    # Prediccion
    preds = model.predict(X_cnn, 
                         batch_size=32, 
                         verbose=1)
    
    most_likely_emotion = preds.argmax(axis=1)

    emotion_pred_num = most_likely_emotion.astype(int).flatten()

    emotions_list = ['female_angry', 'female_disgust', 'female_fear', 'female_happy', 'female_neutral', 'female_sad', 'female_surprised', 'male_angry', 'male_disgust', 'male_fear', 'male_happy', 'male_neutral', 'male_sad', 'male_surprised']

    emotion_pred = emotions_list[emotion_pred_num[0]]

    return emotion_pred


def analyze_text(text, tokenizer, text_sentiment_model):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen= 30)
    prediction = text_sentiment_model.predict(padded_sequences)
    prediction_val = prediction[0][0]
    
    if prediction_val > 0.6:
        sentiment = 'Positive'
    elif prediction_val >= 0.4:
        sentiment = 'Neutral'
    else:
        sentiment = 'Negativo'

    return sentiment


def combined_sentiment(emotion, text_sentiment):
    
    # Se crea diccionario con las combinaciones de "emotion" y "text_sentiment":
    combinations = {
        ("angry", "Positive"): "sarcastic, most likely angry",
        ("angry", "Neutral"): "angry",
        ("angry", "Negative"): "angry",
        ("disgust", "Positive"): "ironic, most likely disgust",
        ("disgust", "Neutral"): "disgust",
        ("disgust", "Negative"): "disgust",
        ("fear", "Positive"): "surprise and likely fear",
        ("fear", "Neutral"): "fear",
        ("fear", "Negative"): "fear",
        ("happy", "Positive"): "happy",
        ("happy", "Neutral"): "happy",
        ("happy", "Negative"): "ironic, most likely happy",
        ("Neutral", "Positive"): "neutral",
        ("Neutral", "Neutral"): "neutral",
        ("Neutral", "Negative"): "most likely neutral",
        ("sad", "Positive"): "disappointment, most likely sad",
        ("sad", "Neutral"): "sad",
        ("sad", "Negative"): "sad",
        ("surprised", "Positive"): "surprised",
        ("surprised", "Neutral"): "surprised",
        ("surprised", "Negative"): "Disbelief or disappointment and surprise",
    }
    
    # Devolvemos el sentimiento combinado correspondiente:
    if (emotion, text_sentiment) in combinations:
        return combinations[(emotion, text_sentiment)]
    else:
        # Se devuelve un mensaje predeterminado si la combinación no está especificada:
        return f"unknown combination:{emotion}, {text_sentiment}"