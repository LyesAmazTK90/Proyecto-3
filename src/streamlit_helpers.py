
import streamlit as st
from keras.models import model_from_json
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


@st.cache_resource
def get_processor():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    return processor


@st.cache_resource
def get_transcription_model():
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    return  model


@st.cache_resource
def get_voice_sentiment_model(model_config_json):
    with open(model_config_json, 'r') as json_file:
        loaded_model_json = json_file.read()
    
    model = model_from_json(loaded_model_json)
    # Se cargan las métricas en el modelo:
    model.load_weights("saved_models/Emotion_Voice_Detection_Model.h5")
    print("Loaded model from disk")
    
    # Definir el optimizador correctamente - igual que al entrenar el modelo
    opt = RMSprop(learning_rate=0.00001, decay=1e-6)

    # evaluate loaded model on test data
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    #TODO: Add label encoder here to get a coherent output (not numerical)
    return model