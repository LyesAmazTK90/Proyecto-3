import streamlit as st
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

@st.cache_resource
def get_processor():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    return processor

@st.cache_resource
def get_model():
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    return  model