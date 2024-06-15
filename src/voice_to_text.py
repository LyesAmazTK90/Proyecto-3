import torch
import librosa

def transcribe_audio(file_path, processor, model):
    audio, sr = librosa.load(file_path, sr=16000)
    input_values = processor(audio, sampling_rate=sr, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription