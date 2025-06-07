import gradio as gr
import torch
import librosa
import numpy as np

from scripts.cnn_model import EmotionCNN
from scripts.crema_dataset import CREMADataset

model = EmotionCNN(num_classes=6)
model.load_state_dict(torch.load("models/emotion_cnn.pth", map_location=torch.device("cpu")))
model.eval()

dummy_dataset = CREMADataset("data/crema_labels.csv")
idx2label = dummy_dataset.idx2label

SAMPLE_RATE = 16000
NUM_SAMPLES = SAMPLE_RATE * 3

def predict_emotion(audio):
    waveform, sr = librosa.load(audio, sr=SAMPLE_RATE)

    if len(waveform) > NUM_SAMPLES:
        waveform = waveform[:NUM_SAMPLES]
    else:
        waveform = np.pad(waveform, (0, NUM_SAMPLES - len(waveform)))

    mel_spec = librosa.feature.melspectrogram(y=waveform, sr=SAMPLE_RATE, n_mels=64)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    mean, std = mel_spec_db.mean(), mel_spec_db.std()
    if std != 0:
        mel_spec_db = (mel_spec_db - mean) / std
    else:
        mel_spec_db = mel_spec_db - mean

    tensor = torch.tensor(mel_spec_db).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        output = model(tensor)
        predicted = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1)[0][predicted].item()

    return f"{idx2label[predicted]} ({confidence*100:.2f}% confidence)"

app = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Audio(sources=["upload", "microphone"], type="filepath", label="Upload or Record Audio"),
    outputs=gr.Text(label="Predicted Emotion"),
    title="🎙️ Real-Time Voice Emotion Detector",
    description="Upload or record a voice sample and get an emotion prediction using a CNN model trained on CREMA-D."
)

if __name__ == "__main__":
    app.launch()
