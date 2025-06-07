# 🎙️ VoiceSense: Real-Time Emotion Detection from Voice

**VoiceSense** is an AI-powered application that predicts emotions from speech using deep learning and audio signal processing.

Built with **PyTorch**, **Librosa**, and **Gradio**, the model is trained on the **CREMA-D dataset** to classify 6 emotions:  
**Anger, Disgust, Fear, Happy, Neutral, Sad**.

---

## 🚀 Features

- 🎧 Upload or record a voice sample
- 🧠 Predicts emotion with confidence score
- 🖼️ Built using Mel Spectrograms and CNN
- 🔬 Achieves ~88% accuracy on CREMA-D test set

---

## 🧠 Model Details

- CNN trained on Mel Spectrograms
- 6-class softmax output
- Trained for 20 epochs
- Accuracy: **88.48%**

---

## 🛠️ Tech Stack

- Python
- PyTorch
- Librosa
- Gradio
- CREMA-D Dataset

---

## 📦 How to Run

1. Clone the repo

```bash
git clone https://github.com/muhammad-musa17/voicesense.git
cd voicesense
