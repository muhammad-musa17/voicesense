# 🎙️ VoiceSense: Real-Time Speech Emotion Detection

VoiceSense is a PyTorch-based project that classifies human emotions (like Happy, Sad, Anger, Fear, Disgust, Neutral) from speech using deep learning and the CREMA-D dataset.

## 🔧 Features
- Preprocessing with librosa and torchaudio
- CNN model trained on Mel-spectrograms
- Gradio web demo for testing with mic or audio upload
- Trained to ~88% accuracy on 6 emotion classes

## 🧠 Tech Stack
- Python, PyTorch, Librosa, Matplotlib
- Jupyter Notebooks for analysis
- Gradio for live interface
- GitHub for version control

## 📊 Model Performance
Achieved 88.48% accuracy after 20 epochs of training.

## 🤝 Contribution
Feel free to fork or submit pull requests!

## 🚀 Run Locally

```bash
git clone https://github.com/muhammad-musa17/voicesense.git
cd voicesense
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py  # Run Gradio app



