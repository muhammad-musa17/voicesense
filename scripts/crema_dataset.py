import os
import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import librosa
import numpy as np


class CREMADataset(Dataset):
    def __init__(self, csv_file, sample_rate=16000, duration=3.0, transform=None):
        df = pd.read_csv(csv_file)
        df = df[df['file_path'].apply(os.path.exists)].reset_index(drop=True)
        self.data = df

        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration)
        self.transform = transform

        self.label2idx = {label: idx for idx, label in enumerate(self.data['emotion'].unique())}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filepath = row['file_path']
        emotion = row['emotion']
        label = self.label2idx[emotion]

        waveform, sr = librosa.load(filepath, sr=self.sample_rate)

        if len(waveform) > self.num_samples:
            waveform = waveform[:self.num_samples]
        else:
            waveform = np.pad(waveform, (0, self.num_samples - len(waveform)))

        mel_spec = librosa.feature.melspectrogram(y=waveform, sr=self.sample_rate, n_mels=64)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max) 

        mean = mel_spec_db.mean()
        std = mel_spec_db.std()
        if std != 0:
            mel_spec_db = (mel_spec_db - mean) / std
        else:
            mel_spec_db = mel_spec_db - mean


        mel_tensor = torch.tensor(mel_spec_db).unsqueeze(0).float()

        return mel_tensor, label
