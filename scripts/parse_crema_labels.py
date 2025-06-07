import os
import pandas as pd

data_dir = "data/CREMA-D"
file_list = [f for f in os.listdir(data_dir) if f.endswith(".wav")]

emotion_map = {
    "ANG": "Anger",
    "DIS": "Disgust",
    "FEA": "Fear",
    "HAP": "Happy",
    "NEU": "Neutral",
    "SAD": "Sad"
}

data = []

for file_name in file_list:
    parts = file_name.split("_")
    actor_id = parts[0]
    emotion_code = parts[2]
    emotion = emotion_map.get(emotion_code, "Unknown")

    full_path = os.path.abspath(os.path.join(data_dir, file_name))
    
    data.append({
        "file_path": full_path,
        "actor_id": int(actor_id),
        "emotion": emotion
    })

df = pd.DataFrame(data)
df.to_csv("data/crema_labels.csv", index=False)

print(f" Created crema_labels.csv with {len(df)} samples.")
