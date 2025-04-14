import pandas as pd 
import os 
import librosa
import numpy as np

audio_file_path = r"C:\Users\manis\OneDrive\Desktop\Realtimevoice\data\sample\sample_audio_video"
feature_output_path = r"C:\Users\manis\OneDrive\Desktop\Realtimevoice\data\processed\audio_features"

os.makedirs(feature_output_path, exist_ok=True)

for filename in os.listdir(audio_file_path):
    if filename.endswith(".wav"):
        filepath = os.path.join(audio_file_path, filename)
        
        try:
            y, sr = librosa.load(filepath, sr=16000)
            
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

            max_len = 300
            if mfcc.shape[1] < max_len:
                pad_width = max_len - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
            else:
                mfcc = mfcc[:, :max_len]

            feature_filename = filename.replace(".wav", ".npy")
            np.save(os.path.join(feature_output_path, feature_filename), mfcc)

            print(f"Processed: {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
