import pandas as pd
import os
import shutil
from moviepy.editor import VideoFileClip

# Fix the path by using raw strings (r prefix) or double backslashes
df = pd.read_csv(r'C:\Users\manis\OneDrive\Desktop\Realtimevoice\data\raw\train_sent_emo.csv')
sample_df = df.sample(5)

video_src_dir = r"D:\MELD\MELD.Raw\train\train_splits"
video_dest_dir = r"C:\Users\manis\OneDrive\Desktop\Realtimevoice\data\sample\sample_train_video"
audio_dest_dir = r"C:\Users\manis\OneDrive\Desktop\Realtimevoice\data\sample\sample_audio_video"

os.makedirs(video_dest_dir, exist_ok=True)
os.makedirs(audio_dest_dir, exist_ok=True)

for _, row in sample_df.iterrows():
    dialog_id = row["Dialogue_ID"]
    utterance_id = row["Utterance_ID"]
    filename = f"dia{dialog_id}_utt{utterance_id}.mp4"

    video_path = os.path.join(video_src_dir, filename)
    video_dst_path = os.path.join(video_dest_dir, filename)
    audio_dst_path = os.path.join(audio_dest_dir, f"dia{dialog_id}_utt{utterance_id}.wav")

    if os.path.exists(video_path):
        # Copy video
        shutil.copy(video_path, video_dst_path)
        print(f"Copied Video: {filename}")

        # Extract audio
        try:
            clip = VideoFileClip(video_path)
            clip.audio.write_audiofile(audio_dst_path)
            print(f"Extracted Audio: {audio_dst_path}")
        except Exception as e:
            print(f"Error extracting audio from {filename}: {e}")
    else:
        print(f"Missing: {filename}")