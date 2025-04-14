import pandas as pd
import os

video_dir = r"C:\Users\manis\OneDrive\Desktop\Realtimevoice\data\processed\video_features"
audio_dir = r"C:\Users\manis\OneDrive\Desktop\Realtimevoice\data\sample\sample_audio_video"
audio_path = r"C:\Users\manis\OneDrive\Desktop\Realtimevoice\data\processed\transcripts\audio_transcripts.csv"
final_output = r"C:\Users\manis\OneDrive\Desktop\Realtimevoice\data\final\multimodal_features.csv"

df_audio = pd.read_csv(audio_path)

final_features = []
existing_files = 0

for _, row in df_audio.iterrows():
    filename = row['filename']
    video_csv = filename.replace(".wav", ".mp4.csv")
    video_file_path = os.path.join(video_dir, video_csv)
    audio_file_path = os.path.join(audio_dir, filename)

    if os.path.exists(video_file_path):
        try:
            df_video = pd.read_csv(video_file_path)
            avg_emotions = df_video.drop(columns=['frame']).mean(numeric_only=True).to_dict()

            final_features.append({
                "filename": filename,
                "audio_path": audio_file_path,
                "transcript": row.get("transcript", ""),
                "sentiment": row.get("sentiment", ""),
                "grammar_errors": row.get("grammar_errors", ""),
                **avg_emotions
            })
            existing_files += 1
        except Exception as e:
            print(f" Error processing {video_csv}: {e}")
    else:
        print(f"Skipping: No video features found for {filename}")

# Save to CSV
if final_features:
    df_final = pd.DataFrame(final_features)
    df_final.to_csv(final_output, index=False)
    print(f"\Final merged CSV saved to: {final_output}")
else:
    print(" No features to save!")

print(f"Processed {existing_files} out of {len(df_audio)} files.")
