import cv2
from fer import FER
import os 
import pandas as pd 

output_dir = r"C:\Users\manis\OneDrive\Desktop\Realtimevoice\data\processed\video_features"
os.makedirs(output_dir, exist_ok=True)

video_path = r"C:\Users\manis\OneDrive\Desktop\Realtimevoice\data\sample\sample_train_video"
emotion_detector = FER()

for filename in os.listdir(video_path):
    if filename.endswith(".mp4"):
        fullpath = os.path.join(video_path, filename)
        cap = cv2.VideoCapture(fullpath)

        frame_count = 0
        emotion_data = []  # 🔁 moved inside the loop

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 10 == 0:
                result = emotion_detector.detect_emotions(frame)
                if result:
                    emotions = result[0]['emotions']
                    emotions['frame'] = frame_count
                    emotion_data.append(emotions)

            frame_count += 1
        cap.release()

        if emotion_data:
            df_emotions = pd.DataFrame(emotion_data)
            output_filename = filename + ".csv"  # 🔁 keep the .mp4.csv format
            output_path = os.path.join(output_dir, output_filename)
            df_emotions.to_csv(output_path, index=False)
            print(f"✅ Saved emotions to {output_path}")
        else:
            print(f"⚠️ No emotions detected in {filename}")


