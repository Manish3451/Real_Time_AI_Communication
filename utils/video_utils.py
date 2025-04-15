import cv2
from fer import FER
import numpy as np

def capture_emotions(duration=20):
    detector = FER(mtcnn=True)
    cap = cv2.VideoCapture(0)
    emotion_scores = {
        "angry": 0, "disgust": 0, "fear": 0,
        "happy": 0, "sad": 0, "surprise": 0, "neutral": 0
    }

    print("📸 Capturing emotions... Look at the camera.")
    frame_count = 0
    while frame_count < duration * 5:  # ~5fps
        ret, frame = cap.read()
        if not ret:
            break
        results = detector.detect_emotions(frame)
        for r in results:
            emotions = r["emotions"]
            for k in emotion_scores:
                emotion_scores[k] += emotions.get(k, 0)
        frame_count += 1

    cap.release()
    print("✅ Emotion capture complete.")
    for k in emotion_scores:
        emotion_scores[k] /= frame_count  # normalize
    return emotion_scores
