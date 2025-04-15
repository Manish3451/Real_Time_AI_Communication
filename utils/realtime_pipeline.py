import threading
from .audio_utils import record_audio, speech_to_text, analyze_sentiment, count_grammar_errors
from .video_utils import capture_emotions
import joblib
import pandas as pd
import os

def process_realtime_input():
    print("Starting real-time communication analysis...")
    
    audio_results = {"audio": None, "transcript": "", "sentiment": 0, "grammar_errors": 0}
    video_results = {"emotions": {}}
    
    def audio_processing():
        audio = record_audio(duration=20)
        audio_results["audio"] = audio
        transcript = speech_to_text(audio)
        audio_results["transcript"] = transcript
        print(f"Transcript: {transcript}")
        audio_results["sentiment"] = analyze_sentiment(transcript)
        audio_results["grammar_errors"] = count_grammar_errors(transcript)
    
    def video_processing():
        emotions = capture_emotions(duration=20)
        video_results["emotions"] = emotions
    
    audio_thread = threading.Thread(target=audio_processing)
    video_thread = threading.Thread(target=video_processing)
    
    print("Starting audio and video capture simultaneously...")
    audio_thread.start()
    video_thread.start()
    
    audio_thread.join()
    video_thread.join()
    print("Audio and video processing complete!")
    
    features = pd.DataFrame({
        "sentiment": [audio_results["sentiment"]],
        "grammar_errors": [audio_results["grammar_errors"]],
        "angry": [video_results["emotions"].get("angry", 0)],
        "disgust": [video_results["emotions"].get("disgust", 0)],
        "fear": [video_results["emotions"].get("fear", 0)],
        "happy": [video_results["emotions"].get("happy", 0)],
        "sad": [video_results["emotions"].get("sad", 0)],
        "surprise": [video_results["emotions"].get("surprise", 0)],
        "neutral": [video_results["emotions"].get("neutral", 0)]
    })
    
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "rf_model.pkl")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        score = model.predict(features)[0]
    else:
        print(f"Model not found at {model_path}. Using a simple average for demonstration.")
        score = (audio_results["sentiment"] + (1 - audio_results["grammar_errors"]/10) + 
                video_results["emotions"].get("happy", 0) + 
                video_results["emotions"].get("neutral", 0)) / 4
    
    print("\n===== Communication Score Analysis =====")
    print(f"Text sentiment: {audio_results['sentiment']:.2f}")
    print(f"Grammar errors: {audio_results['grammar_errors']}")
    print("\nFacial emotion analysis:")
    for emotion, value in video_results["emotions"].items():
        print(f"  - {emotion.capitalize()}: {value:.2f}")
    
    print(f" FINAL COMMUNICATION SCORE: {score:.2f}/1.0")
    
    return {
        "score": score,
        "transcript": audio_results["transcript"],
        "sentiment": audio_results["sentiment"],
        "grammar_errors": audio_results["grammar_errors"],
        "emotions": video_results["emotions"]
    }

if __name__ == "__main__":
    process_realtime_input()