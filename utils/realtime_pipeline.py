import threading
import time
from .audio_utils import record_audio, speech_to_text, analyze_sentiment, count_grammar_errors
from .video_utils import capture_emotions
import joblib
import pandas as pd
import os
import cv2

global_video_frame = None
recording_active = False

def get_current_frame():
    """Return the current video frame for display"""
    return global_video_frame

def process_realtime_input(duration=20, callback=None):
    """Process audio and video input in parallel
    
    Args:
        duration: Recording duration in seconds
        callback: Function to call with each video frame (for display)
    """
    global global_video_frame, recording_active
    
    print("Starting real-time communication analysis...")
    recording_active = True
    
    audio_results = {"audio": None, "transcript": "", "sentiment": 0, "grammar_errors": 0}
    video_results = {"emotions": {}}
    
    def audio_processing():
        audio = record_audio(duration=duration)
        audio_results["audio"] = audio
        transcript = speech_to_text(audio)
        audio_results["transcript"] = transcript
        print(f"Transcript: {transcript}")
        audio_results["sentiment"] = analyze_sentiment(transcript)
        audio_results["grammar_errors"] = count_grammar_errors(transcript)
    
    def video_processing():
        global global_video_frame, recording_active
        
        detector = None 
        cap = cv2.VideoCapture(0)
        emotion_scores = {
            "angry": 0, "disgust": 0, "fear": 0,
            "happy": 0, "sad": 0, "surprise": 0, "neutral": 0
        }
        
        frame_count = 0
        start_time = time.time()
        
        from fer import FER
        detector = FER(mtcnn=True)
        
        while recording_active and (time.time() - start_time) < duration:
            ret, frame = cap.read()
            if not ret:
                break
                
            global_video_frame = frame
            
            if frame_count % 5 == 0:
                results = detector.detect_emotions(frame)
                for r in results:
                    emotions = r["emotions"]
                    for k in emotion_scores:
                        emotion_scores[k] += emotions.get(k, 0)
            
            frame_count += 1
            
            if callback is not None:
                callback(frame)
        
        cap.release()
        global_video_frame = None
        recording_active = False
        
        for k in emotion_scores:
            emotion_scores[k] /= max(frame_count // 5, 1)  # process every 5th frame
        
        video_results["emotions"] = emotion_scores
    
    audio_thread = threading.Thread(target=audio_processing)
    video_thread = threading.Thread(target=video_processing)
    
    print("📹 Starting audio and video capture simultaneously...")
    audio_thread.start()
    video_thread.start()
    
    audio_thread.join()
    video_thread.join()
    print("✅ Audio and video processing complete!")
    
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
    
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "rf_model.pkl")
    
    try:
        model = joblib.load(model_path)
        score = model.predict(features)[0]
    except:
        print(f"Model not found at {model_path}. Using a simple average.")
        score = (audio_results["sentiment"] + (1 - min(audio_results["grammar_errors"], 10)/10) + 
                video_results["emotions"].get("happy", 0) + 
                video_results["emotions"].get("neutral", 0)) / 4
        score = max(0, min(score, 1))  # Ensure score is between 0 and 1
    
    print("\n===== Communication Score Analysis =====")
    print(f"Text sentiment: {audio_results['sentiment']:.2f}")
    print(f"Grammar errors: {audio_results['grammar_errors']}")
    print("\nFacial emotion analysis:")
    for emotion, value in video_results["emotions"].items():
        print(f"  - {emotion.capitalize()}: {value:.2f}")
    
    print(f"FINAL COMMUNICATION SCORE: {score:.2f}/1.0")
    
    return {
        "score": score,
        "transcript": audio_results["transcript"],
        "sentiment": audio_results["sentiment"],
        "grammar_errors": audio_results["grammar_errors"],
        "emotions": video_results["emotions"]
    }

def stop_recording():
    """Stop any active recording"""
    global recording_active
    recording_active = False