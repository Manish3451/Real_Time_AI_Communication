import speech_recognition as sr
from textblob import TextBlob
import language_tool_python

def record_audio(duration=20):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    print("Recording... Speak now!")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, phrase_time_limit=duration)
    print("Recording done.")
    return audio

def speech_to_text(audio):
    recognizer = sr.Recognizer()
    try:
        text = recognizer.recognize_google(audio)
        return text
    except Exception as e:
        print("Speech Recognition Error:", e)
        return ""

def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity

def count_grammar_errors(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    return len(matches)
