import whisper
import os
import pandas as pd
from textblob import TextBlob
import language_tool_python

audio_dir = r"C:\Users\manis\OneDrive\Desktop\Realtimevoice\data\sample\sample_audio_video"
output_dir = r"C:\Users\manis\OneDrive\Desktop\Realtimevoice\data\processed\transcripts"
os.makedirs(output_dir, exist_ok=True)

model = whisper.load_model("base")
tool = language_tool_python.LanguageTool('en-US')

transcripts = []

for file in os.listdir(audio_dir):
    if file.endswith(".wav"):
        print(f"Transcribing: {file}")
        audio_path = os.path.join(audio_dir, file)

        result = model.transcribe(audio_path)
        text = result['text']

        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity

        matches = tool.check(text)
        grammar_errors = len(matches)

        transcripts.append({
            "filename": file,
            "transcript": text,
            "sentiment": sentiment,
            "grammar_errors": grammar_errors
        })

df = pd.DataFrame(transcripts)
df.to_csv(os.path.join(output_dir, "audio_transcripts.csv"), index=False)
print(" Transcriptions + Analysis saved.")
