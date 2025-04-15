import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import time
import cv2
from PIL import Image
import io

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.realtime_pipeline import process_realtime_input, get_current_frame, stop_recording

st.set_page_config(page_title="Communication Score Analyzer", page_icon="🎙️", layout="wide")

st.title("Communication Score Analyzer")
st.write("This app analyzes your verbal and non-verbal communication in real-time.")

video_placeholder = st.empty()

if 'analyzing' not in st.session_state:
    st.session_state.analyzing = False
if 'results' not in st.session_state:
    st.session_state.results = None

def update_video_feed():
    while st.session_state.analyzing:
        frame = get_current_frame()
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            video_placeholder.image(buf, caption="Live Feed", use_column_width=True)
        time.sleep(0.1)  

col1, col2 = st.columns([1, 1])
duration = col1.slider("Recording Duration (seconds)", min_value=10, max_value=60, value=20)

if col2.button("Start Analysis"):
    if not st.session_state.analyzing:
        st.session_state.analyzing = True
        video_placeholder.empty()
        
        import threading
        feed_thread = threading.Thread(target=update_video_feed)
        feed_thread.daemon = True
        feed_thread.start()
        
        progress_bar = st.progress(0)
        start_time = time.time()
        
        try:
            results = process_realtime_input(duration=duration)
            st.session_state.results = results
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Make sure your camera and microphone are working properly.")
        finally:
            st.session_state.analyzing = False
            stop_recording()  

        video_placeholder.empty()

if st.session_state.results is not None:
    results = st.session_state.results
    
    st.success("Analysis complete!")
    
    st.header(f"Your Communication Score: {results['score']:.2f}/1.0")
    st.progress(min(results['score'], 1.0))  # Progress bar for score
    
    st.subheader("Speech Transcript")
    st.write(results['transcript'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Sentiment Score", f"{results['sentiment']:.2f}")
    with col2:
        st.metric("Grammar Errors", results['grammar_errors'])
    
    st.subheader("Facial Emotion Analysis")
    emotions_df = pd.DataFrame({
        "Emotion": list(results['emotions'].keys()),
        "Score": list(results['emotions'].values())
    })
    
    emotions_df = emotions_df.sort_values(by="Score", ascending=False)
    
    st.bar_chart(emotions_df.set_index("Emotion"))
    
    st.subheader("Score Breakdown")
    st.write("""
    Your communication score is calculated based on:
    - Speech sentiment (positive vs negative tone)
    - Grammar correctness
    - Facial expressions (happy/neutral expressions improve the score)
    """)

st.markdown("---")
st.write("Created as a demo project for real-time communication analysis.")