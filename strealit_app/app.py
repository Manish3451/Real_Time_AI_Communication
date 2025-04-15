import streamlit as st
import pandas as pd
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.realtime_pipeline import process_realtime_input

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.realtime_pipeline import process_realtime_input

st.set_page_config(page_title="Real-time Communication Analyzer", page_icon="🎤")

st.title("Real-time Communication Analyzer")
st.write("This app analyzes your verbal and non-verbal communication and provides a score.")

if st.button("Start Analysis"):
    with st.spinner("Recording and analyzing... Please speak clearly and look at the camera."):
        results = process_realtime_input()
    
    st.success("Analysis complete!")
    
    st.subheader(f"Your Communication Score: {results['score']:.2f}/1.0")
    st.progress(results['score'])
    
    st.subheader("Transcript")
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
    st.bar_chart(emotions_df.set_index("Emotion"))