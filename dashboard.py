import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Emotion Analytics Dashboard", layout="wide")

st.title(" Emotion Detection Analytics")

CSV_FILE = "emotion_log.csv"


# Load data 
if not os.path.exists(CSV_FILE):
    st.warning("No emotion data found. Run emotion.py first.")
    st.stop()

df = pd.read_csv(CSV_FILE)

if df.empty:
    st.warning("Emotion log is empty.")
    st.stop()

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")


# Summary metrics
st.subheader("Summary")

col1, col2 = st.columns(2)
col1.metric("Total Records", len(df))
col2.metric("Unique Emotions", df["emotion"].nunique())


# Emotion Distribution
st.subheader("Emotion Distribution")
emotion_counts = df["emotion"].value_counts()
st.bar_chart(emotion_counts)


# Timeline
st.subheader("⏱ Emotion Timeline")
timeline = df.groupby([df["timestamp"].dt.floor("S"), "emotion"]).size().unstack(fill_value=0)
st.line_chart(timeline)


# Raw Data
with st.expander("View Raw Data"):
    st.dataframe(df)


# Download button
st.download_button(
    label="Download Emotion Log",
    data=df.to_csv(index=False),
    file_name="emotion_log.csv",
    mime="text/csv"
)
