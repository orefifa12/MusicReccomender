import streamlit as st
import os
from src.extractor import extract_features
from src.indexer import MusicIndexer

# Initialize indexer and load data (just like in your main.py)
indexer = MusicIndexer()
for file in os.listdir("data"):
    if file.endswith(('.wav', '.aif', '.m4a', '.mp3')):
        vec = extract_features(os.path.join("data", file))
        indexer.add_song(vec, file)

st.title("🎵 AI Music Recommender")

with st.sidebar:
    st.header("Settings")
    k = st.slider("Number of recommendations", 1, 10, 3)

# File uploader
uploaded_file = st.file_uploader("Upload a song to find similarities", type=['wav', 'mp3', 'm4a'])

if uploaded_file is not None:
    # Save temp file
    with open("temp_upload.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Process
    vec = extract_features("temp_upload.wav")
    matches = indexer.search(vec, k=k)
    
    # Display results
    st.subheader("Recommended for you:")
    if matches:
        songs, scores = matches
        for song, score in zip(songs, scores):
            st.write(f"▶️ {song} — **{score:.0f}% match**")
    else:
        st.warning("No songs in the index to search.")
        
    os.remove("temp_upload.wav")

