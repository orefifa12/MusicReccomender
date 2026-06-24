import os
import streamlit as st
from sklearn.preprocessing import StandardScaler

from src.extractor import extract_features
from src.indexer import MusicIndexer
from src.re_ranker import (
    CANDIDATE_POOL,
    fit_clusters,
    predict_cluster,
    rerank_candidates,
)

NUM_CLUSTERS = 5


@st.cache_resource
def load_library():
    all_vectors = []
    all_files = []

    for file in os.listdir("data"):
        if file.endswith((".wav", ".aif", ".m4a", ".mp3")):
            vec = extract_features(os.path.join("data", file))
            if vec is not None:
                all_vectors.append(vec)
                all_files.append(file)

    if not all_vectors:
        return None

    scaler = StandardScaler()
    all_vectors_scaled = scaler.fit_transform(all_vectors)

    kmeans, labels = fit_clusters(all_vectors_scaled, n_clusters=NUM_CLUSTERS)
    song_clusters = {all_files[i]: int(labels[i]) for i in range(len(all_files))}

    indexer = MusicIndexer()
    for i, file in enumerate(all_files):
        indexer.add_song(all_vectors_scaled[i], file, cluster_id=song_clusters[file])

    return indexer, scaler, kmeans, song_clusters


st.title("🎵 AI Music Recommender")

library = load_library()

if library is None:
    st.error("No songs could be processed. Add audio files to data/ and check extractor.py.")
    st.stop()

indexer, scaler, kmeans, song_clusters = library

with st.sidebar:
    st.header("Settings")
    k = st.slider("Number of recommendations", 1, 10, 3)
    same_vibe_only = st.toggle("Same vibe only", value=False)
    st.caption(f"Library: {len(song_clusters)} songs · {NUM_CLUSTERS} clusters")

# File uploader
uploaded_file = st.file_uploader("Upload a song to find similarities", type=['wav', 'mp3', 'm4a'])

if uploaded_file is not None:
    # Save temp file
    with open("temp_upload.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    vec = extract_features("temp_upload.wav")
    if vec is None:
        st.error("Could not extract features from the uploaded file.")
    else:
        vec = scaler.transform([vec])[0]
        query_cluster = predict_cluster(kmeans, vec)

        st.caption(f"Detected cluster: **{query_cluster}**")

        # Stage 1: FAISS retrieval 
        candidates = indexer.search(vec, k=CANDIDATE_POOL)

        st.subheader("Recommended for you:")
        if candidates:
            pool_songs, pool_scores = candidates

            # Stage 2: re-rank by cluster match could be improved by supervised learning
            ranked = rerank_candidates(
                pool_songs, pool_scores, song_clusters, query_cluster
            )

            if same_vibe_only:
                ranked = [r for r in ranked if r["same_cluster"]]

            if not ranked:
                st.warning("No same-vibe matches found. Try turning off the filter.")
            else:
                for result in ranked[:k]:
                    cluster_tag = "same vibe" if result["same_cluster"] else "different vibe"
                    st.write(
                        f"▶️ {result['song']} — **{result['similarity']:.0f}% match** · {cluster_tag}"
                    )
        else:
            st.warning("No songs in the index to search.")

    os.remove("temp_upload.wav")
