# MusicReccomender

My small Python app that recommends similar songs from your library by comparing audio fingerprints. Each track is converted into a 59-dimensional feature vector with [librosa](https://librosa.org/) (MFCC, chroma, spectral contrast), indexed with [FAISS](https://github.com/facebookresearch/faiss), and re-ranked by K-Means cluster ("vibe") with [scikit-learn](https://scikit-learn.org/). (readme augmented by cursor:)

## How it works

1. **Extract** — Load up to 30 seconds of audio and build a fingerprint vector (`src/extractor.py`).
2. **Scale** — Standardize features across the library so no single feature dominates.
3. **Cluster** — K-Means groups songs into vibe clusters (no manual tags needed).
4. **Retrieve** — FAISS returns the top 20 most similar songs (candidate pool).
5. **Re-rank** — Boost same-vibe matches and penalize different-vibe matches (`src/re_ranker.py`).

## Project structure

```
MusicReccomender/
├── data/              # Your audio files (gitignored)
├── src/
│   ├── __init__.py
│   ├── extractor.py   # Audio feature extraction
│   ├── indexer.py     # FAISS L2 index + search
│   └── re_ranker.py   # K-Means clustering + re-ranking
├── app.py             # Streamlit UI (main entry point)
├── main.py            # FastAPI backend (optional)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Add your audio files to `data/`. Supported extensions: `.wav`, `.aif`, `.m4a`, `.mp3`.

## Usage

**Streamlit app (recommended):**

```bash
streamlit run app.py
```

Upload a song to get recommendations. Use the sidebar to adjust how many results to show and toggle **Same vibe only** to filter by cluster.

**FastAPI backend (optional):**

```bash
uvicorn main:app --reload
```

POST an audio file to `/recommend`. Note: the API uses basic FAISS search and does not include scaling or re-ranking yet.

## Dependencies

- `faiss-cpu` — fast vector similarity search
- `librosa` — audio loading and feature extraction
- `numpy` — numerical arrays
- `scikit-learn` — scaling, K-Means clustering, re-ranking
- `streamlit` — web UI
- `fastapi` / `uvicorn` — optional REST API (`main.py`)
