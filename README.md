# MusicReccomender

A small Python project that recommends similar songs by comparing audio fingerprints. Each track is converted into a 13-dimensional MFCC vector with [librosa](https://librosa.org/), then indexed with [FAISS](https://github.com/facebookresearch/faiss) for fast nearest-neighbor search.

## How it works

1. **Extract** — Load up to 30 seconds of audio and compute mean MFCC features (`src/extractor.py`).
2. **Index** — Store feature vectors in a FAISS L2 index with filename metadata (`src/indexer.py`).
3. **Search** — Given a query track, find the closest matches in the index.

## Project structure

```
MusicReccomender/
├── data/              # Your audio files (gitignored)
├── src/
│   ├── __init__.py
│   ├── extractor.py   # MFCC feature extraction
│   ├── indexer.py     # FAISS vector index + search
│   └── service.py     # FastAPI backend (planned)
├── tests/             # Reserved for unit tests
├── main.py            # Entry point / demo script
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

Run the demo:

```bash
python main.py
```

This extracts features from a sample file in `data/`, adds it to the index, and runs a similarity search. Update the file path in `main.py` to use a different track.

## Dependencies

- `faiss-cpu` — vector similarity search
- `librosa` — audio loading and MFCC extraction
- `numpy` — numerical arrays
