import os
from fastapi import FastAPI, UploadFile, File
from src.extractor import extract_features
from src.indexer import MusicIndexer
import shutil

app = FastAPI()
indexer = MusicIndexer()

# Populate indexer on startup (Mock-up logic)
@app.on_event("startup")
def startup_event():
    for file in os.listdir("data"):
        if file.endswith(('.wav', '.aif', '.m4a', 'mp3')):
            path = os.path.join("data", file)
            vec = extract_features(path)
            indexer.add_song(vec, file)

@app.post("/recommend")
async def recommend(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 2. Extract features
    vec = extract_features(temp_path)
    
    # 3. Search
    results, scores = indexer.search(vec, k=3)
    
    # Cleanup temp file
    os.remove(temp_path)
    
    return {"filename": file.filename, "matches": results, "scores": scores}