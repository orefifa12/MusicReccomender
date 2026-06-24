from src.extractor import extract_features
from src.indexer import MusicIndexer

indexer = MusicIndexer()

# 1. Extract
vec = extract_features("data/The_Grinding_Sound.m4a")

# 2. Add to your indexer class
indexer.add_song(vec, "The_Grinding_Sound.m4a")

# 2. Test search
indexer.search(vec, 1)