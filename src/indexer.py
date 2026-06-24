import faiss
import numpy as np

_DISTANCE_SCALE = 100.0
dimensionSize = 59  # 40 MFCCs + 12 Chroma + 7 Spectral Contrast


def distance_to_match_percent(distance: float) -> float:
    """Convert L2 distance to a 0–100 match score (100 = identical)."""
    return round(100.0 * np.exp(-distance / _DISTANCE_SCALE), 1)


class MusicIndexer:
    def __init__(self, dimension=dimensionSize):
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []

    def add_song(self, vector, filename, cluster_id=None):
        vec = vector.astype(np.float32).reshape(1, -1)
        self.index.add(vec)
        self.metadata.append({"filename": filename, "cluster_id": cluster_id})

    def get_cluster(self, filename):
        for entry in self.metadata:
            if entry["filename"] == filename:
                return entry["cluster_id"]
        return None

    def search(self, vector, k=5):
        vec = vector.astype(np.float32).reshape(1, -1)
        closestSongsList = []
        match_scores = []

        if self.index.ntotal == 0:
            return None

        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(vec, k)

        for songID, squared_distance in zip(indices[0], distances[0]):
            if songID != -1:
                closestSongsList.append(self.metadata[songID]["filename"])
                distance = np.sqrt(float(squared_distance))
                match_scores.append(distance_to_match_percent(distance))

        return [closestSongsList, match_scores]
