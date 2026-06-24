from sklearn.cluster import KMeans
import numpy as np

CANDIDATE_POOL = 20
DEFAULT_BOOST = 15
DEFAULT_PENALTY = 10


def fit_clusters(vectors, n_clusters=5):
    """Group songs into n_clusters using K-Means on scaled feature vectors."""
    n_clusters = min(n_clusters, len(vectors))
    if n_clusters < 1:
        return None, np.array([])

    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(vectors)
    return kmeans, labels


def predict_cluster(kmeans, vector):
    """Return the cluster ID for a single feature vector."""
    return int(kmeans.predict(np.asarray(vector).reshape(1, -1))[0])


def rerank_candidates(songs, similarity_scores, song_clusters, query_cluster,
                        boost=DEFAULT_BOOST, penalty=DEFAULT_PENALTY):
    """
    Stage 2: adjust FAISS retrieval scores by cluster match.

    Returns a list of dicts sorted by relevance (highest first).
    """
    ranked = []
    for song, similarity in zip(songs, similarity_scores):
        same_cluster = song_clusters.get(song) == query_cluster
        relevance = similarity + boost if same_cluster else similarity - penalty
        ranked.append({
            "song": song,
            "similarity": similarity,
            "relevance": relevance,
            "same_cluster": same_cluster,
        })

    ranked.sort(key=lambda x: x["relevance"], reverse=True)
    return ranked
