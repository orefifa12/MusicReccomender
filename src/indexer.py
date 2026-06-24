import faiss
import numpy as np

_DISTANCE_SCALE = 100.0


def distance_to_match_percent(distance: float) -> float:
    """Convert L2 distance to a 0–100 match score (100 = identical)."""
    return round(100.0 * np.exp(-distance / _DISTANCE_SCALE), 1)


class MusicIndexer:
    def __init__(self, dimension=13):
        """
        Initializes the MusicIndexer with a FAISS L2 index of specified dimension and an empty metadata list.
        
        Args:
            dimension (int): The dimensionality of the feature vectors to index. Default is 13.
        """
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []
        
   
    def add_song(self, vector, filename):
        """
        Adds a song's feature vector to the FAISS index and stores its filename as metadata.

        Args:
            vector (np.ndarray): The feature vector representing the song (should be 1D, length = dimension).
            filename (str): The name or identifier of the song file.

        Returns:
            None
        """
        vec = vector.astype(np.float32).reshape(1, -1)
        self.index.add(vec)
        self.metadata.append(filename)

        if self.metadata:
            print(self.metadata)
       

    def search(self, vector, k=5):
        """
        Search for the k most similar songs to the given feature vector.

        Args:
            vector (np.ndarray): The feature vector to search for (should be 1D, length = dimension).
            k (int): The number of closest matches to return.

        Returns:
            list: [closestSongsList, match_scores] where
                - closestSongsList is a list of k filenames (metadata) of the closest songs.
                - match_scores is a list of 0–100 match percentages (higher = more similar).

        Prints and returns nothing if the index is empty.
        """
        vec = vector.astype(np.float32).reshape(1, -1)
        closestSongsList = []
        match_scores = []

        if self.index.ntotal != 0:
            k = min(k, self.index.ntotal) # limit by the amount of items in index
            distances, indices = self.index.search(vec, k)
        else:
            return print("There is no Index to search. Try Again!")

        for songID, squared_distance in zip(indices[0], distances[0]):
            if songID != -1:
                closestSongsList.append(self.metadata[songID])
                distance = np.sqrt(float(squared_distance))
                match_scores.append(distance_to_match_percent(distance))

        return [closestSongsList, match_scores]
