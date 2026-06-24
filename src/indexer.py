import faiss
import numpy as np

class MusicIndexer:
    def __init__(self, dimension=13):
        self.index = index = faiss.IndexFlatL2(dimension)
        self.metadata = []
        
    def add_song(self, vector, filename):
        vec = vector.astype(np.float32).reshape(1, -1)
        self.index.add(vec)
        self.metadata.append(filename)

        if self.metadata:
            print(self.metadata)

    def search(self, vector, k=5):
        vec = vector.astype(np.float32).reshape(1, -1)
        closestSongsList = []

        if self.index.ntotal != 0:
            k = min(k, self.index.ntotal) # limit by the amount of items in index
            distances, indices = self.index.search(vec, k)
        else:
            return print("There is no Index to search. Try Again!")

        for songID in indices[0]: 
            #TODO: Go back and add a maximum distance
            # print(self.metadata[songID]) #for debugging/testing
            if songID != -1:
                closestSongsList.append(self.metadata[songID])
        
        return closestSongsList
