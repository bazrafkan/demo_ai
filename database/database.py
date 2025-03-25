import faiss
import numpy as np

from database.i_database import IDatabase
from typing import List, Tuple

class Database(IDatabase):
    """
    A class to store and retrieve embeddings using FAISS.
    """
    def __init__(self, dimension: int) -> None:
        self.__index = faiss.IndexFlatL2(dimension)
        self.__dimension = dimension
        
    @property
    def dimension(self) -> int:
        return self.__dimension
    
   
    @property
    def length(self) -> int:
        return self.__index.ntotal


    def insert(self, embeddings: List[float]) -> None:
        self.__index.add(np.array(embeddings)) 


    def search(self, query_embedding: List[float], k: int) -> Tuple[np.ndarray, np.ndarray]:
        distances, indices = self.__index.search(np.array(query_embedding), k)
        return distances, indices