from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np



class IDatabase(ABC):
    """Interface for the database class"""

    @abstractmethod
    def __init__(self, dimension: int) -> None:
        """
        Initialize the database class.

        Args:
            dimension (int): The dimensionality of the vector embeddings.
        """
        pass
    
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Property to return the dimensionality of the vector embeddings.
        """
        pass
    
    
    @property
    @abstractmethod
    def length(self) -> int:
        """
        Property to return the number of embeddings in the database.
        """
        pass

    
    @abstractmethod
    def insert(self, embeddings: List[float]) -> None:
        """
        Insert embeddings into the database.
        
        Args:
            embeddings (list): The embeddings to insert into the database
        """
        pass


    @abstractmethod  
    def search(self, query_embedding: List[float], k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar embeddings in the database.
        
        Args:
            query_embedding (list): The query embedding to search for.
            k (int): The number of similar embeddings to retrieve.
            
        Returns:
            tuple: A tuple containing the distances and indices of the similar embeddings.
        """
        pass

