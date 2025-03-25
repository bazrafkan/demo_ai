from abc import ABC, abstractmethod
from typing import List
from database import IDatabase

class IRetrieve(ABC):
    """Interface for the Retrieve class"""

    @abstractmethod
    def __init__(self, 
                 documents: List[str], 
                 embedding_model: str = "all-MiniLM-L6-v2", 
                 database: None | IDatabase = None) -> None:
        """
        Initialize the Retrieve class.

        Args:
            documents (list): A list of documents to retrieve and rank.
            embedding_model (str): The name of the Sentence Transformer model to use for embeddings.
            database (Database): An instance of the Database class to store the embeddings.
        """
        pass


    @abstractmethod
    def retrieve_and_rank(self, query: str, k: int=3) -> List[str]:
        """
        Retrieve relevant documents and rank them based on similarity to the query.

        Args:
            query (str): The input query.
            k (int): The number of documents to retrieve.
        Returns:
            list: Ranked retrieved documents
        """
        pass