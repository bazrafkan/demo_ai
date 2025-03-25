from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from database import Database, IDatabase
from retrieve.i_retrieve import IRetrieve
from typing import List

class Retrieve(IRetrieve):
    def __init__(self, 
                 documents: List[str], 
                 embedding_model: str = "all-MiniLM-L6-v2", 
                 database: None | IDatabase = None) -> None:
        """
        Initialize the Retrieve class with the documents and the embedding model.
        
        Args:
            documents (list): A list of documents to retrieve and rank.
            embedding_model (str): The name of the Sentence Transformer model to use for embeddings.
            database (Database): An instance of the Database class to store the embeddings.
        """
        self.__model = SentenceTransformer(embedding_model)
        self.__documents = documents
        self.__embeddings = self.__model.encode(documents)
        # Store in FAISS for retrieval
        dimension = self.__embeddings.shape[1]
        if database is None:
            self.__database = Database(dimension)
        else:
            self.__database = database
        self.__database.insert(self.__embeddings)


    def retrieve_and_rank(self, query: str, k: int=3) -> List[str]:
        query_embedding = self.__model.encode([query])
        distances, indices = self.__database.search(query_embedding, k)
        
        # Extract retrieved texts
        retrieved_docs = [self.__documents[i] for i in indices[0]]

        # Compute similarity scores
        doc_embeddings = self.__model.encode(retrieved_docs)
        scores = cosine_similarity(query_embedding, doc_embeddings)[0]

        # Rank documents by similarity score
        ranked_docs = [doc for _, doc in sorted(zip(scores, retrieved_docs), reverse=True)]
        return ranked_docs