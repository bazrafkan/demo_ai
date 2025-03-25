import os
import voyageai
from sklearn.metrics.pairwise import cosine_similarity
from database import Database, IDatabase
from retrieve.i_retrieve import IRetrieve
from typing import List


class RetrieveVoyageai(IRetrieve):
    def __init__(self, 
                 documents: List[str], 
                 embedding_model: str = "voyage-3-large", 
                 database: None | IDatabase = None) -> None:
        """
        Initialize the Retrieve class with the documents and the embedding model.
        
        Args:
            documents (list): A list of documents to retrieve and rank.
            embedding_model (str): The name of the Sentence Transformer model to use for embeddings.
            database (Database): An instance of the Database class to store the embeddings.
        """
        api_key = os.environ.get("VOYAGEAI_API_KEY", "<your VoyageAI API key>")
        self.__vo = voyageai.Client(api_key=api_key)
        self.__embedding_model = embedding_model
        result = self.__vo.embed(documents, model=embedding_model, input_type="document")
        self.__embeddings = result.embeddings
        
        self.__documents = documents
        # Store in FAISS for retrieval
        dimension = len(self.__embeddings[0])
        if database is None:
            self.__database = Database(dimension)
        else:
            self.__database = database
        self.__database.insert(self.__embeddings)


    def retrieve_and_rank(self, query, k=3) -> list:
        result = self.__vo.embed([query], model=self.__embedding_model, input_type="document")
        query_embedding = result.embeddings
        distances, indices = self.__database.search(query_embedding, k)
        
        # Extract retrieved texts
        retrieved_docs = [self.__documents[i] for i in indices[0]]

        # Compute similarity scores
        result = self.__vo.embed(retrieved_docs, model=self.__embedding_model, input_type="document")
        doc_embeddings = result.embeddings
        scores = cosine_similarity(query_embedding, doc_embeddings)[0]

        # Rank documents by similarity score
        ranked_docs = [doc for _, doc in sorted(zip(scores, retrieved_docs), reverse=True)]
        return ranked_docs