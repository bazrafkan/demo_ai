import unittest

from database import IDatabase
from retrieve.retrieve import Retrieve
from retrieve.i_retrieve import IRetrieve
import numpy as np

class DummyDatabase(IDatabase):
    def __init__(self, dim):
        self.vectors = []
        self.dim = dim
        
    @property
    def dimension(self):
        return self.dim
    
    @property
    def length(self):
        return len(self.vectors)

    def insert(self, embeddings):
        self.vectors = embeddings

    def search(self, query_embedding, k):
        from sklearn.metrics.pairwise import cosine_similarity
        scores = cosine_similarity(query_embedding, self.vectors)[0]
        indices = np.argsort(scores)[-k:][::-1]
        return [scores[i] for i in indices], [indices]


class TestRetrieve(unittest.TestCase):
    def setUp(self):
        self.docs = [
            "The cat sits on the mat.",
            "The dog barks at night.",
            "Birds fly in the sky."
        ]
        self.database = DummyDatabase(dim=768)
        self.retriever = Retrieve(documents=self.docs, 
                                  embedding_model="all-MiniLM-L6-v2", 
                                  database=self.database)

    def test_instance(self):
        self.assertIsInstance(self.database, IDatabase)
        self.assertIsInstance(self.retriever, IRetrieve)

    def test_retrieve_and_rank(self):
        result = self.retriever.retrieve_and_rank("A dog making noise", k=2)
        self.assertEqual(len(result), 2)
        self.assertTrue(any("dog" in doc for doc in result))

if __name__ == "__main__":
    unittest.main()