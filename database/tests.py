import unittest

from database import IDatabase
from database import Database


class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.dimension = 768
        self.length = 10
        self.embeddings = [range(self.dimension)] * self.length
        self.database = Database(dimension=self.dimension)

    def test_instance(self):
        self.assertIsInstance(self.database, IDatabase)

    def test_insert(self):
        self.database.insert(self.embeddings)
        self.assertEqual(self.database.length, self.length)
        self.assertEqual(self.database.dimension, self.dimension)

    def test_search(self):
        self.database.insert(self.embeddings)
        query_embedding = [self.embeddings[0]]
        distances, indices = self.database.search(query_embedding, k=2)
        self.assertEqual(len(distances[0]), 2)
        self.assertEqual(len(indices[0]), 2)