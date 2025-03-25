import unittest
from unittest.mock import patch
from llm import LLM, OpenAILLM, ILLM

class TestLLM(unittest.TestCase):
    @patch("llm.LLM.__init__", return_value=None)
    def setUp(self, mock_init):
        self.llm = LLM()
        
        # Mock the __init__ method
        self.llm._LLM__tokenizer = None
        self.llm._LLM__llm = None

        self.openai_llm = OpenAILLM()
        
    def test_instance(self):
        self.assertIsInstance(self.llm, ILLM)
        self.assertIsInstance(self.openai_llm, ILLM)