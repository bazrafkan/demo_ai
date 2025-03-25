from abc import ABC, abstractmethod
from typing import List

class ILLM(ABC):
    """Interface for the LLM class"""

    @abstractmethod
    def __init__(self, model_name: str) -> None:
        """
        Initialize the LLM class.

        Args:
            model_name (str): The name of the LLM model.
        """
        pass


    @abstractmethod
    def train(self, train_data: List[any], valid_data: List[any]) -> None:
        """
        Train the LLM model on the given training data.

        Args:
            train_data (list): A list of training data.
            valid_data (list): A list of validation data.
        """
        pass


    @abstractmethod
    def predict(self, system: str, userContent: str, max_length: int= 350) -> str:
        """
        Generate a response from the LLM model.

        Args:
            system (str): The system prompt.
            userContent (str): The user content.
            max_length (int): The maximum length of the generated response.
        Returns:
            str: The generated response.
        """
        pass

