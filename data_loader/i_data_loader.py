from abc import ABC, abstractmethod
from typing import Tuple, List


class IDataLoader(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def load(self, dataset_name: str, limit: int = 0, limit_test: int | None = None) -> Tuple[List[str], List[any], List[str], List[any]]:
        """
        Load the dataset.

        Args:
            dataset_name (str): The name of the dataset.
            limit (int): The number of samples to load.

        Returns:
            tuple: The training and testing texts and labels
        """
        pass