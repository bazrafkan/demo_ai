from datasets import load_dataset
from data_loader.i_data_loader  import IDataLoader
from typing import Tuple, List


class DataLoader(IDataLoader):
    def __init__(self) -> None:
        pass

    def load(self, dataset_name: str, limit: int = 0, limit_test: int | None = None) -> Tuple[List[str], List[any], List[str], List[any]]:
        limit_test = limit_test if limit_test is not None else int(limit * 0.2)

        dataset_train = load_dataset(
            dataset_name, 
            split = self.__split_string("train", limit))
        dataset_test = load_dataset(
            dataset_name, 
            split = self.__split_string("test", limit_test))


        # Extract training texts and labels
        train_texts = dataset_train["text"]
        train_labels = dataset_train["label"]

        # Extract test texts and labels
        test_texts = dataset_test["text"]
        test_labels = dataset_test["label"]

        return train_texts, train_labels, test_texts, test_labels
    

    def __split_string(self, text: str, limit: int) -> str:
        return f"{text}[:{limit}]" if limit != 0 else text
