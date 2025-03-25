from transformers import AutoTokenizer, AutoModelForCausalLM
from llm.i_llm import ILLM
from typing import List

class LLM(ILLM):
    __instance = None
    __initialized = False

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self, model_name: str = "mistralai/Mistral-7B-v0.1") -> None:
        if self.__initialized:
            return
        self.__tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.__llm = AutoModelForCausalLM.from_pretrained(model_name)
        self.__initialized = True


    def predict(self, system: str, userContent: str, max_length: int = 350) -> str:
        inputs = self.__tokenizer(system, return_tensors="pt")
        output = self.__llm.generate(**inputs, max_length = max_length)  
        return self.__tokenizer.decode(output[0])
    

    def train(self, train_data: List[any], valid_data: List[any]) -> None:
        # DODO: Implement training
        raise NotImplementedError("Training functionality is not implemented yet.")