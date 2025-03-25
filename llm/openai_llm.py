import os

from openai import OpenAI 
from llm.i_llm import ILLM
from typing import List

class OpenAILLM(ILLM):
    def __init__(self, model_name:str = "gpt-4o-mini") -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("Please set the OPENAI_API_KEY environment variable.")
        self.__client = OpenAI(api_key = api_key)
        self.__model_name = model_name


    def predict(self, system: str, userContent: str, max_length: int = 350) -> str:
        completion = self.__client.chat.completions.create(
            model = self.__model_name,
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": userContent}])
        return completion.choices[0].message.content
    

    def train(self, train_data: List[any], valid_data: List[any]) -> None:
        # DODO: Implement training
        raise NotImplementedError("Training functionality is not implemented yet.")