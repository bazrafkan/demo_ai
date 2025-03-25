import json

from retrieve import Retrieve
from llm import OpenAILLM

# Load the JSON file
with open("my_cv.json", "r") as file:
    data = json.load(file)

# Extract text for embeddings
documents = [entry["text"] for entry in data]

embedding_model = "all-MiniLM-L6-v2"

retrieve = Retrieve(documents, embedding_model)


# Load LLM
llm_model_name = "gpt-4o-mini"
openai_llm = OpenAILLM(llm_model_name)

k = 10

while True:
    print("Enter 'exit' or 'quit' or 'q' to stop the program.")
    user_query = input("Enter your quastion >>> ")
    

    if user_query == "exit" or user_query == "quit" or user_query == "q":
        break

    retrieved_texts = retrieve.retrieve_and_rank(user_query,k)

    for i, text in enumerate(retrieved_texts):
        print(f"Ranked Retrieved Text[{i+1}]: {text}")

    system = f"""Use the following retrieved informations to answer the question accurately.

    Context:
    {documents}
    """


    response = openai_llm.predict(system, user_query)
    print("Response:", response)

print("Thank you for using the program!")
print("Goodbye!")