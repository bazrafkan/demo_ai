import json

from llm import LLM

# Load the JSON file
with open("my_cv.json", "r") as file:
    data = json.load(file)

# Extract text for embeddings
documents = [entry["text"] for entry in data]


# Load LLM
llm_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
llm = LLM(llm_model_name)


while True:
    print("Enter 'exit' or 'quit' or 'q' to stop the program.")
    user_query = input("Enter your quastion >>> ")
    

    if user_query == "exit" or user_query == "quit" or user_query == "q":
        break

    prompt = f"""Use the following retrieved informations to answer the question accurately.

    Context:
    {documents}

    Question:
    {user_query}
    """

    response = llm.predict(prompt, "",max_length=1500)
    print("Response:", response)

print("Thank you for using the program!")
print("Goodbye!")