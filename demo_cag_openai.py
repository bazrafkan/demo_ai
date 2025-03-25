import json
from llm import OpenAILLM


# Load the JSON file
with open("my_cv.json", "r") as file:
    data = json.load(file)

# Extract text for embeddings
documents = [entry["text"] for entry in data]


# Load LLM
llm_model_name = "gpt-4o-mini"
openai_llm = OpenAILLM(llm_model_name)


while True:
    print("Enter 'exit' or 'quit' or 'q' to stop the program.")
    user_query = input("Enter your quastion >>> ")
    

    if user_query == "exit" or user_query == "quit" or user_query == "q":
        break

    system = f"""Use the following retrieved informations to answer the question accurately.

    Context:
    {documents}
    """


    response = openai_llm.predict(system, user_query)
    print("Response:", response)

print("Thank you for using the program!")
print("Goodbye!")