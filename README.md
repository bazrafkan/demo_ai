# RAG and CAG Model Demo

This project demonstrates how to implement **Retrieval-Augmented Generation (RAG)** and **Cached-Augmented Generation (CAG)** models using Python.

---

## 🧪 Demo Scripts

Here is a description of each demo included in this project:

### `demo_rag.py`

- Uses a local model (`LLM`) with a retrieval pipeline based on embeddings.
- Retrieves top-k relevant chunks from your documents using a vector database.
- Constructs a prompt with retrieved context and passes it to the local LLM.

### `demo_rag_openai.py`

- Similar to `demo_rag.py`, but uses OpenAI's API (`OpenAILLM`) instead of a local model.
- Retrieves relevant content from your dataset and uses OpenAI to generate responses.

### `demo_cag.py`

- Demonstrates Cached-Augmented Generation (CAG) using a local LLM.
- Uses previously embedded static content (from JSON) and prompts the model directly without dynamic retrieval.

### `demo_cag_openai.py`

- Similar to `demo_cag.py`, but uses OpenAI's API.
- Embeds static content and uses OpenAI to answer user questions with that context.

### `demo_embedding.py`

- Runs experiments to evaluate retrieval performance using various embedding models.
- Calculates metrics like Recall@K, Precision@K, and MRR to guide embedding model selection.

## 🔍 What Are RAG and CAG?

### RAG (Retrieval-Augmented Generation)

RAG combines:

- A **retriever**: finds the most relevant documents for a given query.
- A **generator**: uses those retrieved documents to generate an answer.

### CAG (Cached-Augmented Generation)

CAG combines:

- A **cache**: stores the most relevant documents for previously seen queries.
- A **generator**: generates responses using cached results to improve performance and reduce redundancy.

---

## 📦 Installation

Install the required packages using pip:

```bash
pip install transformers peft datasets accelerate bitsandbytes faiss-cpu rank-bm25 sentence-transformers colorful_debug
```

---

## ⚙️ Environment Setup

Create a `.env` file in the root directory and add your API keys:

```env
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
VOYAGEAI_API_KEY=<YOUR_VOYAGEAI_API_KEY>
```

---

## ✅ Running Tests

To run all test files in the project, use the following command:

```bash
python -m unittest discover -s . -p "tests.py"
```

---
