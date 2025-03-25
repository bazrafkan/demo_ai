# What is K in Retrieval? Why Do We Use It?

In retrieval-based systems (like RAG), **K** represents the number of top retrieved results considered for ranking, classification, or further processing.

---

## 🔹 Why is K Important?

When retrieving documents from a knowledge base or vector store, we usually don’t retrieve just one result—we retrieve the top **K** results that are the most relevant based on similarity.

- ✅ **Ensures Relevant Documents Are Retrieved** – If we only retrieve the top 1 result (K=1), we risk missing important context.
- ✅ **Improves Accuracy** – Larger K increases the chance that the relevant document is included.
- ✅ **Balances Speed vs. Quality** – A very high K slows down processing, but a small K might not retrieve enough useful data.

---

## 🔹 Examples of K in Different Metrics

### 1️⃣ Recall@K

- **Definition**: Measures whether the correct answer is within the top K retrieved results.
- **Behavior**: Higher K improves recall but may increase noise.

**Example**:  
You retrieve K=3 documents for a legal classification problem.  

- If the correct document is among them → Recall@3 = 1  
- If not → Recall@3 = 0  

🚀 **Interpretation**: Higher K improves recall, but too high can introduce irrelevant results.

---

### 2️⃣ MRR (Mean Reciprocal Rank)

- **Definition**: Checks how early the correct document appears in the top K results.
- **Behavior**:
  - If the correct document is ranked 1st, MRR is 1.0.  
  - If ranked 3rd, MRR is 1/3 = 0.33.

**Example (K=5 results ranked by relevance)**:  
1️⃣ Correct Answer – MRR = 1/1 = 1.0  
2️⃣ Wrong Answer  
3️⃣ Wrong Answer  
4️⃣ Correct Answer – MRR = 1/4 = 0.25  
5️⃣ Wrong Answer  

🚀 **Interpretation**: We want relevant documents to appear earlier (higher rank).

---

### 3️⃣ Precision@K

- **Definition**: Measures how many of the top K retrieved results are actually relevant.
- **Example**:
  - If K=5, and 3 out of 5 documents are correct, then Precision@5 = 3/5 = 0.6.

🚀 **Interpretation**: Helps tune K to avoid retrieving too many irrelevant documents.

---

## 🔹 Choosing the Right K

| **K Value** | **Use Case** |
|-------------|--------------|
| K = 1       | Best for high-precision tasks (e.g., yes/no classification). |
| K = 3-5     | Balanced for most RAG applications (ensures good recall). |
| K = 10+     | Needed for broad knowledge retrieval (e.g., research papers, legal case law). |

- 🔹 **Lower K** → Faster, but may miss relevant knowledge.  
- 🔹 **Higher K** → More accurate but slower & may include irrelevant noise.

---

## 🔹 Final Thought

- In RAG, we use **K** to balance speed vs. accuracy.
- **K** should be tuned based on experiments (e.g., track Recall@K and MRR).
- The ideal **K** depends on your retrieval method and domain.

---

## ✅ Recall@K, Precision@K, MRR@K – Definitions & Ranges

### 🔹 Recall@K

- **Definition**: Measures whether the correct item is found within the top K results.
- **Range**:  
    `0 ≤ Recall@K ≤ 1`  
- **Interpretation**:  
  - `1.0` → Correct item is in the top K results.  
  - `0.0` → Correct item is not in top K.  

✅ **Best for**: Comprehensive retrieval – ensuring relevant docs aren’t missed.

---

### 🔹 Precision@K

- **Definition**: Measures how many of the top K results are actually relevant.
- **Range**:  
    `0 ≤ Precision@K ≤ 1`  
- **Formula**:  
    `Precision@K = (# of relevant results in top K) / K`  
- **Interpretation**:  
  - `1.0` → All top K results are relevant.  
  - `0.0` → None of the top K results are relevant.  

✅ **Best for**: Reducing irrelevant documents – good when K is small.

---

### 🔹 MRR@K (Mean Reciprocal Rank)

- **Definition**: Measures how early in the top K list the first relevant result appears.
- **Range**:  
    `0 < MRR@K ≤ 1`  
- **Formula**:  
    `MRR@K = 1 / rank_of_first_relevant_result`  
- **Interpretation**:
  - `1.0` → Correct answer is ranked 1st.
  - `0.5` → Correct answer is ranked 2nd, etc.
  - If not in top K, MRR is 0.  

✅ **Best for**: Prioritizing early correctness – useful when users look at the top result only.

---

## 📊 Metric Comparison Example (K = 5)

| **Retrieved Docs** | **Correct Label** | **Recall@5** | **Precision@5** | **MRR@5** |
|---------------------|-------------------|--------------|------------------|-----------|
| [A, B, C, D, E]     | B                 | 1            | 1/5 = 0.2        | 1/2 = 0.5 |
| [F, G, H, I, J]     | K                 | 0            | 0                | 0         |
| [L, M, N, O, P]     | L                 | 1            | 1/5 = 0.2        | 1         |

---

## 🧠 How to Use These Ranges

- If **Recall@K** is low, increase K or improve retrieval.
- If **Precision@K** is low, your model is returning too many irrelevant results.
- If **MRR** is low, relevant items are buried deep – use reranking!
