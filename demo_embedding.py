# Calculate accuracy for different values of k, recall, MRR, and precision.
# Please read rag_retrieval_metrics_guide.md for more information.

import time

from data_loader import DataLoader
from retrieve import Retrieve
from utils import debug_log, save_log

data_loader = DataLoader()
train_texts, train_labels, test_texts, test_labels = data_loader.load("ag_news", limit=100)


embedding_model = "all-MiniLM-L6-v2" # "all-MiniLM-L6-v2" | "all-mpnet-base-v2"

retrieve = Retrieve(train_texts, embedding_model)


correct = 0
total = len(test_texts)
# k_values = [1, 3, 5, 10]
k_values = [3,10]
performance_logs = []

def retrive_and_rank_with_metrics(test_text: str, 
                                  test_label: str, 
                                  train_texts: dict, 
                                  train_labels: dict, 
                                  k=3) -> tuple:
    """Retrieve and rank documents based on the query and compute metrics"""

    start_time = time.time()
    ranked_docs = retrieve.retrieve_and_rank(test_text, k)
    retrieval_time = time.time() - start_time

    # Get labels of ranked documents
    ranked_data_label = []
    for doc in ranked_docs:
        index = train_texts.index(doc)
        ranked_data_label.append(train_labels[index])


    recall_k, mrr_score, precision_k = 0, 0, 0

    if test_label in ranked_data_label:
        # Recall@K
        recall_k = 1
        for i, labe in enumerate(ranked_data_label):
            if (labe == test_label and mrr_score == 0):
                # MRR
                mrr_score = 1 / (i + 1)
            
            if (labe == test_label):
                # Precision@K
                precision_k += 1/k


    return recall_k, mrr_score, precision_k, retrieval_time



for k in k_values:
    for i, test_text in enumerate(test_texts):
        test_label = test_labels[i]
        recall_k, mrr_score, precision_k, retrieval_time = retrive_and_rank_with_metrics(
            test_text,
            test_label, 
            train_texts,
            train_labels,
            k)
        
        # Log Performance
        performance_logs.append({
            "Query": test_text,
            "True Label": test_label,
            "K": k,
            "Recall@K": recall_k,
            "MRR": mrr_score,
            "Precision@K": precision_k,
            "Retrieval Time (s)": retrieval_time
        })

debug_log("Performance Retrieval Metrics", performance_logs)
# save_log(f"performance_retrieval_metrics_{embedding_model}", performance_logs)