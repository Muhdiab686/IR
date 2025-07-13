# services/evaluation/bert_evaluator.py

import numpy as np
from tqdm import tqdm
import ir_datasets
from utils.io import PersistenceManager

def save_eval_results(results: dict, path: str):
    PersistenceManager.save_json(results, path)

def load_qrels(dataset_name: str) -> dict[str, set[str]]:
    print(f"Loading qrels for {dataset_name}...")
    dataset = ir_datasets.load(dataset_name)
    qrels = {}
    for qrel in dataset.qrels_iter():
        if getattr(qrel, 'relevance', 1) > 0:
            qrels.setdefault(qrel.query_id, set()).add(qrel.doc_id)
    return qrels

def evaluate_bert_model(searcher, dataset_name: str, dataset_key: str, use_faiss: bool, results_path: str = None):
    if results_path is None:
        method = "faiss" if use_faiss else "bruteforce"
        results_path = f"data/evaluation_results/{dataset_key}_bert_{method}.json"

    dataset = ir_datasets.load(dataset_name)
    qrels = load_qrels(dataset_name)
    precisions_at_10, recalls, reciprocal_ranks, average_precisions = [], [], [], []

    print(f"Evaluating BERT model (using Faiss: {use_faiss})...")
    for i, query in enumerate(tqdm(dataset.queries_iter(), total=dataset.queries_count())):
        query_id = query.query_id
        if query_id not in qrels:
            continue
            
        # Get all ranked results for a full evaluation
        retrieved_docs_with_scores = searcher.search(query.text, top_k=None)
        # يجب أن تعيد قائمة [(doc_id, score), ...]
        if not isinstance(retrieved_docs_with_scores, list):
            print(f"Error: search() did not return a list for query: {query.text}")
            continue

        retrieved_doc_ids = [doc_id for doc_id, score in retrieved_docs_with_scores]
        true_relevant_docs = qrels[query_id]

        retrieved_top_10 = set(retrieved_doc_ids[:10])
        p_at_10 = len(retrieved_top_10.intersection(true_relevant_docs)) / 10
        precisions_at_10.append(p_at_10)

        retrieved_set = set(retrieved_doc_ids)
        if not true_relevant_docs:
            continue
        recall = len(retrieved_set.intersection(true_relevant_docs)) / len(true_relevant_docs)
        recalls.append(recall)

        rr = 0.0
        for rank, doc_id in enumerate(retrieved_doc_ids):
            if doc_id in true_relevant_docs:
                rr = 1 / (rank + 1)
                break
        reciprocal_ranks.append(rr)

        hits = 0
        sum_precisions = 0
        for rank, doc_id in enumerate(retrieved_doc_ids):
            if doc_id in true_relevant_docs:
                hits += 1
                precision_at_k = hits / (rank + 1)
                sum_precisions += precision_at_k
        ap = sum_precisions / len(true_relevant_docs) if true_relevant_docs else 0
        average_precisions.append(ap)

    results = {
        "Precision@10": np.mean(precisions_at_10),
        "Recall": np.mean(recalls),
        "MRR": np.mean(reciprocal_ranks),
        "MAP": np.mean(average_precisions)
    }
    save_eval_results(results, results_path)
    return results
