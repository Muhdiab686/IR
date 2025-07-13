#services\evaluation\tfidf_evaluator.py
import json
import ir_datasets
import numpy as np
from tqdm import tqdm
from utils.io import PersistenceManager
from services.retrieval.tfidf_retrieval import Searcher

def save_eval_results(results: dict, path: str):
    PersistenceManager.save_json(results, path)


def load_qrels(dataset_name: str) -> dict[str, set[str]]:
    """
    تحميل qrels (الأجوبة الصحيحة) وتحويلها إلى قاموس.
    Output: {query_id: {relevant_doc_id1, relevant_doc_id2, ...}}
    """
    print(f"Loading qrels for {dataset_name}...")
    dataset = ir_datasets.load(dataset_name)
    qrels = {}
    for qrel in dataset.qrels_iter():
        if getattr(qrel, 'relevance', 1) > 0:
            qrels.setdefault(qrel.query_id, set()).add(qrel.doc_id)
    return qrels
def evaluate_model(searcher: Searcher, dataset_name: str, dataset_key: str ,results_path: str = None):

    if results_path is None:
        results_path = fr"C:\Users\muhammad\Desktop\ir\data\evaluation_results\{dataset_key}_evaluation_results.json"

    dataset = ir_datasets.load(dataset_name)
    qrels = load_qrels(dataset_name)
    precisions_at_10, recalls, reciprocal_ranks, average_precisions = [], [], [], []

    print("Evaluating model...")
    for i, query in enumerate(tqdm(dataset.queries_iter(), total=dataset.queries_count())):
        query_id = query.query_id
        if query_id not in qrels:
            continue
        retrieved_docs_with_scores = searcher.search(query.text)
        retrieved_doc_ids = [doc_id for doc_id, score in retrieved_docs_with_scores]
        true_relevant_docs = qrels[query_id]

        retrieved_top_10 = set(retrieved_doc_ids[:10])
        p_at_10 = len(retrieved_top_10.intersection(true_relevant_docs)) / 10
        precisions_at_10.append(p_at_10)

        retrieved_set = set(retrieved_doc_ids)
        recall = len(retrieved_set.intersection(true_relevant_docs)) / len(true_relevant_docs)
        recalls.append(recall)

        rr = 0.0
        for i, doc_id in enumerate(retrieved_doc_ids):
            if doc_id in true_relevant_docs:
                rr = 1 / (i + 1)
                break
        reciprocal_ranks.append(rr)

        hits = 0
        sum_precisions = 0
        for i, doc_id in enumerate(retrieved_doc_ids):
            if doc_id in true_relevant_docs:
                hits += 1
                precision_at_k = hits / (i + 1)
                sum_precisions += precision_at_k
        ap = sum_precisions / len(true_relevant_docs) if true_relevant_docs else 0
        average_precisions.append(ap)

        if (i+1) % 10000 == 0:
            print(f"Progress: {i+1} queries processed...")

    results = {
        "Precision@10": np.mean(precisions_at_10),
        "Recall": np.mean(recalls),
        "MRR": np.mean(reciprocal_ranks),
        "MAP": np.mean(average_precisions)
    }
    save_eval_results(results, results_path)
    return results
