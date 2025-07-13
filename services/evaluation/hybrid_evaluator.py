#services\evaluation\hybrid_evaluator.py

import ir_datasets
import numpy as np
from tqdm import tqdm
from services.retrieval.hybrid_retrieval import HybridSearcher
from utils.io import PersistenceManager
import sqlite3

def evaluate_hybrid(dataset_key: str, dataset_name: str, results_path: str = None):
    """
    تقييم أداء النموذج الهجين (BM25 + BERT) على مجموعة بيانات محددة.
    التحسينات: الاتصال بقاعدة البيانات مرة واحدة واستخدام BERT batch.
    """
    print(f"--- Starting evaluation for Hybrid Model on '{dataset_key}' ---")
    
    if results_path is None:
        results_path = f"data/evaluation_results/{dataset_key}_hybrid_evaluation_results.json"

    searcher = HybridSearcher(dataset_key)   # داخل الكلاس الآن الاتصال يفتح مرة واحدة فقط!
    dataset = ir_datasets.load(dataset_name)
    
    qrels = {}
    for qrel in dataset.qrels_iter():
        if getattr(qrel, 'relevance', 1) > 0:
            qrels.setdefault(qrel.query_id, set()).add(qrel.doc_id)

    # بدء عملية التقييم
    precisions_at_10, recalls, reciprocal_ranks, average_precisions = [], [], [], []
    
    print("Evaluating hybrid model...")

    for query in tqdm(dataset.queries_iter(), total=dataset.queries_count()):
        query_id = query.query_id
        if query_id not in qrels:
            continue

        retrieved_docs_with_scores = searcher.search(query.text, top_k=100)
        retrieved_doc_ids = [doc_id for doc_id, score in retrieved_docs_with_scores]
        true_relevant_docs = qrels[query_id]

        # حساب Precision@10
        retrieved_top_10 = set(retrieved_doc_ids[:10])
        p_at_10 = len(retrieved_top_10.intersection(true_relevant_docs)) / 10
        precisions_at_10.append(p_at_10)

        # حساب Recall
        retrieved_set = set(retrieved_doc_ids)
        recall = len(retrieved_set.intersection(true_relevant_docs)) / len(true_relevant_docs) if true_relevant_docs else 0
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

    # تجميع النتائج النهائية
    final_results = {
        "Model": f"Hybrid (BM25 + BERT) on {dataset_key}",
        "Precision@10": np.mean(precisions_at_10),
        "Recall": np.mean(recalls),
        "MRR": np.mean(reciprocal_ranks),
        "MAP": np.mean(average_precisions)
    }
    
    PersistenceManager.save_json(final_results, results_path)
    print(f"✅ Evaluation finished. Results saved to {results_path}")
    print(final_results)

    searcher.close()
    return final_results
