# services/retrieval/bm25_retrieval.py
import numpy as np
from utils.io import PersistenceManager
from services.preprocessing.preprocessor import preprocess_bm25
from services.vectorization import vector_store

class BM25Searcher:
    def __init__(self, dataset_key: str):
        print(f"Loading BM25 search artifacts for '{dataset_key}'...")
        bm25_path = vector_store.BM25_CONFIG[dataset_key]
        data = PersistenceManager.load_joblib(bm25_path)
        self.bm25 = data['bm25']
        self.doc_ids = data['doc_ids']
        self.doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}
        inverted_index_path = vector_store.IND_DIR / f"inverted_index_{dataset_key}_bm25.json"
        self.inverted_index = PersistenceManager.load_json(inverted_index_path)
        print("✅ BM25 Searcher initialized.")

    def find_candidate_docs(self, query_tokens: list[str]) -> list[str]:
        candidate_docs = set()
        for token in query_tokens:
            if token in self.inverted_index:
                candidate_docs.update(self.inverted_index[token])
        # الحفاظ على ترتيب doc_ids الأصلي
        return [doc_id for doc_id in self.doc_ids if doc_id in candidate_docs]

    def search(self, query: str, top_k: int = None, k1: float = None, b: float = None) -> list[tuple[str, float]]:
        # إذا الكلاسم bm25 يدعم تغيير القيم:
        if k1 is not None:
            try:
                self.bm25.k1 = k1
            except Exception:
                pass  # أو raise خطأ واضح حسب رغبتك
        if b is not None:
            try:
                self.bm25.b = b
            except Exception:
                pass

        query_tokens = preprocess_bm25(query)
        candidate_doc_ids = self.find_candidate_docs(query_tokens)

        if not candidate_doc_ids:
            return []

        candidate_indices = [self.doc_id_to_idx[doc_id] for doc_id in candidate_doc_ids]
        scores = self.bm25.get_batch_scores(query_tokens, candidate_indices)

        if top_k is None:
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            doc_id = candidate_doc_ids[idx]
            score = scores[idx]
            results.append((doc_id, score))
        return results