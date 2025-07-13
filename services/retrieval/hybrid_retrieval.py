import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
from services.retrieval.bm25_retrieval import BM25Searcher
from services.retrieval.bert_retrieval import BertSearcher
from services.document_store.config import DB_PATHS

class HybridSearcher:
    def __init__(self, dataset_key: str):
        self.dataset_key = dataset_key
        self.db_path = DB_PATHS[dataset_key]
        self.bm25_searcher = BM25Searcher(dataset_key)
        self.bert_searcher = BertSearcher(dataset_key)
        print("✅ HybridSearcher initialized successfully.")

    def _fetch_texts_by_ids(self, doc_ids: list[str]) -> dict[str, str]:
        """
        تابع مساعد لجلب نصوص المستندات من قاعدة البيانات.
        الاتصال يُنشأ مؤقتًا داخل التابع.
        """
        if not doc_ids:
            return {}
        placeholders = ",".join(["?"] * len(doc_ids))
        query = f"SELECT doc_id, text FROM docs WHERE doc_id IN ({placeholders})"
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(query, doc_ids)
            texts = dict(cur.fetchall())
        return texts

    # يمكنك حذف الدالة close() الآن، لأنها لم تعد ضرورية
    # def close(self):
    #     pass

    def search(self, query: str, top_k: int = 10, rerank_candidates: int = 100) -> list[tuple[str, float]]:
        # --- المرحلة الأولى: جلب المرشحين باستخدام BM25 ---
        candidate_results = self.bm25_searcher.search(query, top_k=rerank_candidates)
        if not candidate_results:
            return []

        candidate_ids = [doc_id for doc_id, score in candidate_results]
        candidate_texts_map = self._fetch_texts_by_ids(candidate_ids)
        candidate_texts_list = [candidate_texts_map.get(doc_id, "") for doc_id in candidate_ids]

        # --- BERT batching ---
        query_vector = self.bert_searcher.model.encode([query])  # shape: (1, dim)
        candidate_vectors = self.bert_searcher.model.encode(candidate_texts_list, batch_size=32, show_progress_bar=False)

        # حساب cosine similarity دفعة واحدة
        rerank_scores = cosine_similarity(query_vector, candidate_vectors).flatten()

        reranked_results = list(zip(candidate_ids, rerank_scores))
        reranked_results.sort(key=lambda item: item[1], reverse=True)
        return reranked_results[:top_k]
