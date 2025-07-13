# services/retrieval/tfidf_retrieval.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils.io import PersistenceManager
from services.preprocessing.preprocessor import preprocess_text_and_tokenize
from services.vectorization import vector_store
import joblib

class Searcher:
    def __init__(self, dataset_key: str):
        print(f"Loading search artifacts for '{dataset_key}'...")
        self.dataset_key = dataset_key
        tfidf_config = vector_store.TFIDF_CONFIG[dataset_key]
        self.vectorizer = PersistenceManager.load_joblib(tfidf_config["model"])
        vectors_data = PersistenceManager.load_joblib(tfidf_config["vectors"])
        self.tfidf_matrix = vectors_data['matrix']
        self.doc_ids = vectors_data['doc_ids']
        self.doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}
        inverted_index_path = vector_store.IND_DIR / f"inverted_index_{dataset_key}.json"
        self.inverted_index = PersistenceManager.load_json(inverted_index_path)


    def find_candidate_docs(self, query_tokens: list[str]) -> list[str]:
        candidate_docs = set()
        for token in query_tokens:
            if token in self.inverted_index:
                candidate_docs.update(self.inverted_index[token])
        return [doc_id for doc_id in self.doc_ids if doc_id in candidate_docs]

    def search(self, query: str, top_k: int = None) -> list[tuple[str, float]]:
        query_tokens = preprocess_text_and_tokenize(query)
        candidate_doc_ids = self.find_candidate_docs(query_tokens)
        if not candidate_doc_ids:
            # ولا مستند مرشح
            return []

        candidate_indices = [self.doc_id_to_idx[doc_id] for doc_id in candidate_doc_ids]
        query_vector = self.vectorizer.transform([" ".join(query_tokens)])

        # الكلاسترينج اختياري إذا كان مفعل وبياناته متوفرة

        candidate_matrix = self.tfidf_matrix[candidate_indices]

        if candidate_matrix.shape[0] == 0:
            return []

        similarities = cosine_similarity(query_vector, candidate_matrix).flatten()

        if top_k is None:
            sorted_indices = np.argsort(similarities)[::-1]
        else:
            sorted_indices = np.argsort(similarities)[::-1][:top_k]

        results = [(candidate_doc_ids[idx], float(similarities[idx])) for idx in sorted_indices]
        return results