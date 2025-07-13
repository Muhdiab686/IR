# services/retrieval/bert_retrieval.py

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils.io import PersistenceManager
from services.preprocessing.preprocessor import preprocess_bert
from services.vectorization import vector_store
from utils.db_utils import fetch_docs_content_by_ids


from services.rag.generator import TextGenerator  # تأكد من مكان الكلاس الجديد

class BertSearcher:
    def __init__(self, dataset_key: str, use_faiss: bool = False, use_rag: bool = False, generator_model: str = "t5-small"):
        print(f"Loading BERT artifacts for '{dataset_key}' (Faiss enabled: {use_faiss}, RAG enabled: {use_rag})...")
        self.use_faiss = use_faiss
        self.use_rag = use_rag
        
        config = vector_store.EMBEDDING_CONFIG[dataset_key]
        self.model = PersistenceManager.load_joblib(config["model"])

        vectors_data = PersistenceManager.load_joblib(config["vectors"])
        self.doc_ids = vectors_data['doc_ids']

        if self.use_faiss:
            index_path = vector_store.IND_DIR / f"faiss_index_{dataset_key}.bin"
            self.faiss_index = faiss.read_index(str(index_path))
            print("✅ Faiss index loaded.")
        else:
            self.doc_vectors = vectors_data['vectors']
            print("✅ Brute-force vectors loaded.")

        # تفعيل مولد النصوص في حال تفعيل RAG
        if self.use_rag:
            self.generator = TextGenerator(model_name=generator_model)
        else:
            self.generator = None
        print("✅ BERT Searcher initialized successfully.")

    def search(self, query: str, top_k: int = 10, return_rag: bool = False) -> list[tuple[str, float]] | str:
        processed_query = preprocess_bert(query)
        query_vector = self.model.encode([processed_query])

        if self.use_faiss:
            query_vector = query_vector.astype('float32')
            faiss.normalize_L2(query_vector)
            similarities, top_indices = self.faiss_index.search(query_vector, top_k)
            similarities = similarities[0]
            top_indices = top_indices[0]
        else:
            similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
            top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx, sim in zip(top_indices, similarities):
            if idx == -1:
                continue
            if idx >= len(self.doc_ids):
                print(f"Warning: idx {idx} out of range for doc_ids with length {len(self.doc_ids)}")
                continue
            results.append((self.doc_ids[idx], float(sim)))

        # --- دعم RAG ---
        if self.use_rag or return_rag:
            doc_ids = [doc_id for doc_id, _ in results]
            docs_content = fetch_docs_content_by_ids(self.dataset_key, doc_ids)
            context = ""
            for doc_id in doc_ids:
                # يمكنك اختيار النص الأنسب هنا (مثلاً clean_text_bert)
                context += docs_content[doc_id]["clean_text_bert"] + "\n"
            answer = self.generator.answer_from_context(context, query)
            return answer
        else:
            return results