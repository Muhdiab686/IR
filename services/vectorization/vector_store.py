# services/vectorization/vector_store.py

from pathlib import Path

VEC_DIR = Path("data/vectorizers")
IND_DIR = Path("data/inverted_index")

VEC_DIR.mkdir(exist_ok=True, parents=True)

BM25_CONFIG = {
    "quora": VEC_DIR / "bm25_quora_model.joblib",
    "antique": VEC_DIR / "bm25_antique_model.joblib",
}

TFIDF_CONFIG = {
    "antique": {
        "model": VEC_DIR / "tfidf_antique_model.joblib",
        "vectors": VEC_DIR / "tfidf_antique_vectors.joblib"
    },
    "quora" : {
    "model": VEC_DIR / "tfidf_quora_model.joblib",
    "vectors": VEC_DIR / "tfidf_quora_vectors.joblib"
    }
}

EMBEDDING_CONFIG = {
   "quora" : {
    "model": VEC_DIR / "bert_quora_model.joblib",
    "vectors": VEC_DIR / "bert_quora_vectors.joblib"
    },
    "antique": {
        "model": VEC_DIR / "bert_antique_model.joblib",
        "vectors": VEC_DIR / "bert_antique_vectors.joblib"
    },
}

