# services/vectorization/bm25_model.py
import sqlite3
from rank_bm25 import BM25Okapi
from utils.io import PersistenceManager
from services.vectorization import vector_store
from services.preprocessing.preprocessor import preprocess_bm25
from services.indexing.build_inverted_index import build_inverted_index, save_inverted_index
from pathlib import Path

def fetch_documents(db_path: str) -> dict[str, str]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT doc_id, text FROM docs")
    docs = dict(cur.fetchall())
    conn.close()
    return docs

def process_dataset(dataset_key: str, force=False):
    print(f"--- Starting BM25 processing for dataset: '{dataset_key}' ---")

    db_path = f"data/{dataset_key}_docs.db"
    bm25_path = vector_store.BM25_CONFIG[dataset_key]
    inverted_index_path = vector_store.IND_DIR / f"inverted_index_{dataset_key}_bm25.json"

    original_docs = fetch_documents(db_path)
    doc_ids_list = list(original_docs.keys())
    print(f"Processing {len(original_docs)} documents...")

    # استخدم نفس الدالة للتنظيف والتقطيع
    tokenized_docs = [preprocess_bm25(text) for text in original_docs.values()]

    # بناء نموذج BM25
    print("Building BM25 model...")
    bm25 = BM25Okapi(tokenized_docs)

    # حفظ النموذج مع doc_ids
    PersistenceManager.save_joblib({'bm25': bm25, 'doc_ids': doc_ids_list}, bm25_path)
    print("✅ BM25 model saved.")

    # بناء الفهرس المعكوس
    inverted_index = build_inverted_index({doc_id: tokens for doc_id, tokens in zip(doc_ids_list, tokenized_docs)})
    save_inverted_index(inverted_index, inverted_index_path)
    print("✅ BM25 Inverted Index saved.")

    print(f"--- Finished BM25 processing for '{dataset_key}' ---\n")