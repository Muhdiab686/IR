# services/vectorization/tfidf_vectorizer.py

import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.io import PersistenceManager
from services.vectorization import vector_store
from services.preprocessing.preprocessor import preprocess_text_and_tokenize
from services.indexing.build_inverted_index import build_inverted_index, save_inverted_index
from pathlib import Path

def fetch_documents(db_path: str) -> dict[str, str]:
    """جلب المستندات من قاعدة البيانات على شكل قاموس {doc_id: text}."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT doc_id, text FROM docs")
    docs = dict(cur.fetchall())
    conn.close()
    return docs

def process_dataset(dataset_key: str, force=False):
    print(f"--- Starting offline processing for dataset: '{dataset_key}' ---")
    
    db_path = f"data/{dataset_key}_docs.db"
    tfidf_config = vector_store.TFIDF_CONFIG[dataset_key]
    inverted_index_path = vector_store.IND_DIR / f"inverted_index_{dataset_key}.json"

    original_docs = fetch_documents(db_path)
    print(f"Processing {len(original_docs)} documents...")

    doc_ids_list = list(original_docs.keys())

    print("Building TF-IDF model...")

    vectorizer = TfidfVectorizer(
        tokenizer=preprocess_text_and_tokenize,  
        preprocessor=None,                       
    )

    tfidf_matrix = vectorizer.fit_transform([text for text in original_docs.values()])

    PersistenceManager.save_joblib(vectorizer, tfidf_config["model"])
    PersistenceManager.save_joblib({'matrix': tfidf_matrix, 'doc_ids': doc_ids_list}, tfidf_config["vectors"])
    print("✅ TF-IDF model and vectors saved.")

    # بناء الفهرس المعكوس من التوكنات المنتجة بعد التنظيف
    processed_docs = {doc_id: preprocess_text_and_tokenize(text) for doc_id, text in original_docs.items()}
    inverted_index = build_inverted_index(processed_docs)
    save_inverted_index(inverted_index, inverted_index_path)
    
    print(f"--- Finished processing for '{dataset_key}' ---\n")
