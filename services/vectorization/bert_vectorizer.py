# services/vectorization/bert_vectorizer.py
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.io import PersistenceManager
from services.vectorization import vector_store

from services.preprocessing.preprocessor import preprocess_bert

def train_bert_embeddings(dataset_key: str):
    db_path = f"data/{dataset_key}_docs.db"
    embedding_config = vector_store.EMBEDDING_CONFIG[dataset_key]
    model_path = embedding_config["model"]
    vectors_path = embedding_config["vectors"]

    print(f"--- Starting BERT processing for dataset: '{dataset_key}' ---")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT doc_id, text FROM docs")
    rows = cur.fetchall()
    conn.close()

    doc_ids, original_texts = zip(*rows)
    print(f"Applying light preprocessing to {len(original_texts)} documents...")
    processed_texts = [preprocess_bert(text) for text in original_texts]

    print("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Generating BERT embeddings for {len(processed_texts)} documents...")
    vectors = model.encode(processed_texts, show_progress_bar=True, convert_to_numpy=True)

    PersistenceManager.save_joblib(model, model_path)
    print(f"✅ BERT model saved to {model_path}")

    PersistenceManager.save_joblib({'vectors': vectors, 'doc_ids': list(doc_ids)}, vectors_path)
    print(f"✅ BERT vectors and doc_ids saved to {vectors_path}")
    print(f"--- Finished BERT processing for '{dataset_key}' ---\n")
