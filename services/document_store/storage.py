### services/document_store/storage.py
import sqlite3
from .config import DB_PATHS
from .loader import load_dataset

def store_documents(dataset_key):
    dataset = load_dataset(dataset_key)
    db_name = DB_PATHS[dataset_key]
    conn = sqlite3.connect(db_name)
    cur = conn.cursor()

    cur.execute('''
        CREATE TABLE IF NOT EXISTS docs (
            doc_id TEXT PRIMARY KEY,
            text TEXT
        )
    ''')

    count = 0
    for doc in dataset.docs_iter():
        cur.execute("INSERT OR IGNORE INTO docs (doc_id, text) VALUES (?, ?)", (doc.doc_id, doc.text))
        count += 1

    cur.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON docs(doc_id)")
    conn.commit()
    conn.close()
    print(f"✅ Stored {count} documents in: {db_name}")

def store_antique_from_txt(file_path, db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute('''
        CREATE TABLE IF NOT EXISTS docs (
            doc_id TEXT PRIMARY KEY,
            text TEXT
        )
    ''')

    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            doc_id, text = parts
            cur.execute("INSERT OR IGNORE INTO docs (doc_id, text) VALUES (?, ?)", (doc_id, text))
            count += 1

    cur.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON docs(doc_id)")
    conn.commit()
    conn.close()
    print(f" Stored {count} documents from ANTIQUE in: {db_path}")
