import sqlite3
from utils.io import save_joblib
from collections import defaultdict
from pathlib import Path

INVERTED_INDEX_DIR = Path("data/inverted_index")
INVERTED_INDEX_DIR.mkdir(parents=True, exist_ok=True)

def build_inverted_index(dataset_key, db_path):
    print(f"\n🔧 Building unigram inverted index for: {dataset_key}")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT doc_id, clean FROM docs")
    rows = cur.fetchall()
    conn.close()

    index = defaultdict(lambda: defaultdict(int))

    for doc_id, clean in rows:
        tokens = clean.split() 
        for token in tokens:
            index[token][doc_id] += 1

    index = {
        term: dict(freqs) for term, freqs in index.items()
        if sum(freqs.values())
    }

    path = INVERTED_INDEX_DIR / f"{dataset_key}_inverted_index_for_BM.joblib"
    save_joblib(index, path)
    print(f"✅ Saved unigram inverted index at: {path}")

