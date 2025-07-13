import sqlite3
from services.document_store.config import DB_PATHS
from services.preprocessing.preprocessor import tokenize_query

BATCH_SIZE = 10000  # عدّل الرقم حسب حجم الرام



def process_and_update_docs(db_name):
    from services.preprocessing.preprocessor import preprocess_text_and_tokenize, preprocess_bm25, preprocess_bert

    conn = sqlite3.connect(db_name)
    cur = conn.cursor()
    cur.execute("SELECT doc_id, text FROM docs")
    docs = cur.fetchall()
    total = len(docs)
    print(f"Total documents: {total}")

    for batch_start in range(0, total, BATCH_SIZE):
        batch = docs[batch_start:batch_start + BATCH_SIZE]
        updates = []
        for doc_id, text in batch:
            stem_tokens = preprocess_text_and_tokenize(text)
            bm25_tokens = preprocess_bm25(text)
            bert_text = preprocess_bert(text)

            stem_clean = ' '.join(stem_tokens)
            bm25_clean = ' '.join(bm25_tokens)
            bert_clean = bert_text

            updates.append((
                bm25_clean,
                stem_clean,
                bert_clean,
                doc_id
            ))

        cur.executemany(
            "UPDATE docs SET clean_text_bm25=?, clean_text_stem=?, clean_text_bert=? WHERE doc_id=?",
            updates
        )
        conn.commit()
        print(f"Processed and committed {batch_start + len(batch)} / {total}")
    conn.close()
    print(f"✅ Finished processing all {total} documents in {db_name}")
if __name__ == "__main__":
    for db_key, db_path in DB_PATHS.items():
        print(f"\n=== Processing dataset: {db_key} ===")
        process_and_update_docs(db_path)
