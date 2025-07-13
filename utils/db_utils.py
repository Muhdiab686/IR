# يمكنك وضع هذه الدالة في ملف مساعد utils
import sqlite3

def fetch_docs_content_by_ids(dataset_key: str, doc_ids: list[str]) -> dict[str, dict]:
    db_path = f"data/{dataset_key}_docs.db"
    conn = sqlite3.connect(db_path)
    placeholders = ','.join('?' for _ in doc_ids)
    query = f"""
        SELECT doc_id, text, clean_text_bm25, clean_text_stem, clean_text_bert
        FROM docs WHERE doc_id IN ({placeholders})
    """
    rows = conn.execute(query, doc_ids).fetchall()
    conn.close()
    # إرجاع dict: {doc_id: {"text":..., "bm25":..., "stem":..., "bert":...}}
    result = {}
    for doc_id, text, bm25, stem, bert in rows:
        result[doc_id] = {
            "text": text,
            "clean_text_bm25": bm25,
            "clean_text_stem": stem,
            "clean_text_bert": bert,
        }
    return result

def fetch_docs_content_by_id(dataset_key, doc_ids):
    # تحديد اسم قاعدة البيانات حسب الداتا سيت
    db_path = f"data/{dataset_key}.db"
    placeholders = ','.join('?' for _ in doc_ids)  # مثل (?, ?, ...)
    query = f"SELECT doc_id, text, clean_text_bm25, clean_text_bert FROM documents WHERE doc_id IN ({placeholders})"

    docs_map = {}

    # افتح الاتصال بشكل مؤقت فقط داخل هذه الدالة
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(query, doc_ids)
        for row in cur.fetchall():
            doc_id, text, clean_text_bm25, clean_text_bert = row
            docs_map[doc_id] = {
                "text": text,
                "clean_text_bm25": clean_text_bm25,
                "clean_text_bert": clean_text_bert
            }
    return docs_map
