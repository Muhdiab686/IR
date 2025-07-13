import time
from fastapi import APIRouter, Query
from services.retrieval.bert_retrieval import BertSearcher  # أو اسم الكلاس المناسب عندك
from utils.db_utils import fetch_docs_content_by_ids

router = APIRouter()

searchers = {
    "antique": BertSearcher("antique"),
    "quora": BertSearcher("quora"),
}

@router.get("/")
def vector_search(
    query: str = Query(..., description="User query"),
    dataset: str = Query(
        "antique",
        enum=["antique", "quora"],
        description="Dataset to search in"
    ),
    k: int = Query(10, description="Number of results to return")
):
    start = time.time()
    searcher = searchers[dataset]
    results = searcher.search(query, top_k=k)
    doc_ids = [doc_id for doc_id, score in results]
    docs_map = fetch_docs_content_by_ids(dataset, doc_ids)
    execution_time = time.time() - start

    return {
        "results": [
            {
                "doc_id": doc_id,
                "score": float(score),
                "text": docs_map.get(doc_id, {}).get("text", ""),
                "clean_text_bert": docs_map.get(doc_id, {}).get("clean_text_bert", ""),
            }
            for doc_id, score in results
        ],
        "execution_time": execution_time
    }
