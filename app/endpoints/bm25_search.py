# app/endpoints/bm25_search.py
from fastapi import APIRouter, Query
from services.retrieval.bm25_retrieval import BM25Searcher
from utils.db_utils import fetch_docs_content_by_ids

router = APIRouter()

searchers = {
    "antique": BM25Searcher("antique"),
    "quora": BM25Searcher("quora"),
}

@router.get("/")
def bm25_search(
    query: str = Query(..., description="User query"),
    dataset: str = Query(
        "antique",
        enum=["antique", "quora"],
        description="Dataset to search in"
    ),
    k: int = Query(10, description="Number of results to return"),
    k1: float = Query(None, description="BM25 k1 parameter (optional, default as index)"),
    b: float = Query(None, description="BM25 b parameter (optional, default as index)"),
):
    searcher = searchers[dataset]
    results = searcher.search(query, top_k=k, k1=k1, b=b)
    doc_ids = [doc_id for doc_id, score in results]
    docs_map = fetch_docs_content_by_ids(dataset, doc_ids)

    return [
        {
            "doc_id": doc_id,
            "score": float(score),
            "text": docs_map.get(doc_id, {}).get("text", ""),
            "clean_text_bm25": docs_map.get(doc_id, {}).get("clean_text_bm25", ""),
        }
        for doc_id, score in results
    ]
