from fastapi import APIRouter, Query
from services.retrieval.tfidf_retrieval import Searcher
from utils.db_utils import fetch_docs_content_by_ids

router = APIRouter()

searchers = {
    "antique": Searcher("antique"),
    "quora": Searcher("quora"),
}


@router.get("/")
def search(
    query: str = Query(..., description="User query"),
    dataset: str = Query(
        "antique",
        enum=["antique", "quora"],
        description="Dataset to search in"
    ),
    k: int = Query(10, description="Number of results retrieved")
):
    searcher = searchers[dataset]
    results = searcher.search(query, top_k=k)
    doc_ids = [doc_id for doc_id, score in results]

    # جلب النصوص الأصلية والمنظفة من الداتابيز
    docs_map = fetch_docs_content_by_ids(dataset, doc_ids)

    return [
        {
            "doc_id": doc_id,
            "score": float(score),
            "text": docs_map.get(doc_id, {}).get("text", ""),  # النص الأصلي
            "clean_text_tf": docs_map.get(doc_id, {}).get("clean_text_stem", ""),  # النص المنظف (حسب العمود الذي يمثل TF)
        }
        for doc_id, score in results
    ]
