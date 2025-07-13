# app/endpoints/hybrid_search.py

from fastapi import APIRouter, Request, Query
from services.retrieval.hybrid_retrieval import HybridSearcher
from utils.db_utils import fetch_docs_content_by_ids

router = APIRouter()

# هذه آمنة إذا لم تحفظ داخلها اتصال db كمتغير object
searchers = {
    "antique": HybridSearcher("antique"),
    "quora": HybridSearcher("quora"),
}

def do_hybrid_search(query, dataset_key="antique", top_k=10):
    searcher = searchers[dataset_key]
    results = searcher.search(query, top_k=top_k)

    # استخرج معرّفات الوثائق
    if results and isinstance(results[0], dict):
        doc_ids = [r["doc_id"] for r in results]
    else:
        doc_ids = [doc_id for doc_id, score in results]

    # الدالة الآن آمنة ولن تسبب مشاكل الخيوط
    docs_map = fetch_docs_content_by_ids(dataset_key, doc_ids)

    out = []
    for item in results:
        if isinstance(item, dict):
            doc_id = item["doc_id"]
            score = item.get("score", None)
        else:
            doc_id, score = item

        doc_texts = docs_map.get(doc_id, {})
        out.append({
            "doc_id": doc_id,
            "score": float(score) if score is not None else None,
            "text": doc_texts.get("text", ""),
            "clean_text_bm25": doc_texts.get("clean_text_bm25", ""),
            "clean_text_bert": doc_texts.get("clean_text_bert", ""),
        })

    return {"results": out}

# --- POST endpoint ---
@router.post("/")
async def hybrid_search_post(request: Request):
    data = await request.json()
    query = data["query"]
    dataset_key = data.get("dataset_key", "antique")
    top_k = data.get("top_k", 10)
    return do_hybrid_search(query, dataset_key, top_k)

# --- GET endpoint ---
@router.get("/")
def hybrid_search_get(
    query: str = Query(..., description="User query"),
    dataset_key: str = Query("antique", description="Dataset to search in"),
    top_k: int = Query(10, description="Number of results")
):
    return do_hybrid_search(query, dataset_key, top_k)
