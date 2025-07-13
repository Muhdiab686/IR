# app/rag_endpoints.py
from fastapi import APIRouter, Query
from services.retrieval.bert_retrieval import BertSearcher
from services.rag.generator import TextGenerator
from utils.db_utils import fetch_docs_content_by_ids
from pydantic import BaseModel

router = APIRouter()

searchers = {
    "antique": BertSearcher("antique", use_faiss=True),
    "quora": BertSearcher("quora", use_faiss=True),
}
generator = TextGenerator()

class RagRequest(BaseModel):
    query: str
    dataset_key: str = "antique"

def do_rag_search(query: str, dataset_key: str):
    searcher = searchers[dataset_key]
    retrieved = searcher.search(query, top_k=10)
    retrieved_ids = [doc_id for doc_id, score in retrieved]
    docs_map = fetch_docs_content_by_ids(dataset_key, retrieved_ids)
    context = " ".join([docs_map.get(doc_id, {}).get("text", "") for doc_id in retrieved_ids])
    generated_answer = generator.answer_from_context(context, query)
    return {
        "query": query,
        "generated_answer": generated_answer,
        "source_documents": [
            {"doc_id": doc_id, "content": docs_map.get(doc_id, {}).get("text", "")}
            for doc_id in retrieved_ids
        ]
    }

@router.post("/")
async def rag_search_post(data: RagRequest):
    return do_rag_search(data.query, data.dataset_key)

@router.get("/")
def rag_search_get(
    query: str = Query(..., description="User query"),
    dataset_key: str = Query("antique", description="Dataset to search in")
):
    return do_rag_search(query, dataset_key)
