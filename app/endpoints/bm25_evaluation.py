from fastapi import APIRouter, Query
from services.evaluation.bm25_evaluator import evaluate_bm25_model

router = APIRouter()

@router.get("/")
def evaluate_bm25_model(
    dataset: str = Query(
        "antique",
        enum=["antique", "quora"],
        description="Dataset to evaluate"
    ),
    top_k: int = Query(10, description="Number of top documents to consider"),
    max_queries: int = Query(None, description="Optional limit on number of queries")
):
    return evaluate_bm25_model(dataset_key=dataset, top_k=top_k, max_queries=max_queries)
