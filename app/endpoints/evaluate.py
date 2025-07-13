# app/endpoints/evaluation.py
from fastapi import APIRouter, Query
from services.evaluation.tfidf_evaluator import evaluate_model

router = APIRouter()

@router.get("/")
def evaluate_model_endpoint(
    dataset: str = Query(
        "antique",
        enum=["antique", "quora"],
        description="Dataset to evaluate"
    ),
    top_k: int = Query(10, description="Number of top documents to consider"),
    max_queries: int = Query(None, description="Optional limit on number of queries")
):
    return evaluate_model(dataset_key=dataset, top_k=top_k, max_queries=max_queries)
