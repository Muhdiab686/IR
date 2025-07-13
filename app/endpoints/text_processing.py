# app/endpoints/text_processing.py

from fastapi import APIRouter, Query
from typing import Literal, Optional
from services.preprocessing.preprocessor import preprocess_bm25, preprocess_bert, preprocess_text_and_tokenize

router = APIRouter()

@router.get("/")
def text_processing(
    text: str = Query(..., description="Text to process"),
    type: Optional[Literal["bm25", "bert", "stem", "all"]] = Query("all", description="نوع المعالجة: bm25 أو bert أو stem أو all")
):
    results = {"original_text": text}
    if type in ("bm25", "all"):
        results["clean_text_bm25"] = " ".join(preprocess_bm25(text))
    if type in ("bert", "all"):
        results["clean_text_bert"] = preprocess_bert(text)
    if type in ("stem", "all"):
        results["clean_text_stem"] = " ".join(preprocess_text_and_tokenize(text))
    return results
