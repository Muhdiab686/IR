# app/main.py

from fastapi import FastAPI
from app.endpoints.search import router as search_router
from app.endpoints.evaluate import router as eval_router  
from app.endpoints.bm25_search import router as bm25_router 
from app.endpoints.bm25_evaluation import router as bm25_eval_router
from app.endpoints.search_bert import router as bert_search_router
from app.endpoints.evaluate_bert import router as eval_bert_router
from app.endpoints.hybrid_search import router as hybrid_router
from app.endpoints.rag import router as rag_search
from app.endpoints.vectordb_search import router as vector_router
from app.endpoints.text_processing import router as text_processing

app = FastAPI(title="IR Search API")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # أو ضع رابط الواجهة تحديدًا مثلاً: ["http://127.0.0.1:8080"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# تسجيل المسارات
app.include_router(search_router, prefix="/search", tags=["Search"])
app.include_router(bm25_router, prefix="/bm25", tags=["BM25 Search"])
app.include_router(bert_search_router, prefix="/search-bert", tags=["Search - BERT"])
app.include_router(hybrid_router, prefix="/hybrid", tags=["Hybrid Search"])
app.include_router(rag_search, prefix="/rag", tags=["Documents"])
app.include_router(vector_router, prefix="/vector", tags=["Vector Search"])
app.include_router(text_processing, prefix="/process", tags=["Processing"])
