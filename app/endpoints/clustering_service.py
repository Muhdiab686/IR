from fastapi import APIRouter
import joblib
from sklearn.metrics.pairwise import cosine_similarity

router = APIRouter()

# تحميل النماذج
tfidf_clusters = joblib.load("data/clusters/tfidf_clusters.joblib")
bert_clusters = joblib.load("data/clusters/bert_clusters.joblib")

@router.post("/")
def cluster_search(query_vector: list[float], model: str = "tfidf", top_k: int = 10):
    if model == "tfidf":
        cluster_data = tfidf_clusters
        vectors = joblib.load("data/vectorizers/tfidf_antique_vectors.joblib")['matrix']
    else:
        cluster_data = bert_clusters
        vectors = joblib.load("data/vectorizers/bert_antique_vectors.joblib")['vectors']
    
    cluster_model = cluster_data['model']
    labels = cluster_data['labels']
    
    query_label = cluster_model.predict([query_vector])[0]
    cluster_indices = [i for i, lbl in enumerate(labels) if lbl == query_label]
    
    cluster_vectors = vectors[cluster_indices]
    sims = cosine_similarity([query_vector], cluster_vectors).flatten()
    
    sorted_indices = sims.argsort()[::-1][:top_k]
    doc_indices = [cluster_indices[i] for i in sorted_indices]
    scores = sims[sorted_indices].tolist()

    return {"doc_indices": doc_indices, "scores": scores}
