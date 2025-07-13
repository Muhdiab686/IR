import joblib
from sklearn.cluster import KMeans
from utils.io import PersistenceManager

def train_bert_clusters(bert_vectors_path, output_path, num_clusters=10):
    data = joblib.load(bert_vectors_path)
    vectors, doc_ids = data['vectors'], data['doc_ids']

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vectors)

    PersistenceManager.save_joblib({
        "model": kmeans,
        "labels": labels,
        "doc_ids": doc_ids
    }, output_path)

if __name__ == "__main__":
    train_bert_clusters(
        bert_vectors_path="data/vectorizers/bert_antique_vectors.joblib",
        output_path="data/clusters/bert_antique_clusters.joblib",
        num_clusters=5
    )
    train_bert_clusters(
        bert_vectors_path="data/vectorizers/bert_quora_vectors.joblib",
        output_path="data/clusters/bert_quora_clusters.joblib",
        num_clusters=6 
    )
