# run_search.py
from services.retrieval.tfidf_retrieval import Searcher

if __name__ == "__main__":
    dataset_key = "antique"
    searcher = Searcher(dataset_key)
    query = "your query text here"
    results = searcher.search(query, top_k=10)
    print("Top results:")
    for doc_id, score , text in results:
        print(f"ID: {doc_id}\nScore: {score}\nText: {text}\n---")
