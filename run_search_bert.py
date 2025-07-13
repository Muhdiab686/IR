# run_search_bert.py

from services.retrieval.bert_retrieval import retrieve_top_k_bert

print("\n📚 BERT Search - ANTIQUE:")
query1 = "how do I pay my credit card debt?"
results1 = retrieve_top_k_bert("antique", query1, k=5)
for doc_id, score in results1:
    print(f"{doc_id}: {score:.4f}")

print("\n📚 BERT Search quora:")
query2 = "global climate policies and renewable energy"
results2 = retrieve_top_k_bert("quora", query2, k=5)
for doc_id, score in results2:
    print(f"{doc_id}: {score:.4f}")
