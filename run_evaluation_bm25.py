from services.evaluation.bm25_evaluator import evaluate_bm25

print("\n📊 BM25 Evaluation - ANTIQUE:")
print(evaluate_bm25("antique", top_k=10 ))

print("\n📊 BM25 Evaluation - quora:")
print(evaluate_bm25("quora", top_k=10))
        