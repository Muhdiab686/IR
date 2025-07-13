# run_evaluation.py

from services.evaluation.tfidf_evaluator import evaluate

print("\n📊 Evaluation ANTIQUE:")
print(evaluate("antique", top_k=10))

print("\n\n\n\n\n\\n📊 Evaluation quora:")
print(evaluate("quora", top_k=10))
