# run_evaluation_bert.py

from services.evaluation.bert_evaluator import evaluate_bert

# print("\n Evaluation - ANTIQUE using BERT:")
# print(evaluate_bert("antique", top_k=10 ))

print("\n Evaluation - quora using BERT:")
print(evaluate_bert("quora", top_k=10))
