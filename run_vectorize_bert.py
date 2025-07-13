# run_vectorize_bert.py
from services.vectorization.bert_vectorizer import train_bert_embeddings

if __name__ == "__main__":
    # تعالج كل داتا سيت على حدة (تنتج موديل، فيكتورز، فهرس معكوس)
    train_bert_embeddings("antique")
    train_bert_embeddings("quora")
