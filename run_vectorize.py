# run_vectorize.py
from services.vectorization.tfidf_vectorizer import process_dataset

if __name__ == "__main__":
    # تعالج كل داتا سيت على حدة (تنتج موديل، فيكتورز، فهرس معكوس)
    process_dataset("antique")
    process_dataset("quora")
