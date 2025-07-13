# run_bm25.py

from services.vectorization.bm25_model import process_dataset
from services.document_store.config import DB_PATHS

def train_bm25(dataset_key: str, force=False):
    return process_dataset(dataset_key, force=force)

if __name__ == "__main__":
    datasets_to_train = list(DB_PATHS.keys())

    for dataset_key in datasets_to_train:
        print(f"==== تدريب BM25 على: {dataset_key} ====")
        train_bm25(dataset_key, force=False)
