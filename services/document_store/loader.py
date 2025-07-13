# services/document_store/loader.py

import ir_datasets
from .config import DATASETS

def load_dataset(name):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset name: {name}")
    print(f"Download data from: {DATASETS[name]}")
    return ir_datasets.load(DATASETS[name])


def get_stats(dataset):
    print("Number of documents:", dataset.docs_count())
    
    try:
        print("Number of inquiries:", dataset.queries_count())
    except AttributeError:
        try:
            print("Number of inquiries:", len(list(dataset.queries_iter())))
        except AttributeError:
            print("There are no direct queries in this group.")
        except Exception as e:
            print(f" An error occurred while trying to count queries: {e}")

    try:
        print("Number of qrels ratings:", len(list(dataset.qrels_iter())))
    except Exception as e: 
        print(f" There is no qrels rating data in this group or an error occurred: {e}")