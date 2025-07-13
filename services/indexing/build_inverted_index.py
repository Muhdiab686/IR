# services\indexing\build_inverted_index.py
from collections import defaultdict
from utils.io import PersistenceManager

def build_inverted_index(processed_docs: dict[str, list[str]]) -> dict[str, list[str]]:
   
    print("Building inverted index...")
    inverted_index = defaultdict(list)
    for doc_id, tokens in processed_docs.items():
        for token in set(tokens):
            inverted_index[token].append(doc_id)
    print("Inverted index built successfully.")
    return dict(inverted_index)  # حتى يكون JSON serializable

def save_inverted_index(index: dict, path: str):
    PersistenceManager.save_json(index, path)
