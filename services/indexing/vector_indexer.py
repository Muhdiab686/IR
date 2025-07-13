import numpy as np
import faiss
from utils.io import PersistenceManager
from services.vectorization import vector_store
import argparse # Used for command-line arguments

def build_faiss_index(dataset_key: str):
    """
    Builds and saves a Faiss index for the stored BERT vectors.
    """
    print(f"--- Building Faiss index for dataset: '{dataset_key}' ---")
    
    # 1. Load the stored BERT vectors
    embedding_config = vector_store.EMBEDDING_CONFIG[dataset_key]
    vectors_data = PersistenceManager.load_joblib(embedding_config["vectors"])
    doc_vectors = vectors_data['vectors'].astype('float32')
    
    # 2. Normalize the vectors (essential for using Inner Product for cosine similarity)
    faiss.normalize_L2(doc_vectors)
    
    # 3. Initialize the Faiss index
    dimension = doc_vectors.shape[1]
    # IndexFlatIP is equivalent to Cosine Similarity on normalized vectors
    index = faiss.IndexFlatIP(dimension) 
    
    print(f"Adding {len(doc_vectors)} vectors to the Faiss index...")
    index.add(doc_vectors)
    print(f"Total vectors in index: {index.ntotal}")
    
    # 4. Save the index to a file
    index_path = vector_store.IND_DIR / f"faiss_index_{dataset_key}.bin"
    faiss.write_index(index, str(index_path))
    print(f"✅ Faiss index saved to {index_path}")
    print(f"--- Finished building Faiss index for '{dataset_key}' ---\n")

# --- Main execution block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build Faiss index for specified datasets.")
    parser.add_argument(
        'datasets', 
        nargs='*', 
        default=['quora', 'antique'], 
        help="List of datasets to process (e.g., quora antique). Defaults to all."
    )
    args = parser.parse_args()

    for dataset in args.datasets:
        build_faiss_index(dataset)