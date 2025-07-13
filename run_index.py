from services.indexing.build_inverted_index import build_inverted_index
from services.document_store.config import DB_PATHS

for key in DB_PATHS.keys():
    print(f" Building inverted index for {key}")
    build_inverted_index(key, DB_PATHS[key])
