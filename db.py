"""
Shared ChromaDB collection handle used by both ingest.py and rag.py.
"""

import os
import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "data", "chroma_db")
COLLECTION_NAME = "resume_bot"


def get_collection():
    print("[db] Creating ChromaDB client...", flush=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    print("[db] Loading ONNX embedding model...", flush=True)
    ef = ONNXMiniLM_L6_V2()
    print("[db] Embedding model loaded. Opening collection...", flush=True)
    return client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=ef,
    )
