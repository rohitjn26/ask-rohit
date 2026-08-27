"""
Semantic cache for the resume bot.

Stores question→answer pairs in a separate ChromaDB collection, embedded by
question text. On each request, searches for a similar cached question. If
similarity exceeds CACHE_SIMILARITY_THRESHOLD, returns the cached answer
without hitting the LLM.

Cache is wiped by ingest.py whenever new PDFs are added, so answers never
go stale after a re-ingest.
"""

import hashlib
import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

from db import CHROMA_DIR

CACHE_COLLECTION_NAME = "resume_bot_cache"
CACHE_SIMILARITY_THRESHOLD = 0.65

_cache_col = None


def _get_col():
    global _cache_col
    if _cache_col is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        ef = ONNXMiniLM_L6_V2()
        _cache_col = client.get_or_create_collection(
            CACHE_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=ef,
        )
    return _cache_col


def get_cached_answer(question):
    col = _get_col()
    if col.count() == 0:
        return None

    results = col.query(query_texts=[question], n_results=1)
    if not results["distances"][0]:
        return None

    distance = results["distances"][0][0]
    similarity = 1.0 - distance
    cached_q = results["documents"][0][0]
    answer = results["metadatas"][0][0].get("answer", "")

    print(f"[cache] similarity={similarity:.3f} | cached_q={cached_q[:60]!r}")
    if similarity >= CACHE_SIMILARITY_THRESHOLD and isinstance(answer, str) and answer:
        print("[cache] HIT — returning cached answer")
        return answer

    print("[cache] MISS — forwarding to LLM")
    return None


def store_in_cache(question, answer):
    if not answer or not isinstance(answer, str):
        return
    col = _get_col()
    qid = hashlib.sha256(question.encode()).hexdigest()[:20]
    col.upsert(
        ids=[qid],
        documents=[question],
        metadatas=[{"answer": answer}],
    )
    print(f"[cache] Stored answer for: {question[:60]!r}")


def clear_cache():
    col = _get_col()
    count = col.count()
    if count > 0:
        all_ids = col.get()["ids"]
        col.delete(ids=all_ids)
        print(f"[cache] Cleared {count} cached entries.")
    else:
        print("[cache] Cache already empty.")
