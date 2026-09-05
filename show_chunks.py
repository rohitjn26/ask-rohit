"""
Show chunks stored in ChromaDB.

Usage:
    python show_chunks.py                          # all chunks (truncated)
    python show_chunks.py e476b325 70114327        # specific chunks by ID prefix
    python show_chunks.py --source resume          # chunks whose source contains 'resume'
    python show_chunks.py --source faq             # chunks from faq.md
"""

import sys
import re
from db import get_collection

col = get_collection()
data = col.get()
docs = data["documents"]
metadatas = data["metadatas"]
ids = data["ids"]

# --source filter
source_filter = None
args = sys.argv[1:]
if "--source" in args:
    idx = args.index("--source")
    source_filter = args[idx + 1].lower()
    args = args[:idx] + args[idx + 2:]

if source_filter:
    rows = [(i, d, m) for i, d, m in zip(ids, docs, metadatas)
            if source_filter in m.get("source", "").lower()]
    print(f"Chunks matching source '{source_filter}': {len(rows)}\n")
    for cid, doc, meta in rows:
        print(f"--- {cid[:8]} | source: {meta.get('source', '?')} ---")
        print(doc)
        print()
    sys.exit(0)

# Specific chunk IDs passed as args
if args:
    raw = " ".join(args)
    prefixes = [p.lower() for p in re.findall(r"[0-9a-f]{6,}", raw)]
    for prefix in prefixes:
        matches = [(i, d, m) for i, d, m in zip(ids, docs, metadatas) if i.startswith(prefix)]
        if not matches:
            print(f"No chunk found with ID starting with '{prefix}'")
        for cid, doc, meta in matches:
            print(f"--- {cid} | source: {meta.get('source', '?')} ---")
            print(doc)
            print()
    sys.exit(0)

# All chunks (truncated)
print(f"Total chunks: {len(docs)}\n")
for cid, doc, meta in zip(ids, docs, metadatas):
    print(f"--- {cid[:8]} | source: {meta.get('source', '?')} ---")
    print(doc[:300])
    print()
