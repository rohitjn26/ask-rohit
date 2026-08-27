import sys
from db import get_collection

col = get_collection()
data = col.get()
docs = data["documents"]
metadatas = data["metadatas"]
ids = data["ids"]

# If chunk ID prefixes passed as args, show those chunks in full
# Accepts: python show_chunks.py e476b325 70114327
# Or:      python show_chunks.py "['e476b325', '70114327', 'abc12345']"
if len(sys.argv) > 1:
    raw = " ".join(sys.argv[1:])
    import re
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

# Otherwise show all chunks (truncated)
print(f"Total chunks: {len(docs)}\n")
for cid, doc, meta in zip(ids, docs, metadatas):
    print(f"--- {cid[:8]} | source: {meta.get('source', '?')} ---")
    print(doc[:300])
    print()
