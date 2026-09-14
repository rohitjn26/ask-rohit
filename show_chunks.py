"""
Show chunks stored in ChromaDB.

Usage:
    python show_chunks.py                           # all chunks (truncated)
    python show_chunks.py e476b325 70114327         # specific chunks by ID prefix
    python show_chunks.py --source resume           # chunks whose source contains 'resume'
    python show_chunks.py --type experience         # chunks by chunk_type
    python show_chunks.py --type experience --source resume
"""

import sys
import re
from db import get_collection

col = get_collection()
data = col.get()
docs = data["documents"]
metadatas = data["metadatas"]
ids = data["ids"]


def meta_label(meta):
    ctype = meta.get("chunk_type", "?")
    if ctype == "experience":
        company = meta.get("company", "?")
        role = meta.get("role", "")
        start = meta.get("start_year", "")
        end = meta.get("end_year", "")
        date = f"{start}–{end}" if start and end else str(start or "")
        return f"experience | {company} | {role} | {date}"
    if ctype == "deep_dive":
        qnum = meta.get("question_num", "")
        section = meta.get("section", "")
        return f"deep_dive | Q{qnum} | {section[:50]}" if qnum else f"deep_dive | {section[:60]}"
    return f"{ctype} | {meta.get('source', '?')}"


args = sys.argv[1:]

# Parse --source and --type flags
source_filter = None
type_filter = None
for flag in ("--source", "--type"):
    if flag in args:
        idx = args.index(flag)
        val = args[idx + 1].lower()
        args = args[:idx] + args[idx + 2:]
        if flag == "--source":
            source_filter = val
        else:
            type_filter = val

def matches_filters(meta):
    if source_filter and source_filter not in meta.get("source", "").lower():
        return False
    if type_filter and type_filter not in meta.get("chunk_type", "").lower():
        return False
    return True

# Specific chunk IDs
if args:
    raw = " ".join(args)
    prefixes = [p.lower() for p in re.findall(r"[0-9a-f]{6,}", raw)]
    for prefix in prefixes:
        matches = [(i, d, m) for i, d, m in zip(ids, docs, metadatas) if i.startswith(prefix)]
        if not matches:
            print(f"No chunk found with ID starting with '{prefix}'")
        for cid, doc, meta in matches:
            print(f"--- {cid} | {meta_label(meta)} ---")
            print(doc)
            print()
    sys.exit(0)

# Filter mode or all-chunks mode
rows = [(i, d, m) for i, d, m in zip(ids, docs, metadatas) if matches_filters(m)]
full_text = source_filter or type_filter  # show full text when filtered

if source_filter or type_filter:
    label = " + ".join(filter(None, [
        f"source='{source_filter}'" if source_filter else "",
        f"type='{type_filter}'" if type_filter else "",
    ]))
    print(f"Chunks matching {label}: {len(rows)}\n")
else:
    print(f"Total chunks: {len(rows)}\n")

for cid, doc, meta in rows:
    print(f"--- {cid[:8]} | {meta_label(meta)} ---")
    print(doc if full_text else doc[:300])
    print()
