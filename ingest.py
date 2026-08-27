"""
Ingest one or more PDFs into ChromaDB for the resume bot.

Usage:
    python ingest.py path/to/file1.pdf path/to/file2.pdf ...
    python ingest.py --force path/to/file1.pdf   # re-ingest even if unchanged

Strategy (hybrid):
  1. Extract text + font-size metadata with pdfplumber.
  2. Try RULE-BASED chunking first: detect headers by font size relative to
     body text, group each header with the paragraphs beneath it. Free,
     instant, no API call.
  3. If heuristic isn't confident (font sizes too uniform), FALL BACK to an
     LLM rewrite (Claude Haiku) that produces self-contained chunks by reading
     for meaning instead of layout.

Idempotency:
  A manifest at data/.ingest_manifest.json tracks a content hash per source
  PDF. On rerun, an unchanged PDF is skipped. Use --force to bypass.
  When a PDF does change, old chunks for that source are deleted from ChromaDB
  before new ones are upserted.
"""

import os
import sys
import json
import hashlib
import re

YEAR_RE = re.compile(r'\b(19|20)\d{2}\b')

from dotenv import load_dotenv
load_dotenv()

import pdfplumber
import anthropic

from db import get_collection
from cache import clear_cache

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MANIFEST_PATH = os.path.join(DATA_DIR, ".ingest_manifest.json")
MODEL_NAME = "claude-haiku-4-5-20251001"

HEADER_SIZE_RATIO = 1.15
BULLET_CHARS = ('•', '▪', '▸', '●', '◦', '◆')
SUB_BULLET_CHARS = ('–', '—', '−')  # en-dash, em-dash, minus — used as sub-bullets

REWRITE_PROMPT = """You are preparing content for a retrieval-augmented \
chatbot that answers questions about Rohit Jain, based on documents he \
provides (resume, project write-ups, technical deep-dives, certificates, etc.).

I will give you raw text extracted from one of his PDF documents. Rewrite \
it into a set of self-contained paragraphs suitable as retrieval chunks:

- Each paragraph must make full sense on its own, without needing any other \
paragraph for context — because at answer time, only one paragraph at a \
time gets retrieved and shown to the model.
- Carry forward identifying context into EVERY paragraph that needs it — \
e.g. if the document describes a job, each paragraph about that job should \
repeat the company name, title, and dates; if it's a technical document, \
each paragraph should name the concept or system it's about.
- Rewrite bullet points as flowing prose sentences.
- Preserve every factual detail, number, and date exactly — do not \
invent, summarize away, or round any specifics.
- Separate paragraphs with a single blank line. Do not add headers, \
titles, numbering, or any markdown formatting — plain paragraphs only.
- Do not add commentary, preamble, or a summary — output only the \
rewritten paragraphs.

Raw extracted text:
---
{raw_text}
---

Rewritten paragraphs:"""


# --- Manifest (idempotency) -----------------------------------------------

def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(manifest):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


# --- Rule-based extraction (no API) ---------------------------------------

def extract_words_with_meta(pdf_path):
    lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            words = page.extract_words(extra_attrs=["size", "fontname"])
            if not words:
                continue
            words.sort(key=lambda w: (round(w["top"]), w["x0"]))
            current_top = None
            current_line = []
            for w in words:
                top = round(w["top"])
                if current_top is None or abs(top - current_top) <= 2:
                    current_line.append(w)
                    current_top = top if current_top is None else current_top
                else:
                    if current_line:
                        text = " ".join(x["text"] for x in current_line)
                        avg_size = sum(x["size"] for x in current_line) / len(current_line)
                        bold_count = sum(1 for x in current_line if "bold" in x.get("fontname", "").lower())
                        bold_ratio = bold_count / len(current_line)
                        lines.append((text, avg_size, bold_ratio))
                    current_line = [w]
                    current_top = top
            if current_line:
                text = " ".join(x["text"] for x in current_line)
                avg_size = sum(x["size"] for x in current_line) / len(current_line)
                bold_count = sum(1 for x in current_line if "bold" in x.get("fontname", "").lower())
                bold_ratio = bold_count / len(current_line)
                lines.append((text, avg_size, bold_ratio))
    return lines


def classify_line(text, size, bold_ratio, body_size, header_threshold):
    """
    Classify a line into one of:
      SECTION    — big bold or large font (EXPERIENCE, EDUCATION, SKILLS)
      JOB        — fully bold short line at body size (company, title, dates)
      BULLET     — line starting with a top-level bullet char (•)
      SUB_BULLET — line starting with a sub-bullet char (–, —)
      BODY       — everything else (regular text, continuations)
    """
    stripped = text.strip()
    if not stripped:
        return 'EMPTY', ''

    # Bullet detection takes priority — structure beats font
    if stripped[0] in BULLET_CHARS:
        return 'BULLET', stripped[1:].strip()
    if stripped[0] in SUB_BULLET_CHARS and len(stripped) > 2:
        return 'SUB_BULLET', stripped[1:].strip()

    is_large = size >= header_threshold
    is_bold_line = bold_ratio >= 0.75
    is_short = len(stripped) < 80
    is_all_caps = stripped.replace(' ', '').isupper() and len(stripped) > 2

    # Large font = section header (name banner, etc.)
    if is_large:
        return 'SECTION', stripped

    # Fully bold + ALL CAPS + short = section header (EXPERIENCE, EDUCATION)
    if is_bold_line and is_all_caps and is_short:
        return 'SECTION', stripped

    # Fully bold + has year or separator (|, —, –) + starts with letter = job/company header
    # Length limit is generous (150) since company lines can be long.
    # Must start with a letter so date-only continuation lines like "2020 – Dec 2021"
    # don't accidentally overwrite the job context.
    has_year = bool(YEAR_RE.search(stripped))
    has_separator = '|' in stripped or ' — ' in stripped or ' – ' in stripped
    if (is_bold_line and size >= body_size * 0.95 and len(stripped) < 150
            and stripped[0].isalpha() and (has_year or has_separator)):
        return 'JOB', stripped

    return 'BODY', stripped


def build_chunks(lines, body_size, header_threshold):
    """
    Walk classified lines top-to-bottom and group into self-contained chunks.

    Each • bullet (with its – sub-bullets and body continuations) becomes one
    chunk, prefixed with the current section + job context so every chunk is
    self-contained for retrieval.

    Sections/jobs with no bullets (e.g. Skills, Summary) are collected as a
    single chunk flushed when the next section or job header appears.
    """
    chunks = []
    current_section = ""
    current_job = ""
    current_lines = []

    def context_prefix():
        parts = [p for p in [current_section, current_job] if p]
        return " | ".join(parts)

    def flush():
        nonlocal current_lines
        if current_lines:
            prefix = context_prefix()
            body = "\n".join(current_lines)
            chunk = f"{prefix}\n{body}" if prefix else body
            chunks.append(chunk.strip())
            current_lines = []

    for text, size, bold_ratio in lines:
        kind, content = classify_line(text, size, bold_ratio, body_size, header_threshold)

        if kind == 'EMPTY':
            continue

        elif kind == 'SECTION':
            flush()
            current_section = content
            current_job = ""

        elif kind == 'JOB':
            flush()
            current_job = content

        elif kind == 'BULLET':
            flush()
            current_lines = [content]

        elif kind == 'SUB_BULLET':
            current_lines.append(f"– {content}")

        elif kind == 'BODY':
            current_lines.append(content)

    flush()
    return [c for c in chunks if c.strip()]


def try_rule_based_chunks(pdf_path):
    lines = extract_words_with_meta(pdf_path)
    if not lines:
        print("  [detect] no text lines extracted.")
        return None

    sizes = [size for _, size, _ in lines]
    body_size = max(set(sizes), key=sizes.count)
    header_threshold = body_size * HEADER_SIZE_RATIO

    print(f"  [detect] body font size: {body_size:.1f} | "
          f"header threshold (>={HEADER_SIZE_RATIO}x): {header_threshold:.1f}")

    chunks = build_chunks(lines, body_size, header_threshold)

    if len(chunks) <= 1:
        print(f"  [detect] only {len(chunks)} chunk(s) — falling back to LLM.")
        return None

    print(f"  [detect] rule-based chunking succeeded — {len(chunks)} chunks.")
    return chunks


# --- Markdown chunking ----------------------------------------------------

def chunk_markdown(md_path):
    """
    Chunk a markdown file by header hierarchy + bullet structure.

    # H1 and ## H2 = section/sub-section boundaries (context carriers)
    ### H3 = chunk boundary within a section
    - bullet lines = individual chunk or appended to current chunk
    Regular paragraphs = body, flushed at next header
    """
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    chunks = []
    current_h1 = ""
    current_h2 = ""
    current_lines = []

    def context_prefix():
        parts = [p for p in [current_h1, current_h2] if p]
        return " | ".join(parts)

    def flush():
        nonlocal current_lines
        if current_lines:
            prefix = context_prefix()
            body = "\n".join(current_lines)
            chunk = f"{prefix}\n{body}" if prefix else body
            chunks.append(chunk.strip())
            current_lines = []

    for raw in lines:
        line = raw.rstrip()
        if not line:
            continue

        if line.startswith("# "):
            flush()
            current_h1 = line[2:].strip()
            current_h2 = ""
        elif line.startswith("## "):
            flush()
            current_h2 = line[3:].strip()
        elif line.startswith("### "):
            flush()
            current_lines = [line[4:].strip()]
        elif line.startswith(("- ", "* ", "• ")):
            flush()
            current_lines = [line[2:].strip()]
        elif line.startswith("  - ") or line.startswith("  * "):
            current_lines.append(f"– {line.strip()[2:]}")
        else:
            current_lines.append(line.strip())

    flush()
    result = [c for c in chunks if c.strip()]
    print(f"  [markdown] chunked into {len(result)} chunks.")
    return result


# --- DOCX chunking --------------------------------------------------------

def extract_docx_lines(docx_path):
    """
    Extract lines from a DOCX with the same (text, avg_size, bold_ratio)
    format as extract_words_with_meta() so both share classify_line/build_chunks.
    """
    from docx import Document

    doc = Document(docx_path)
    lines = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        runs = [r for r in para.runs if r.text.strip()]
        if not runs:
            lines.append((text, 12.0, 0.0))
            continue
        sizes = []
        bold_count = 0
        for run in runs:
            pt = run.font.size / 12700 if run.font.size else None
            sizes.append(pt if pt else 12.0)
            if run.bold:
                bold_count += 1
        avg_size = sum(sizes) / len(sizes)
        bold_ratio = bold_count / len(runs)
        lines.append((text, avg_size, bold_ratio))
    return lines


def chunk_docx(docx_path):
    lines = extract_docx_lines(docx_path)
    if not lines:
        print("  [docx] no text extracted.")
        return []

    sizes = [s for _, s, _ in lines]
    body_size = max(set(round(s, 1) for s in sizes), key=lambda x: sum(1 for s in sizes if round(s, 1) == x))
    header_threshold = body_size * HEADER_SIZE_RATIO

    print(f"  [docx] body font size: {body_size:.1f} | header threshold: {header_threshold:.1f}")

    chunks = build_chunks(lines, body_size, header_threshold)
    if not chunks:
        print("  [docx] no chunks produced.")
        return []

    print(f"  [docx] chunked into {len(chunks)} chunks.")
    return chunks


# --- LLM fallback ---------------------------------------------------------

def extract_pdf_text(pdf_path):
    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                parts.append(text)
    return "\n\n".join(parts)


def rewrite_as_chunks(raw_text, client):
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=4000,
        messages=[{"role": "user", "content": REWRITE_PROMPT.format(raw_text=raw_text)}],
    )
    return response.content[0].text.strip()


# --- Semantic deduplication -----------------------------------------------

DEDUP_THRESHOLD = 0.92


def dedup_chunks(chunks, source_name, collection):
    """
    Drop chunks that are near-duplicates of chunks already in ChromaDB
    from a DIFFERENT source. Same-source chunks are always kept (re-ingest).
    """
    if collection.count() == 0:
        return chunks

    unique = []
    skipped = 0
    for chunk in chunks:
        results = collection.query(query_texts=[chunk], n_results=1)
        if not results["distances"][0]:
            unique.append(chunk)
            continue
        similarity = 1.0 - results["distances"][0][0]
        existing_source = results["metadatas"][0][0].get("source", "")
        if similarity >= DEDUP_THRESHOLD and existing_source != source_name:
            skipped += 1
            print(f"  [dedup] skipped (sim={similarity:.3f} vs {existing_source}): {chunk[:60]!r}")
            continue
        unique.append(chunk)

    if skipped:
        print(f"  [dedup] {skipped} duplicate(s) removed, {len(unique)} unique chunks kept.")
    return unique


# --- Main ingestion flow --------------------------------------------------

def ingest(file_path, collection, client):
    source_name = os.path.basename(file_path)

    print(f"Reading {file_path} ...")

    if file_path.lower().endswith(".md"):
        chunks = chunk_markdown(file_path)
    elif file_path.lower().endswith(".docx"):
        chunks = chunk_docx(file_path)
    else:
        rule_chunks = try_rule_based_chunks(file_path)
        if rule_chunks:
            chunks = rule_chunks
        else:
            print("  Rule-based chunking wasn't confident — falling back to LLM rewrite.")
            raw_text = extract_pdf_text(file_path)
            if not raw_text.strip():
                print("  WARNING: no extractable text (scanned PDF?). Skipping.")
                return False
            rewritten = rewrite_as_chunks(raw_text, client)
            print(f"  [llm] raw output preview:\n{rewritten[:500]}\n  ...")
            chunks = [p.strip() for p in rewritten.split("\n\n") if p.strip()]
            if len(chunks) <= 1:
                chunks = [p.strip() for p in rewritten.split("\n") if p.strip()]
                print(f"  [llm] fell back to single-newline split — {len(chunks)} chunks")

    # Drop near-duplicates of chunks already in DB from other sources.
    chunks = dedup_chunks(chunks, source_name, collection)
    if not chunks:
        print(f"  All chunks were duplicates of existing content — skipping {source_name}.")
        return False

    # Remove stale chunks for this source before upserting fresh ones.
    existing = collection.get(where={"source": source_name})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        print(f"  Removed {len(existing['ids'])} stale chunks for {source_name}.")

    ids = [
        hashlib.sha256(f"{source_name}:{chunk}".encode()).hexdigest()[:20]
        for chunk in chunks
    ]
    collection.upsert(
        ids=ids,
        documents=chunks,
        metadatas=[{"source": source_name} for _ in chunks],
    )
    print(f"  Upserted {len(chunks)} chunks for {source_name} into ChromaDB.")
    return True


def main():
    args = sys.argv[1:]
    force = "--force" in args
    pdf_paths = [a for a in args if a != "--force"]

    if not pdf_paths:
        print("Usage: python ingest.py [--force] path/to/file1.pdf [path/to/file2.pdf ...]")
        sys.exit(1)

    manifest = load_manifest()
    collection = get_collection()
    client = anthropic.Anthropic()

    for pdf_path in pdf_paths:
        if not pdf_path.lower().endswith((".pdf", ".md", ".docx")):
            print(f"Skipping unsupported file type: {pdf_path}")
            continue
        if not os.path.exists(pdf_path):
            print(f"File not found: {pdf_path}")
            continue

        current_hash = file_hash(pdf_path)
        if not force and manifest.get(pdf_path) == current_hash:
            print(f"Skipping {pdf_path} — unchanged. Use --force to redo.")
            continue

        success = ingest(pdf_path, collection, client)
        if success:
            manifest[pdf_path] = current_hash
            save_manifest(manifest)
            clear_cache()

    print(f"\nDone. Total chunks in DB: {collection.count()}")


if __name__ == "__main__":
    main()
