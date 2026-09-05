"""
Debug tool — shows exactly which chunks were retrieved and how the answer
was generated for a given question.

Usage:
    python debug_query.py "what did Rohit work on at Medable?"
    python debug_query.py --top-k 15 "Where did Rohit work before Medable?"
"""

import sys
import argparse
from dotenv import load_dotenv
load_dotenv()

from rag import build_index, _retrieve_hybrid, SYSTEM_PROMPT, MODEL_NAME, CONFIDENCE_THRESHOLD, TOP_K
import anthropic

parser = argparse.ArgumentParser()
parser.add_argument("--top-k", type=int, default=TOP_K, help=f"Number of chunks to retrieve (default: {TOP_K})")
parser.add_argument("question", nargs="*", help="Question to ask")
args = parser.parse_args()

build_index()

question = " ".join(args.question) if args.question else input("Question: ")
top_k = args.top_k

print(f"\n{'='*60}")
print(f"QUESTION : {question}")
print(f"TOP-K    : {top_k}")
print(f"{'='*60}\n")

docs, confidence = _retrieve_hybrid(question, top_k=top_k)

print(f"\nCONFIDENCE: {confidence:.3f} (threshold: {CONFIDENCE_THRESHOLD})")
print(f"CHUNKS RETRIEVED: {len(docs)}\n")

for i, doc in enumerate(docs):
    print(f"--- chunk {i+1} ---")
    print(doc)
    print()

print(f"{'='*60}")
print("ANSWER:")
print(f"{'='*60}\n")

context = "\n\n---\n\n".join(docs)
client = anthropic.Anthropic()
response = client.messages.create(
    model=MODEL_NAME,
    max_tokens=900,
    system=SYSTEM_PROMPT.format(context=context),
    messages=[{"role": "user", "content": question}],
)
print(response.content[0].text)
