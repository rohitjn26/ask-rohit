"""
Debug tool — shows exactly which chunks were retrieved and how the answer
was generated for a given question.

Usage:
    python debug_query.py "what did Rohit work on at Medable?"
"""

import sys
from dotenv import load_dotenv
load_dotenv()

from rag import build_index, _retrieve_hybrid, SYSTEM_PROMPT, MODEL_NAME, CONFIDENCE_THRESHOLD
import anthropic

build_index()

question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Question: ")

print(f"\n{'='*60}")
print(f"QUESTION: {question}")
print(f"{'='*60}\n")

docs, confidence = _retrieve_hybrid(question)

print(f"CONFIDENCE: {confidence:.3f} (threshold: {CONFIDENCE_THRESHOLD})")
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
    max_tokens=600,
    system=SYSTEM_PROMPT.format(context=context),
    messages=[{"role": "user", "content": question}],
)
print(response.content[0].text)
