from dotenv import load_dotenv
load_dotenv()

import os
import threading
import gradio as gr

from rag import retrieve_and_answer_stream, build_index, CONFIDENCE_THRESHOLD
from notify import notify_unanswered

# Load index in background so Gradio binds the port immediately.
_index_ready = False

def _load_index():
    global _index_ready
    try:
        print("[startup] Loading index in background thread...", flush=True)
        build_index()
        _index_ready = True
        print("[startup] Background index load complete — ready.", flush=True)
    except Exception as e:
        import traceback
        print(f"[startup] ERROR loading index: {e}", flush=True)
        traceback.print_exc()

threading.Thread(target=_load_index, daemon=True).start()
print("[startup] Gradio starting — index loading in background.", flush=True)


def respond(message, history):
    if not _index_ready:
        yield "Still loading, please try again in a few seconds..."
        return

    partial = ""
    confidence = 1.0
    error = None

    for partial, confidence, error in retrieve_and_answer_stream(message, history):
        yield partial

    if error:
        notify_unanswered(message, reason="error", detail=error)
        return

    if confidence < CONFIDENCE_THRESHOLD:
        notify_unanswered(message, reason="low_confidence")
        yield partial + "\n\n_(I've flagged this question for Rohit to follow up on directly.)_"


demo = gr.ChatInterface(
    fn=respond,
    title="Ask Rohit's Resume",
    description=(
        "Ask me about Rohit's experience, projects, or background. "
        "I'll do my best to answer from his resume and LinkedIn — and if "
        "I don't know something, I'll flag it for him to follow up."
    ),
    examples=[
        "What did Rohit work on at Medable?",
        "Why is Rohit job searching right now?",
        "What's Rohit's experience with data platforms and AI?",
    ],
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
