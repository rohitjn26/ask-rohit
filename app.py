from dotenv import load_dotenv
load_dotenv()

import gradio as gr

from rag import retrieve_and_answer, build_index, CONFIDENCE_THRESHOLD
from notify import notify_unanswered

# Build the retrieval index once at startup, not on every request.
build_index()


def respond(message, history):
    answer, confidence, error = retrieve_and_answer(message, history)

    if error:
        # Something broke (API credit exhausted, rate limited, etc.)
        # Still tell Rohit, but don't spam him for every single rate-limit blip
        # beyond what notify.py's internal rate limit already allows.
        notify_unanswered(message, reason="error", detail=error)
        return answer

    if confidence < CONFIDENCE_THRESHOLD:
        notify_unanswered(message, reason="low_confidence")
        answer += (
            "\n\n_(I've flagged this question for Rohit to follow up on directly.)_"
        )

    return answer


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
    demo.launch()
