---
title: Ask Rohit
emoji: 💼
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Ask Rohit's Resume — RAG chatbot

A small RAG chatbot over Rohit's resume/LinkedIn content, built to run cheaply
on Claude Haiku 4.5 and deploy on Hugging Face Spaces (Gradio).

## Architecture

- **Retrieval**: local `sentence-transformers` embeddings (free, no API cost)
  over paragraph-level chunks from `data/*.md`.
- **Generation**: Claude Haiku 4.5 via the Anthropic API — cheap and fast,
  well suited to short, grounded Q&A like this.
- **Fallback handling**: if retrieval confidence is low, or the API call
  fails (rate limited, out of credit, etc.), the bot answers gracefully and
  emails a notification via SendGrid instead of showing an error or making
  something up.

## Setup

1. `pip install -r requirements.txt`
2. Add your content as `.md` files under `data/` (one file per job/topic,
   paragraphs become retrieval chunks) — either by hand, or by running
   `ingest.py` on your PDFs (see below).
3. Set environment variables (in Hugging Face Spaces: Settings → Repository
   secrets):
   - `ANTHROPIC_API_KEY` — from console.anthropic.com
   - `SENDGRID_API_KEY` — from your SendGrid account
   - `NOTIFY_FROM_EMAIL` — a SendGrid-verified sender address
   - `NOTIFY_TO_EMAIL` — where you want alerts sent
4. `python app.py` locally, or push to a Hugging Face Space for hosting.

## Adding more documents (ingest.py)

As you add more source material (project write-ups, certificates,
recommendation letters, etc.), use `ingest.py` instead of hand-formatting
markdown each time:

```
export ANTHROPIC_API_KEY=your-key-here
python ingest.py path/to/some-document.pdf path/to/another.pdf
```

For each PDF, it:
1. Extracts the raw text (`pdfplumber`).
2. Sends it to Claude Haiku with instructions to rewrite it into
   self-contained paragraphs — each one repeating its own context
   (company/dates, project name, etc.) so it still makes sense when
   retrieved on its own, since only one chunk at a time gets shown to
   the model at answer time.
3. Saves the result as `data/<filename>.md`.

**Always spot-check the generated file before deploying** — skim it for
anything the rewrite may have dropped, blurred, or gotten wrong, since it's
an LLM rewrite of your source content, not a verbatim copy. Restart the app
(or call `rag.build_index()` again) to pick up new files.

This step costs a small amount of Anthropic API usage per document
(one Haiku call, input scaled to document length) — negligible compared to
ongoing chat usage, but it does draw from the same balance, so factor it in
if you're ingesting many documents right before your spend cap.

## Cost control (important before going live)

This is billed pay-as-you-go against your Anthropic Console balance —
there's no subscription tie-in. Before linking the bot anywhere public:

1. Go to console.anthropic.com → Billing.
2. Add prepaid credit (e.g. $5).
3. **Turn off auto-reload.** This is what makes the cap real — without it,
   the balance just runs out and stops, instead of silently topping itself
   up.
4. Optionally set a spend alert (e.g. at $3-4) so you get a heads-up before
   it runs out.

With Haiku 4.5 pricing, a full conversation (several question/answer turns)
typically costs well under a cent, so $5 covers a lot of visitor traffic —
this cap is really just a safety net against something unexpected (bot
loops, scraper traffic, etc.), not a limit you should expect to hit under
normal use.

When the balance does run out, API calls start failing — `rag.py` catches
this (`401`/`403`/`429`/other errors) and returns a friendly fallback message
to the visitor instead of a raw error, and still notifies you by email.

## Tuning

- `rag.py` → `CONFIDENCE_THRESHOLD`: raise it if the bot is answering things
  it shouldn't be confident about; lower it if it's flagging things it
  actually handled fine.
- `notify.py` → `MAX_EMAILS_PER_HOUR`: adjust if you're getting flooded or
  missing real questions.
