"""
Sends Rohit an email via Gmail SMTP when the bot can't confidently answer
a question, or when something breaks.

Rate-limited so a spammy/adversarial visitor can't flood the inbox.
"""

import os
import time
import smtplib
from email.mime.text import MIMEText
from collections import deque

GMAIL_USER = os.environ.get("NOTIFY_TO_EMAIL", "")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")
TO_EMAIL = os.environ.get("NOTIFY_TO_EMAIL", "")

MAX_EMAILS_PER_HOUR = 10
_sent_timestamps = deque()


def _under_rate_limit():
    now = time.time()
    while _sent_timestamps and now - _sent_timestamps[0] > 3600:
        _sent_timestamps.popleft()
    return len(_sent_timestamps) < MAX_EMAILS_PER_HOUR


def notify_unanswered(question, reason="low_confidence", detail=""):
    """
    reason: 'low_confidence' (bot didn't know) or 'error' (something broke).
    """
    if not GMAIL_APP_PASSWORD:
        print(f"[notify] GMAIL_APP_PASSWORD not set, skipping email. "
              f"Question: {question!r} reason={reason}")
        return False

    if not _under_rate_limit():
        print("[notify] rate limit hit, skipping email for:", question)
        return False

    subject = (
        "Resume bot: couldn't answer a question"
        if reason == "low_confidence"
        else "Resume bot: error (check API credit / status)"
    )
    body = f"Reason: {reason}\nQuestion: {question}\n"
    if detail:
        body += f"Detail: {detail}\n"

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = GMAIL_USER
    msg["To"] = TO_EMAIL

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_USER, TO_EMAIL, msg.as_string())
        _sent_timestamps.append(time.time())
        return True
    except Exception as e:
        print(f"[notify] Gmail send failed: {e}")
        return False
