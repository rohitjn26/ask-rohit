#!/usr/bin/env python3
"""
Eval script for the resume bot golden test set.

Each answer is judged by Claude Haiku (binary PASS/FAIL + one-line reason).
Off-topic cases use exact phrase match instead of the LLM judge.

Usage:
    python tests/eval.py
    python tests/eval.py --verbose          # show actual answer for each case
    python tests/eval.py --id specific_02   # run a single test by id
"""

import sys
import os
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

import anthropic

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden.json")
RESULTS_DIR = os.path.dirname(__file__)
OFF_TOPIC_PHRASE = "I'm only able to answer questions about Rohit's background and experience."

JUDGE_PROMPT = """\
You are evaluating an AI assistant that answers questions about a person named Rohit Jain \
based only on retrieved context from his resume and project notes.

Question asked:
{question}

Expected answer (ground truth written by Rohit):
{expected}

Actual answer from the bot:
{actual}

Judge whether the actual answer is PASS or FAIL:
- PASS: the answer covers the key facts from the expected answer correctly, even if worded differently
- FAIL: the answer is wrong, incomplete in a meaningful way, refuses when it should answer, or answers when it should refuse

Reply in exactly this format (two lines, nothing else):
VERDICT: PASS or FAIL
REASON: one sentence explaining why
"""

_judge_client = None


def _get_judge():
    global _judge_client
    if _judge_client is None:
        _judge_client = anthropic.Anthropic()
    return _judge_client


def llm_judge(question, expected, actual):
    prompt = JUDGE_PROMPT.format(question=question, expected=expected, actual=actual)
    response = _get_judge().messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=120,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    passed = "VERDICT: PASS" in text
    reason = ""
    for line in text.splitlines():
        if line.startswith("REASON:"):
            reason = line[len("REASON:"):].strip()
    return passed, reason


def collect_answer(question):
    from rag import retrieve_and_answer_stream
    full, confidence, error = "", 1.0, None
    for full, confidence, error in retrieve_and_answer_stream(question):
        pass
    return full.strip(), confidence


def score_case(case, actual):
    if case["category"] == "off_topic":
        passed = OFF_TOPIC_PHRASE.lower() in actual.lower()
        reason = "correct refusal" if passed else f"expected refusal phrase not found"
        return passed, reason, "exact"

    passed, reason = llm_judge(case["question"], case["expected_answer"], actual)
    return passed, reason, "llm"


def run_eval(cases, verbose=False):
    from rag import build_index
    print("Building index...", flush=True)
    build_index()

    skipped = [c for c in cases if not c["expected_answer"]]
    runnable = [c for c in cases if c["expected_answer"]]

    print(f"\nRunning {len(runnable)} test cases ({len(skipped)} skipped — no expected answer)...\n")

    results = []
    by_category = {}

    for case in runnable:
        actual, confidence = collect_answer(case["question"])
        passed, reason, method = score_case(case, actual)

        status = "PASS" if passed else "FAIL"
        print(f"{status}  {case['id']:<22}  {case['question']}")
        if not passed or verbose:
            print(f"       reason  : {reason}")
        if verbose:
            print(f"       actual  : {actual[:200]}")
        if not passed or verbose:
            print()

        cat = case["category"]
        by_category.setdefault(cat, {"pass": 0, "total": 0})
        by_category[cat]["total"] += 1
        if passed:
            by_category[cat]["pass"] += 1

        results.append({
            "id": case["id"],
            "category": cat,
            "question": case["question"],
            "expected_answer": case["expected_answer"],
            "actual_answer": actual,
            "confidence": confidence,
            "method": method,
            "passed": passed,
            "reason": reason,
        })

    total_pass = sum(1 for r in results if r["passed"])

    print(f"\n{'─' * 64}")
    print(f"Results: {total_pass}/{len(runnable)} passed\n")
    print("By category:")
    for cat, counts in by_category.items():
        bar = "█" * counts["pass"] + "░" * (counts["total"] - counts["pass"])
        print(f"  {cat:<22}  {counts['pass']}/{counts['total']}  {bar}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"results_{ts}.json")
    with open(out_path, "w") as f:
        json.dump({"timestamp": ts, "pass": total_pass, "total": len(runnable), "cases": results}, f, indent=2)
    print(f"\nResults saved → {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Eval resume bot against golden test set.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show actual answer for each case")
    parser.add_argument("--id", help="Run only the test with this id")
    args = parser.parse_args()

    with open(GOLDEN_PATH) as f:
        cases = json.load(f)

    if args.id:
        cases = [c for c in cases if c["id"] == args.id]
        if not cases:
            print(f"No test with id={args.id!r}")
            sys.exit(1)

    run_eval(cases, verbose=args.verbose)


if __name__ == "__main__":
    main()
