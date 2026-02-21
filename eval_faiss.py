"""
Evaluate the FAISS RAG system against the ex2.json test set.

Answers are produced by faiss2.answer_question() — the exact same pipeline
users experience (hybrid retrieval, query rewriting, LLM-as-judge).

No OpenAI API key required — everything runs through Nebius.
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from classification_workflow import get_test_questions
import faiss2
from faiss2 import answer_question, create_or_load_faiss_index, NEBIUS_API_KEY, NEBIUS_BASE_URL, LLM_MODEL

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SUMMARY_PATH = Path(__file__).parent / "evaluation_summary.json"
RUN_LABEL    = "baseline run"

BASELINES = {
    "GPT-4o": {
        "correct": 5, "abstention": 14, "hallucination_rate": 0.00, "citation_rate": 0.30,
        "by_domain": {"דירה": 2, "עסקים": 1, "רכב": 0, "בריאות": 2, "נסיעות": 0},
        "by_qtype":  {"yes_no_policy": 2, "factual_lookup": 3},
    },
    "GPT-5.2": {
        "correct": 9, "abstention": 10, "hallucination_rate": 0.05, "citation_rate": 0.30,
        "by_domain": {"דירה": 1, "עסקים": 3, "רכב": 1, "בריאות": 3, "נסיעות": 1},
        "by_qtype":  {"yes_no_policy": 3, "factual_lookup": 6},
    },
}
DOMAIN_TOTALS = {"דירה": 7, "עסקים": 5, "רכב": 3, "בריאות": 3, "נסיעות": 2}
QTYPE_TOTALS  = {"yes_no_policy": 12, "factual_lookup": 8}

FACTUAL_KEYWORDS = ["מהי", "מהו", "מתי", "מה המחיר", "עלות", "מספר", "תעריף"]

# ---------------------------------------------------------------------------
# Eval scoring judge (compares generated answer against ground-truth reference)
# ---------------------------------------------------------------------------
JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for an insurance Q&A system.
You will receive a question, a reference answer (ground truth), and a generated answer.
Evaluate the generated answer and respond with a JSON object containing:

1. "correctness": one of "correct", "partially_correct", "wrong", "abstention"
   - "correct": The generated answer conveys the same factual information as the reference.
   - "partially_correct": The answer is on the right track but missing key details or has minor errors.
   - "wrong": The answer asserts something factually different from the reference.
   - "abstention": The answer says "I don't know" or similar refusal to answer.

2. "hallucination": true/false
   - true if the generated answer asserts specific facts that contradict the reference answer.
   - false if the answer is correct, abstains, or hedges without asserting false facts.

3. "has_citation": true/false
   - true if the generated answer includes any source reference (URL, document name, page number).

4. "explanation": A brief (1-2 sentence) explanation of your assessment.

Respond ONLY with the JSON object, no markdown formatting."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify_question_type(question: str) -> str:
    for kw in FACTUAL_KEYWORDS:
        if kw in question:
            return "factual_lookup"
    return "yes_no_policy"


def judge(client: OpenAI, question: str, reference: str, generated: str) -> dict:
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {question}\n\nReference answer (ground truth): {reference}\n\nGenerated answer: {generated}"},
            ],
            temperature=0,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        return json.loads(raw)
    except Exception as e:
        return {"correctness": "error", "hallucination": False, "has_citation": False, "explanation": str(e)}


def detect_abstention(text: str) -> bool:
    patterns = [r"לא יודע", r"אין לי מידע", r"לא מצאתי", r"לא ניתן לקבוע",
                r"לא הצלחתי", r"I don't know", r"cannot verify", r"אין מספיק מידע"]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def compute_aggregates(per_question: list) -> dict:
    n = len(per_question)
    correctness_counts = {}
    hallucination_count = 0
    citation_count = 0
    by_domain = {}
    by_qtype = {}

    for q in per_question:
        j = q["judge"]
        c = j.get("correctness", "error")
        correctness_counts[c] = correctness_counts.get(c, 0) + 1
        if j.get("hallucination"):
            hallucination_count += 1
        if j.get("has_citation"):
            citation_count += 1

        d = q["domain"]
        by_domain.setdefault(d, {"total": 0, "correct": 0, "abstention": 0})
        by_domain[d]["total"] += 1
        if c == "correct":
            by_domain[d]["correct"] += 1
        if c == "abstention":
            by_domain[d]["abstention"] += 1

        qt = q["question_type"]
        by_qtype.setdefault(qt, {"total": 0, "correct": 0, "abstention": 0})
        by_qtype[qt]["total"] += 1
        if c == "correct":
            by_qtype[qt]["correct"] += 1

    return {
        "judge_correctness": correctness_counts,
        "hallucination_rate": hallucination_count / n,
        "citation_rate": citation_count / n,
        "abstention_rate": correctness_counts.get("abstention", 0) / n,
        "by_domain": by_domain,
        "by_question_type": by_qtype,
    }


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison(agg: dict, n: int):
    jc           = agg.get("judge_correctness", {})
    rag_correct  = jc.get("correct", 0)
    rag_abstain  = jc.get("abstention", 0)
    rag_halluc   = agg.get("hallucination_rate", 0.0)
    rag_citation = agg.get("citation_rate", 0.0)

    def fmt(num, total):
        return f"{num}/{total} ({num/total:.0%})"

    W = 72
    b4  = BASELINES["GPT-4o"]
    b52 = BASELINES["GPT-5.2"]

    print("\n" + "=" * W)
    print("  EVALUATION COMPARISON")
    print("=" * W)
    print(f"  {'Metric':<32} {'GPT-4o':>10} {'GPT-5.2':>10} {'FAISS RAG':>10}")
    print("-" * W)
    print(f"  {'Correct answers':<32} {fmt(b4['correct'],n):>10} {fmt(b52['correct'],n):>10} {fmt(rag_correct,n):>10}")
    print(f"  {'Abstentions':<32} {fmt(b4['abstention'],n):>10} {fmt(b52['abstention'],n):>10} {fmt(rag_abstain,n):>10}")
    print(f"  {'Hallucination rate':<32} {b4['hallucination_rate']:>9.0%} {b52['hallucination_rate']:>10.0%} {rag_halluc:>10.0%}")
    print(f"  {'Citation rate':<32} {b4['citation_rate']:>9.0%} {b52['citation_rate']:>10.0%} {rag_citation:>10.0%}")

    print(f"\n  {'By Domain':<32} {'GPT-4o':>10} {'GPT-5.2':>10} {'FAISS RAG':>10}")
    print("-" * W)
    by_domain = agg.get("by_domain", {})
    for domain, total in DOMAIN_TOTALS.items():
        rag_d = by_domain.get(domain, {}).get("correct", "?")
        rag_s = fmt(rag_d, total) if isinstance(rag_d, int) else "?"
        print(f"  {domain:<32} {fmt(b4['by_domain'][domain], total):>10} {fmt(b52['by_domain'][domain], total):>10} {rag_s:>10}")

    print(f"\n  {'By Question Type':<32} {'GPT-4o':>10} {'GPT-5.2':>10} {'FAISS RAG':>10}")
    print("-" * W)
    by_qtype = agg.get("by_question_type", {})
    for qtype, total in QTYPE_TOTALS.items():
        rag_q = by_qtype.get(qtype, {}).get("correct", "?")
        rag_s = fmt(rag_q, total) if isinstance(rag_q, int) else "?"
        print(f"  {qtype:<32} {fmt(b4['by_qtype'][qtype], total):>10} {fmt(b52['by_qtype'][qtype], total):>10} {rag_s:>10}")

    print("=" * W)


# ---------------------------------------------------------------------------
# Run summary (persisted across runs)
# ---------------------------------------------------------------------------

def append_to_summary(agg: dict, n: int, label: str):
    jc = agg.get("judge_correctness", {})
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "label": label,
        "model": LLM_MODEL,
        "correct": jc.get("correct", 0),
        "partially_correct": jc.get("partially_correct", 0),
        "wrong": jc.get("wrong", 0),
        "abstention": jc.get("abstention", 0),
        "accuracy": f"{jc.get('correct', 0)/n:.0%}",
        "hallucination_rate": f"{agg.get('hallucination_rate', 0):.0%}",
        "citation_rate": f"{agg.get('citation_rate', 0):.0%}",
        "by_domain": {
            d: f"{s.get('correct', 0)}/{s['total']}"
            for d, s in agg.get("by_domain", {}).items()
        },
        "by_question_type": {
            qt: f"{s.get('correct', 0)}/{s['total']}"
            for qt, s in agg.get("by_question_type", {}).items()
        },
    }

    runs = []
    if SUMMARY_PATH.exists():
        with open(SUMMARY_PATH, encoding="utf-8") as f:
            runs = json.load(f)
    runs.append(entry)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(runs, f, ensure_ascii=False, indent=2)
    print(f"Run appended to {SUMMARY_PATH} ({len(runs)} total runs)")


def print_summary_history():
    if not SUMMARY_PATH.exists():
        return
    with open(SUMMARY_PATH, encoding="utf-8") as f:
        runs = json.load(f)
    if not runs:
        return

    W = 80
    print("\n" + "=" * W)
    print("  ALL RUNS HISTORY")
    print("=" * W)
    print(f"  {'#':<3} {'Timestamp':<17} {'Label':<25} {'Correct':>8} {'Abstain':>8} {'Halluc':>8} {'Cite':>7}")
    print("-" * W)
    for i, r in enumerate(runs, 1):
        n = r["correct"] + r["partially_correct"] + r["wrong"] + r["abstention"]
        print(
            f"  {i:<3} {r['timestamp']:<17} {r['label']:<25}"
            f" {r['correct']}/{n} ({r['accuracy']}){' ':>1}"
            f" {r['abstention']}/{n}{' ':>3}"
            f" {r['hallucination_rate']:>8}"
            f" {r['citation_rate']:>7}"
        )
    print("=" * W)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(label: str = RUN_LABEL):
    test_cases = get_test_questions()
    print(f"Loaded {len(test_cases)} test questions.\n")

    # Initialise faiss2's global state (index + hybrid retriever + table store)
    create_or_load_faiss_index()
    print("Ready.\n")

    nebius_client = OpenAI(base_url=NEBIUS_BASE_URL, api_key=NEBIUS_API_KEY)

    per_question = []
    for i, case in enumerate(test_cases):
        question  = case["prompt"]
        reference = case["reference"]
        domain    = case["type"]
        qtype     = classify_question_type(question)

        print(f"[{i+1}/{len(test_cases)}] {domain}: {question[:60]}...")

        t0 = time.time()
        generated = answer_question(question)   # exact same function as the chatbot
        latency = round(time.time() - t0, 2)

        judge_result = judge(nebius_client, question, reference, generated)
        abstain = detect_abstention(generated)

        print(f"  → {judge_result.get('correctness','?')} | hallucination={judge_result.get('hallucination')} | citation={judge_result.get('has_citation')}")

        per_question.append({
            "domain": domain,
            "question_type": qtype,
            "prompt": question,
            "reference": reference,
            "generated": generated,
            "abstention_heuristic": abstain,
            "latency_seconds": latency,
            "judge": judge_result,
        })

    agg = compute_aggregates(per_question)
    results = {
        "model_name": f"faiss2-rag ({LLM_MODEL})",
        "num_questions": len(test_cases),
        "aggregates": agg,
        "per_question": per_question,
    }

    out_path = "evaluation_results_faiss2-rag.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")

    append_to_summary(agg, len(test_cases), label)
    print_comparison(agg, len(test_cases))
    print_summary_history()


if __name__ == "__main__":
    import sys
    label = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else RUN_LABEL
    main(label)
