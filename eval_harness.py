"""
Evaluation Harness for Harel Insurance Chatbot
================================================
Reusable framework for evaluating any answering system against the reference set.

Usage:
    from eval_harness import EvaluationHarness

    harness = EvaluationHarness(test_set_path="ex2.json")
    results = harness.run(answer_fn=my_system, model_name="my-model")
    harness.print_summary(results)
    harness.save(results, "results.json")
"""

import json
import os
import re
import time
from typing import Callable, Optional

import numpy as np
from openai import OpenAI

# ---------------------
# Configuration
# ---------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = "text-embedding-3-large"
JUDGE_MODEL = "gpt-4o"

# Questions that expect a yes/no/simple answer (indices in flattened test_cases list)
YES_NO_QUESTION_KEYWORDS = ["האם", "זכאי", "מכוסה", "דורשת"]
FACTUAL_QUESTION_KEYWORDS = ["מהי", "מהו", "מתי", "מה המחיר", "עלות", "מספר", "תעריף"]

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

JUDGE_USER_TEMPLATE = """Question: {question}

Reference answer (ground truth): {reference}

Generated answer: {generated}"""


class EvaluationHarness:
    def __init__(self, test_set_path: str = "ex2.json", use_judge: bool = True):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.use_judge = use_judge

        with open(test_set_path, "r", encoding="utf-8") as f:
            test_samples = json.load(f)

        self.test_cases = []
        for insurance_type, samples in test_samples.items():
            for sample in samples:
                q_text = sample["שאלה"]
                q_type = self._classify_question_type(q_text)
                self.test_cases.append({
                    "prompt": q_text,
                    "reference": sample["תשובה"],
                    "domain": insurance_type,
                    "question_type": q_type,
                    "source": sample.get("מקור", {}),
                })

    # ---------------------
    # Public API
    # ---------------------

    def run(
        self,
        answer_fn: Callable[[str], str],
        model_name: str = "unknown",
        verbose: bool = True,
    ) -> dict:
        """Run evaluation on all test cases.

        Args:
            answer_fn: A callable that takes a question string and returns an answer string.
            model_name: Name label for this evaluation run.
            verbose: Print progress.

        Returns:
            A dict with per-question results and aggregate metrics.
        """
        per_question = []

        for i, case in enumerate(self.test_cases):
            if verbose:
                print(f"[{i+1}/{len(self.test_cases)}] {case['domain']}: {case['prompt'][:60]}...")

            t0 = time.time()
            try:
                generated = answer_fn(case["prompt"])
            except Exception as e:
                generated = f"[ERROR] {e}"
            latency = time.time() - t0

            # Cosine similarity
            sim = self._cosine_similarity(generated, case["reference"])

            # Abstention detection (heuristic)
            abstain_heuristic = self._detect_abstention(generated)

            # LLM judge
            judge_result = None
            if self.use_judge:
                judge_result = self._judge(case["prompt"], case["reference"], generated)

            entry = {
                "domain": case["domain"],
                "question_type": case["question_type"],
                "prompt": case["prompt"],
                "reference": case["reference"],
                "generated": generated,
                "cosine_similarity": sim,
                "abstention_heuristic": abstain_heuristic,
                "latency_seconds": round(latency, 2),
            }
            if judge_result:
                entry["judge"] = judge_result

            per_question.append(entry)

        aggregates = self._compute_aggregates(per_question)

        return {
            "model_name": model_name,
            "num_questions": len(self.test_cases),
            "aggregates": aggregates,
            "per_question": per_question,
        }

    def print_summary(self, results: dict):
        """Print a formatted summary table."""
        agg = results["aggregates"]
        name = results["model_name"]
        n = results["num_questions"]

        print(f"\n{'='*60}")
        print(f"  Evaluation Summary: {name}")
        print(f"{'='*60}")
        print(f"  Questions evaluated: {n}")
        print(f"  Mean cosine similarity: {agg['mean_cosine_similarity']:.3f}")
        print(f"  Abstention rate (heuristic): {agg['abstention_rate_heuristic']:.0%}")

        if "judge_correctness" in agg:
            jc = agg["judge_correctness"]
            print(f"\n  LLM Judge Correctness:")
            print(f"    Correct:           {jc.get('correct', 0)}/{n} ({jc.get('correct', 0)/n:.0%})")
            print(f"    Partially correct: {jc.get('partially_correct', 0)}/{n}")
            print(f"    Wrong:             {jc.get('wrong', 0)}/{n}")
            print(f"    Abstention:        {jc.get('abstention', 0)}/{n} ({jc.get('abstention', 0)/n:.0%})")
            print(f"  Hallucination rate:  {agg.get('hallucination_rate', 0):.0%}")
            print(f"  Citation rate:       {agg.get('citation_rate', 0):.0%}")

        if "by_domain" in agg:
            print(f"\n  By Domain:")
            for domain, stats in agg["by_domain"].items():
                correct = stats.get("correct", 0)
                total = stats["total"]
                print(f"    {domain:10s}  {correct}/{total} correct ({correct/total:.0%})")

        if "by_question_type" in agg:
            print(f"\n  By Question Type:")
            for qtype, stats in agg["by_question_type"].items():
                correct = stats.get("correct", 0)
                total = stats["total"]
                print(f"    {qtype:15s}  {correct}/{total} correct ({correct/total:.0%})")

        print(f"{'='*60}\n")

    def save(self, results: dict, path: str):
        """Save results to JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {path}")

    # ---------------------
    # Internal methods
    # ---------------------

    def _classify_question_type(self, question: str) -> str:
        for kw in FACTUAL_QUESTION_KEYWORDS:
            if kw in question:
                return "factual_lookup"
        return "yes_no_policy"

    def _cosine_similarity(self, text_a: str, text_b: str) -> float:
        try:
            resp = self.client.embeddings.create(model=EMBED_MODEL, input=[text_a, text_b])
            a = np.array(resp.data[0].embedding)
            b = np.array(resp.data[1].embedding)
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        except Exception:
            return 0.0

    def _detect_abstention(self, text: str) -> bool:
        patterns = [
            r"לא יודע",
            r"לא ידוע",
            r"אין לי מידע",
            r"לא מצאתי",
            r"לא ניתן לקבוע",
            r"לא הצלחתי",
            r"לא יכול לאמת",
            r"I don't know",
            r"I do not know",
            r"cannot verify",
        ]
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in patterns)

    def _judge(self, question: str, reference: str, generated: str) -> Optional[dict]:
        try:
            resp = self.client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": JUDGE_USER_TEMPLATE.format(
                            question=question,
                            reference=reference,
                            generated=generated,
                        ),
                    },
                ],
                temperature=0,
            )
            raw = resp.choices[0].message.content.strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)
            return json.loads(raw)
        except Exception as e:
            return {"correctness": "error", "hallucination": False, "has_citation": False, "explanation": str(e)}

    def _compute_aggregates(self, per_question: list) -> dict:
        n = len(per_question)
        sims = [q["cosine_similarity"] for q in per_question]
        abstentions = [q["abstention_heuristic"] for q in per_question]

        agg = {
            "mean_cosine_similarity": float(np.mean(sims)),
            "std_cosine_similarity": float(np.std(sims)),
            "abstention_rate_heuristic": sum(abstentions) / n,
        }

        # Judge-based aggregates
        if per_question[0].get("judge"):
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

                # By domain
                d = q["domain"]
                if d not in by_domain:
                    by_domain[d] = {"total": 0, "correct": 0, "abstention": 0}
                by_domain[d]["total"] += 1
                if c == "correct":
                    by_domain[d]["correct"] += 1
                if c == "abstention":
                    by_domain[d]["abstention"] += 1

                # By question type
                qt = q["question_type"]
                if qt not in by_qtype:
                    by_qtype[qt] = {"total": 0, "correct": 0, "abstention": 0}
                by_qtype[qt]["total"] += 1
                if c == "correct":
                    by_qtype[qt]["correct"] += 1
                if c == "abstention":
                    by_qtype[qt]["abstention"] += 1

            agg["judge_correctness"] = correctness_counts
            agg["hallucination_rate"] = hallucination_count / n
            agg["citation_rate"] = citation_count / n
            agg["by_domain"] = by_domain
            agg["by_question_type"] = by_qtype

        return agg


# ---------------------
# CLI: re-evaluate existing result files
# ---------------------
def evaluate_existing_results(results_path: str):
    """Re-evaluate a previously generated results file (from baseline.py) using the judge."""
    harness = EvaluationHarness(use_judge=True)

    with open(results_path, "r", encoding="utf-8") as f:
        raw_results = json.load(f)

    per_question = []
    for i, item in enumerate(raw_results):
        prompt = item["prompt"]
        reference = item["reference_response"]
        generated = item["generated_response"]

        # Find matching test case for metadata
        match = next((tc for tc in harness.test_cases if tc["prompt"] == prompt), None)
        domain = item.get("type", match["domain"] if match else "unknown")
        q_type = match["question_type"] if match else "unknown"

        print(f"[{i+1}/{len(raw_results)}] Judging: {prompt[:60]}...")

        judge_result = harness._judge(prompt, reference, generated)
        abstain = harness._detect_abstention(generated)

        per_question.append({
            "domain": domain,
            "question_type": q_type,
            "prompt": prompt,
            "reference": reference,
            "generated": generated,
            "cosine_similarity": item.get("similarity_score", 0),
            "abstention_heuristic": abstain,
            "latency_seconds": 0,
            "judge": judge_result,
        })

    model_name = os.path.splitext(os.path.basename(results_path))[0]
    aggregates = harness._compute_aggregates(per_question)

    results = {
        "model_name": model_name,
        "num_questions": len(per_question),
        "aggregates": aggregates,
        "per_question": per_question,
    }

    harness.print_summary(results)

    out_path = results_path.replace(".json", "_judged.json")
    harness.save(results, out_path)
    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Re-evaluate existing results
        evaluate_existing_results(sys.argv[1])
    else:
        print("Usage:")
        print("  Evaluate existing results:  python eval_harness.py evaluation_results_gpt-5.2.json")
        print()
        print("  Programmatic usage:")
        print("    from eval_harness import EvaluationHarness")
        print("    harness = EvaluationHarness('ex2.json')")
        print("    results = harness.run(answer_fn=my_system, model_name='my-model')")
        print("    harness.print_summary(results)")
