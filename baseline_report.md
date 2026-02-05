# Baseline Report: GPT-4o vs GPT-5.2 on Harel Insurance Q&A

## 1. Executive Summary

We evaluated **GPT-4o** and **GPT-5.2** (with web-search tool) against a 20-question reference set spanning 5 Harel insurance domains. Neither model has access to Harel's policy PDFs; both rely on web search over the public site.

| Metric | GPT-4o | GPT-5.2 |
|--------|--------|---------|
| **Correct answers** | 5/20 (25%) | 9/20 (45%) |
| **Abstentions ("I don't know")** | 14/20 (70%) | 10/20 (50%) |
| **Hallucinations** | 0/20 (0%) | 1/20 (5%) |
| **Partially correct** | 1/20 (5%) | 0/20 (0%) |
| **Mean cosine similarity** | 0.442 | 0.375 |

**Key takeaway:** GPT-5.2 is meaningfully better (45% vs 25% accuracy) but still fails on more than half the questions. The cosine similarity metric is misleading — GPT-4o scores higher because short Hebrew "I don't know" responses share embedding space with short Hebrew reference answers. **Manual correctness assessment is essential.**

---

## 2. Methodology

### 2.1 Setup
- **System prompt:** Answer from Harel website only, cite sources, say "I don't know" if unable to verify.
- **Tool:** OpenAI `web_search` enabled for both models.
- **Embedding model:** `text-embedding-3-large` for cosine similarity.
- **Reference set:** 20 questions from `ex2.json` with ground-truth answers sourced from Harel policy documents.

### 2.2 Evaluation Dimensions
Each response was assessed on four axes:

| Dimension | Definition |
|-----------|------------|
| **Correctness** | Does the answer match the ground truth? (Correct / Partial / Wrong / Abstention) |
| **Hallucination** | Does the model assert false information as fact? |
| **Citation** | Is a source provided? Does it point to the relevant document? |
| **Completeness** | Does the answer include the key detail from the reference (e.g., "200L limit", "60 days")? |

---

## 3. Results by Domain

### 3.1 Apartment (דירה) — 7 questions

| # | Question (short) | Reference | GPT-4o | GPT-5.2 |
|---|-----------------|-----------|--------|---------|
| 1 | Laptop coverage? | לא | Abstain | Abstain (hedges) |
| 2 | Plants extension extra cost? | לא (no extra) | **Correct** | **Correct** |
| 3 | When is dwelling "unoccupied"? | >60 days | **Correct** (60 days) | Abstain |
| 4 | Lightning coverage? | כן | Abstain | Abstain |
| 5 | Electric bicycle theft outside? | לא (non-motorized only) | Partial (correct "no", wrong reason) | Abstain |
| 6 | Solar heater 300L covered? | לא (>200L excluded) | Abstain | **HALLUCINATION** (says may be covered) |
| 7 | Rain moisture in ceiling? | לא | Abstain | Abstain (hedges) |

| Metric | GPT-4o | GPT-5.2 |
|--------|--------|---------|
| Correct | 2/7 (29%) | 1/7 (14%) |
| Abstain | 4/7 | 5/7 |
| Hallucinate | 0 | 1 |

**Analysis:** This is the hardest domain. Most answers require reading specific clauses in the "Adira" policy PDFs (exclusions for >200L boilers, non-motorized bicycle definitions, 60-day vacancy rules). Web search surfaces marketing pages, not policy text.

### 3.2 Business (עסקים) — 5 questions

| # | Question (short) | Reference | GPT-4o | GPT-5.2 |
|---|-----------------|-----------|--------|---------|
| 8 | Statute of limitations? | 3 years | Abstain | **Correct** (3 yrs + 5 yrs note) |
| 9 | Doctor license — legal costs? | כן | Abstain | **Correct** (yes, with extension) |
| 10 | Claims phone number? | 03-9294000 | **Correct** | **Correct** |
| 11 | Moshav — land subsidence? | לא | Abstain | Abstain |
| 12 | Asbestos = "massive structure"? | כן | Abstain | Abstain |

| Metric | GPT-4o | GPT-5.2 |
|--------|--------|---------|
| Correct | 1/5 (20%) | 3/5 (60%) |
| Abstain | 4/5 | 2/5 |

**Analysis:** GPT-5.2 excels here on legal/regulatory questions that are partly covered on public web pages. It fails on Moshav-specific policy definitions that live in PDFs.

### 3.3 Car (רכב) — 3 questions

| # | Question (short) | Reference | GPT-4o | GPT-5.2 |
|---|-----------------|-----------|--------|---------|
| 13 | CMH — fuel tank coverage? | לא | Abstain | Abstain |
| 14 | Tav Socher statute of limitations? | 7 years | Abstain | **Correct** (7 years) |
| 15 | CMH — battery jump damage? | לא | Abstain | Abstain |

| Metric | GPT-4o | GPT-5.2 |
|--------|--------|---------|
| Correct | 0/3 (0%) | 1/3 (33%) |
| Abstain | 3/3 | 2/3 |

**Analysis:** CMH (construction equipment) policy questions are entirely PDF-locked. The statute-of-limitations question is answerable from public legal knowledge.

### 3.4 Health (בריאות) — 3 questions

| # | Question (short) | Reference | GPT-4o | GPT-5.2 |
|---|-----------------|-----------|--------|---------|
| 16 | Home visit copay? | 25 NIS | **Correct** | **Correct** |
| 17 | Online psych copay? | 100 NIS | **Correct** | **Correct** |
| 18 | Surgery insurance monthly (age 14)? | 38.27 NIS | Abstain | **Correct** |

| Metric | GPT-4o | GPT-5.2 |
|--------|--------|---------|
| Correct | 2/3 (67%) | 3/3 (100%) |

**Analysis:** Best domain for both models. Health service pages on harel-group.co.il expose copay amounts and pricing tables publicly, making them accessible to web search.

### 3.5 Travel (נסיעות) — 2 questions

| # | Question (short) | Reference | GPT-4o | GPT-5.2 |
|---|-----------------|-----------|--------|---------|
| 19 | First Class — 1.5 year trip? | לא (365 day max) | Abstain | **Correct** (365 day limit) |
| 20 | First Class daily rate to India? | 2.2 USD/day | Abstain | Abstain |

| Metric | GPT-4o | GPT-5.2 |
|--------|--------|---------|
| Correct | 0/2 (0%) | 1/2 (50%) |

**Analysis:** The duration limit is on the policy page; the pricing tariff is in a PDF.

---

## 4. Results by Question Type

| Question type | Count | GPT-4o correct | GPT-5.2 correct |
|--------------|-------|----------------|-----------------|
| **Yes/No policy** (coverage questions) | 12 | 2 (17%) | 3 (25%) |
| **Factual lookup** (numbers, definitions) | 8 | 3 (38%) | 6 (75%) |

Both models perform dramatically better on factual lookups (phone numbers, copays, statutory periods) than on yes/no policy-coverage questions that require reading exclusion clauses.

---

## 5. Failure Analysis

### 5.1 Where GPT-5.2 Succeeds (9/20)
These questions share a pattern — **the answer is surfaced on a public web page**:

| Pattern | Examples |
|---------|----------|
| Copay/pricing on product pages | Home visit 25 NIS, psych 100 NIS, surgery 38.27 NIS |
| Legal/regulatory public knowledge | Statute of limitations (3 yrs, 7 yrs) |
| Product feature on marketing page | First Class 365-day limit, plants extension free |
| Contact information | Claims phone 03-9294000 |

### 5.2 Where GPT-5.2 Fails Without Domain Grounding (11/20)
These failures share a pattern — **the answer requires reading policy PDF clauses**:

| Failure mode | Count | Examples |
|-------------|-------|----------|
| **Abstention on PDF-locked info** | 9 | 200L boiler limit, electric bicycle motor exclusion, CMH fuel exclusion, Moshav definitions |
| **Hallucination from shallow web reading** | 1 | Said 300L boiler "may be covered" — missed the 200L limit in the policy PDF |
| **Numeric lookup in PDF tariff** | 1 | Daily rate from tariff PDF inaccessible to web search |

### 5.3 Root Causes

1. **PDF blindness**: Web search can read HTML pages but not policy PDFs. ~55% of reference answers come from PDF-only content.
2. **Cautious refusal**: The system prompt's "say I don't know" instruction leads to safe-but-useless abstentions. The models are well-calibrated but unhelpful.
3. **Shallow policy pages**: Harel's ASPX pages are marketing-oriented summaries. The precise coverage conditions, exclusions, and definitions live in the actual policy documents (PDFs).
4. **One dangerous hallucination**: GPT-5.2 on the 300L boiler question — it accessed the policy PDF partially but missed the critical 200L limit, asserting coverage may exist when it does not. This is the most harmful failure mode for an insurance chatbot.

---

## 6. Cosine Similarity: A Flawed Metric

Mean cosine similarity is **inversely correlated** with actual accuracy in our results:

| Model | Accuracy | Mean Cosine Sim |
|-------|----------|-----------------|
| GPT-4o | 25% | **0.442** |
| GPT-5.2 | 45% | 0.375 |

**Why:** GPT-4o's Hebrew "אני לא יודע" responses are short and topically similar to the short Hebrew reference answers, inflating cosine similarity. GPT-5.2's longer, more detailed (but sometimes hedging) answers diverge further in embedding space from concise references.

**Recommendation:** Use cosine similarity only as a secondary signal. Primary evaluation must use **LLM-as-judge correctness** or **manual classification**.

---

## 7. Baseline Summary & Target for RAG System

### What the RAG system must beat

| Metric | GPT-5.2 Baseline | Target |
|--------|------------------|--------|
| Overall accuracy | 45% (9/20) | >80% |
| Yes/No policy questions | 25% (3/12) | >75% |
| Factual lookups | 75% (6/8) | >90% |
| Hallucination rate | 5% (1/20) | 0% |
| Abstention rate | 50% (10/20) | <15% |
| Citation to specific doc + page | ~30% | >90% |

### Key requirements for the RAG pipeline
1. **Ingest policy PDFs** — the single highest-impact improvement. ~55% of failures stem from PDF-locked content.
2. **Chunk with clause-level granularity** — to retrieve specific exclusions and conditions.
3. **Cite document + page** — the reference set includes source file and page number.
4. **Reduce abstention without increasing hallucination** — the model should answer from retrieved context rather than defaulting to "I don't know".

---

## 8. Evaluation Harness

The reusable evaluation harness is in `eval_harness.py`. It supports:

- **Multiple metrics**: cosine similarity, LLM-as-judge correctness, abstention detection, hallucination flagging, citation presence
- **Any answering system**: pass a callable `answer_fn(question) -> str`
- **Structured output**: JSON results with per-question and aggregate scores
- **Domain breakdown**: automatic grouping by insurance domain
- **Question-type breakdown**: yes/no vs factual lookup analysis

Usage:
```python
from eval_harness import EvaluationHarness

harness = EvaluationHarness(test_set_path="ex2.json")
results = harness.run(answer_fn=my_rag_system, model_name="rag-v1")
harness.print_summary(results)
harness.save(results, "evaluation_results_rag-v1.json")
```
