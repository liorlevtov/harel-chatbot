"""
table_store.py — Structured lookup layer for pricing tables and numeric thresholds.

Parses all markdown tables from data_prepared into a flat JSON store.
At query time, uses an LLM to decide whether the question is a "table question"
and if so looks up the answer directly before falling back to RAG.

Build the store:
    python table_store.py          # scans data_prepared, writes table_store.json

Use from code:
    from table_store import TableStore
    ts = TableStore.load()
    result = ts.lookup(client, question)   # returns str or None
"""

import json
import os
import re
from pathlib import Path

PROJECT_ROOT       = Path(__file__).parent
DATA_PREPARED_DIR  = PROJECT_ROOT / "data_prepared"
TABLE_STORE_PATH   = PROJECT_ROOT / "table_store.json"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_markdown_tables(text: str, source: str) -> list[dict]:
    """
    Extract all markdown pipe-tables from text.
    Returns list of {"source": ..., "headers": [...], "rows": [[...], ...]}.
    """
    tables = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # A table row starts and ends with |
        if line.startswith("|") and line.endswith("|"):
            raw_table = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                raw_table.append(lines[i].strip())
                i += 1
            if len(raw_table) < 2:
                continue
            # Parse header row
            headers = [c.strip() for c in raw_table[0].strip("|").split("|")]
            # Skip separator row (---|---)
            data_rows = []
            for row_line in raw_table[1:]:
                if re.match(r"^\|[-| :]+\|$", row_line):
                    continue
                cells = [c.strip() for c in row_line.strip("|").split("|")]
                if len(cells) == len(headers):
                    data_rows.append(cells)
            if data_rows:
                tables.append({"source": source, "headers": headers, "rows": data_rows})
        else:
            i += 1
    return tables


def build_table_store() -> list[dict]:
    """Walk data_prepared, parse every markdown file, collect all tables."""
    OUTDATED_YEARS = [str(y) for y in range(2013, 2022)]

    def is_outdated(filename: str) -> bool:
        if filename.startswith("archive_"):
            return True
        return any(year in filename for year in OUTDATED_YEARS)

    all_tables = []
    for root, _, files in os.walk(DATA_PREPARED_DIR):
        for file in files:
            if not file.endswith(".md") or is_outdated(file):
                continue
            path = os.path.join(root, file)
            try:
                with open(path, encoding="utf-8") as f:
                    text = f.read()
            except Exception:
                continue
            tables = _parse_markdown_tables(text, source=path)
            all_tables.extend(tables)

    print(f"Extracted {len(all_tables)} tables from data_prepared.")
    return all_tables


def save_table_store(tables: list[dict], path: Path = TABLE_STORE_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tables, f, ensure_ascii=False, indent=2)
    print(f"Table store saved to {path} ({len(tables)} tables)")


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

LOOKUP_SYSTEM_PROMPT = """You are a precise lookup engine for an Israeli insurance chatbot.
You will receive a user question and a set of tables extracted from policy documents.

Your job:
1. Decide if any table contains an EXACT, DIRECT answer to the question.
2. If YES: return a JSON object:
   {"found": true, "answer": "<concise answer in the same language as the question>", "source": "<file path>"}
3. If NO (or uncertain): return:
   {"found": false, "answer": null, "source": null}

Strict rules:
- Only answer if the specific item asked about is EXPLICITLY named or described in a table cell.
- Do not infer or extrapolate from general categories (e.g. "water damage" does not imply coverage for rain seepage through roof unless that is explicitly stated).
- If the question asks about a specific plan or product (e.g. "רפואה אישית אונליין פלוס"), only answer from tables clearly belonging to that exact plan.
- If multiple tables give different values for what appears to be the same question (conflicting data), return found:false.
- Do not infer, extrapolate, or combine data from multiple rows/tables.
- Return ONLY valid JSON. No markdown, no explanation outside the JSON.
"""


# ---------------------------------------------------------------------------
# Domain detection
# ---------------------------------------------------------------------------

_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "health":    ["בריאות", "רפואה", "ניתוח", "רופא", "מחלה", "ביטוח בריאות", "כירורגי", "תרופות", "השתתפות עצמית", "קופת חולים", "אשפוז"],
    "travel":    ["נסיעות", "חו\"ל", "חול", "טיול", "תיירות", "דרכון", "abroad", "נסיעה לחו", "ביטוח נסיעות"],
    "car":       ["רכב", "מכונית", "צמ\"ה", "ציוד מכני", "מנוע", "כלי רכב", "ביטוח רכב", "תאונת דרכים", "רכוב"],
    "apartment": ["דירה", "מבנה", "תכולה", "דיור", "רכוש", "ביטוח דירה", "שריפה", "פריצה", "נזקי מים"],
    "business":  ["עסקים", "מושב", "חבר מושב", "עסק", "מקצועי", "רישיון", "ביטוח עסקים", "אחריות מקצועית"],
}


def _detect_domain(question: str) -> str | None:
    """Return the data_prepared subdirectory name that best matches the question, or None."""
    q_lower = question.lower()
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            return domain
    return None


def _score_table(table: dict, question: str) -> int:
    """Count keyword overlaps between question tokens and table content."""
    question_tokens = set(re.sub(r"[^\w\u0590-\u05FF]", " ", question).lower().split())
    content = " ".join(table["headers"]) + " " + " ".join(
        " ".join(row) for row in table["rows"]
    )
    content_tokens = set(re.sub(r"[^\w\u0590-\u05FF]", " ", content).lower().split())
    return len(question_tokens & content_tokens)


def _select_relevant_tables(tables: list[dict], question: str, top_n: int = 10) -> list[dict]:
    """Return the top_n tables most likely to contain the answer.

    1. If a domain can be detected, first restrict to tables from that domain directory.
    2. Requires at least 2 overlapping tokens to avoid loosely-related noise.
    """
    domain = _detect_domain(question)
    if domain:
        domain_tables = [t for t in tables if f"/{domain}/" in t["source"].replace("\\", "/")]
        # Fall back to all tables if domain filter leaves nothing
        candidate_tables = domain_tables if domain_tables else tables
    else:
        candidate_tables = tables

    scored = [(t, _score_table(t, question)) for t in candidate_tables]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [t for t, score in scored[:top_n] if score >= 2]


def _tables_to_text(tables: list[dict]) -> str:
    """Serialise a list of table dicts into a readable text block for the LLM."""
    blocks = []
    for t in tables:
        header_line = " | ".join(t["headers"])
        row_lines = [" | ".join(row) for row in t["rows"]]
        block = f"Source: {t['source']}\n{header_line}\n" + "\n".join(row_lines)
        blocks.append(block)
    return "\n\n".join(blocks)


class TableStore:
    def __init__(self, tables: list[dict]):
        self.tables = tables

    @classmethod
    def load(cls, path: Path = TABLE_STORE_PATH) -> "TableStore":
        if not path.exists():
            raise FileNotFoundError(
                f"Table store not found at {path}. Run: python table_store.py"
            )
        with open(path, encoding="utf-8") as f:
            tables = json.load(f)
        print(f"Loaded {len(tables)} tables from {path}")
        return cls(tables)

    def lookup(self, client, question: str, llm_model: str) -> str | None:
        """
        Try to answer `question` from the structured table store.
        Returns the answer string if found, or None to fall back to RAG.
        Only fires when a specific insurance domain can be detected from the question.
        """
        if _detect_domain(question) is None:
            return None  # can't narrow domain → skip table store to avoid false positives
        relevant = _select_relevant_tables(self.tables, question, top_n=10)
        if not relevant:
            return None
        tables_text = _tables_to_text(relevant)
        try:
            resp = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": LOOKUP_SYSTEM_PROMPT},
                    {"role": "user", "content": f"QUESTION:\n{question}\n\nTABLES:\n{tables_text}"},
                ],
                temperature=0,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            result = json.loads(raw)
            if result.get("found"):
                return result.get("answer")
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# CLI: build the store
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tables = build_table_store()
    save_table_store(tables)
