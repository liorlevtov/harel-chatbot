"""
FAISS v3 RAG pipeline for Harel Insurance chatbot.

Key improvements over faiss2:
- DoclingLoader + HybridChunker for structure-aware document loading & chunking
- Real page numbers from PDF provenance stored in FAISS metadata
- Source URL from manifest stored per chunk
- answer_question_with_sources() returns page numbers + URLs for citations
"""

import json
import re
from pathlib import Path
from typing import Optional

from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent
DATA_PREPARED_DIR = PROJECT_ROOT / "data_prepared"
FAISS_INDEX_PATH = PROJECT_ROOT / "faiss3_index"

NEBIUS_API_KEY = (
    "v1.CmMKHHN0YXRpY2tleS1lMDBldHBiMzYyY3JuMngxcXYSIXNlcnZpY2VhY2NvdW50LWUwM"
    "GtieTJqN2p6ajljYXJuczILCKeFo8wGEL26q1s6DAiliLuXBxDA0NfVA0ACWgNlMDA.AAAAAAAA"
    "AAGNSitzi_mVnjLQCBIM0OeiIYDXqXQJwYLBqfTkFWqTVMAo_oZW5fhZCxCmfkh7rz9-U72xMI"
    "LMxWQ7a8fAxkYG"
)
NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
LLM_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "BAAI/bge-multilingual-gemma2"

TOP_K = 5
FETCH_K = 20
MIN_SCORE = 0.30

# ---------------------------------------------------------------------------
# RAG prompts (v3 - with page citation instructions)
# ---------------------------------------------------------------------------

RAG_HEB_TEMPLATE_V3 = """אתה עוזר וירטואלי של קבוצת ביטוח הראל.
ענה על שאלת המשתמש אך ורק באמצעות המידע שסופק בקטעי המקור.
כל קטע מסומן ב-[מקור: <שם קובץ>, עמוד <מספר>].
אם התשובה לא נמצאת בהקשר, אמור שאין לך מספיק מידע.
היה תמציתי, מדויק ומועיל. השב באותה שפה שבה נשאלה השאלה.

בסוף תשובתך הוסף בלוק מקורות בפורמט הבא (השתמש אך ורק במקורות שמהם לקחת מידע):
מקורות:
- קובץ: <שם הקובץ>, עמוד: <מספר העמוד>
"""

JUDGE_SYSTEM_PROMPT_V3 = """You are a strict quality-control judge for an insurance chatbot (Harel Insurance, Israel).
You will receive:
  - QUESTION: the user's question
  - CONTEXT: document excerpts retrieved from the knowledge base (each tagged with source file and page)
  - ANSWER: the chatbot's draft answer

Return a JSON object with exactly these fields:
{
  "verdict": "<approved|hallucinated|incomplete|no_context>",
  "critique": "<one sentence>",
  "refined_answer": "<improved answer or null>"
}

Verdict definitions:
- "approved": Answer is accurate, fully supported by context, complete, and cites sources.
- "hallucinated": Answer contains facts NOT present in context.
- "incomplete": Correct but missing important details present in context.
- "no_context": Context has no relevant info.

Rules:
- If verdict is "approved" or "no_context", set refined_answer to null.
- If verdict is "hallucinated", set refined_answer to null.
- If verdict is "incomplete", write a corrected refined_answer using ONLY the provided context.
- refined_answer must include source citations in the same format as the original.
- Return ONLY valid JSON. No markdown outside the JSON.
"""

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

_raw_client = OpenAI(base_url=NEBIUS_BASE_URL, api_key=NEBIUS_API_KEY)


def _make_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=NEBIUS_API_KEY,
        openai_api_base=NEBIUS_BASE_URL,
        check_embedding_ctx_length=False,
    )


embeddings = _make_embeddings()
vector_store: Optional[FAISS] = None

# ---------------------------------------------------------------------------
# Manifest & helpers
# ---------------------------------------------------------------------------


def _load_prepared_manifest() -> dict:
    """Load data_prepared/manifest.json."""
    manifest_path = DATA_PREPARED_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


OUTDATED_YEARS = [str(y) for y in range(2013, 2022)]


def _is_outdated(filename: str) -> bool:
    if filename.startswith("archive_"):
        return True
    return any(year in filename for year in OUTDATED_YEARS)


def _extract_page_number(metadata: dict) -> int:
    """Extract page number from DoclingLoader dl_meta."""
    dl_meta = metadata.get("dl_meta", {})
    doc_items = []
    if isinstance(dl_meta, dict):
        doc_items = dl_meta.get("doc_items", [])
    elif isinstance(dl_meta, list):
        doc_items = dl_meta
    for item in doc_items:
        if isinstance(item, dict):
            prov = item.get("prov", [])
            if prov and isinstance(prov, list):
                return prov[0].get("page_no", 0)
    return 0


# ---------------------------------------------------------------------------
# Index building (DoclingLoader + HybridChunker → FAISS)
# ---------------------------------------------------------------------------


def build_faiss3_index() -> FAISS:
    """
    Build the v3 FAISS index using DoclingLoader with HybridChunker.

    For each document in the manifest, loads and chunks it using docling's
    structure-aware HybridChunker, then embeds into FAISS.
    """
    from langchain_docling import DoclingLoader
    from docling.chunking import HybridChunker
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    pdf_pipeline_opts = PdfPipelineOptions(allow_external_plugins=True)
    converter = DocumentConverter(
        format_options={
            "pdf": PdfFormatOption(pipeline_options=pdf_pipeline_opts),
        }
    )
    chunker = HybridChunker(tokenizer=EMBEDDING_MODEL, max_tokens=512)

    manifest = _load_prepared_manifest()
    all_files = [f for f in manifest.get("files", []) if f.get("status") == "success"]
    print(f"Building v3 FAISS index from {len(all_files)} documents...")

    all_docs = []

    for idx, file_entry in enumerate(all_files, 1):
        source_path = file_entry.get("source_filepath", "")
        name = Path(source_path).name
        safe_name = name.encode("utf-8", errors="replace").decode("ascii", errors="replace")

        if _is_outdated(name):
            print(f"  [{idx}/{len(all_files)}] [SKIP] Outdated: {safe_name}")
            continue

        if not Path(source_path).exists():
            print(f"  [{idx}/{len(all_files)}] [SKIP] Not found: {safe_name}")
            continue

        try:
            loader = DoclingLoader(
                file_path=source_path,
                converter=converter,
                chunker=chunker,
            )
            docs = loader.load()
        except Exception as e:
            print(f"  [{idx}/{len(all_files)}] [FAIL] {safe_name}: {e}")
            continue

        # Enrich each chunk's metadata from manifest + extract page numbers
        for doc in docs:
            page_no = _extract_page_number(doc.metadata)
            dl_meta = doc.metadata.get("dl_meta", {})
            headings = dl_meta.get("headings", []) if isinstance(dl_meta, dict) else []
            breadcrumb = " > ".join(
                h.strip() for h in headings if isinstance(h, str) and h.strip()
            )

            # Replace with clean, flat metadata for FAISS storage
            doc.metadata = {
                "source": file_entry.get("output_filepath", source_path),
                "source_url": file_entry.get("source_url", ""),
                "domain": file_entry.get("domain", "general"),
                "page_number": page_no,
                "header_breadcrumb": breadcrumb,
                "chunk_index": 0,
            }

        all_docs.extend(docs)
        print(f"  [{idx}/{len(all_files)}] [OK] {safe_name} ({len(docs)} chunks)")

    print(f"Total chunks to embed: {len(all_docs)}")

    if not all_docs:
        raise ValueError("No chunks produced. Check data files.")

    # Assign global chunk indices
    for i, doc in enumerate(all_docs):
        doc.metadata["chunk_index"] = i

    # Build FAISS index in batches
    EMBED_BATCH = 50
    vs = FAISS.from_documents(all_docs[:EMBED_BATCH], embeddings)
    print(f"  Embedded {min(EMBED_BATCH, len(all_docs))}/{len(all_docs)} chunks")

    for batch_start in range(EMBED_BATCH, len(all_docs), EMBED_BATCH):
        batch_end = min(batch_start + EMBED_BATCH, len(all_docs))
        vs.add_documents(all_docs[batch_start:batch_end])
        print(f"  Embedded {batch_end}/{len(all_docs)} chunks")

    FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(FAISS_INDEX_PATH))
    print(f"v3 FAISS index saved to {FAISS_INDEX_PATH}")
    return vs


def create_or_load_faiss_index() -> FAISS:
    """Load existing v3 FAISS index or build from scratch."""
    global vector_store
    if FAISS_INDEX_PATH.exists() and any(FAISS_INDEX_PATH.iterdir()):
        print(f"Loading v3 FAISS index from {FAISS_INDEX_PATH}...")
        vector_store = FAISS.load_local(
            str(FAISS_INDEX_PATH),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        print("v3 index loaded.")
    else:
        vector_store = build_faiss3_index()
    return vector_store


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def _retrieve(question: str) -> list[tuple]:
    """Retrieve top-k chunks with relevance score filtering."""
    if vector_store is None:
        raise RuntimeError("FAISS index not loaded. Call create_or_load_faiss_index() first.")
    candidates = vector_store.similarity_search_with_relevance_scores(question, k=FETCH_K)
    filtered = [(doc, score) for doc, score in candidates if score >= MIN_SCORE]
    return filtered[:TOP_K]


def _build_context(results: list[tuple]) -> str:
    """Build context string with page numbers for the LLM."""
    blocks = []
    for i, (doc, score) in enumerate(results):
        source = doc.metadata.get("source", "unknown")
        page_no = doc.metadata.get("page_number", 0)
        breadcrumb = doc.metadata.get("header_breadcrumb", "")
        filename = Path(source).name

        page_label = f", עמוד {page_no}" if page_no > 0 else ""
        context_header = f"[מקור: {filename}{page_label}]"
        if breadcrumb:
            context_header += f" נושא: {breadcrumb}"

        text = doc.page_content
        if len(text) > 1500:
            text = text[:1500] + "…"

        blocks.append(f"{context_header}\n{text}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Generation and validation
# ---------------------------------------------------------------------------


def validate_answer(question: str, context: str, answer: str) -> dict:
    """Run LLM-as-judge on a generated answer."""
    user_msg = f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}"
    resp = _raw_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT_V3},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"verdict": "approved", "critique": "parse error", "refined_answer": None}


def answer_question_with_sources(question: str) -> dict:
    """
    Full v3 RAG pipeline: retrieve -> generate -> judge -> return with sources.

    Returns:
        {
            "answer": str,
            "sources": [
                {
                    "filepath": str,
                    "page_number": int,   # real PDF page; 0 for HTML
                    "domain": str,
                    "url": str,           # original harel-group.co.il URL
                }
            ]
        }
    """
    # 1. Retrieve
    results = _retrieve(question)
    if not results:
        return {
            "answer": "לא מצאתי מידע רלוונטי. נסה לנסח מחדש את השאלה.",
            "sources": [],
        }

    context = _build_context(results)

    # 2. Generate
    resp = _raw_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": RAG_HEB_TEMPLATE_V3},
            {"role": "user", "content": f"הקשר:\n{context}\n\nשאלה: {question}"},
        ],
        temperature=0.1,
    )
    draft = resp.choices[0].message.content.strip()

    # 3. Judge
    judgment = validate_answer(question, context, draft)
    verdict = judgment.get("verdict", "approved")
    print(f"  [judge] verdict={verdict} | {judgment.get('critique', '')}")

    if verdict == "approved":
        final_answer = draft
    elif verdict in ("hallucinated", "no_context"):
        return {"answer": "אין לי מספיק מידע כדי לענות על שאלה זו.", "sources": []}
    elif verdict == "incomplete":
        final_answer = judgment.get("refined_answer") or draft
    else:
        final_answer = draft

    # 4. Collect sources (deduplicate by url+page)
    seen = set()
    sources = []
    for doc, _score in results:
        filepath = doc.metadata.get("source", "")
        page_no = doc.metadata.get("page_number", 0)
        domain = doc.metadata.get("domain", "")
        source_url = doc.metadata.get("source_url", "")
        key = (source_url, page_no)
        if key not in seen and source_url:
            seen.add(key)
            sources.append({
                "filepath": filepath,
                "page_number": page_no,
                "domain": domain,
                "url": source_url,
            })

    return {"answer": final_answer, "sources": sources}


def answer_question(question: str) -> str:
    """Compatibility shim returning just the answer text."""
    return answer_question_with_sources(question)["answer"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    create_or_load_faiss_index()
    print("\nv3 FAISS index ready.\n")
    while True:
        q = input("שאלה: ").strip()
        if q.lower() in ("exit", "quit", "יציאה"):
            break
        if not q:
            continue
        result = answer_question_with_sources(q)
        print(f"\n{result['answer']}\n")
        for src in result["sources"]:
            page_str = f" עמוד {src['page_number']}" if src['page_number'] > 0 else ""
            print(f"  - {src['url']}{page_str}")


if __name__ == "__main__":
    main()
