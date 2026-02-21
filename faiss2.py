import json
import os
import pickle
import re
import faiss
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.messages import SystemMessage, HumanMessage
from pathlib import Path
from table_store import TableStore


RAG_HEB_TEMPLATE = """אתה עוזר וירטואלי של קבוצת ביטוח הראל.
ענה על שאלת המשתמש אך ורק באמצעות המידע שסופק בקבצים.
אם התשובה לא נמצאת בהקשר, אמור שאין לך מספיק מידע.
היה תמציתי, מדויק ומועיל. השב באותה שפה שבה נשאלה השאלה
הוסף שורת מקור בפורמט (לדוגמא):
      "מקור":
        "קובץ": "apartment/files/www.harel-group.co.il--2025.pdf",
        "עמוד": 6
לא להשתמש בדוגמא שסיפקתי, זהו רק פורמט.
תוודא שהקובץ והעמוד תואמים למידע שסיפקת בתשובתך ועונה באופן תמציתי על השאלה, ושהוא קיים בעץ התיקיות.
"""

JUDGE_SYSTEM_PROMPT = """You are a strict quality-control judge for an insurance chatbot (Harel Insurance, Israel).
You will receive:
  - QUESTION: the user's question
  - CONTEXT: document excerpts retrieved from the knowledge base
  - ANSWER: the chatbot's draft answer

Your job is to evaluate the answer and return a JSON object with exactly these fields:
{
  "verdict": "<approved|hallucinated|incomplete|no_context>",
  "critique": "<one sentence explaining your decision>",
  "refined_answer": "<improved answer in the same language as the question, or null>"
}

Verdict definitions:
- "approved"     : Answer is accurate, fully supported by the context, and complete.
- "hallucinated" : Answer contains facts NOT present in the context (fabricated info).
- "incomplete"   : Answer is partially correct but misses important details that ARE in the context.
- "no_context"   : The context contains no relevant information to answer the question.

Rules:
- If verdict is "approved" or "no_context", set refined_answer to null.
- If verdict is "hallucinated", set refined_answer to null (we will refuse the answer entirely).
- If verdict is "incomplete", write a corrected refined_answer using ONLY the provided context.
- refined_answer must be in the same language as the question.
- refined_answer must include the source citation in the same format as the original answer.
- Return ONLY valid JSON. No markdown, no explanation outside the JSON.
"""

TOP_K = 5



PROJECT_ROOT = Path(__file__).parent
DATA_PREPARED_DIR = PROJECT_ROOT / "data_prepared"
FAISS_INDEX_PATH = PROJECT_ROOT / "faiss_index"
DOCS_CACHE_PATH  = FAISS_INDEX_PATH / "docs.pkl"



NEBIUS_API_KEY = (
    "v1.CmMKHHN0YXRpY2tleS1lMDBldHBiMzYyY3JuMngxcXYSIXNlcnZpY2VhY2NvdW50LWUwM"
    "GtieTJqN2p6ajljYXJuczILCKeFo8wGEL26q1s6DAiliLuXBxDA0NfVA0ACWgNlMDA.AAAAAAAA"
    "AAGNSitzi_mVnjLQCBIM0OeiIYDXqXQJwYLBqfTkFWqTVMAo_oZW5fhZCxCmfkh7rz9-U72xMI"
    "LMxWQ7a8fAxkYG"
)
NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
LLM_MODEL = "openai/gpt-oss-120b"
#"Qwen/Qwen3-Next-80B-A3B-Thinking"

# "Qwen/Qwen3-30B-A3B-Thinking-2507"
    # "openai/gpt-oss-120b"
# "Qwen/Qwen3-30B-A3B-Thinking-2507

 #"Qwen/Qwen3-32B"

EMBEDDING_MODEL = "BAAI/bge-multilingual-gemma2"
# EMBEDDING_MODEL = "text-embedding-3-large"


_client = OpenAI(base_url=NEBIUS_BASE_URL, api_key=NEBIUS_API_KEY)


def _make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=NEBIUS_API_KEY,
        openai_api_base=NEBIUS_BASE_URL,
        temperature=0,
    )
model = _make_llm()


def _make_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=NEBIUS_API_KEY,
        openai_api_base=NEBIUS_BASE_URL,
        check_embedding_ctx_length=False,  # skip tiktoken validation for custom models
    )
embeddings = _make_embeddings()


embedding_dim = len(embeddings.embed_query(RAG_HEB_TEMPLATE.format(context="", question="")))
index = faiss.IndexFlatL2(embedding_dim)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

_retriever = None   # set by create_or_load_faiss_index()
_table_store = None # set by create_or_load_faiss_index()


def _build_ensemble(docs):
    """Return a (bm25, faiss_retriever) pair for hybrid retrieval."""
    print(f"Building BM25 index from {len(docs)} chunks...")
    bm25 = BM25Retriever.from_documents(docs, k=TOP_K)
    faiss_ret = vector_store.as_retriever(search_kwargs={"k": TOP_K})
    return bm25, faiss_ret


def _hybrid_retrieve(retrievers, query: str) -> list:
    """
    Reciprocal Rank Fusion over two retrievers.
    weights = [0.4 BM25, 0.6 FAISS] — favour semantic slightly.
    """
    bm25_ret, faiss_ret = retrievers
    bm25_docs  = bm25_ret.invoke(query)
    faiss_docs = faiss_ret.invoke(query)

    RRF_K = 60
    scores: dict[str, float] = {}
    doc_map: dict[str, object] = {}

    for rank, doc in enumerate(bm25_docs):
        key = hash(doc.page_content)
        scores[key] = scores.get(key, 0) + 0.4 / (RRF_K + rank + 1)
        doc_map[key] = doc

    for rank, doc in enumerate(faiss_docs):
        key = hash(doc.page_content)
        scores[key] = scores.get(key, 0) + 0.6 / (RRF_K + rank + 1)
        doc_map[key] = doc

    ranked = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    return [doc_map[k] for k in ranked[:TOP_K]]


def create_or_load_faiss_index():
    global vector_store, _retriever, _table_store
    # Load structured table store if available
    try:
        _table_store = TableStore.load()
    except FileNotFoundError:
        print("Table store not found — run 'python table_store.py' to build it.")
        _table_store = None
    if FAISS_INDEX_PATH.exists():
        print(f"Loading existing FAISS index from {FAISS_INDEX_PATH}...")
        vector_store = FAISS.load_local(
            str(FAISS_INDEX_PATH),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        print("Index loaded successfully.")

        if DOCS_CACHE_PATH.exists():
            with open(DOCS_CACHE_PATH, "rb") as f:
                cached_docs = pickle.load(f)
            _retriever = _build_ensemble(cached_docs)
        else:
            print("No docs cache found — using FAISS-only retrieval. Rebuild the index to enable hybrid search.")
            faiss_ret = vector_store.as_retriever(search_kwargs={"k": TOP_K})
            _retriever = (None, faiss_ret)

    else:
        docs = []

        OUTDATED_YEARS = [str(y) for y in range(2013, 2022)]

        def is_outdated(filename: str) -> bool:
            if filename.startswith("archive_"):
                return True
            return any(year in filename for year in OUTDATED_YEARS)

        for root, _, files in os.walk(DATA_PREPARED_DIR):
            for file in files:
                if file.endswith(".md"):
                    if is_outdated(file):
                        print(f"Skipping outdated file: {file}".encode("utf-8", errors="replace").decode("ascii", errors="replace"))
                        continue
                    print(f"Loading document: {file}".encode("utf-8", errors="replace").decode("ascii", errors="replace"))
                    path = os.path.join(root, file)
                    docs.append(TextLoader(path, encoding="utf-8").load()[0])

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )
        all_splits = text_splitter.split_documents(docs)
        print(f"Split files into {len(all_splits)} sub-documents.")

        document_ids = vector_store.add_documents(
            documents=all_splits,
            ids=[f"{doc.metadata['source']}_{i}" for i, doc in enumerate(all_splits)],
        )
        print(document_ids[:3])

        FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(FAISS_INDEX_PATH))

        with open(DOCS_CACHE_PATH, "wb") as f:
            pickle.dump(all_splits, f)
        print(f"Index and docs cache saved to {FAISS_INDEX_PATH}")

        _retriever = _build_ensemble(all_splits)

    return vector_store


def _retrieve(query: str) -> list:
    """Use hybrid retriever if available, otherwise fall back to FAISS-only."""
    if _retriever is not None:
        bm25_ret, faiss_ret = _retriever
        if bm25_ret is not None:
            return _hybrid_retrieve(_retriever, query)
        return faiss_ret.invoke(query)
    return vector_store.similarity_search(query, k=TOP_K)


REWRITE_SYSTEM_PROMPT = """You are a search query optimizer for an Israeli insurance knowledge base.
Given a user question in Hebrew (or English), generate 3 alternative search queries that:
1. Use different wording / synonyms to maximise recall
2. Mirror how insurance policy documents phrase the same concept
3. Are concise (one line each)

Return ONLY a JSON array of 3 strings. No explanation, no markdown.
Example output: ["query 1", "query 2", "query 3"]
"""


def rewrite_query(question: str) -> list[str]:
    """Generate 3 alternative phrasings of the question for better retrieval recall."""
    try:
        resp = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.3,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        alternatives = json.loads(raw)
        if isinstance(alternatives, list):
            return [question] + [q for q in alternatives if isinstance(q, str)]
    except Exception:
        pass
    return [question]  # fallback: original only


# ---------------------------------------------------------------------------
# Source quality re-ranking
# ---------------------------------------------------------------------------

# Files that are unlikely to contain the actual policy clauses we need
_BAD_SOURCE_PREFIXES = ("index_", "faq_", "default_")
_BAD_SOURCE_SUBSTRINGS = ("tofes", "טופס", "הצעה")

# Files that ARE policy conditions (inner booklets, terms, etc.)
_GOOD_SOURCE_SUBSTRINGS = ("pnimhoveret", "תנאי", "כתב-שירות", "גילוי-נאות", "polisa", "פוליסה")


def _source_priority(doc) -> int:
    """Return a priority score: higher = show first in context."""
    fname = os.path.basename(doc.metadata.get("source", "")).lower()
    if any(fname.startswith(p) for p in _BAD_SOURCE_PREFIXES):
        return -1
    if any(s in fname for s in _BAD_SOURCE_SUBSTRINGS):
        return -1
    if any(s in fname for s in _GOOD_SOURCE_SUBSTRINGS):
        return 2
    return 1


def _rerank_by_source(docs: list) -> list:
    """Re-rank retrieved docs: policy condition docs first, index/form pages last."""
    return sorted(docs, key=_source_priority, reverse=True)


def _retrieve_multi_query(queries: list[str]) -> list:
    """Retrieve docs for each query variant using hybrid search, merge, deduplicate, and re-rank."""
    seen = set()
    merged = []
    for q in queries:
        for doc in _retrieve(q):
            key = hash(doc.page_content)
            if key not in seen:
                seen.add(key)
                merged.append(doc)
    return _rerank_by_source(merged)


def validate_answer(question: str, context: str, answer: str) -> dict:
    """Run LLM-as-judge on a generated answer. Returns verdict dict."""
    user_msg = f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}"
    resp = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
    )
    raw = resp.choices[0].message.content.strip()
    # strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # fallback: treat as approved so we don't silently drop answers
        return {"verdict": "approved", "critique": "parse error", "refined_answer": None}


def answer_question(question: str) -> str:
    """Full pipeline: table lookup → rewrite → retrieve (hybrid) → generate → judge → refine/reject."""
    # 0. Structured table lookup (fast exact answer for prices / numeric thresholds)
    if _table_store is not None:
        table_answer = _table_store.lookup(_client, question, LLM_MODEL)
        if table_answer:
            print("  [table] answered from structured store")
            return table_answer

    # 1. Query rewriting — generate alternative phrasings
    queries = rewrite_query(question)
    print(f"  [rewrite] {len(queries)} queries: {queries[1:]}")

    # 2. Retrieve with all query variants, merge & deduplicate
    docs = _retrieve_multi_query(queries)
    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source', '')}]\n{doc.page_content}"
        for doc in docs
    )

    # 2. Generate initial answer
    resp = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": RAG_HEB_TEMPLATE},
            {"role": "user", "content": f"הקשר:\n{context}\n\nשאלה: {question}"},
        ],
        temperature=0.1,
    )
    draft = resp.choices[0].message.content.strip()

    # 3. Judge
    judgment = validate_answer(question, context, draft)
    verdict = judgment.get("verdict", "approved")

    print(f"  [judge] verdict={verdict} | {judgment.get('critique', '')}")

    if verdict in ("hallucinated", "no_context"):
        return "אין לי מספיק מידע כדי לענות על שאלה זו."
    elif verdict == "incomplete":
        refined = judgment.get("refined_answer")
        return refined if refined else draft
    else:
        return draft


def run_agent():
    print("Harel Insurance Chatbot (type 'exit' to quit)\n")
    while True:
        query = input("שאלה: ").strip()
        if query.lower() in ("exit", "quit", "יציאה"):
            break
        if not query:
            continue
        answer = answer_question(query)
        print(f"\n{answer}\n")


def vector_store_retrieve(
    vector_store: FAISS,
    query: str,
    k: int = 5,
    fetch_k: int = 20,
    score_threshold: float = 0.4,
) -> list:
    """
    Search FAISS index with relevance filtering.

    Fetches `fetch_k` candidates, discards any below `score_threshold`,
    then returns the top `k` survivors (higher score = more relevant).
    """
    candidates = vector_store.similarity_search_with_relevance_scores(query, k=fetch_k)
    filtered = [(doc, score) for doc, score in candidates if score >= score_threshold]
    return filtered[:k]


def _build_context_block(results: list, max_chars_per_chunk: int = 1500) -> str:
    """Turn FAISS search results into a context string for the model."""
    blocks = []
    for i, (doc, score) in enumerate(results):
        filename = doc.metadata.get("source", "unknown")
        text = doc.page_content
        if max_chars_per_chunk and len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk] + "…"
        blocks.append(f"[{i}] {filename} (score={score:.4f})\n{text}")
    return "\n\n".join(blocks).strip()


def ask_rag_with_manual_retrieval(
        vector_store: FAISS,
        min_top_score: float = 0.3,
        k: int = 5,
    ) -> dict:

    while True:
        question = input("\nEnter your question (or 'exit' to quit): ").strip()
        if question.lower() in ["exit", "quit"]:
            break

        # 1) Retrieval
        results = vector_store_retrieve(vector_store, question, k=k)
        top_score = results[0][1] if results else None

        # 2) Gate (relevance scores are in [0, 1]; higher = more similar)
        if not results or top_score < min_top_score:
            return {
                "answer": "לא מצאתי מידע רלוונטי מספיק. נסה לנסח מחדש את השאלה.",
                "top_score": top_score,
                "used_results": [],
            }

        # 3) Generate from retrieved context
        context = _build_context_block(results)
        messages = [
            SystemMessage(content=RAG_HEB_TEMPLATE),
            HumanMessage(content=f"הקשר:\n{context}\n\nשאלה:\n{question}"),
        ]
        response = model.invoke(messages)

        res = {
            "answer": response.content,
            "top_score": top_score,
            "used_results": results,
        }
        print(res['answer'])



def main():
    create_or_load_faiss_index()
    print("\nFAISS index is ready.\n")
    result = ask_rag_with_manual_retrieval(vector_store)
    print(f"Score: {result['top_score']:.4f}")
    print(f"\nAnswer:\n{result['answer']}")
    # run_agent()


if __name__ == "__main__":
    main()
