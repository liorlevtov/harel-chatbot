"""
FAISS-based RAG for Harel Insurance Chatbot.

Features:
- Ingests markdown documents from data_prepared/ across all insurance domains
- Builds a single combined FAISS index (persisted to disk)
- Uses Nebius API (OpenAI-compatible) for embeddings and LLM generation
- Provides a simple query(question) interface via LangChain
"""

from pathlib import Path


from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NEBIUS_API_KEY = (
    "v1.CmMKHHN0YXRpY2tleS1lMDBldHBiMzYyY3JuMngxcXYSIXNlcnZpY2VhY2NvdW50LWUwM"
    "GtieTJqN2p6ajljYXJuczILCKeFo8wGEL26q1s6DAiliLuXBxDA0NfVA0ACWgNlMDA.AAAAAAAA"
    "AAGNSitzi_mVnjLQCBIM0OeiIYDXqXQJwYLBqfTkFWqTVMAo_oZW5fhZCxCmfkh7rz9-U72xMI"
    "LMxWQ7a8fAxkYG"
)
NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"

EMBEDDING_MODEL = "BAAI/bge-multilingual-gemma2"
LLM_MODEL = "google/gemma-3-27b-it-fast"

PROJECT_ROOT = Path(__file__).parent
DATA_PREPARED_DIR = PROJECT_ROOT / "data_prepared"
FAISS_INDEX_PATH = PROJECT_ROOT / "faiss_index"

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Retrieval
DEFAULT_TOP_K = 6

DOMAINS = [
    "apartment",
    "business",
    "car",
    "dental",
    "health",
    "life",
    "mortgage",
    "travel",
    "general",
]

RAG_PROMPT_TEMPLATE = """You are a helpful insurance assistant for Harel Insurance Group.
Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say you don't have enough information.
Be concise, accurate, and helpful. You may respond in the same language as the question.

Context:
{context}

Question: {question}

Answer:"""


RAG_HEB_TEMPLATE = """אתה עוזר וירטואלי של קבוצת ביטוח הראל.
ענה על שאלת המשתמש אך ורק באמצעות המידע שסופק להלן.
אם התשובה לא נמצאת בהקשר, אמור שאין לך מספיק מידע.
היה תמציתי, מדויק ומועיל. השב באותה שפה שבה נשאלה השאלה.
הקשר:
{context}
שאלה: {question}
תשובה:"""


# ---------------------------------------------------------------------------
# LangChain clients (Nebius via OpenAI-compatible API)
# ---------------------------------------------------------------------------

def _make_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=NEBIUS_API_KEY,
        openai_api_base=NEBIUS_BASE_URL,
        check_embedding_ctx_length=False,  # skip tiktoken validation for custom models
    )


def _make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=NEBIUS_API_KEY,
        openai_api_base=NEBIUS_BASE_URL,
        temperature=0.1,
    )


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def load_domain_documents(domain: str) -> list[Document]:
    """Load all markdown files for a given insurance domain."""
    domain_dir = DATA_PREPARED_DIR / domain / "files"
    if not domain_dir.exists():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    docs: list[Document] = []
    for md_file in domain_dir.glob("*.md"):
        try:
            text = md_file.read_text(encoding="utf-8")
            if not text.strip():
                continue
            chunks = splitter.create_documents(
                [text],
                metadatas=[{"source": str(md_file), "domain": domain, "filename": md_file.name}],
            )
            docs.extend(chunks)
        except Exception as e:
            print(f"  Warning: could not read {md_file.name}: {e}")

    return docs


# ---------------------------------------------------------------------------
# FAISS index management
# ---------------------------------------------------------------------------

def ingest(force: bool = False) -> None:
    """
    Build and persist a single FAISS index from all domains.

    Args:
        force: If True, re-build even if an index already exists on disk.
    """
    if FAISS_INDEX_PATH.exists() and not force:
        print(f"Index already exists at {FAISS_INDEX_PATH} – skipping (use force=True to rebuild).")
        return

    print("Starting ingestion process...")
    embeddings = _make_embeddings()
    all_docs: list[Document] = []

    for domain in DOMAINS:
        print(f"[{domain}] Loading documents...")
        docs = load_domain_documents(domain)
        if not docs:
            print(f"[{domain}] No documents found – skipping.")
            continue
        all_docs.extend(docs)
        print(f"[{domain}] {len(docs)} chunks loaded.")

    if not all_docs:
        print("No documents found across any domain.")
        return

    print(f"\nEmbedding {len(all_docs)} chunks total...")
    vectorstore = FAISS.from_documents(all_docs, embeddings)
    FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_INDEX_PATH))
    print(f"Index saved to {FAISS_INDEX_PATH}")


def load_vectorstore() -> FAISS:
    """Load the persisted FAISS index."""
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"No FAISS index found at {FAISS_INDEX_PATH}. "
            "Run faiss_rag.ingest() first."
        )
    embeddings = _make_embeddings()
    return FAISS.load_local(str(FAISS_INDEX_PATH), embeddings, allow_dangerous_deserialization=True)


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def query(
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """
    Retrieve relevant context and generate an answer.

    Args:
        question: The user's question.
        top_k:    Number of chunks to retrieve.

    Returns:
        Dict with keys:
            'answer'   – generated answer string
            'sources'  – list of source file names used
    """
    vectorstore = load_vectorstore()

    prompt = PromptTemplate(
        template=RAG_HEB_TEMPLATE,
        input_variables=["context", "question"],
    )

    chain = RetrievalQA.from_chain_type(
        llm=_make_llm(),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": top_k}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    while True:
        question = input("\nEnter your question (or 'exit' to quit): ").strip()
        if question.lower() in ["exit", "quit"]:
            break

        result = chain.invoke({"query": question})
        answer = result.get("result", "").strip()
        source_docs = result.get("source_documents", [])
        sources = list({doc.metadata.get("filename", "") for doc in source_docs})
        print(f"\nAnswer:\n{answer}")
        print(f"Sources: {', '.join(sources) or 'none'}")
    return 


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    ingest()
    print("\nFAISS index is ready. You can now query it using the 'query' function.\n")
    query()
    exit(0)


    import argparse

    parser = argparse.ArgumentParser(description="Harel FAISS RAG")
    sub = parser.add_subparsers(dest="cmd")

    # ingest sub-command
    ingest_p = sub.add_parser("ingest", help="Build FAISS index from data_prepared/")
    ingest_p.add_argument("--force", action="store_true", help="Rebuild existing index")

    # query sub-command
    query_p = sub.add_parser("query", help="Query the RAG")
    query_p.add_argument("question", help="Question to ask")

    args = parser.parse_args()

    if args.cmd == "ingest":
        ingest(force=args.force)
    elif args.cmd == "query":
        out = query(args.question)
        print(f"Sources: {', '.join(out['sources']) or 'none'}")
        print(f"\nAnswer :\n{out['answer']}")
    else:
        parser.print_help()
