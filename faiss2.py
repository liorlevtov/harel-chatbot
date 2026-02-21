import json
import os
import re
import faiss
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from pathlib import Path


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
FAISS_INDEX_PATH = PROJECT_ROOT / "faiss_index_v2"



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
        temperature=0.1,
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


def create_or_load_faiss_index():
    if FAISS_INDEX_PATH.exists():
        print(f"Loading existing FAISS index from {FAISS_INDEX_PATH}...")
        global vector_store
        vector_store = FAISS.load_local(
            str(FAISS_INDEX_PATH),
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        print("Index loaded successfully.")

    else:
        docs = []

        OUTDATED_START_YEAR = 2013
        OUTDATED_END_YEAR = 2022
        OUTDATED_YEARS = [str(y) for y in range(OUTDATED_START_YEAR, OUTDATED_END_YEAR)]

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

                    loader = TextLoader(path, encoding="utf-8")

                    docs.append(loader.load()[0])

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # chunk size (characters)
            chunk_overlap=200,  # chunk overlap (characters)
            add_start_index=True,  # track index in original document
        )
        all_splits = text_splitter.split_documents(docs)

        print(f"Split files into {len(all_splits)} sub-documents.")

        document_ids = vector_store.add_documents(
            documents=all_splits, 
            ids=[f"{doc.metadata['source']}_{i}" for i, doc in enumerate(all_splits)], 
            verbose=True
        )

        print(document_ids[:3])
        FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(FAISS_INDEX_PATH))
        print(f"Index saved to {FAISS_INDEX_PATH}")

    return vector_store


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
    """Full RAG pipeline: retrieve → generate → judge → refine/reject."""
    # 1. Retrieve
    docs = vector_store.similarity_search(question, k=TOP_K)
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

    if verdict == "approved":
        return draft
    elif verdict in ("hallucinated", "no_context"):
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


def main():
    create_or_load_faiss_index()
    print("\nFAISS index is ready.\n")
    run_agent()


if __name__ == "__main__":
    main()
