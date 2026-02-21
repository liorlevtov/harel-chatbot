import os
import faiss
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.tools import tool
from langchain.agents import create_agent
from pathlib import Path


RAG_HEB_TEMPLATE = """אתה עוזר וירטואלי של קבוצת ביטוח הראל.
ענה על שאלת המשתמש אך ורק באמצעות המידע שסופק בקבצים.
אם התשובה לא נמצאת בהקשר, אמור שאין לך מספיק מידע.
היה תמציתי, מדויק ומועיל. השב באותה שפה שבה נשאלה השאלה
"""



PROJECT_ROOT = Path(__file__).parent
DATA_PREPARED_DIR = PROJECT_ROOT / "data_prepared" 
FAISS_INDEX_PATH = PROJECT_ROOT / "faiss_index"



NEBIUS_API_KEY = (
    "v1.CmMKHHN0YXRpY2tleS1lMDBldHBiMzYyY3JuMngxcXYSIXNlcnZpY2VhY2NvdW50LWUwM"
    "GtieTJqN2p6ajljYXJuczILCKeFo8wGEL26q1s6DAiliLuXBxDA0NfVA0ACWgNlMDA.AAAAAAAA"
    "AAGNSitzi_mVnjLQCBIM0OeiIYDXqXQJwYLBqfTkFWqTVMAo_oZW5fhZCxCmfkh7rz9-U72xMI"
    "LMxWQ7a8fAxkYG"
)
NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
LLM_MODEL = "Qwen/Qwen3-32B"

EMBEDDING_MODEL = "BAAI/bge-multilingual-gemma2"
# EMBEDDING_MODEL = "text-embedding-3-large"


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
        vector_store.load_local(
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


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


def run_agent(vector_store: FAISS):

    tools = [retrieve_context]
    agent = create_agent(model, tools, system_prompt=RAG_HEB_TEMPLATE)

    while True:
        query = input("Enter your question (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        
        events = agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
        )
        for event in events: 
            response = event["messages"][-1]
            response.pretty_print()


def main():
    vector_store = create_or_load_faiss_index()
    print("\nFAISS index is ready. You can now query it using the 'run_agent' function.\n")
    run_agent(vector_store)


if __name__ == "__main__":
    main()
