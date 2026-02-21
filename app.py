import asyncio
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, List, Literal, Optional

import faiss2 as rag

BASE_DIR = Path(__file__).parent

_executor = ThreadPoolExecutor(max_workers=1)

# Mapping from local file path (relative, e.g. "data_prepared/car/files/xxx.md")
# to the original source URL on harel-group.co.il
_filepath_to_url: Dict[str, str] = {}


def _load_manifest():
    """Load data_prepared/manifest.json to build filepath → URL lookup."""
    manifest_path = BASE_DIR / "data_prepared" / "manifest.json"
    if not manifest_path.exists():
        print(f"Warning: manifest not found at {manifest_path}, sources will be empty.")
        return
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    for entry in manifest.get("files", []):
        output_path = entry.get("output_filepath", "")
        source_url = entry.get("source_url", "")
        if output_path and source_url:
            _filepath_to_url[output_path] = source_url


def _resolve_sources(raw_sources: list) -> List["Source"]:
    """Convert raw source dicts from faiss2 into Source objects with URLs."""
    seen = set()
    sources = []
    for src in raw_sources:
        filepath = src.get("filepath", "")
        # Normalize: FAISS metadata stores absolute paths, manifest uses relative
        # Strip the project root prefix to get relative path
        rel_path = filepath
        project_root = str(BASE_DIR)
        if rel_path.startswith(project_root):
            rel_path = rel_path[len(project_root):].lstrip("/")

        url = _filepath_to_url.get(rel_path, "")
        if not url:
            # Try matching by filename (last component) as fallback
            filename = Path(rel_path).stem  # e.g. "xxx_hash"
            for manifest_path, manifest_url in _filepath_to_url.items():
                if Path(manifest_path).stem == filename:
                    url = manifest_url
                    break
        if not url or url in seen:
            continue
        seen.add(url)
        sources.append(Source(link=url, page=src.get("start_index") or 0))
    return sources


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load manifest and FAISS index at startup."""
    _load_manifest()
    print(f"Loaded {len(_filepath_to_url)} source URL mappings from manifest.")
    print("Loading FAISS index from disk...")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_executor, rag.create_or_load_faiss_index)
    print("FAISS index ready.")
    yield
    _executor.shutdown(wait=True)


app = FastAPI(title="Harel Insurance Chatbot", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# ---------------------------------------------------------------------------
# Pydantic models (compatible with ex2_evaluation_script_v2/completions_api.py)
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "harel-rag"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False


class Source(BaseModel):
    link: str
    page: int


class Choice(BaseModel):
    index: int
    text: str
    sources: List[Source] = []
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"]
    created: float
    model: str
    choices: List[Choice]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(str(BASE_DIR / "static" / "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok"}


def _extract_question(request: ChatCompletionRequest) -> str:
    for msg in reversed(request.messages):
        if msg.role == "user":
            return msg.content.strip()
    raise ValueError("No user message found in request")


async def _process(request: ChatCompletionRequest) -> ChatCompletionResponse:
    try:
        question = _extract_question(request)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        _executor, rag.answer_question_with_sources, question
    )

    sources = _resolve_sources(result.get("sources", []))

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        object="chat.completion",
        created=time.time(),
        model=request.model,
        choices=[
            Choice(
                index=0,
                text=result["answer"],
                sources=sources,
                finish_reason="stop",
            )
        ],
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def v1_completions(request: ChatCompletionRequest):
    return await _process(request)


@app.post("/chat/completions", response_model=ChatCompletionResponse)
async def completions(request: ChatCompletionRequest):
    return await _process(request)
