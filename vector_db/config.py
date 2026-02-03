"""Vector database configuration settings."""

from pathlib import Path

# Project root (assumes vector_db is a package in the root)
PROJECT_ROOT = Path(__file__).parent.parent

# Milvus connection
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
COLLECTION_NAME = "harel_docs"

# Data paths (absolute paths based on project root)
DATA_PREPARED_DIR = PROJECT_ROOT / "data_prepared"
MANIFEST_PATH = DATA_PREPARED_DIR / "manifest.json"

# Chunking settings
MIN_CHUNK_SIZE = 500  # tokens
MAX_CHUNK_SIZE = 800  # tokens
CHUNK_OVERLAP = 100   # tokens

# Embedding settings
EMBEDDING_MODEL = "BAAI/bge-m3"
DENSE_DIM = 1024

# Search settings
DEFAULT_TOP_K = 10
DENSE_WEIGHT = 0.7
SPARSE_WEIGHT = 0.3

# Ingestion settings
BATCH_SIZE = 50
MAX_WORKERS = 4

# Insurance domains
DOMAINS = [
    "car",
    "life",
    "travel",
    "health",
    "dental",
    "mortgage",
    "business",
    "apartment",
    "general",
]
