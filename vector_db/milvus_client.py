"""Milvus client for collection management and data operations."""

from typing import List, Dict, Any
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
)

from .config import (
    MILVUS_HOST,
    MILVUS_PORT,
    COLLECTION_NAME,
    DENSE_DIM,
)


def connect():
    """Connect to Milvus server."""
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT,
    )


def disconnect():
    """Disconnect from Milvus server."""
    connections.disconnect(alias="default")


def collection_exists() -> bool:
    """Check if collection exists."""
    return utility.has_collection(COLLECTION_NAME)


def create_collection() -> Collection:
    """Create the collection with schema and indexes."""
    # Define schema
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
        FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=DENSE_DIM),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="source_url", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="domain", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
        FieldSchema(name="total_chunks", dtype=DataType.INT64),
    ]

    schema = CollectionSchema(
        fields=fields,
        description="Harel Insurance document chunks",
        enable_dynamic_field=False,
    )

    # Create collection
    collection = Collection(
        name=COLLECTION_NAME,
        schema=schema,
    )

    # Create indexes
    # Dense vector index (HNSW for fast ANN search)
    collection.create_index(
        field_name="dense_vector",
        index_params={
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200},
        },
    )

    # Sparse vector index
    collection.create_index(
        field_name="sparse_vector",
        index_params={
            "metric_type": "IP",
            "index_type": "SPARSE_INVERTED_INDEX",
            "params": {"drop_ratio_build": 0.2},
        },
    )

    # Scalar index for domain filtering
    collection.create_index(
        field_name="domain",
        index_params={"index_type": "INVERTED"},
    )

    print(f"Collection '{COLLECTION_NAME}' created with indexes.")
    return collection


def get_collection() -> Collection:
    """Get the collection, creating it if it doesn't exist."""
    if not collection_exists():
        return create_collection()
    return Collection(COLLECTION_NAME)


def drop_collection():
    """Drop the collection if it exists."""
    if collection_exists():
        utility.drop_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' dropped.")


def insert_batch(records: List[Dict[str, Any]]) -> int:
    """
    Insert a batch of records into the collection.

    Args:
        records: List of dicts with fields matching the schema

    Returns:
        Number of records inserted
    """
    if not records:
        return 0

    collection = get_collection()

    # Prepare data for insertion
    data = [
        [r["id"] for r in records],
        [r["chunk_text"] for r in records],
        [r["dense_vector"] for r in records],
        [r["sparse_vector"] for r in records],
        [r["source_url"] for r in records],
        [r["domain"] for r in records],
        [r["filename"] for r in records],
        [r["chunk_index"] for r in records],
        [r["total_chunks"] for r in records],
    ]

    collection.insert(data)
    return len(records)


def load_collection():
    """Load collection into memory for searching."""
    collection = get_collection()
    collection.load()
    print(f"Collection '{COLLECTION_NAME}' loaded into memory.")


def get_collection_stats() -> Dict[str, Any]:
    """Get collection statistics."""
    collection = get_collection()
    return {
        "name": COLLECTION_NAME,
        "num_entities": collection.num_entities,
    }
