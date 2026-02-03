"""Ingestion script to load data from data_prepared into Milvus."""

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Iterator, Tuple

from tqdm import tqdm

from .config import (
    DATA_PREPARED_DIR,
    MANIFEST_PATH,
    BATCH_SIZE,
)
from .chunker import chunk_markdown
from .embedder import embed_texts
from .milvus_client import (
    connect,
    disconnect,
    get_collection,
    insert_batch,
    drop_collection,
    load_collection,
    get_collection_stats,
)


def load_manifest() -> Dict[str, Any]:
    """Load the manifest from data_prepared."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found at {MANIFEST_PATH}. "
            "Please run data preparation first: make run-prep-data"
        )

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_chunk_id(filepath: str, chunk_index: int) -> str:
    """Generate a unique ID for a chunk."""
    content = f"{filepath}:{chunk_index}"
    return hashlib.md5(content.encode()).hexdigest()[:16]


def iter_documents(manifest: Dict[str, Any]) -> Iterator[Tuple[Dict[str, Any], str]]:
    """
    Iterate over documents, yielding (metadata, content) pairs.

    Args:
        manifest: The manifest dict from data_prepared

    Yields:
        Tuple of (file_entry, markdown_content)
    """
    files = manifest.get("files", [])

    for file_entry in files:
        if file_entry.get("status") != "success":
            continue

        output_filepath = file_entry.get("output_filepath")
        if not output_filepath:
            continue

        filepath = Path(output_filepath)
        if not filepath.exists():
            print(f"Warning: File not found: {filepath}")
            continue

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            yield file_entry, content
        except Exception as e:
            print(f"Warning: Error reading {filepath}: {e}")
            continue


def process_document(
    file_entry: Dict[str, Any],
    chunks: List[str],
    dense_vectors,
    sparse_vectors,
) -> List[Dict[str, Any]]:
    """
    Process a document into records ready for Milvus insertion.

    Args:
        file_entry: Metadata from manifest
        chunks: Pre-chunked text pieces
        dense_vectors: Pre-computed dense embeddings
        sparse_vectors: Pre-computed sparse embeddings

    Returns:
        List of record dicts
    """
    if not chunks:
        return []

    # Validate lengths match
    if len(dense_vectors) != len(chunks) or len(sparse_vectors) != len(chunks):
        raise ValueError(
            f"Embedding mismatch: {len(chunks)} chunks, "
            f"{len(dense_vectors)} dense, {len(sparse_vectors)} sparse"
        )

    output_filepath = file_entry.get("output_filepath", "")
    source_url = file_entry.get("source_url", "")
    domain = file_entry.get("domain", "general")
    filename = Path(output_filepath).name

    records = []
    for i, (chunk, dense_vec, sparse_vec) in enumerate(
        zip(chunks, dense_vectors, sparse_vectors)
    ):
        # Truncate chunk_text if too long (Milvus VARCHAR limit)
        chunk_text = chunk[:8000] if len(chunk) > 8000 else chunk

        record = {
            "id": generate_chunk_id(output_filepath, i),
            "chunk_text": chunk_text,
            "dense_vector": dense_vec.tolist() if hasattr(dense_vec, 'tolist') else list(dense_vec),
            "sparse_vector": sparse_vec,
            "source_url": source_url[:500] if len(source_url) > 500 else source_url,
            "domain": domain,
            "filename": filename[:250] if len(filename) > 250 else filename,
            "chunk_index": i,
            "total_chunks": len(chunks),
        }
        records.append(record)

    return records


def run_ingestion(fresh_start: bool = False):
    """
    Run the full ingestion pipeline.

    Args:
        fresh_start: If True, drop existing collection and start fresh
    """
    print("=" * 60)
    print("Harel Insurance Vector Database Ingestion")
    print("=" * 60)

    # Load manifest
    print(f"\n[1/4] Loading manifest from {MANIFEST_PATH}...")
    manifest = load_manifest()
    total_files = len([f for f in manifest.get("files", []) if f.get("status") == "success"])
    print(f"Found {total_files} documents to process.")

    # Connect to Milvus
    print("\n[2/4] Connecting to Milvus...")
    try:
        connect()
    except Exception as e:
        print(f"\nError: Could not connect to Milvus at localhost:19530")
        print("Make sure Milvus is running: make milvus-start")
        print(f"Details: {e}")
        sys.exit(1)

    if fresh_start:
        print("Dropping existing collection...")
        drop_collection()

    # Get/create collection
    collection = get_collection()
    print(f"Collection '{collection.name}' ready.")

    # Process documents
    print(f"\n[3/4] Processing documents...")

    all_records = []
    total_chunks = 0
    processed_docs = 0
    failed_docs = 0

    # Collect all chunks first for batch embedding
    doc_chunks_map = []  # List of (file_entry, chunks)

    for file_entry, content in tqdm(
        iter_documents(manifest),
        total=total_files,
        desc="Chunking documents",
    ):
        chunks = chunk_markdown(content)
        if chunks:
            doc_chunks_map.append((file_entry, chunks))
            total_chunks += len(chunks)

    print(f"Total chunks to embed: {total_chunks}")

    # Embed in batches and create records
    print("\n[4/4] Embedding and inserting...")

    batch_records = []
    inserted_count = 0

    for file_entry, chunks in tqdm(doc_chunks_map, desc="Embedding & inserting"):
        try:
            # Embed all chunks for this document
            dense_vectors, sparse_vectors = embed_texts(chunks)

            # Create records (now passing chunks directly)
            records = process_document(file_entry, chunks, dense_vectors, sparse_vectors)

            batch_records.extend(records)
            processed_docs += 1

            # Insert when batch is full
            if len(batch_records) >= BATCH_SIZE:
                insert_batch(batch_records)
                inserted_count += len(batch_records)
                batch_records = []

        except Exception as e:
            print(f"\nError processing {file_entry.get('output_filepath')}: {e}")
            failed_docs += 1
            continue

    # Insert remaining records
    if batch_records:
        insert_batch(batch_records)
        inserted_count += len(batch_records)

    # Flush and load collection
    collection.flush()
    load_collection()

    # Summary
    stats = get_collection_stats()
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"Documents processed: {processed_docs}")
    print(f"Documents failed: {failed_docs}")
    print(f"Total chunks inserted: {inserted_count}")
    print(f"Collection entities: {stats['num_entities']}")
    print("=" * 60)

    disconnect()


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest data_prepared documents into Milvus vector database"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Drop existing collection and start fresh",
    )

    args = parser.parse_args()

    try:
        run_ingestion(fresh_start=args.fresh)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
