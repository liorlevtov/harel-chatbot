"""Hybrid search functionality for the vector database."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from pymilvus import Collection, AnnSearchRequest, WeightedRanker

from .config import (
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    DENSE_WEIGHT,
    SPARSE_WEIGHT,
)
from .embedder import embed_query
from .milvus_client import connect, disconnect, get_collection, load_collection


@dataclass
class SearchResult:
    """A single search result."""
    chunk_text: str
    source_url: str
    domain: str
    filename: str
    score: float
    chunk_index: int
    total_chunks: int


def hybrid_search(
    query: str,
    domain: Optional[str] = None,
    domains: Optional[List[str]] = None,
    top_k: int = DEFAULT_TOP_K,
    dense_weight: float = DENSE_WEIGHT,
    sparse_weight: float = SPARSE_WEIGHT,
) -> List[SearchResult]:
    """
    Perform hybrid search combining dense and sparse vectors.

    Args:
        query: Search query text
        domain: Single domain to filter by (e.g., "car", "health")
        domains: List of domains to filter by
        top_k: Number of results to return
        dense_weight: Weight for dense vector search (0-1)
        sparse_weight: Weight for sparse vector search (0-1)

    Returns:
        List of SearchResult objects sorted by relevance
    """
    if not query or not query.strip():
        return []

    try:
        # Connect and load collection
        connect()
        collection = get_collection()

        # Ensure collection is loaded
        try:
            collection.load()
        except Exception:
            pass  # Already loaded

        # Embed query
        dense_vector, sparse_vector = embed_query(query)

        # Build filter expression
        filter_expr = None
        if domain:
            filter_expr = f'domain == "{domain}"'
        elif domains:
            domain_list = '", "'.join(domains)
            filter_expr = f'domain in ["{domain_list}"]'

        # Create search requests for hybrid search
        dense_search = AnnSearchRequest(
            data=[dense_vector.tolist()],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"ef": 100}},
            limit=top_k * 2,  # Get more candidates for reranking
        )

        sparse_search = AnnSearchRequest(
            data=[sparse_vector],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {}},
            limit=top_k * 2,
        )

        # Perform hybrid search with weighted ranker
        ranker = WeightedRanker(dense_weight, sparse_weight)

        results = collection.hybrid_search(
            reqs=[dense_search, sparse_search],
            rerank=ranker,
            limit=top_k,
            expr=filter_expr,
            output_fields=[
                "chunk_text",
                "source_url",
                "domain",
                "filename",
                "chunk_index",
                "total_chunks",
            ],
        )

        # Handle empty results
        if not results or len(results) == 0:
            return []

        # Convert to SearchResult objects
        search_results = []
        for hits in results:
            for hit in hits:
                entity = hit.entity
                result = SearchResult(
                    chunk_text=entity.get("chunk_text", ""),
                    source_url=entity.get("source_url", ""),
                    domain=entity.get("domain", ""),
                    filename=entity.get("filename", ""),
                    score=hit.score,
                    chunk_index=entity.get("chunk_index", 0),
                    total_chunks=entity.get("total_chunks", 1),
                )
                search_results.append(result)

        return search_results

    finally:
        disconnect()


def search_simple(
    query: str,
    domain: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
) -> List[Dict[str, Any]]:
    """
    Simple search interface returning dicts.

    Args:
        query: Search query text
        domain: Domain to filter by
        top_k: Number of results

    Returns:
        List of result dicts with chunk_text, source_url, domain, filename, score
    """
    results = hybrid_search(query, domain=domain, top_k=top_k)

    return [
        {
            "chunk_text": r.chunk_text,
            "source_url": r.source_url,
            "domain": r.domain,
            "filename": r.filename,
            "score": r.score,
        }
        for r in results
    ]


# Convenience function for direct usage
def search(
    query: str,
    domain: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
) -> List[Dict[str, Any]]:
    """
    Search the vector database.

    Args:
        query: Search query (Hebrew or English)
        domain: Optional domain filter (car, health, travel, etc.)
        top_k: Number of results to return

    Returns:
        List of dicts with keys: chunk_text, source_url, domain, filename, score

    Example:
        >>> from vector_db import search
        >>> results = search("מה זה ביטוח חובה?", domain="car", top_k=5)
        >>> for r in results:
        ...     print(f"{r['domain']}: {r['chunk_text'][:100]}...")
    """
    return search_simple(query, domain=domain, top_k=top_k)
