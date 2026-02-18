"""
Vector store setup and interactive search loop.

Usage:
    python integration.py           # assumes vector store already populated
    python integration.py --ingest  # run ingestion first, then start search loop
    python integration.py --fresh   # drop & rebuild the collection, then start search loop
"""

import argparse
import sys

from vector_db.ingest import run_ingestion
from vector_db.search import hybrid_search
from vector_db.milvus_client import connect, load_collection, collection_exists
from vector_db.config import DEFAULT_TOP_K, DOMAINS, DOMAINS_HEB


def build_vector_store(fresh_start: bool = False):
    """Run ingestion pipeline to populate the vector store."""
    print("Building vector store...")
    try:
        run_ingestion(fresh_start=fresh_start)
    except ConnectionError as e:
        print(f"\nError: {e}")
        print("\nTo start Milvus locally, run:  make milvus-start")
        sys.exit(1)
    print("Vector store ready.\n")


def ensure_collection_loaded():
    """Connect to Milvus and load the collection into memory."""
    connect()
    if not collection_exists():
        print(
            "Error: collection does not exist. "
            "Run with --ingest or --fresh to populate it first."
        )
        sys.exit(1)
    load_collection()


def display_results(results, query: str):
    """Pretty-print search results."""
    if not results:
        print("  (no results found)\n")
        return

    print(f"\n  Found {len(results)} result(s) for: \"{query}\"\n")
    for i, r in enumerate(results, 1):
        score_str = f"{r.score:.4f}" if hasattr(r, "score") else str(r.get("score", ""))
        domain = r.domain if hasattr(r, "domain") else r.get("domain", "")
        filename = r.filename if hasattr(r, "filename") else r.get("filename", "")
        chunk_text = r.chunk_text if hasattr(r, "chunk_text") else r.get("chunk_text", "")
        source_url = r.source_url if hasattr(r, "source_url") else r.get("source_url", "")

        print(f"  [{i}] score={score_str}  domain={domain}  file={filename}")
        if source_url:
            print(f"       url: {source_url}")
        # Print a preview of the chunk (first 300 chars)
        preview = chunk_text.replace("\n", " ").strip()[:300]
        print(f"       {preview}...")
        print()


def interactive_loop(top_k: int = DEFAULT_TOP_K):
    """
    Run an interactive Q&A loop.

    The user can optionally prefix their query with a domain filter:
        car: מה זה ביטוח חובה?
        health: כיצד מגישים תביעה?

    Type 'quit' or 'exit' to stop.
    Type 'help' to see available domains.
    """
    print("=" * 60)
    print("Harel Insurance Vector Search")
    print("=" * 60)
    print(f"Available domains: {', '.join(DOMAINS)}")
    print("Tip: prefix your query with a domain name and colon to filter.")
    print("     e.g.  car: מה זה ביטוח חובה?")
    print(" or   בריאות: כיצד מגישים תביעה?")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            raw = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not raw:
            continue

        lower = raw.lower()
        if lower in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        if lower == "help":
            print(f"  Domains: {', '.join(DOMAINS)}")
            print(f"  Hebrew domains: {', '.join(DOMAINS_HEB.values())}\n")
            continue

        # Parse optional domain prefix  "car: some question"
        domain = None
        query = raw
        if ":" in raw:
            prefix, _, rest = raw.partition(":")
            candidate = prefix.strip().lower()

            if candidate in DOMAINS:
                domain = candidate
                query = rest.strip()
            elif candidate in DOMAINS_HEB.values():
                # Map Hebrew domain name back to English
                domain = next(k for k, v in DOMAINS_HEB.items() if v == candidate)
                query = rest.strip()

        if not query:
            print("  Empty query after domain prefix. Please enter a question.\n")
            continue

        try:
            results = hybrid_search(query, domain=domain, top_k=top_k)
            display_results(results, query)
        except Exception as e:
            print(f"  Search error: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build the Harel vector store and run an interactive search loop."
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run ingestion before starting the search loop (adds to existing data).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Drop the existing collection, re-ingest, then start the search loop.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of results to return per query (default: {DEFAULT_TOP_K}).",
    )
    args = parser.parse_args()

    if args.fresh or args.ingest:
        build_vector_store(fresh_start=args.fresh)
    else:
        # Just connect and load the already-populated collection.
        ensure_collection_loaded()

    interactive_loop(top_k=args.top_k)


if __name__ == "__main__":
    main()
