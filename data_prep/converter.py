"""Document conversion and chunking using docling."""

import json
from pathlib import Path

from docling.document_converter import DocumentConverter


def convert_document(input_path: Path, output_path: Path) -> None:
    """
    Convert a document (PDF, HTML, ASPX) to markdown using docling.

    Args:
        input_path: Path to source file
        output_path: Path for output markdown file

    Raises:
        Exception: If conversion fails
    """
    converter = DocumentConverter()
    result = converter.convert(str(input_path))

    # Export to markdown
    markdown_content = result.document.export_to_markdown()

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write markdown with UTF-8 encoding for Hebrew support
    output_path.write_text(markdown_content, encoding="utf-8")


def convert_and_chunk(input_path: Path, output_path: Path, chunks_path: Path) -> int:
    """
    Convert a document to markdown AND produce page-aware chunks using
    docling's HybridChunker.

    Saves:
      - output_path: markdown file (same as convert_document)
      - chunks_path: JSON file with chunked data including page numbers

    Returns:
        Number of chunks produced.
    """
    import tiktoken
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

    converter = DocumentConverter()
    result = converter.convert(str(input_path))
    dl_doc = result.document

    # Save markdown
    output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_content = dl_doc.export_to_markdown()
    output_path.write_text(markdown_content, encoding="utf-8")

    # Chunk with HybridChunker (token-aware, structure-preserving)
    tokenizer = OpenAITokenizer(
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        max_tokens=512,
    )
    chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
    doc_chunks = list(chunker.chunk(dl_doc=dl_doc))

    # Extract chunk data with page numbers
    chunks_data = []
    for i, chunk in enumerate(doc_chunks):
        # Get page number from provenance
        page_no = 0
        for item in chunk.meta.doc_items:
            if hasattr(item, "prov") and item.prov:
                page_no = item.prov[0].page_no
                break

        # Get headings breadcrumb
        headings = chunk.meta.headings or []

        # Get the enriched text (with headings context prepended)
        enriched_text = chunker.contextualize(chunk)

        chunks_data.append({
            "chunk_index": i,
            "text": chunk.text,
            "enriched_text": enriched_text,
            "page_number": page_no,
            "headings": headings,
        })

    # Save chunks JSON
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)

    return len(chunks_data)
