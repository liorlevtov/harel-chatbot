"""Markdown-aware text chunking."""

import re
from typing import List

from .config import MIN_CHUNK_SIZE, MAX_CHUNK_SIZE, CHUNK_OVERLAP


def estimate_tokens(text: str) -> int:
    """Estimate token count (rough approximation: 1 token ~ 4 chars for mixed Hebrew/English)."""
    return len(text) // 4


def split_by_headers(text: str) -> List[str]:
    """Split markdown by headers while keeping header with content."""
    # Pattern matches markdown headers (# ## ### etc.)
    pattern = r'(^#{1,6}\s+.+$)'
    parts = re.split(pattern, text, flags=re.MULTILINE)

    sections = []
    current_section = ""

    for part in parts:
        if re.match(r'^#{1,6}\s+', part):
            # This is a header - start new section
            if current_section.strip():
                sections.append(current_section.strip())
            current_section = part + "\n"
        else:
            current_section += part

    if current_section.strip():
        sections.append(current_section.strip())

    return sections


def split_by_paragraphs(text: str) -> List[str]:
    """Split text by double newlines (paragraphs)."""
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]


def merge_small_chunks(chunks: List[str], min_size: int) -> List[str]:
    """Merge chunks that are too small."""
    if not chunks:
        return []

    merged = []
    current = chunks[0]

    for chunk in chunks[1:]:
        if estimate_tokens(current) < min_size:
            current += "\n\n" + chunk
        else:
            merged.append(current)
            current = chunk

    if current:
        merged.append(current)

    return merged


def split_large_chunk(text: str, max_size: int, overlap: int) -> List[str]:
    """Split a large chunk into smaller pieces with overlap."""
    tokens_estimate = estimate_tokens(text)
    if tokens_estimate <= max_size:
        return [text]

    # Split by sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""
    overlap_buffer = ""

    for sentence in sentences:
        test_chunk = current_chunk + " " + sentence if current_chunk else sentence

        if estimate_tokens(test_chunk) > max_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Keep last part for overlap
            words = current_chunk.split()
            overlap_words = words[-overlap:] if len(words) > overlap else words
            overlap_buffer = " ".join(overlap_words)
            current_chunk = overlap_buffer + " " + sentence
        else:
            current_chunk = test_chunk

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def chunk_markdown(text: str) -> List[str]:
    """
    Chunk markdown text into pieces suitable for embedding.

    Strategy:
    1. Split by headers to preserve document structure
    2. Split large sections by paragraphs
    3. Merge small chunks together
    4. Split remaining large chunks with overlap
    """
    if not text or not text.strip():
        return []

    # Step 1: Split by headers
    sections = split_by_headers(text)

    # Step 2: Process each section
    chunks = []
    for section in sections:
        section_tokens = estimate_tokens(section)

        if section_tokens <= MAX_CHUNK_SIZE:
            chunks.append(section)
        else:
            # Split by paragraphs first
            paragraphs = split_by_paragraphs(section)

            for para in paragraphs:
                if estimate_tokens(para) <= MAX_CHUNK_SIZE:
                    chunks.append(para)
                else:
                    # Split large paragraphs
                    sub_chunks = split_large_chunk(para, MAX_CHUNK_SIZE, CHUNK_OVERLAP)
                    chunks.extend(sub_chunks)

    # Step 3: Merge small chunks
    chunks = merge_small_chunks(chunks, MIN_CHUNK_SIZE)

    # Step 4: Final pass - split any remaining large chunks
    final_chunks = []
    for chunk in chunks:
        if estimate_tokens(chunk) > MAX_CHUNK_SIZE:
            final_chunks.extend(split_large_chunk(chunk, MAX_CHUNK_SIZE, CHUNK_OVERLAP))
        else:
            final_chunks.append(chunk)

    return [c for c in final_chunks if c.strip()]
