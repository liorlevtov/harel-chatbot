"""Async file downloader for HTML pages and PDFs."""

import asyncio
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set
from urllib.parse import unquote, urlparse

import aiohttp

from .config import (
    HEADERS,
    MAX_CONCURRENT_REQUESTS,
    OUTPUT_DIR,
    CONNECTION_TIMEOUT,
    READ_TIMEOUT,
    CHUNK_SIZE,
    is_pdf_url,
    categorize_url,
)


def generate_filename(url: str, is_pdf: bool) -> str:
    """Generate a unique filename from URL using hash to avoid race conditions."""
    parsed = urlparse(url)
    path = unquote(parsed.path)

    # Get base filename from path
    base_name = path.split("/")[-1] or "index"

    # Remove unsafe characters
    base_name = re.sub(r'[<>:"/\\|?*]', '-', base_name)
    base_name = re.sub(r'\s+', '_', base_name)

    # Remove existing extension
    name_without_ext = base_name.rsplit('.', 1)[0] if '.' in base_name else base_name

    # Truncate if too long
    if len(name_without_ext) > 150:
        name_without_ext = name_without_ext[:150]

    # Add URL hash for uniqueness (prevents race conditions)
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]

    # Determine extension
    ext = ".pdf" if is_pdf else ".html"

    return f"{name_without_ext}_{url_hash}{ext}"


async def record_error(
    url: str,
    error: str,
    results: List[Dict],
    lock: asyncio.Lock,
) -> None:
    """Record a download error."""
    async with lock:
        results.append({
            "url": url,
            "status": "error",
            "error": error,
        })


async def download_file(
    session: aiohttp.ClientSession,
    url: str,
    semaphore: asyncio.Semaphore,
    results: List[Dict],
    lock: asyncio.Lock,
) -> None:
    """Download a single file and save it to disk with streaming."""
    async with semaphore:
        try:
            timeout = aiohttp.ClientTimeout(
                total=READ_TIMEOUT,
                connect=CONNECTION_TIMEOUT,
            )

            async with session.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True) as response:
                if response.status != 200:
                    print(f"[Download] HTTP {response.status}: {url}")
                    await record_error(url, f"HTTP {response.status}", results, lock)
                    return

                # Determine file type
                content_type = response.headers.get("Content-Type", "")
                is_pdf = is_pdf_url(url, content_type)

                # Categorize and create directory
                domain = categorize_url(url)
                domain_dir = Path(OUTPUT_DIR) / domain / "files"
                domain_dir.mkdir(parents=True, exist_ok=True)

                # Generate unique filename (hash-based to avoid race conditions)
                filename = generate_filename(url, is_pdf)
                filepath = domain_dir / filename

                # Stream content to disk to avoid memory issues with large files
                total_size = 0
                try:
                    with open(filepath, "wb") as f:
                        async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                            f.write(chunk)
                            total_size += len(chunk)
                except IOError as e:
                    print(f"[Download] File write error {url}: {e}")
                    # Clean up partial file
                    if filepath.exists():
                        filepath.unlink()
                    await record_error(url, f"File write error: {e}", results, lock)
                    return

                # Record success AFTER file is written
                async with lock:
                    results.append({
                        "url": url,
                        "status": "success",
                        "domain": domain,
                        "filepath": str(filepath),
                        "filename": filepath.name,
                        "type": "pdf" if is_pdf else "html",
                        "size_bytes": total_size,
                        "content_type": content_type,
                    })

                print(f"[Download] Saved: {filepath} ({total_size:,} bytes)")

        except asyncio.TimeoutError:
            print(f"[Download] Timeout: {url}")
            await record_error(url, "Timeout", results, lock)
        except aiohttp.ClientError as e:
            print(f"[Download] Network error {url}: {e}")
            await record_error(url, f"Network error: {e}", results, lock)
        except Exception as e:
            print(f"[Download] Error {url}: {e}")
            await record_error(url, str(e), results, lock)


async def download_all(urls: Set[str], max_concurrent: int = MAX_CONCURRENT_REQUESTS) -> List[Dict]:
    """Download all URLs concurrently."""
    results: List[Dict] = []
    lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max_concurrent)

    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    connector = aiohttp.TCPConnector(limit=max_concurrent * 2, ssl=False)

    print(f"[Download] Starting download of {len(urls)} files with {max_concurrent} concurrent connections...")

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            download_file(session, url, semaphore, results, lock)
            for url in urls
        ]
        await asyncio.gather(*tasks)

    # Save manifest
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "total_urls": len(urls),
        "successful": sum(1 for r in results if r.get("status") == "success"),
        "failed": sum(1 for r in results if r.get("status") == "error"),
        "files": results,
    }

    manifest_path = Path(OUTPUT_DIR) / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[Download] Manifest saved to: {manifest_path}")

    return results
