"""URL discovery via sitemap parsing and web crawling."""

import asyncio
import re
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, urlparse
from typing import Set

import aiohttp
from bs4 import BeautifulSoup

from .config import (
    SITEMAP_URL,
    HEADERS,
    INCLUDE_PATTERNS,
    EXCLUDE_PATTERNS,
    START_URLS,
    CONNECTION_TIMEOUT,
    READ_TIMEOUT,
    is_pdf_url,
    clean_url,
)


def is_valid_url(url: str) -> bool:
    """Check if URL should be scraped based on include/exclude patterns."""
    parsed = urlparse(url)
    if not parsed.netloc.endswith("harel-group.co.il"):
        return False

    for pattern in EXCLUDE_PATTERNS:
        if pattern in url:
            return False

    for pattern in INCLUDE_PATTERNS:
        if pattern in url:
            return True

    return False


def extract_pdf_links(soup: BeautifulSoup, base_url: str) -> Set[str]:
    """Extract all PDF links from a parsed HTML page."""
    pdf_urls = set()

    # Find PDFs in href attributes
    for tag in soup.find_all(href=re.compile(r"\.pdf", re.I)):
        href = tag.get("href", "")
        if href:
            absolute_url = urljoin(base_url, href)
            pdf_urls.add(absolute_url)

    # Find PDFs in src attributes (embedded)
    for tag in soup.find_all(src=re.compile(r"\.pdf", re.I)):
        src = tag.get("src", "")
        if src:
            absolute_url = urljoin(base_url, src)
            pdf_urls.add(absolute_url)

    return pdf_urls


async def fetch_sitemap(session: aiohttp.ClientSession) -> Set[str]:
    """Fetch and parse sitemap.xml for insurance-related URLs."""
    urls = set()

    try:
        timeout = aiohttp.ClientTimeout(total=CONNECTION_TIMEOUT)
        async with session.get(SITEMAP_URL, headers=HEADERS, timeout=timeout) as response:
            if response.status == 200:
                content = await response.text()

                # Parse XML safely
                try:
                    root = ET.fromstring(content)
                except ET.ParseError as e:
                    print(f"[Sitemap] XML parse error: {e}")
                    return urls

                # Handle different sitemap formats
                namespaces = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

                # Try with namespace
                for loc in root.findall('.//ns:loc', namespaces):
                    url = loc.text.strip() if loc.text else ""
                    if url and is_valid_url(url):
                        urls.add(url)

                # Try without namespace
                for loc in root.findall('.//loc'):
                    url = loc.text.strip() if loc.text else ""
                    if url and is_valid_url(url):
                        urls.add(url)

                print(f"[Sitemap] Found {len(urls)} insurance-related URLs")
            else:
                print(f"[Sitemap] Failed to fetch: HTTP {response.status}")
    except Exception as e:
        print(f"[Sitemap] Error fetching sitemap: {e}")

    return urls


async def crawl_page(
    session: aiohttp.ClientSession,
    url: str,
    visited: Set[str],
    to_visit: asyncio.Queue,
    found_urls: Set[str],
    lock: asyncio.Lock,
) -> None:
    """Crawl a single page and extract links."""
    try:
        timeout = aiohttp.ClientTimeout(total=READ_TIMEOUT)
        async with session.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True) as response:
            if response.status != 200:
                return

            content_type = response.headers.get("Content-Type", "")

            # If it's a PDF, just add it to found URLs
            if is_pdf_url(url, content_type):
                async with lock:
                    found_urls.add(url)
                return

            # Parse HTML for links
            if "text/html" in content_type or "aspx" in url.lower():
                async with lock:
                    found_urls.add(url)

                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")

                # Find all regular links
                for tag in soup.find_all(["a", "link"], href=True):
                    href = tag.get("href", "")
                    if not href:
                        continue

                    absolute_url = urljoin(url, href)
                    cleaned = clean_url(absolute_url)

                    if is_valid_url(cleaned):
                        async with lock:
                            if cleaned not in visited:
                                visited.add(cleaned)
                                await to_visit.put(cleaned)

                # Extract PDF links
                pdf_links = extract_pdf_links(soup, url)
                async with lock:
                    found_urls.update(pdf_links)

    except asyncio.TimeoutError:
        print(f"[Crawl] Timeout: {url}")
    except Exception as e:
        print(f"[Crawl] Error crawling {url}: {e}")


async def crawl_worker(
    session: aiohttp.ClientSession,
    to_visit: asyncio.Queue,
    visited: Set[str],
    found_urls: Set[str],
    lock: asyncio.Lock,
    shutdown_event: asyncio.Event,
) -> None:
    """Worker that processes URLs from the queue."""
    while not shutdown_event.is_set():
        try:
            url = await asyncio.wait_for(to_visit.get(), timeout=2.0)
            await crawl_page(session, url, visited, to_visit, found_urls, lock)
            to_visit.task_done()
        except asyncio.TimeoutError:
            # Queue empty, check if we should shutdown
            if to_visit.empty():
                break
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[Worker] Error: {e}")


async def discover_urls(max_workers: int = 10) -> Set[str]:
    """Discover all URLs to scrape via sitemap and crawling."""
    found_urls: Set[str] = set()
    visited: Set[str] = set()
    to_visit: asyncio.Queue = asyncio.Queue()
    lock = asyncio.Lock()
    shutdown_event = asyncio.Event()

    connector = aiohttp.TCPConnector(limit=max_workers * 2, ssl=False)

    async with aiohttp.ClientSession(connector=connector) as session:
        # Step 1: Fetch sitemap
        print("[Discovery] Fetching sitemap...")
        sitemap_urls = await fetch_sitemap(session)
        found_urls.update(sitemap_urls)

        # Step 2: Initialize crawl queue with start URLs
        print("[Discovery] Starting crawl from root pages...")
        for url in START_URLS:
            visited.add(url)
            await to_visit.put(url)

        # Also add sitemap URLs to crawl for deeper discovery
        for url in sitemap_urls:
            if url not in visited and not is_pdf_url(url):
                visited.add(url)
                await to_visit.put(url)

        # Step 3: Run crawl workers
        workers = [
            asyncio.create_task(
                crawl_worker(session, to_visit, visited, found_urls, lock, shutdown_event)
            )
            for _ in range(max_workers)
        ]

        # Wait for queue to be processed
        await to_visit.join()

        # Signal shutdown and wait for workers
        shutdown_event.set()
        await asyncio.gather(*workers, return_exceptions=True)

    print(f"[Discovery] Total URLs found: {len(found_urls)}")
    return found_urls
