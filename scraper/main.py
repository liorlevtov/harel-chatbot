"""Main entry point for the Harel Insurance scraper."""

import asyncio
import argparse
import sys
from pathlib import Path

from .crawler import discover_urls
from .downloader import download_all
from .config import MAX_CONCURRENT_REQUESTS, OUTPUT_DIR


async def run_scraper(
    max_concurrent: int = MAX_CONCURRENT_REQUESTS,
    output_dir: str = OUTPUT_DIR,
    skip_discovery: bool = False,
    urls_file: str | None = None,
) -> None:
    """Run the complete scraping pipeline."""
    print("=" * 60)
    print("Harel Insurance Scraper")
    print("=" * 60)

    # Step 1: Discover URLs
    if skip_discovery and urls_file:
        print(f"\n[1/2] Loading URLs from {urls_file}...")
        try:
            with open(urls_file, "r") as f:
                urls = set(line.strip() for line in f if line.strip())
            print(f"Loaded {len(urls)} URLs")
        except FileNotFoundError:
            print(f"Error: URLs file not found: {urls_file}")
            sys.exit(1)
        except IOError as e:
            print(f"Error reading URLs file: {e}")
            sys.exit(1)
    else:
        print("\n[1/2] Discovering URLs (sitemap + crawling)...")
        urls = await discover_urls(max_workers=max_concurrent // 2 or 1)

    if not urls:
        print("No URLs found. Exiting.")
        return

    # Save discovered URLs for reference
    urls_path = Path(output_dir) / "discovered_urls.txt"
    urls_path.parent.mkdir(parents=True, exist_ok=True)
    with open(urls_path, "w") as f:
        for url in sorted(urls):
            f.write(url + "\n")
    print(f"Saved URL list to: {urls_path}")

    # Step 2: Download all files
    print(f"\n[2/2] Downloading {len(urls)} files...")
    results = await download_all(urls, max_concurrent=max_concurrent)

    # Summary
    successful = sum(1 for r in results if r.get("status") == "success")
    failed = sum(1 for r in results if r.get("status") == "error")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total URLs discovered: {len(urls)}")
    print(f"Successfully downloaded: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {output_dir}/")
    print("=" * 60)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Scrape Harel Insurance website for policy documents"
    )
    parser.add_argument(
        "-c", "--concurrent",
        type=int,
        default=MAX_CONCURRENT_REQUESTS,
        help=f"Maximum concurrent downloads (default: {MAX_CONCURRENT_REQUESTS})"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--urls-file",
        type=str,
        help="Skip discovery and use URLs from file (one URL per line)"
    )

    args = parser.parse_args()

    # Update config with CLI args
    import scraper.config as config
    config.OUTPUT_DIR = args.output

    # Run scraper
    asyncio.run(run_scraper(
        max_concurrent=args.concurrent,
        output_dir=args.output,
        skip_discovery=bool(args.urls_file),
        urls_file=args.urls_file,
    ))


if __name__ == "__main__":
    main()
