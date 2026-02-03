"""Main entry point for data preparation."""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from .config import (
    INPUT_DIR,
    INPUT_MANIFEST,
    MAX_CONCURRENT_WORKERS,
    OUTPUT_DIR,
    OUTPUT_MANIFEST,
    SUPPORTED_TYPES,
)
from .converter import convert_document


def load_manifest(manifest_path: Path) -> dict:
    """Load the scraper manifest."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest: dict, manifest_path: Path) -> None:
    """Save the prepared data manifest."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def get_output_path(source_filepath: str, input_dir: Path, output_dir: Path) -> Path:
    """
    Generate output path mirroring the input directory structure.

    Example: data/car/files/policy.pdf -> data_prepared/car/files/policy.md
    """
    source_path = Path(source_filepath)
    relative_path = source_path.relative_to(input_dir)
    output_path = output_dir / relative_path.with_suffix(".md")
    return output_path


def process_file(
    file_entry: dict,
    input_dir: Path,
    output_dir: Path,
    index: int,
    total: int,
    skip_existing: bool = True,
) -> dict:
    """
    Process a single file and return result entry for manifest.

    Args:
        file_entry: Entry from scraper manifest
        input_dir: Base input directory
        output_dir: Base output directory
        index: Current file index (1-based)
        total: Total number of files
        skip_existing: Skip files that already have output

    Returns:
        Dict with processing result for output manifest
    """
    source_filepath = file_entry["filepath"]
    source_path = Path(source_filepath)
    output_path = get_output_path(source_filepath, input_dir, output_dir)

    result = {
        "source_filepath": source_filepath,
        "source_url": file_entry["url"],
        "source_type": file_entry["type"],
        "domain": file_entry["domain"],
        "output_filepath": str(output_path),
    }

    # Skip if output already exists
    if skip_existing and output_path.exists():
        result["status"] = "success"
        result["skipped"] = True
        print(f"  [{index}/{total}] [SKIP] {source_path.name}", flush=True)
        return result

    try:
        convert_document(source_path, output_path)
        result["status"] = "success"
        print(f"  [{index}/{total}] [OK] {source_path.name}", flush=True)
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        print(f"  [{index}/{total}] [FAIL] {source_path.name}: {e}", flush=True)
        raise

    return result


def run_prep(
    input_dir: Path = INPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
    max_workers: int = MAX_CONCURRENT_WORKERS,
) -> None:
    """Run the data preparation pipeline."""
    print("=" * 60)
    print("Harel Insurance Data Preparation")
    print("=" * 60)

    # Load scraper manifest
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}")
        print("Please run the scraper first: make run-scrape")
        sys.exit(1)

    print(f"\n[1/2] Loading manifest from {manifest_path}...")
    source_manifest = load_manifest(manifest_path)

    # Filter to supported file types
    files_to_process = [
        f for f in source_manifest["files"]
        if f.get("type") in SUPPORTED_TYPES and f.get("status") == "success"
    ]

    total_files = len(files_to_process)
    print(f"Found {total_files} files to process ({', '.join(SUPPORTED_TYPES)})")

    # Process files concurrently
    print(f"\n[2/2] Converting documents (workers: {max_workers})...")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    successful = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_file, f, input_dir, output_dir, i + 1, total_files, True
            ): f
            for i, f in enumerate(files_to_process)
        }

        for future in as_completed(futures):
            file_entry = futures[future]
            try:
                result = future.result()
                results.append(result)
                if result["status"] == "success":
                    successful += 1
            except Exception as e:
                # Stop on first error for debugging
                print(f"\nError processing {file_entry['filepath']}: {e}")
                print("Stopping due to error. Fix the issue and re-run.")
                executor.shutdown(wait=False, cancel_futures=True)
                sys.exit(1)

    # Create output manifest
    output_manifest = {
        "timestamp": datetime.now().isoformat(),
        "source_manifest": str(manifest_path),
        "total_files": total_files,
        "successful": successful,
        "failed": failed,
        "files": results,
    }

    manifest_output_path = output_dir / "manifest.json"
    save_manifest(output_manifest, manifest_output_path)
    print(f"\nSaved manifest to: {manifest_output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total files processed: {total_files}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {output_dir}/")
    print("=" * 60)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare scraped data for vector database ingestion"
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=INPUT_DIR,
        help=f"Input directory (default: {INPUT_DIR})"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=MAX_CONCURRENT_WORKERS,
        help=f"Max concurrent workers (default: {MAX_CONCURRENT_WORKERS})"
    )

    args = parser.parse_args()

    run_prep(
        input_dir=args.input,
        output_dir=args.output,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
