"""Data preparation configuration settings."""

from pathlib import Path

# Input directory (output from scraper)
INPUT_DIR = Path("data")

# Output directory for prepared data
OUTPUT_DIR = Path("data_prepared")

# Input manifest from scraper
INPUT_MANIFEST = INPUT_DIR / "manifest.json"

# Output manifest for prepared data
OUTPUT_MANIFEST = OUTPUT_DIR / "manifest.json"

# Concurrency settings
MAX_CONCURRENT_WORKERS = 4

# Supported file types for conversion
SUPPORTED_TYPES = {"pdf", "html", "aspx"}
