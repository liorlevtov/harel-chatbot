#!/usr/bin/env python3
"""Convenience script to run the data preparation pipeline."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data_prep.main import run_prep


if __name__ == "__main__":
    print("Starting Harel Insurance Data Preparation...")
    print("This will convert scraped documents to markdown using docling")
    print()

    run_prep()
