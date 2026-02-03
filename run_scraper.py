#!/usr/bin/env python3
"""Convenience script to run the Harel Insurance scraper."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scraper.main import run_scraper


if __name__ == "__main__":
    print("Starting Harel Insurance Scraper...")
    print("This will scrape ~350 documents from harel-group.co.il")
    print()

    asyncio.run(run_scraper())
