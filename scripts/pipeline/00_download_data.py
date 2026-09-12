"""
Step 0 — Download and cache the Civil Comments dataset locally via the
Hugging Face `datasets` library. This is a convenience/verification
script; scripts/01_extract_embeddings.py will also trigger this
download automatically on first run.

Usage
-----
    python scripts/00_download_data.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_civil_comments


def main():
    print("Downloading google/civil_comments (cached under ~/.cache/huggingface)...")
    dataset = load_civil_comments()
    print(dataset)
    print("\nExample record:")
    print(dataset["train"][0])


if __name__ == "__main__":
    main()
