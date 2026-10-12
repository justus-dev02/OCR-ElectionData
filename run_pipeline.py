#!/usr/bin/env python3
"""Entry point without installation: python run_pipeline.py <command> ..."""

import sys
from pathlib import Path

src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from election_ocr.cli import cli  # noqa: E402

if __name__ == "__main__":
    cli()
