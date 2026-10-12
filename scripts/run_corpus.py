"""Process every source PDF that has a document profile (one run per document) and time it.

    python scripts/run_corpus.py --workers 5
    python scripts/run_corpus.py --workers 1 --only 080_2611_61001.pdf --db .work/seq.db

Writes into the configured database (``--db`` overrides it) and appends one JSON line
per document to ``<db>.timing.jsonl`` (pages, wall time, seconds per page, workers).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from election_ocr.config import load_config
    from election_ocr.ingestion.registry import sha256_of
    from election_ocr.persistence.db import make_engine, make_session
    from election_ocr.pipeline.runner import DocumentRunner, default_workers
    from election_ocr.reference import ReferenceRepository

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--db", type=Path, default=None, help="SQLite file (default: config database_url)")
    ap.add_argument("--output-dir", type=Path, default=None, help="page images etc. (default: config paths.output_dir)")
    ap.add_argument("--only", nargs="*", default=None, help="PDF file names")
    args = ap.parse_args()

    cfg = load_config()
    if args.db is not None:
        cfg.database_url = f"sqlite:///{args.db.resolve()}"
    if args.output_dir is not None:
        cfg.paths.output_dir = args.output_dir.resolve()
    ref = ReferenceRepository.load(cfg.paths.reference_dir)
    session = make_session(make_engine(cfg.database_url))
    timing_path = Path(cfg.database_url.replace("sqlite:///", "") + ".timing.jsonl")
    seen = set()
    for pdf in sorted((ROOT / "data").glob("*.pdf")):
        if args.only and pdf.name not in args.only:
            continue
        digest = sha256_of(pdf)
        if digest in seen:
            print(f"skip {pdf.name}: byte-identical to an earlier file", flush=True)
            continue
        seen.add(digest)
        if ref.profile_for_sha256(digest) is None:
            print(f"skip {pdf.name}: no document profile", flush=True)
            continue
        t0 = time.time()
        res = DocumentRunner(cfg, session, ref).run(pdf, workers=args.workers)
        wall = time.time() - t0
        pages = len(res.pages_processed)
        line = {
            "document": pdf.name,
            "run_id": res.run_id,
            "pages": pages,
            "page_errors": len(res.page_errors),
            "workers": args.workers,
            "wall_seconds": round(wall, 1),
            "seconds_per_page": round(wall / max(1, pages), 2),
        }
        with open(timing_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")
        print(json.dumps(line), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
