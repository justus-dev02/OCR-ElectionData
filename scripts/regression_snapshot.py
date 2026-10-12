"""Golden-master regression check on real source pages.

``capture`` runs the full pipeline on a fixed set of sample pages (one run and
database per case) and writes a canonical JSON dump of everything the run
stored: pages, tables, cells, OCR candidates, result rows, issues, review
cases. Volatile values (run ids, timestamps, surrogate keys) are normalised so
that two captures of the same code are byte-identical.

``compare`` diffs two captures. It is used to prove that a refactoring or a
change of the execution model (e.g. parallel page extraction) does not change
a single stored value.

    python scripts/regression_snapshot.py capture --out .work/regression/before
    python scripts/regression_snapshot.py capture --out .work/regression/after --workers 4
    python scripts/regression_snapshot.py compare .work/regression/before .work/regression/after

Needs the source PDFs in data/ and the OCR/ML models in models/.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# (case name, source PDF, pages): every table layout of the seven volumes
CASES = [
    ("bw49_gemeinden", "010_2611_49001.pdf", "12,25,40,55"),
    ("he53_gemeinden", "Hessen_Bundestagswahl_Statistik1953.pdf", "50,59"),
    ("bw53_gemeinden", "011_2611_53001.pdf", "39,41"),
    ("bw53_wahlkreise", "011_2611_53001.pdf", "29,31"),
    ("bw57_gemeinden", "043_2611_57001.pdf", "49,51,117"),
    ("bw57_wahlkreise", "043_2611_57001.pdf", "41,43"),
    ("bw61_gemeinden", "080_2611_61001.pdf", "35-38"),
    ("bw61_wahlkreise", "080_2611_61001.pdf", "31"),
    ("bw65_gemeinden", "121_2611_65001.pdf", "33-37"),
    ("bw65_wahlkreise", "121_2611_65001.pdf", "20,22,24,26,28"),
    ("bw69_gemeinden", "166_2611_69001.pdf", "32,33,40"),
]

DUMPED_TABLES = [
    "pages",
    "tables",
    "cells",
    "ocr_candidates",
    "result_rows",
    "issues",
    "review_cases",
    "state_transitions",
    "published_results",
]
VOLATILE_COLUMNS = {
    "started_at",
    "finished_at",
    "created_at",
    "updated_at",
    "at",
    "published_at",
    "registered_at",
    # autoincrement surrogate keys
    "candidate_id",
    "issue_id",
    "transition_id",
    "case_id",
    "published_id",
}


def _normalise(value, run_id: str, out: str):
    if isinstance(value, str):
        return value.replace(out, "OUT").replace(run_id, "RUN")
    return value


def dump(db_path: Path) -> dict:
    out_root = str(db_path.parent.resolve())
    con = sqlite3.connect(db_path)
    (run_id,) = con.execute("select run_id from processing_runs order by started_at desc limit 1").fetchone()
    run = dict(
        zip(
            ["status", "pages_requested", "stats"],
            con.execute("select status, pages_requested, stats from processing_runs where run_id=?", (run_id,)).fetchone(),
        )
    )
    out: dict = {"run": {k: _normalise(v, run_id, out_root) for k, v in run.items()}}
    for table in DUMPED_TABLES:
        cur = con.execute(f'select * from "{table}"')
        cols = [d[0] for d in cur.description]
        rows = []
        for raw in cur.fetchall():
            rows.append({c: _normalise(v, run_id, out_root) for c, v in zip(cols, raw) if c not in VOLATILE_COLUMNS})
        out[table] = sorted(rows, key=lambda r: json.dumps(r, sort_keys=True, default=str))
    return out


def capture(out_dir: Path, workers: int | None, only: list[str] | None) -> None:
    from election_ocr.config import load_config
    from election_ocr.persistence.db import make_engine, make_session
    from election_ocr.pipeline.runner import DocumentRunner
    from election_ocr.reference import ReferenceRepository

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, pdf, pages in CASES:
        if only and name not in only:
            continue
        db = out_dir / f"{name}.db"
        db.unlink(missing_ok=True)
        cfg = load_config()
        cfg.database_url = f"sqlite:///{db.resolve()}"
        cfg.paths.output_dir = out_dir.resolve() / "output"
        session = make_session(make_engine(cfg.database_url))
        ref = ReferenceRepository.load(cfg.paths.reference_dir)
        runner = DocumentRunner(cfg, session, ref)
        kwargs = {}
        if workers is not None and "workers" in inspect.signature(runner.run).parameters:
            kwargs["workers"] = workers
        t0 = time.time()
        result = runner.run(ROOT / "data" / pdf, pages=pages, **kwargs)
        session.close()
        (out_dir / f"{name}.json").write_text(json.dumps(dump(db), sort_keys=True, indent=1, default=str))
        print(f"{name}: {len(result.pages_processed)} pages, errors={result.page_errors}, {time.time() - t0:.0f}s", flush=True)


def compare(a: Path, b: Path) -> int:
    failures = 0
    for name, _, _ in CASES:
        fa, fb = a / f"{name}.json", b / f"{name}.json"
        if not fa.exists() or not fb.exists():
            print(f"{name}: missing ({fa.exists()=}, {fb.exists()=})")
            failures += 1
            continue
        da, db = json.loads(fa.read_text()), json.loads(fb.read_text())
        diffs = []
        for key in sorted(set(da) | set(db)):
            if da.get(key) != db.get(key):
                ra, rb = da.get(key) or [], db.get(key) or []
                if isinstance(ra, list) and isinstance(rb, list):
                    sa = {json.dumps(r, sort_keys=True) for r in ra}
                    sb = {json.dumps(r, sort_keys=True) for r in rb}
                    diffs.append(f"{key}: {len(sa - sb)} rows only in A, {len(sb - sa)} rows only in B")
                    for r in sorted(sa - sb)[:2]:
                        diffs.append(f"   A: {r[:400]}")
                    for r in sorted(sb - sa)[:2]:
                        diffs.append(f"   B: {r[:400]}")
                else:
                    diffs.append(f"{key}: {str(ra)[:300]} != {str(rb)[:300]}")
        if diffs:
            failures += 1
            print(f"{name}: DIFFERENT")
            print("\n".join("  " + d for d in diffs))
        else:
            print(f"{name}: identical ({sum(len(v) for v in da.values() if isinstance(v, list))} rows)")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--out", type=Path, required=True)
    cap.add_argument("--workers", type=int, default=None)
    cap.add_argument("--only", nargs="*", default=None, help="case names")
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("a", type=Path)
    cmp_.add_argument("b", type=Path)
    args = ap.parse_args()
    if args.cmd == "capture":
        capture(args.out, args.workers, args.only)
        return 0
    return compare(args.a, args.b)


if __name__ == "__main__":
    raise SystemExit(main())
