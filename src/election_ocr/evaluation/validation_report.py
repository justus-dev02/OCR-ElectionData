"""
Validation report over the latest completed run of every document in a database.

Nothing is published while the gazetteer is missing and the reference data are
unverified (see README). This report therefore measures what the machine part of
the pipeline achieves *before* that human step:

Cell level
    share of numeric cells on which >= 2 independent OCR engines agree (no dissent)

Row level (publishable roles), by the blocking issues that remain
    ``reference_only``  blocked ONLY by missing/unverified reference data: every field agreed,
                        every arithmetic check executed and passed, the section sum verified
    ``data_verified``   additionally allowed: the section sum could not be verified because
                        OTHER rows of the section are unresolved
    ``needs_review``    any other blocking issue (OCR dissent, structure, arithmetic, ...)

Section level
    section totals whose members add up EXACTLY to the printed total (independent check)

External (``evaluation.external``)
    exact agreement of machine-agreed values with independently digitised official data
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from election_ocr.evaluation.external import (
    FieldComparison,
    compare_hessen_1953,
    load_hessen_1953,
)
from election_ocr.persistence.db import Cell, IssueRow, ProcessingRun, ResultRow, SourceDocument, TableRow

REFERENCE_CODES = frozenset({"REFERENCE_DATA_MISSING", "REFERENCE_UNVERIFIED"})
OTHER_ROWS_CODES = frozenset({"HIERARCHY_UNVERIFIED"})
SUM_CODES = frozenset({"HIERARCHY_UNVERIFIED", "HIERARCHICAL_SUM_MISMATCH"})
TIERS = ("reference_only", "data_verified", "needs_review")


@dataclass
class ExternalResult:
    source: str
    rows_considered: int
    rows_matched: int
    rows_ambiguous: int
    rows_unmatched: int
    cells_compared: int
    cells_equal: int
    per_field: Dict[str, Dict[str, int]]
    mismatches: List[dict]

    @classmethod
    def of(cls, source: str, c: FieldComparison, max_mismatches: int = 50) -> ExternalResult:
        return cls(
            source=source,
            rows_considered=c.rows_considered,
            rows_matched=c.rows_matched,
            rows_ambiguous=c.rows_ambiguous,
            rows_unmatched=c.rows_unmatched,
            cells_compared=c.total_compared,
            cells_equal=c.total_equal,
            per_field={k: {"compared": c.compared[k], "equal": c.equal[k]} for k in sorted(c.compared)},
            mismatches=c.mismatches[:max_mismatches],
        )


@dataclass
class DocumentValidation:
    document: str
    run_id: str
    profile: Optional[str]
    pages: int
    page_errors: int
    tables: int
    tables_without_rows: int
    numeric_cells: Dict[str, int]
    rows: int
    tiers: Dict[str, int]
    section_totals: int
    section_totals_verified: int
    section_totals_mismatch: int
    top_blockers: Dict[str, int]
    external: List[ExternalResult] = field(default_factory=list)

    @property
    def agreed_share(self) -> float:
        total = sum(self.numeric_cells.values())
        return self.numeric_cells.get("AGREED", 0) / total if total else 0.0


def latest_runs(session: Session) -> List[ProcessingRun]:
    """Latest COMPLETED run per document."""
    runs = session.execute(
        select(ProcessingRun).where(ProcessingRun.status == "COMPLETED").order_by(ProcessingRun.started_at)
    ).scalars()
    latest: Dict[str, ProcessingRun] = {}
    for r in runs:
        latest[r.document_id] = r
    return list(latest.values())


def blocking_codes(session: Session, run_id: str) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = defaultdict(set)
    q = select(IssueRow.result_id, IssueRow.code).where(
        IssueRow.run_id == run_id, IssueRow.severity == "BLOCKING", IssueRow.acknowledged.is_(False)
    )
    for rid, code in session.execute(q):
        if rid is not None:
            out[rid].add(code)
    return out


def row_tier(codes: Set[str]) -> str:
    if codes <= REFERENCE_CODES:
        return "reference_only"
    if codes <= REFERENCE_CODES | OTHER_ROWS_CODES:
        return "data_verified"
    return "needs_review"


def row_tiers(session: Session, run_id: str) -> Dict[str, str]:
    """result_id -> tier for every publishable row of the run."""
    blockers = blocking_codes(session, run_id)
    rows = session.execute(
        select(ResultRow.result_id).where(ResultRow.run_id == run_id, ResultRow.publishable_role.is_(True))
    ).scalars()
    return {rid: row_tier(blockers.get(rid, set())) for rid in rows}


def validate_document(
    session: Session, run: ProcessingRun, gpwed_dir: Optional[Path] = None, max_mismatches: int = 50
) -> DocumentValidation:
    doc = session.get(SourceDocument, run.document_id)
    blockers = blocking_codes(session, run.run_id)
    rows = list(
        session.execute(select(ResultRow).where(ResultRow.run_id == run.run_id, ResultRow.publishable_role.is_(True))).scalars()
    )
    tiers = Counter(row_tier(blockers.get(r.result_id, set())) for r in rows)
    top = Counter(code for r in rows for code in blockers.get(r.result_id, set()) - REFERENCE_CODES)
    cell_query = (
        select(Cell.status, func.count())
        .where(Cell.run_id == run.run_id, Cell.value_type.in_(("integer", "decimal")))
        .group_by(Cell.status)
    )
    cells: Dict[str, int] = {status: int(n) for status, n in session.execute(cell_query)}
    tables = session.execute(select(TableRow.table_id).where(TableRow.run_id == run.run_id)).scalars().all()
    with_rows = {r.table_id for r in rows}
    totals, verified, mismatch = _section_sums(rows, blockers)
    stats = run.stats or {}
    result = DocumentValidation(
        document=doc.filename if doc else run.document_id,
        run_id=run.run_id,
        profile=run.document_profile_id,
        pages=int(stats.get("pages_processed", 0)),
        page_errors=len(stats.get("page_errors") or {}),
        tables=len(tables),
        tables_without_rows=sum(1 for t in tables if t not in with_rows),
        numeric_cells=cells,
        rows=len(rows),
        tiers={t: tiers.get(t, 0) for t in TIERS},
        section_totals=totals,
        section_totals_verified=verified,
        section_totals_mismatch=mismatch,
        top_blockers=dict(top.most_common(8)),
    )
    if gpwed_dir is not None:
        result.external = _external(session, run, gpwed_dir, rows, blockers, max_mismatches)
    return result


def _section_sums(rows: List[ResultRow], blockers: Dict[str, Set[str]]):
    """(section totals with members, verified exactly, sum mismatches) over absolute total rows."""
    members: Dict[tuple, int] = Counter(
        (r.section_key, r.role_id) for r in rows if r.row_kind == "DATA" and r.role_kind == "absolute"
    )
    totals = verified = mismatch = 0
    for r in rows:
        if r.row_kind not in ("SECTION_TOTAL", "GRAND_TOTAL") or r.role_kind != "absolute":
            continue
        if r.row_kind == "SECTION_TOTAL" and not members.get((r.section_key, r.role_id)):
            continue  # a single printed line (e.g. an independent city): nothing to add up
        totals += 1
        codes = blockers.get(r.result_id, set())
        if "HIERARCHICAL_SUM_MISMATCH" in codes:
            mismatch += 1
        elif not codes & SUM_CODES:
            verified += 1
    return totals, verified, mismatch


def _external(
    session: Session,
    run: ProcessingRun,
    gpwed_dir: Path,
    rows: List[ResultRow],
    blockers: Dict[str, Set[str]],
    max_mismatches: int,
) -> List[ExternalResult]:
    out: List[ExternalResult] = []
    election_ids = {r.election_id for r in rows}
    hessen_file = gpwed_dir / "BW_Hessen-1953.xlsx"
    if hessen_file.exists() and any(r.state_name == "Hessen" for r in rows) and "btw_1953" in election_ids:
        official = load_hessen_1953(hessen_file)
        out.append(
            ExternalResult.of(
                "Hessen 1953 municipalities, all machine-agreed cells",
                compare_hessen_1953(session, run.run_id, official),
                max_mismatches,
            )
        )
        verified_rows = [r.result_id for r in rows if row_tier(blockers.get(r.result_id, set())) != "needs_review"]
        out.append(
            ExternalResult.of(
                "Hessen 1953 municipalities, data-verified rows only",
                compare_hessen_1953(session, run.run_id, official, row_ids=verified_rows),
                max_mismatches,
            )
        )
    return out


# ---------------------------------------------------------------- rendering
def to_json(results: List[DocumentValidation]) -> dict:
    return {"documents": [{**asdict(r), "agreed_share": round(r.agreed_share, 4)} for r in results]}


def _pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.1f} %" if d else "–"


def to_markdown(results: List[DocumentValidation]) -> str:
    lines = [
        "# Validation report",
        "",
        "Generated by `election-ocr report validation` from the latest completed run of every document.",
        "Nothing is *published* yet: the municipality gazetteer and the human verification of the",
        "reference data are still missing (see README, 'Status'). The numbers below measure the",
        "machine part of the pipeline before that step.",
        "",
        "## Extraction and internal consistency",
        "",
        "| Document | Pages | Numeric cells | Agreed (≥2 engines) | Rows | Reference-only | Data-verified | Needs review | Section sums exact |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        cells = sum(r.numeric_cells.values())
        lines.append(
            f"| {r.document} | {r.pages} | {cells:,} | {_pct(r.numeric_cells.get('AGREED', 0), cells)} | {r.rows:,} | "
            f"{r.tiers['reference_only']:,} | {r.tiers['data_verified']:,} ({_pct(r.tiers['data_verified'] + r.tiers['reference_only'], r.rows)} incl. ref.-only) | "
            f"{r.tiers['needs_review']:,} | {r.section_totals_verified}/{r.section_totals} |"
        )
    tot_rows = sum(r.rows for r in results)
    tot_ok = sum(r.tiers["reference_only"] + r.tiers["data_verified"] for r in results)
    tot_cells = sum(sum(r.numeric_cells.values()) for r in results)
    tot_agreed = sum(r.numeric_cells.get("AGREED", 0) for r in results)
    lines += [
        f"| **Total** | {sum(r.pages for r in results)} | {tot_cells:,} | {_pct(tot_agreed, tot_cells)} | {tot_rows:,} | "
        f"{sum(r.tiers['reference_only'] for r in results):,} | {tot_ok - sum(r.tiers['reference_only'] for r in results):,} ({_pct(tot_ok, tot_rows)} incl. ref.-only) | "
        f"{sum(r.tiers['needs_review'] for r in results):,} | {sum(r.section_totals_verified for r in results)}/{sum(r.section_totals for r in results)} |",
        "",
        "* **Reference-only**: blocked only by missing/unverified reference data - every field agreed by two",
        "  independent OCR engines, every arithmetic identity of the row executed and satisfied, section sum exact.",
        "* **Data-verified**: as above, but the section sum could not be checked because *other* rows of the",
        "  section are still unresolved.",
        "* **Section sums exact**: printed section/state totals whose members add up exactly (all fields).",
        "",
        "### Most frequent remaining blockers (rows, reference-data codes excluded)",
        "",
    ]
    for r in results:
        blockers = ", ".join(f"{k} {v:,}" for k, v in r.top_blockers.items()) or "–"
        lines.append(f"* **{r.document}**: {blockers}")
    externals = [(r, e) for r in results for e in r.external]
    lines += ["", "## External validation against independently digitised official data", ""]
    if not externals:
        lines.append("_No external data available (run `scripts/fetch_gpwed.sh`)._")
    else:
        lines += [
            "| Document | Comparison | Rows matched | Cells compared | Exactly equal |",
            "|---|---|---:|---:|---:|",
        ]
        for r, e in externals:
            lines.append(
                f"| {r.document} | {e.source} | {e.rows_matched:,}/{e.rows_considered:,} | {e.cells_compared:,} | "
                f"{e.cells_equal:,} ({_pct(e.cells_equal, e.cells_compared)}) |"
            )
        lines += ["", "### Per field", ""]
        for r, e in externals:
            per = ", ".join(f"{k} {v['equal']}/{v['compared']}" for k, v in e.per_field.items())
            lines.append(f"* {r.document} - {e.source}: {per}")
        lines += ["", "### Mismatches (first entries per comparison)", ""]
        for r, e in externals:
            if not e.mismatches:
                continue
            lines += [
                f"**{r.document} - {e.source}**",
                "",
                "| Page | Unit | Vote | Field | Pipeline | Official |",
                "|---:|---|---|---|---:|---:|",
            ]
            lines += [
                f"| {m['page']} | {m['unit']} | {m['vote_type']} | {m['field']} | {m['pipeline']} | {m['official']} |"
                for m in e.mismatches
            ]
            lines.append("")
    return "\n".join(lines) + "\n"


def write_report(results: List[DocumentValidation], out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md, js = out_dir / "validation_report.md", out_dir / "validation_report.json"
    md.write_text(to_markdown(results), encoding="utf-8")
    js.write_text(json.dumps(to_json(results), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return [md, js]
