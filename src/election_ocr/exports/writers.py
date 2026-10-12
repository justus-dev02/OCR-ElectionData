"""
Export writers.

* ``evidence_json``  - everything of one run: document, run metadata, pages,
                       tables (structure evidence), cells with ALL OCR
                       candidates, result rows with statuses and issues.
* ``canonical_json`` - published results only (current, not superseded).
* ``markdown``       - human-readable rendering of the published results.
* ``csv`` / ``parquet`` - flat published results (one line per party result).

Markdown and JSON are derived views; the database is the system of record and
the source PDF (by SHA-256) is the evidence.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from election_ocr.persistence.db import (
    Cell,
    IssueRow,
    OCRCandidateRow,
    Page,
    ProcessingRun,
    PublishedPartyResult,
    PublishedResult,
    ResultRow,
    SourceDocument,
    TableRow,
)
from election_ocr.reference.repository import ReferenceRepository


def _dump(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2, default=str)
    return path


def evidence_json(session: Session, run_id: str, out_path: Path) -> Path:
    """Everything one run stored, nested document -> pages -> tables -> cells -> candidates."""
    run = session.get(ProcessingRun, run_id)
    if run is None:
        raise ValueError(f"run {run_id} not found")
    doc = session.get(SourceDocument, run.document_id)
    if doc is None:  # pragma: no cover - referential integrity
        raise ValueError(f"document {run.document_id} of run {run_id} not found")
    pages = list(session.execute(select(Page).where(Page.run_id == run_id).order_by(Page.page_number)).scalars())
    tables = list(session.execute(select(TableRow).where(TableRow.run_id == run_id)).scalars())
    cells = list(session.execute(select(Cell).where(Cell.run_id == run_id)).scalars())
    cands = list(session.execute(select(OCRCandidateRow).where(OCRCandidateRow.run_id == run_id)).scalars())
    rows = list(session.execute(select(ResultRow).where(ResultRow.run_id == run_id)).scalars())
    issues = list(session.execute(select(IssueRow).where(IssueRow.run_id == run_id)).scalars())

    cands_by_cell: Dict[str, List[dict]] = defaultdict(list)
    for cand in cands:
        cands_by_cell[cand.cell_id].append(_candidate_json(cand))
    cells_by_table: Dict[str, List[dict]] = defaultdict(list)
    for c in cells:
        cells_by_table[c.table_id].append(_cell_json(c, cands_by_cell.get(c.cell_id, [])))
    issues_by_row: Dict[str, List[dict]] = defaultdict(list)
    for i in issues:
        issues_by_row[i.result_id or ""].append(_issue_json(i))
    data = {
        "document": _document_json(doc),
        "run": _run_json(run),
        "pages": [_page_json(p, [t for t in tables if t.page_id == p.page_id], cells_by_table) for p in pages],
        "results": [_result_json(r, issues_by_row.get(r.result_id, [])) for r in rows],
    }
    return _dump(out_path, data)


def _candidate_json(c: OCRCandidateRow) -> dict:
    return {
        "engine": c.engine,
        "engine_version": c.engine_version,
        "variant": c.variant,
        "independence_group": c.independence_group,
        "raw_text": c.raw_text,
        "confidence": c.confidence,
        "parse_status": c.parse_status,
        "parsed_value": c.parsed_value,
        "normalization_steps": c.normalization_steps,
        "used_confusion_map": c.used_confusion_map,
        "role": c.role,
        "note": c.note,
        "meta": c.meta,
    }


def _cell_json(c: Cell, candidates: List[dict]) -> dict:
    return {
        "cell_id": c.cell_id,
        "column_index": c.column_index,
        "column_key": c.column_key,
        "value_type": c.value_type,
        "line_index": c.line_index,
        "bbox_px": c.box,
        "bbox_pdf_pt": c.box_pdf,
        "assignment_status": c.assignment_status,
        "crosses_boundary": c.crosses_boundary,
        "status": c.status,
        "value": c.value,
        "supporting_groups": c.supporting_groups,
        "dissenting_groups": c.dissenting_groups,
        "labels": c.labels,
        "candidates": candidates,
    }


def _issue_json(i: IssueRow) -> dict:
    return {
        "code": i.code,
        "severity": i.severity,
        "scope": i.scope,
        "field": i.field,
        "message": i.message,
        "acknowledged": i.acknowledged,
        "details": i.details,
    }


def _document_json(doc: SourceDocument) -> dict:
    return {
        "document_id": doc.document_id,
        "sha256": doc.sha256,
        "filename": doc.filename,
        "page_count": doc.page_count,
        "text_layer_pages": doc.text_layer_pages,
        "pdf_metadata": doc.pdf_metadata,
        "renderer_version": doc.renderer_version,
    }


def _run_json(run: ProcessingRun) -> dict:
    return {
        "run_id": run.run_id,
        "status": run.status,
        "document_profile_id": run.document_profile_id,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "config_fingerprint": run.config_fingerprint,
        "reference_snapshot": run.reference_snapshot,
        "code_version": run.code_version,
        "engine_versions": run.engine_versions,
        "model_info": run.model_info,
        "stats": run.stats,
        "config": run.config_snapshot,
    }


def _page_json(p: Page, tables: List[TableRow], cells_by_table: Dict[str, List[dict]]) -> dict:
    return {
        "page_id": p.page_id,
        "page_number": p.page_number,
        "image_path": p.image_path,
        "image_sha256": p.image_sha256,
        "dpi": p.dpi,
        "deskew_angle": p.deskew_angle,
        "deskew_applied": p.deskew_applied,
        "has_text_layer": p.has_text_layer,
        "quality": p.quality,
        "issues": p.issues,
        "tables": [
            {
                "table_id": t.table_id,
                "table_profile_id": t.table_profile_id,
                "variant_id": t.variant_id,
                "detection_score": t.detection_score,
                "region_px": t.region,
                "columns_px": t.columns,
                "ml_columns_px": t.ml_columns,
                "column_agreement": t.column_agreement,
                "header_matches": t.header_matches,
                "section_headers": t.section_headers,
                "line_pitch": t.line_pitch,
                "issues": t.issues,
                "cells": cells_by_table.get(t.table_id, []),
            }
            for t in tables
        ],
    }


def _result_json(r: ResultRow, issues: List[dict]) -> dict:
    return {
        "result_id": r.result_id,
        "page_number": r.page_number,
        "row_kind": r.row_kind,
        "marker": r.marker,
        "vote_type": r.vote_type,
        "election_id": r.election_id,
        "publishable_role": r.publishable_role,
        "status": r.status,
        "unit_id": r.unit_id,
        "unit_name": r.unit_name,
        "unit_level": r.unit_level,
        "unit_resolution": r.unit_resolution,
        "section_key": r.section_key,
        "label_readings": r.label_readings,
        "field_cells": r.field_cells,
        "party_cells": r.party_cells,
        "issues": issues,
    }


def _published(session: Session, election_id: Optional[str] = None) -> List[dict]:
    q = (
        select(PublishedResult)
        .where(PublishedResult.superseded_at.is_(None))
        .order_by(PublishedResult.election_id, PublishedResult.unit_level, PublishedResult.unit_name, PublishedResult.vote_type)
    )
    if election_id:
        q = q.where(PublishedResult.election_id == election_id)
    published = list(session.execute(q).scalars())
    parties_by_result: Dict[int, List[PublishedPartyResult]] = defaultdict(list)
    if published:
        ids = [p.published_id for p in published]
        party_query = (
            select(PublishedPartyResult).where(PublishedPartyResult.published_id.in_(ids)).order_by(PublishedPartyResult.id)
        )
        for x in session.execute(party_query).scalars():
            parties_by_result[x.published_id].append(x)
    out = []
    for p in published:
        parties = parties_by_result.get(p.published_id, [])
        out.append(
            {
                "published_id": p.published_id,
                "election_id": p.election_id,
                "unit_id": p.unit_id,
                "unit_name": p.unit_name,
                "unit_level": p.unit_level,
                "vote_type": p.vote_type,
                "eligible_voters": p.eligible_voters,
                "voters": p.voters,
                "valid_votes": p.valid_votes,
                "invalid_votes": p.invalid_votes,
                "turnout_percent": p.turnout_percent,
                "party_results": [
                    {"party_id": x.party_id, "votes": x.votes, "source": x.source, "cell_id": x.cell_id} for x in parties
                ],
                "provenance": p.provenance,
                "published_at": p.published_at,
            }
        )
    return out


def canonical_json(session: Session, out_path: Path, election_id: Optional[str] = None) -> Path:
    return _dump(out_path, {"published_results": _published(session, election_id)})


def markdown(session: Session, reference: ReferenceRepository, out_path: Path, election_id: Optional[str] = None) -> Path:
    results = _published(session, election_id)
    lines: List[str] = []
    by_election: Dict[str, List[dict]] = defaultdict(list)
    for r in results:
        by_election[r["election_id"]].append(r)
    if not results:
        lines.append("# Published election results\n\n_No published results._\n")
    for eid, rows in by_election.items():
        e = reference.elections.get(eid)
        lines.append(f"# {e.name if e else eid} ({e.election_date if e else ''})\n")
        party_ids = sorted({p["party_id"] for r in rows for p in r["party_results"]})
        names = [reference.parties[p].canonical_name if p in reference.parties else p for p in party_ids]
        by_vote: Dict[str, List[dict]] = defaultdict(list)
        for r in rows:
            by_vote[r["vote_type"]].append(r)
        for vt, vrows in by_vote.items():
            lines.append(f"## {vt}\n")
            header = ["Unit", "Level", "Eligible", "Voters", "Valid", "Invalid", "Turnout %", *names, "Source"]
            lines.append("| " + " | ".join(header) + " |")
            lines.append("|" + "|".join(["---"] * 2 + ["---:"] * (len(header) - 3) + ["---"]) + "|")
            for r in vrows:
                pv = {p["party_id"]: p["votes"] for p in r["party_results"]}
                src = f"p. {r['provenance'].get('page_number')}"
                cells = (
                    [
                        r["unit_name"],
                        r["unit_level"],
                        _fmt(r["eligible_voters"]),
                        _fmt(r["voters"]),
                        _fmt(r["valid_votes"]),
                        _fmt(r["invalid_votes"]),
                        r["turnout_percent"] or "",
                    ]
                    + [_fmt(pv.get(p)) for p in party_ids]
                    + [src]
                )
                lines.append("| " + " | ".join(str(c) for c in cells) + " |")
            lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _fmt(v) -> str:
    return "" if v is None else str(v)


def flat_records(session: Session, election_id: Optional[str] = None) -> List[dict]:
    rows = []
    for r in _published(session, election_id):
        base = {
            k: r[k]
            for k in (
                "election_id",
                "unit_id",
                "unit_name",
                "unit_level",
                "vote_type",
                "eligible_voters",
                "voters",
                "valid_votes",
                "invalid_votes",
                "turnout_percent",
            )
        }
        base["source_page"] = r["provenance"].get("page_number")
        base["result_id"] = r["provenance"].get("result_key")
        if not r["party_results"]:
            rows.append({**base, "party_id": None, "votes": None})
        for p in r["party_results"]:
            rows.append({**base, "party_id": p["party_id"], "votes": p["votes"], "value_source": p["source"]})
    return rows


def csv_export(session: Session, out_path: Path, election_id: Optional[str] = None) -> Path:
    import pandas as pd

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(flat_records(session, election_id)).to_csv(out_path, index=False, encoding="utf-8")
    return out_path


def parquet_export(session: Session, out_path: Path, election_id: Optional[str] = None) -> Path:
    import pandas as pd

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(flat_records(session, election_id)).to_parquet(out_path, index=False)
    return out_path


def review_report(session: Session, out_path: Path, run_id: Optional[str] = None) -> Path:
    """Markdown overview of open review cases grouped by issue code."""
    from election_ocr.persistence.db import ReviewCase

    q = select(ReviewCase).where(ReviewCase.status == "OPEN")
    if run_id:
        q = q.where(ReviewCase.run_id == run_id)
    cases = list(session.execute(q).scalars())
    counts: Dict[str, int] = defaultdict(int)
    for c in cases:
        for code in c.codes or []:
            counts[code] += 1
    lines = [f"# Open review cases ({len(cases)})\n", "| Issue code | Cases |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
