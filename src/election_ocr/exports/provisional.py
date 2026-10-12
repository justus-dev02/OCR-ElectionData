"""
Provisional export: machine-verified rows that are NOT published.

Publication needs a resolved municipality (gazetteer) and human-verified reference
data. Until then, rows of the validation tiers ``reference_only`` / ``data_verified``
(see ``evaluation.validation_report``) can be exported for inspection and for
external comparisons. Every line carries ``status = UNPUBLISHED`` and its tier;
unit names are the OCR readings (agreed by >= 2 engines where possible), not
gazetteer entries. Values are built exactly like the publication gate builds them
(``DocumentService.build_values`` + blank-zero proof).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from election_ocr.config import PipelineConfig
from election_ocr.evaluation.validation_report import latest_runs, row_tiers
from election_ocr.issues import IssueFactory
from election_ocr.parsing.text import exact_key
from election_ocr.persistence.db import ResultRow, SourceDocument
from election_ocr.persistence.repository import Repository
from election_ocr.pipeline.document_service import DocumentService
from election_ocr.reference.repository import ReferenceRepository
from election_ocr.validation.rules import apply_blank_zero

_TIER_SETS = {"reference_only": {"reference_only"}, "data_verified": {"reference_only", "data_verified"}}
_METRICS = ("eligible_voters", "voters", "invalid_votes", "valid_votes", "turnout_percent")


def agreed_text(readings: Dict[str, str], min_sources: int = 2) -> Optional[str]:
    """The reading all engine groups agree on (normalised), if at least ``min_sources`` read it."""
    texts = [t for t in readings.values() if t]
    if len(texts) >= min_sources and len({exact_key(t) for t in texts}) == 1:
        return texts[0]
    return None


def provisional_csv(
    session: Session, reference: ReferenceRepository, cfg: PipelineConfig, out_dir: Path, tier: str
) -> List[Path]:
    allowed = _TIER_SETS[tier]
    repo = Repository(session)
    service = DocumentService(repo, reference, cfg, IssueFactory(cfg.issue_severity_overrides))
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for run in latest_runs(session):
        tiers = row_tiers(session, run.run_id)
        profile = service.profile_of_run(run.run_id)
        tables = repo.tables(run.run_id)
        cells = repo.cells(run.run_id)
        records = []
        rows = session.execute(
            select(ResultRow).where(ResultRow.run_id == run.run_id).order_by(ResultRow.page_number, ResultRow.result_key)
        ).scalars()
        for r in rows:
            if tiers.get(r.result_id) not in allowed or r.role_kind != "absolute":
                continue
            t = tables[r.table_id]
            tp, variant = service.table_spec(profile, t.table_profile_id, t.variant_id)
            v = service.build_values(r, cells, tp, variant, {}, {})
            apply_blank_zero(v, service.issues)
            context = r.context or {}
            unit_readings = context.get("_unit_name_readings") or {}
            section_readings = context.get("_section_name_readings") or {}
            record = {
                "status": "UNPUBLISHED",
                "validation_tier": tiers[r.result_id],
                "election_id": r.election_id,
                "state": r.state_name,
                "vote_type": r.vote_type,
                "row_kind": r.row_kind,
                "unit_level": r.unit_level,
                "section": r.section_name or agreed_text(section_readings) or "",
                "unit_name": agreed_text(unit_readings) or "",
                "unit_name_readings": " | ".join(f"{g}: {t}" for g, t in sorted(unit_readings.items())),
                "unit_index": context.get("_unit_index") or "",
                "source_page": r.page_number,
                **{m: _num(v.value(m)) for m in _METRICS if m in v.metrics},
                **{f"party_{p}": _num(fv.value) for p, fv in v.parties.items()},
                "run_id": run.run_id,
                "result_id": r.result_id,
            }
            records.append(record)
        if not records:
            continue
        doc = session.get(SourceDocument, run.document_id)
        stem = Path(doc.filename).stem if doc else run.document_id
        path = out_dir / f"provisional_{stem}.csv"
        pd.DataFrame(records).to_csv(path, index=False, encoding="utf-8")
        written.append(path)
    return written


def _num(v) -> Optional[str]:
    return None if v is None else str(v)
