"""
Gold-standard evaluation.

Gold files (``gold/<document_id>/page_NNNN.yaml``) are manual, independent
transcriptions of source pages. A file without ``verified_by`` is rejected -
unverified data must never serve as ground truth.

Key metrics (per field value):
* agreed_correct   - cell AGREED and equal to gold
* agreed_wrong     - cell AGREED but different from gold  (false accept - must be 0)
* abstained        - not AGREED (routed to review)
* published_wrong  - published value differs from gold     (the KPI that matters most)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from election_ocr.persistence.db import Cell, FieldOverride, PublishedPartyResult, PublishedResult, ResultRow
from election_ocr.reference.repository import ReferenceRepository


class GoldRow(BaseModel):
    label_index: Optional[str] = None
    row_kind: str = "DATA"
    marker: Optional[str] = None
    values: Dict[str, Optional[str]] = Field(description="metric:<name> / party:<id> -> printed value ('-' allowed)")

    @field_validator("values", mode="before")
    @classmethod
    def _stringify(cls, v):
        return {k: (None if x is None else str(x)) for k, x in (v or {}).items()}


class GoldPage(BaseModel):
    document_sha256: str
    page: int
    verified_by: str = Field(min_length=1)
    verified_at: str
    notes: Optional[str] = None
    rows: List[GoldRow]


@dataclass
class EvaluationReport:
    pages: int = 0
    gold_rows: int = 0
    matched_rows: int = 0
    unmatched_rows: List[str] = field(default_factory=list)
    fields: int = 0
    agreed_correct: int = 0
    agreed_wrong: int = 0
    abstained: int = 0
    published_fields: int = 0
    published_wrong: int = 0
    wrong_details: List[dict] = field(default_factory=list)

    @property
    def false_accept_rate(self) -> float:
        agreed = self.agreed_correct + self.agreed_wrong
        return self.agreed_wrong / agreed if agreed else 0.0

    @property
    def automation_rate(self) -> float:
        return (self.agreed_correct + self.agreed_wrong) / self.fields if self.fields else 0.0


def load_gold(gold_dir: Path, document_id: str, sha256: str) -> List[GoldPage]:
    pages = []
    for p in sorted((gold_dir / document_id).glob("page_*.yaml")):
        gp = GoldPage(**yaml.safe_load(p.read_text(encoding="utf-8")))
        if gp.document_sha256 != sha256:
            raise ValueError(f"{p}: gold file belongs to another document")
        pages.append(gp)
    return pages


def _gold_value(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    s = v.strip().replace(" ", "")
    if s in {"-", "–", "—"}:
        return "0"
    return s.replace(",", ".")


def evaluate_run(session: Session, run_id: str, gold_pages: List[GoldPage], reference: ReferenceRepository) -> EvaluationReport:
    rep = EvaluationReport(pages=len(gold_pages))
    rows = list(session.execute(select(ResultRow).where(ResultRow.run_id == run_id)).scalars())
    cells = {c.cell_id: c for c in session.execute(select(Cell).where(Cell.run_id == run_id)).scalars()}
    overrides: Dict[str, Dict[str, str]] = {}
    for o in session.execute(select(FieldOverride).where(FieldOverride.result_id.like(f"{run_id}:%"))).scalars():
        overrides.setdefault(o.result_id, {})[o.field] = o.value
    published: Dict[str, Dict[str, str]] = {}
    for p in session.execute(select(PublishedResult).where(PublishedResult.run_id == run_id)).scalars():
        vals = {
            f"metric:{k}": getattr(p, k) for k in ("eligible_voters", "voters", "valid_votes", "invalid_votes", "turnout_percent")
        }
        for pr in session.execute(
            select(PublishedPartyResult).where(PublishedPartyResult.published_id == p.published_id)
        ).scalars():
            vals[f"party:{pr.party_id}"] = pr.votes
        published[p.result_id] = {k: str(v) for k, v in vals.items() if v is not None}

    idx_rx = re.compile(r"^\s*(\d+)\s")
    for gp in gold_pages:
        page_rows = [r for r in rows if r.page_number == gp.page]
        totals_seen: Dict[Optional[str], int] = {}
        for g in gp.rows:
            rep.gold_rows += 1
            cand = [r for r in page_rows if r.row_kind == g.row_kind and r.marker == g.marker]
            if g.row_kind == "DATA":
                cand = [
                    r
                    for r in cand
                    if any(
                        (m := idx_rx.match(lr.get("text", ""))) and m.group(1) == g.label_index for lr in (r.label_readings or [])
                    )
                ]
            else:
                k = totals_seen.get(g.marker, 0)
                totals_seen[g.marker] = k + 1
                cand = sorted(cand, key=lambda r: r.group_index)[k : k + 1]
            key = f"p{gp.page}:{g.row_kind}:{g.label_index}:{g.marker}"
            if len(cand) != 1:
                rep.unmatched_rows.append(key)
                continue
            r = cand[0]
            rep.matched_rows += 1
            cell_ids = {f"metric:{k}": v for k, v in r.field_cells.items()} | {f"party:{k}": v for k, v in r.party_cells.items()}
            for fname, gval in g.values.items():
                gold = _gold_value(gval)
                if gold is None:
                    continue
                rep.fields += 1
                ov = overrides.get(r.result_id, {}).get(fname)
                c = cells.get(cell_ids.get(fname) or "")
                agreed_value = ov if ov is not None else (c.value if c is not None and c.status == "AGREED" else None)
                if agreed_value is None:
                    rep.abstained += 1
                elif agreed_value == gold:
                    rep.agreed_correct += 1
                else:
                    rep.agreed_wrong += 1
                    rep.wrong_details.append(
                        {"row": key, "field": fname, "gold": gold, "extracted": agreed_value, "cell_id": c.cell_id if c else None}
                    )
                pub = published.get(r.result_id, {}).get(fname)
                if pub is not None:
                    rep.published_fields += 1
                    if pub != gold:
                        rep.published_wrong += 1
    return rep


def gold_template(session: Session, run_id: str, page: int, sha256: str) -> dict:
    """Blank gold skeleton for one page (no values prefilled, to avoid anchoring bias)."""
    rows = sorted(
        (
            r
            for r in session.execute(select(ResultRow).where(ResultRow.run_id == run_id, ResultRow.page_number == page)).scalars()
        ),
        key=lambda r: (r.group_index, r.marker or ""),
    )
    out_rows = []
    for r in rows:
        keys = [f"metric:{k}" for k in r.field_cells] + [f"party:{k}" for k in r.party_cells]
        out_rows.append(
            {
                "label_index": None if r.row_kind != "DATA" else "",
                "row_kind": r.row_kind,
                "marker": r.marker,
                "values": {k: None for k in keys},
            }
        )
    return {"document_sha256": sha256, "page": page, "verified_by": "", "verified_at": "", "notes": "", "rows": out_rows}
