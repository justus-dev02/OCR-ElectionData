"""
Human-in-the-loop review.

A reviewer works on review cases (one per blocked, publishable result row, or
one per table that produced no rows). Every action is stored as an immutable
``ReviewDecision``; human values live in ``field_overrides`` and never
overwrite OCR evidence. After decisions, ``approve`` re-validates the whole
run (hierarchical sums depend on other rows) and publishes what passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from sqlalchemy import select

from election_ocr.config import PipelineConfig
from election_ocr.issues import IssueCode, IssueFactory
from election_ocr.parsing.numbers import DecimalFormat, IntegerFormat, ParseStatus, parse_decimal, parse_integer
from election_ocr.persistence.db import (
    Acknowledgement,
    FieldOverride,
    IssueRow,
    ResultRow,
    ReviewCase,
    ReviewDecision,
)
from election_ocr.persistence.repository import Repository
from election_ocr.pipeline.document_service import DocumentService
from election_ocr.pipeline.state_machine import ResultStatus
from election_ocr.reference.models import NumberFormat
from election_ocr.reference.repository import ReferenceRepository


class ReviewError(ValueError):
    pass


@dataclass
class FieldEvidence:
    field: str
    cell_id: Optional[str]
    status: Optional[str]
    value: Optional[str]
    override: Optional[str]
    candidates: List[dict] = dataclass_field(default_factory=list)
    crop_path: Optional[str] = None


@dataclass
class CaseEvidence:
    case: ReviewCase
    row: Optional[ResultRow]
    issues: List[IssueRow]
    fields: List[FieldEvidence]
    label_readings: List[dict]
    unit_resolution: Optional[dict]
    row_crop_path: Optional[str]
    arithmetic_hints: List[str]


class ReviewService:
    def __init__(self, repo: Repository, reference: ReferenceRepository, config: PipelineConfig):
        self.repo = repo
        self.ref = reference
        self.cfg = config
        self.issue_factory = IssueFactory(config.issue_severity_overrides)
        self.docs = DocumentService(repo, reference, config, self.issue_factory)

    # ------------------------------------------------------------------ queries
    def list_cases(self, status: str = "OPEN", code: Optional[str] = None, run_id: Optional[str] = None) -> List[ReviewCase]:
        q = select(ReviewCase).where(ReviewCase.status == status).order_by(ReviewCase.case_id)
        if run_id:
            q = q.where(ReviewCase.run_id == run_id)
        cases = list(self.repo.s.execute(q).scalars())
        if code:
            cases = [c for c in cases if code in (c.codes or [])]
        return cases

    def _case(self, case_id: int) -> ReviewCase:
        case = self.repo.s.get(ReviewCase, case_id)
        if case is None:
            raise ReviewError(f"review case {case_id} not found")
        return case

    def _row(self, case: ReviewCase) -> ResultRow:
        if case.kind != "ROW" or not case.result_id:
            raise ReviewError(f"case {case.case_id} is a {case.kind} case; use 'review table' actions")
        row = self.repo.row(case.result_id)
        if row is None:  # pragma: no cover - referential integrity
            raise ReviewError("result row missing")
        return row

    def evidence(self, case_id: int, export_crops: bool = True) -> CaseEvidence:
        case = self._case(case_id)
        out_dir = self.cfg.paths.review_dir / f"case_{case_id:06d}"
        if case.kind != "ROW":
            t = self.repo.tables(case.run_id)[case.table_id or ""]
            page = self.repo.pages(case.run_id)[t.page_id]
            crop = None
            if export_crops:
                out_dir.mkdir(parents=True, exist_ok=True)
                crop = str(self._crop(page.image_path, t.region, out_dir / "table.png", margin=10))
            return CaseEvidence(case, None, [], [], [], None, crop, [])
        row = self._row(case)
        cells = self.repo.cells(row.run_id)
        overrides = self.repo.overrides(row.run_id).get(row.result_id, {})
        page = next(p for p in self.repo.pages(row.run_id).values() if p.page_number == row.page_number)
        fields: List[FieldEvidence] = []
        items = [(f"metric:{k}", v) for k, v in row.field_cells.items()] + [(f"party:{k}", v) for k, v in row.party_cells.items()]
        if export_crops:
            out_dir.mkdir(parents=True, exist_ok=True)
        for key, cid in items:
            c = cells.get(cid) if cid else None
            cands = []
            if c is not None:
                for cand in self.repo.candidates(c.cell_id):
                    cands.append(
                        {
                            "engine": cand.engine,
                            "variant": cand.variant,
                            "raw": cand.raw_text,
                            "parsed": cand.parsed_value,
                            "confidence": cand.confidence,
                            "role": cand.role,
                            "steps": cand.normalization_steps,
                        }
                    )
            crop = None
            if export_crops and c is not None:
                crop = str(self._crop(page.image_path, c.box, out_dir / f"{key.replace(':', '_')}.png", margin=12))
            fields.append(
                FieldEvidence(
                    field=key,
                    cell_id=cid,
                    status=c.status if c else "ABSENT",
                    value=c.value if c else None,
                    override=overrides[key].value if key in overrides else None,
                    candidates=cands,
                    crop_path=crop,
                )
            )
        row_crop = None
        if export_crops:
            t = self.repo.tables(row.run_id)[row.table_id]
            boxes = [
                cells[cid].box
                for cid in list(row.field_cells.values()) + list(row.party_cells.values()) + list(row.label_cell_ids or [])
                if cid and cid in cells
            ]
            if boxes:
                y0 = min(b[1] for b in boxes)
                y1 = max(b[3] for b in boxes)
                row_crop = str(self._crop(page.image_path, [t.region[0], y0, t.region[2], y1], out_dir / "row.png", margin=25))
        return CaseEvidence(
            case=case,
            row=row,
            issues=self.repo.issues_for(row.result_id),
            fields=fields,
            label_readings=row.label_readings or [],
            unit_resolution=row.unit_resolution,
            row_crop_path=row_crop,
            arithmetic_hints=self._hints(row, fields),
        )

    @staticmethod
    def _crop(image_path: str, box, out: Path, margin: int) -> Path:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ReviewError(f"page image missing: {image_path}")
        h, w = img.shape
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        crop = img[max(0, y0 - margin) : min(h, y1 + margin), max(0, x0 - margin) : min(w, x1 + margin)]
        cv2.imwrite(str(out), crop)
        return out

    def _hints(self, row: ResultRow, fields: List[FieldEvidence]) -> List[str]:
        """Show which single candidate value would make 'sum of parties == valid votes' hold.

        Hints only - the resolver never uses them: every dissent goes to review (docs/decisions.md, D1).
        """
        vals: Dict[str, Optional[Decimal]] = {}
        alts: Dict[str, List[str]] = {}
        for f in fields:
            v = f.override or (f.value if f.status in ("AGREED",) else None)
            vals[f.field] = Decimal(v) if v is not None else None
            alts[f.field] = sorted({c["parsed"] for c in f.candidates if c.get("parsed") is not None})
        valid = vals.get("metric:valid_votes")
        parties = [k for k in vals if k.startswith("party:")]
        unknown = [k for k in parties if vals[k] is None]
        hints: List[str] = []
        if valid is not None and len(unknown) == 1:
            known = sum((x for k in parties if (x := vals[k]) is not None), Decimal(0))
            needed = valid - known
            for alt in alts.get(unknown[0], []):
                ok = Decimal(alt) == needed
                hints.append(
                    f"{unknown[0]} = {alt} {'would satisfy' if ok else 'would NOT satisfy'} party sum = valid ({needed} needed)"
                )
        return hints

    # ------------------------------------------------------------------ actions
    def _decision(self, case: ReviewCase, row_id: str, action: str, target, previous, new, reviewer: str, reason: str) -> None:
        if not reason or not reason.strip():
            raise ReviewError("a reason is mandatory for every review decision")
        self.repo.s.add(
            ReviewDecision(
                case_id=case.case_id,
                result_id=row_id,
                action=action,
                target=target,
                previous=previous,
                new=new,
                reviewer=reviewer,
                reason=reason,
            )
        )

    def _mark_reviewed(self, row: ResultRow, reviewer: str) -> None:
        if ResultStatus(row.status) == ResultStatus.NEEDS_REVIEW:
            self.repo.set_status(row, ResultStatus.REVIEWED, reviewer, reason="human decision recorded")

    def set_value(self, case_id: int, field_key: str, raw_value: str, reviewer: str, reason: str) -> None:
        case = self._case(case_id)
        row = self._row(case)
        if field_key not in {f"metric:{k}" for k in row.field_cells} | {f"party:{k}" for k in row.party_cells}:
            raise ReviewError(f"unknown field {field_key}; expected metric:<name> or party:<id>")
        fmt = self._field_format(row, field_key)
        if fmt.value_type == "integer":
            pr = parse_integer(raw_value, fmt.integer_format or IntegerFormat())
        else:
            pr = parse_decimal(raw_value, fmt.decimal_format or DecimalFormat())
        name = field_key.split(":", 1)[1]
        if pr.status not in (ParseStatus.OK, ParseStatus.DASH) or pr.value is None or pr.used_confusion_map:
            raise ReviewError(f"'{raw_value}' is not a valid {fmt.value_type} for {field_key}: {pr.reason}")
        existing = (
            self.repo.s.execute(
                select(FieldOverride).where(FieldOverride.result_id == row.result_id, FieldOverride.field == field_key)
            )
            .scalars()
            .first()
        )
        previous = {"override": existing.value} if existing else {"cell_id": (row.field_cells | row.party_cells).get(name)}
        if existing:
            existing.value, existing.decimals, existing.reviewer, existing.reason = pr.value, pr.decimals, reviewer, reason
        else:
            self.repo.s.add(
                FieldOverride(
                    result_id=row.result_id,
                    field=field_key,
                    value=pr.value,
                    decimals=pr.decimals,
                    reviewer=reviewer,
                    reason=reason,
                )
            )
        self._decision(case, row.result_id, "SET_VALUE", field_key, previous, {"value": pr.value}, reviewer, reason)
        self._mark_reviewed(row, reviewer)
        self.repo.commit()

    def _field_format(self, row: ResultRow, field_key: str) -> NumberFormat:
        """Number format of the field on THIS row: the profile binding of the row's role (column + slot),
        so percent/difference rows and continuation-page columns are parsed like the pipeline parsed them."""
        profile = self.docs.profile_of_run(row.run_id)
        t = self.repo.tables(row.run_id)[row.table_id]
        tp, variant = self.docs.table_spec(profile, t.table_profile_id, t.variant_id)
        bound_field = field_key[len("metric:") :] if field_key.startswith("metric:") else field_key
        binding = next((b for b in tp.bindings(variant).get(self.docs.role_id_of(row), []) if b.field == bound_field), None)
        if binding is None:
            raise ReviewError(f"field {field_key} is not bound to a column for role {self.docs.role_id_of(row)}")
        return binding.number_format

    def set_unit(self, case_id: int, unit_id: str, reviewer: str, reason: str) -> None:
        case = self._case(case_id)
        row = self._row(case)
        unit = self.ref.units.get(unit_id)
        if unit is None:
            raise ReviewError(f"unit {unit_id} not in gazetteer - add it to reference/administrative_units first")
        election = self.ref.elections[row.election_id]
        if not unit.valid_on(election.election_date):
            raise ReviewError(f"unit {unit_id} is not valid on {election.election_date}")
        if unit.level.upper() != row.unit_level.upper():
            raise ReviewError(f"unit level {unit.level} does not match row level {row.unit_level}")
        existing = (
            self.repo.s.execute(
                select(FieldOverride).where(FieldOverride.result_id == row.result_id, FieldOverride.field == "unit")
            )
            .scalars()
            .first()
        )
        if existing:
            existing.value, existing.reviewer, existing.reason = unit_id, reviewer, reason
        else:
            self.repo.s.add(FieldOverride(result_id=row.result_id, field="unit", value=unit_id, reviewer=reviewer, reason=reason))
        self._decision(case, row.result_id, "SET_UNIT", "unit", {"unit_id": row.unit_id}, {"unit_id": unit_id}, reviewer, reason)
        self._mark_reviewed(row, reviewer)
        self.repo.commit()

    def acknowledge(self, case_id: int, code: str, reviewer: str, reason: str) -> None:
        case = self._case(case_id)
        row = self._row(case)
        IssueCode(code)  # validates the code
        self._ack(row, code, reviewer, reason)
        self._decision(case, row.result_id, "ACK", code, None, None, reviewer, reason)
        self._mark_reviewed(row, reviewer)
        self.repo.commit()

    def acknowledge_table(self, table_id: str, code: str, reviewer: str, reason: str) -> int:
        IssueCode(code)
        rows = [r for r in self.repo.s.execute(select(ResultRow).where(ResultRow.table_id == table_id)).scalars()]
        if not rows:
            raise ReviewError(f"no result rows for table {table_id}")
        for row in rows:
            self._ack(row, code, reviewer, reason)
            case = self.repo.case_for(row.result_id)
            if case is not None:
                self._decision(case, row.result_id, "ACK", code, None, {"table_id": table_id}, reviewer, reason)
            self._mark_reviewed(row, reviewer)
        self.repo.commit()
        return len(rows)

    def _ack(self, row: ResultRow, code: str, reviewer: str, reason: str) -> None:
        if not reason.strip():
            raise ReviewError("a reason is mandatory")
        exists = (
            self.repo.s.execute(
                select(Acknowledgement).where(Acknowledgement.result_id == row.result_id, Acknowledgement.code == code)
            )
            .scalars()
            .first()
        )
        if not exists:
            self.repo.s.add(Acknowledgement(result_id=row.result_id, code=code, reviewer=reviewer, reason=reason))

    def approve(self, case_id: int, reviewer: str, reason: str) -> dict:
        case = self._case(case_id)
        row = self._row(case)
        self._decision(case, row.result_id, "APPROVE", None, {"status": row.status}, None, reviewer, reason)
        self._mark_reviewed(row, reviewer)
        self.repo.commit()
        summary = self.docs.validate_run(row.run_id, actor=reviewer)
        published = self.docs.publish_run(row.run_id, actor=reviewer)
        self.repo.s.refresh(row)
        remaining = [i.code for i in self.repo.issues_for(row.result_id) if i.severity == "BLOCKING" and not i.acknowledged]
        return {
            "status": row.status,
            "remaining_blocking": sorted(set(remaining)),
            "published_in_run": published,
            "summary": summary.__dict__,
        }

    def reject(self, case_id: int, reviewer: str, reason: str) -> None:
        case = self._case(case_id)
        row = self._row(case)
        self._decision(case, row.result_id, "REJECT", None, {"status": row.status}, None, reviewer, reason)
        self.repo.set_status(row, ResultStatus.REJECTED, reviewer, reason=reason)
        case.status = "REJECTED"
        self.repo.commit()
