"""
Repository: all database access of the pipeline, review and export layers.

Evidence rows are append-only per run. Status changes go through the state
machine and are logged in ``state_transitions``.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Dict, Iterable, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from election_ocr import __version__
from election_ocr.issues import Issue
from election_ocr.persistence.db import (
    Acknowledgement,
    Cell,
    FieldOverride,
    IssueRow,
    OCRCandidateRow,
    Page,
    ProcessingRun,
    ResultRow,
    ReviewCase,
    StateTransitionRow,
    TableRow,
)
from election_ocr.pipeline.models import CellRecord, PageRecord, RowRecord, TableRecord
from election_ocr.pipeline.state_machine import (
    ResultStatus,
    RunStatus,
    result_transition,
    run_transition,
)


def new_run_id() -> str:
    return f"run_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}"


def _issue_json(issues: Iterable[Issue]) -> List[dict]:
    return [i.model_dump(mode="json") for i in issues]


class Repository:
    def __init__(self, session: Session):
        self.s = session

    # --------------------------------------------------------------- runs
    def create_run(
        self,
        document_id: str,
        document_profile_id: Optional[str],
        config_snapshot: dict,
        config_fingerprint: str,
        reference_snapshot: str,
        pages_requested: Optional[List[int]],
    ) -> ProcessingRun:
        run = ProcessingRun(
            run_id=new_run_id(),
            document_id=document_id,
            document_profile_id=document_profile_id,
            status=RunStatus.CREATED.value,
            pages_requested=pages_requested,
            config_snapshot=config_snapshot,
            config_fingerprint=config_fingerprint,
            reference_snapshot=reference_snapshot,
            code_version=__version__,
        )
        self.s.add(run)
        self.s.commit()
        return run

    def set_run_status(self, run: ProcessingRun, target: RunStatus, error: Optional[str] = None) -> None:
        run.status = run_transition(RunStatus(run.status), target).value
        if target in (RunStatus.COMPLETED, RunStatus.FAILED):
            run.finished_at = datetime.now(UTC)
        if error:
            run.error = error
        self.s.add(StateTransitionRow(entity_type="run", entity_id=run.run_id, to_state=target.value, actor="pipeline"))
        self.s.commit()

    def get_run(self, run_id: str) -> Optional[ProcessingRun]:
        return self.s.get(ProcessingRun, run_id)

    def latest_run(self, document_id: Optional[str] = None) -> Optional[ProcessingRun]:
        q = select(ProcessingRun).order_by(ProcessingRun.started_at.desc())
        if document_id:
            q = q.where(ProcessingRun.document_id == document_id)
        return self.s.execute(q).scalars().first()

    # --------------------------------------------------------------- evidence
    def save_page(self, run_id: str, document_id: str, page: PageRecord) -> None:
        """Persist one page with its tables, cells, OCR candidates and candidate result rows."""
        self.s.add(_page_row(run_id, page))
        self.s.flush()  # explicit FK order: page -> tables -> cells -> candidates -> rows
        for t in page.tables:
            self.s.add(_table_row(run_id, page.page_id, t))
            self.s.flush()
            self.s.add_all(_cell_row(run_id, t.table_id, c) for c in t.cells)
            self.s.flush()
            self.s.add_all(cand for c in t.cells for cand in _candidate_rows(run_id, c))
            self.s.flush()
            for row in t.rows:
                rid = f"{run_id}:{row.result_key}"
                self.s.add(_result_row(run_id, document_id, rid, row))
                self.s.add(
                    StateTransitionRow(
                        entity_type="result", entity_id=rid, to_state=ResultStatus.EXTRACTED.value, actor="pipeline"
                    )
                )
        self.s.commit()

    # --------------------------------------------------------------- queries
    def rows(self, run_id: str) -> List[ResultRow]:
        return list(
            self.s.execute(
                select(ResultRow).where(ResultRow.run_id == run_id).order_by(ResultRow.page_number, ResultRow.result_key)
            ).scalars()
        )

    def row(self, result_id: str) -> Optional[ResultRow]:
        return self.s.get(ResultRow, result_id)

    def cells(self, run_id: str) -> Dict[str, Cell]:
        return {c.cell_id: c for c in self.s.execute(select(Cell).where(Cell.run_id == run_id)).scalars()}

    def candidates(self, cell_id: str) -> List[OCRCandidateRow]:
        return list(self.s.execute(select(OCRCandidateRow).where(OCRCandidateRow.cell_id == cell_id)).scalars())

    def tables(self, run_id: str) -> Dict[str, TableRow]:
        return {t.table_id: t for t in self.s.execute(select(TableRow).where(TableRow.run_id == run_id)).scalars()}

    def pages(self, run_id: str) -> Dict[str, Page]:
        return {p.page_id: p for p in self.s.execute(select(Page).where(Page.run_id == run_id)).scalars()}

    def overrides(self, run_id: str) -> Dict[str, Dict[str, FieldOverride]]:
        out: Dict[str, Dict[str, FieldOverride]] = {}
        q = select(FieldOverride).where(FieldOverride.result_id.like(f"{run_id}:%"))
        for o in self.s.execute(q).scalars():
            out.setdefault(o.result_id, {})[o.field] = o
        return out

    def acknowledgements(self, run_id: str) -> Dict[str, Dict[str, Acknowledgement]]:
        out: Dict[str, Dict[str, Acknowledgement]] = {}
        q = select(Acknowledgement).where(Acknowledgement.result_id.like(f"{run_id}:%"))
        for a in self.s.execute(q).scalars():
            out.setdefault(a.result_id, {})[a.code] = a
        return out

    def issues_for(self, result_id: str) -> List[IssueRow]:
        return list(self.s.execute(select(IssueRow).where(IssueRow.result_id == result_id)).scalars())

    def replace_issues(self, run_id: str, result_id: str, issues: List[Issue], acked_codes: Iterable[str]) -> None:
        acked = set(acked_codes)
        for old in self.issues_for(result_id):
            self.s.delete(old)
        for i in issues:
            self.s.add(
                IssueRow(
                    run_id=run_id,
                    result_id=result_id,
                    scope=i.scope.value,
                    scope_ref=str(i.details.get("cell_id") or "") or None,
                    code=i.code.value,
                    severity=i.severity.value,
                    message=i.message,
                    field=i.field,
                    details=i.details,
                    acknowledged=i.code.value in acked,
                )
            )

    # --------------------------------------------------------------- status
    def set_status(self, row: ResultRow, target: ResultStatus, actor: str, reason: Optional[str] = None) -> None:
        current = ResultStatus(row.status)
        if current == target:
            return
        row.status = result_transition(current, target).value
        self.s.add(
            StateTransitionRow(
                entity_type="result",
                entity_id=row.result_id,
                from_state=current.value,
                to_state=target.value,
                actor=actor,
                reason=reason,
            )
        )

    # --------------------------------------------------------------- review
    def case_for(self, result_id: str) -> Optional[ReviewCase]:
        return self.s.execute(select(ReviewCase).where(ReviewCase.result_id == result_id)).scalars().first()

    def upsert_case(self, row: ResultRow, codes: List[str]) -> ReviewCase:
        case = self.case_for(row.result_id)
        if case is None:
            case = ReviewCase(kind="ROW", result_id=row.result_id, run_id=row.run_id, status="OPEN", codes=codes)
            self.s.add(case)
        else:
            case.codes = codes
            if case.status == "RESOLVED":
                case.status = "OPEN"
        return case

    def upsert_table_case(self, run_id: str, table_id: str, codes: List[str]) -> ReviewCase:
        case = self.s.execute(select(ReviewCase).where(ReviewCase.table_id == table_id)).scalars().first()
        if case is None:
            case = ReviewCase(kind="TABLE", table_id=table_id, run_id=run_id, status="OPEN", codes=codes)
            self.s.add(case)
        else:
            case.codes = codes
        return case

    def close_case(self, row: ResultRow, status: str) -> None:
        case = self.case_for(row.result_id)
        if case is not None and case.status == "OPEN":
            case.status = status

    def commit(self) -> None:
        self.s.commit()


# ------------------------------------------------------------------ record -> ORM row mapping
def _page_row(run_id: str, page: PageRecord) -> Page:
    return Page(
        page_id=page.page_id,
        run_id=run_id,
        page_number=page.page_number,
        width_px=page.width_px,
        height_px=page.height_px,
        dpi=page.dpi,
        image_path=page.image_path,
        image_sha256=page.image_sha256,
        deskew_angle=page.deskew_angle,
        deskew_applied=page.deskew_applied,
        rotation_matrix=page.rotation_matrix,
        has_text_layer=page.has_text_layer,
        text_layer_word_count=page.text_layer_word_count,
        quality=page.quality,
        issues=_issue_json(page.issues),
    )


def _table_row(run_id: str, page_id: str, t: TableRecord) -> TableRow:
    return TableRow(
        table_id=t.table_id,
        page_id=page_id,
        run_id=run_id,
        page_number=t.page_number,
        table_profile_id=t.table_profile_id,
        variant_id=t.variant_id,
        detection_score=t.detection_score,
        region=t.region,
        header_bottom_y=t.header_bottom_y,
        body=t.body,
        columns=t.columns,
        ml_columns=t.ml_columns,
        column_agreement=t.column_agreement,
        header_matches=[m.model_dump(mode="json") for m in t.header_matches],
        section_headers=[h.model_dump(mode="json") for h in t.section_headers],
        line_pitch=t.line_pitch,
        glyph_height=t.glyph_height,
        line_count=t.line_count,
        issues=_issue_json(t.issues),
    )


def _cell_row(run_id: str, table_id: str, c: CellRecord) -> Cell:
    r = c.resolution
    return Cell(
        cell_id=c.cell_id,
        table_id=table_id,
        run_id=run_id,
        column_index=c.column_index,
        column_key=c.column_key,
        value_type=c.value_type,
        line_index=c.line_index,
        box=c.box,
        box_pdf=c.box_pdf,
        assignment_status=c.assignment_status,
        crosses_boundary=c.crosses_boundary,
        line_distance_ratio=c.line_distance_ratio,
        status=r.status.value,
        value=r.value,
        decimals=r.decimals,
        supporting_groups=r.supporting_groups,
        dissenting_groups=r.dissenting_groups,
        labels=[label.model_dump(mode="json") for label in c.labels],
        shape=c.shape,
        shape_conflict=c.shape_conflict,
        format_id=c.format_id,
    )


def _candidate_rows(run_id: str, c: CellRecord) -> List[OCRCandidateRow]:
    return [
        OCRCandidateRow(
            cell_id=c.cell_id,
            run_id=run_id,
            engine=cand.engine,
            engine_version=cand.engine_version,
            variant=cand.variant,
            independence_group=cand.independence_group,
            raw_text=cand.raw_text,
            confidence=cand.confidence,
            parse_status=cand.parse.status.value,
            parsed_value=cand.parse.value,
            normalized=cand.parse.normalized,
            normalization_steps=cand.parse.steps,
            used_confusion_map=cand.parse.used_confusion_map,
            role=cand.role,
            note=cand.note,
            meta=cand.meta,
        )
        for cand in c.resolution.candidates
    ]


def _result_row(run_id: str, document_id: str, result_id: str, row: RowRecord) -> ResultRow:
    return ResultRow(
        result_id=result_id,
        run_id=run_id,
        document_id=document_id,
        result_key=row.result_key,
        table_id=row.table_id,
        page_number=row.page_number,
        group_index=row.group_index,
        line_indices=row.line_indices,
        row_kind=row.row_kind,
        marker=row.marker,
        role_id=row.role_id,
        role_kind=row.role_kind,
        percent_of_role=row.percent_of_role,
        related_table_ids=row.related_table_ids or None,
        vote_type=row.vote_type,
        election_id=row.election_id,
        publishable_role=row.publishable_role,
        unit_level=row.unit_level,
        section_key=row.section_key,
        section_kind=row.section_kind,
        section_name=row.section_name,
        section_level=row.section_level,
        state_name=row.state,
        context={
            **row.context,
            "_section_name_readings": row.section_name_readings,
            "_unit_name_readings": row.unit_name_readings,
            "_unit_index_cell_id": row.unit_index_cell_id,
        },
        label_readings=[label.model_dump(mode="json") for label in row.label_readings],
        label_cell_ids=row.label_cell_ids,
        field_cells=row.field_cells,
        party_cells=row.party_cells,
        marker_cell_id=row.marker_cell_id,
        status=ResultStatus.EXTRACTED.value,
        row_issues=_issue_json(row.issues),
    )
