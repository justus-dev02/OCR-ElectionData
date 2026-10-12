"""
Relational schema (SQLAlchemy 2.0). SQLite by default, PostgreSQL-compatible.

Logical layers:
    evidence    source_documents, processing_runs, pages, tables, cells, ocr_candidates
    processing  result_rows, issues, state_transitions
    review      review_cases, review_decisions, field_overrides
    published   published_results, published_party_results   (analytics read ONLY these)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


# ----------------------------------------------------------------- evidence
class SourceDocument(Base):
    __tablename__ = "source_documents"
    document_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    path: Mapped[str] = mapped_column(String(1024), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, nullable=False)
    text_layer_pages: Mapped[int] = mapped_column(Integer, nullable=False)
    pdf_metadata: Mapped[Any] = mapped_column(JSON, nullable=True)
    renderer_version: Mapped[str] = mapped_column(String(128), nullable=False)
    registered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class ProcessingRun(Base):
    __tablename__ = "processing_runs"
    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    document_id: Mapped[str] = mapped_column(String(64), ForeignKey("source_documents.document_id"), nullable=False, index=True)
    document_profile_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    pages_requested: Mapped[Any] = mapped_column(JSON, nullable=True)
    config_snapshot: Mapped[Any] = mapped_column(JSON, nullable=False)
    config_fingerprint: Mapped[str] = mapped_column(String(32), nullable=False)
    reference_snapshot: Mapped[str] = mapped_column(String(32), nullable=False)
    code_version: Mapped[str] = mapped_column(String(64), nullable=False)
    engine_versions: Mapped[Any] = mapped_column(JSON, nullable=True)
    model_info: Mapped[Any] = mapped_column(JSON, nullable=True)
    stats: Mapped[Any] = mapped_column(JSON, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class Page(Base):
    __tablename__ = "pages"
    page_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), ForeignKey("processing_runs.run_id"), nullable=False, index=True)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    width_px: Mapped[int] = mapped_column(Integer, nullable=False)
    height_px: Mapped[int] = mapped_column(Integer, nullable=False)
    dpi: Mapped[int] = mapped_column(Integer, nullable=False)
    image_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    image_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    deskew_angle: Mapped[float] = mapped_column(Float, nullable=False)
    deskew_applied: Mapped[bool] = mapped_column(Boolean, nullable=False)
    rotation_matrix: Mapped[Any] = mapped_column(JSON, nullable=True)
    has_text_layer: Mapped[bool] = mapped_column(Boolean, nullable=False)
    text_layer_word_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    quality: Mapped[Any] = mapped_column(JSON, nullable=True)
    issues: Mapped[Any] = mapped_column(JSON, nullable=True)


class TableRow(Base):
    __tablename__ = "tables"
    table_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    page_id: Mapped[str] = mapped_column(String(96), ForeignKey("pages.page_id"), nullable=False, index=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    table_profile_id: Mapped[str] = mapped_column(String(128), nullable=False)
    variant_id: Mapped[str] = mapped_column(String(128), nullable=False)
    detection_score: Mapped[float] = mapped_column(Float, nullable=False)
    region: Mapped[Any] = mapped_column(JSON, nullable=False)
    header_bottom_y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    body: Mapped[Any] = mapped_column(JSON, nullable=False)
    columns: Mapped[Any] = mapped_column(JSON, nullable=False)
    ml_columns: Mapped[Any] = mapped_column(JSON, nullable=False)
    column_agreement: Mapped[Any] = mapped_column(JSON, nullable=True)
    header_matches: Mapped[Any] = mapped_column(JSON, nullable=True)
    section_headers: Mapped[Any] = mapped_column(JSON, nullable=True)
    line_pitch: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    glyph_height: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    line_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    issues: Mapped[Any] = mapped_column(JSON, nullable=True)


class Cell(Base):
    __tablename__ = "cells"
    cell_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    table_id: Mapped[str] = mapped_column(String(128), ForeignKey("tables.table_id"), nullable=False, index=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    column_index: Mapped[int] = mapped_column(Integer, nullable=False)
    column_key: Mapped[str] = mapped_column(String(64), nullable=False)
    value_type: Mapped[str] = mapped_column(String(16), nullable=False)
    line_index: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    box: Mapped[Any] = mapped_column(JSON, nullable=False)
    box_pdf: Mapped[Any] = mapped_column(JSON, nullable=False)
    assignment_status: Mapped[str] = mapped_column(String(16), nullable=False)
    crosses_boundary: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    line_distance_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    value: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    decimals: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    supporting_groups: Mapped[Any] = mapped_column(JSON, nullable=True)
    dissenting_groups: Mapped[Any] = mapped_column(JSON, nullable=True)
    labels: Mapped[Any] = mapped_column(JSON, nullable=True)
    shape: Mapped[Optional[str]] = mapped_column(String(16), nullable=True, doc="geometric token class: dash | narrow | wide")
    shape_conflict: Mapped[Optional[str]] = mapped_column(Text, nullable=True, doc="agreed value contradicts the token shape")
    format_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, doc="number format the cell was resolved with")


class OCRCandidateRow(Base):
    __tablename__ = "ocr_candidates"
    candidate_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cell_id: Mapped[str] = mapped_column(String(160), ForeignKey("cells.cell_id"), nullable=False, index=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    engine: Mapped[str] = mapped_column(String(32), nullable=False)
    engine_version: Mapped[str] = mapped_column(String(128), nullable=False)
    variant: Mapped[str] = mapped_column(String(32), nullable=False)
    independence_group: Mapped[str] = mapped_column(String(32), nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    parse_status: Mapped[str] = mapped_column(String(16), nullable=False)
    parsed_value: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    normalized: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    normalization_steps: Mapped[Any] = mapped_column(JSON, nullable=True)
    used_confusion_map: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    meta: Mapped[Any] = mapped_column(JSON, nullable=True)


# --------------------------------------------------------------- processing
class ResultRow(Base):
    __tablename__ = "result_rows"
    result_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), ForeignKey("processing_runs.run_id"), nullable=False, index=True)
    document_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    result_key: Mapped[str] = mapped_column(String(160), nullable=False)
    table_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    group_index: Mapped[int] = mapped_column(Integer, nullable=False)
    line_indices: Mapped[Any] = mapped_column(JSON, nullable=False)
    row_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    marker: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)
    role_id: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True, doc="result role of the table profile (e.g. abs_e, pct_z)"
    )
    role_kind: Mapped[Optional[str]] = mapped_column(String(16), nullable=True, doc="absolute | percent")
    percent_of_role: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True, doc="percent roles: role whose values they describe"
    )
    related_table_ids: Mapped[Any] = mapped_column(JSON, nullable=True, doc="further tables contributing cells (page spreads)")
    vote_type: Mapped[str] = mapped_column(String(16), nullable=False)
    election_id: Mapped[str] = mapped_column(String(64), nullable=False)
    publishable_role: Mapped[bool] = mapped_column(Boolean, nullable=False)
    unit_level: Mapped[str] = mapped_column(String(32), nullable=False)
    section_key: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    section_kind: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    section_name: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    section_level: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    state_name: Mapped[str] = mapped_column(String(128), nullable=False)
    context: Mapped[Any] = mapped_column(JSON, nullable=True)
    label_readings: Mapped[Any] = mapped_column(JSON, nullable=True)
    label_cell_ids: Mapped[Any] = mapped_column(JSON, nullable=True)
    field_cells: Mapped[Any] = mapped_column(JSON, nullable=False)
    party_cells: Mapped[Any] = mapped_column(JSON, nullable=False)
    marker_cell_id: Mapped[Optional[str]] = mapped_column(String(160), nullable=True)
    unit_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    unit_name: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    unit_resolution: Mapped[Any] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    row_issues: Mapped[Any] = mapped_column(JSON, nullable=True, doc="structural issues found at interpretation time")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)
    __table_args__ = (UniqueConstraint("run_id", "result_key"),)


class IssueRow(Base):
    __tablename__ = "issues"
    issue_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    result_id: Mapped[Optional[str]] = mapped_column(String(200), nullable=True, index=True)
    scope: Mapped[str] = mapped_column(String(16), nullable=False)
    scope_ref: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    code: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    field: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    details: Mapped[Any] = mapped_column(JSON, nullable=True)
    acknowledged: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class StateTransitionRow(Base):
    __tablename__ = "state_transitions"
    transition_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entity_type: Mapped[str] = mapped_column(String(32), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    from_state: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    to_state: Mapped[str] = mapped_column(String(16), nullable=False)
    actor: Mapped[str] = mapped_column(String(128), nullable=False)
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


# ------------------------------------------------------------------- review
class ReviewCase(Base):
    __tablename__ = "review_cases"
    case_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(8), nullable=False, default="ROW")  # ROW | TABLE (table/page without usable rows)
    result_id: Mapped[Optional[str]] = mapped_column(String(200), nullable=True, unique=True)
    table_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, unique=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, index=True)  # OPEN | RESOLVED | REJECTED
    codes: Mapped[Any] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)


class ReviewDecision(Base):
    __tablename__ = "review_decisions"
    decision_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_id: Mapped[int] = mapped_column(Integer, ForeignKey("review_cases.case_id"), nullable=False, index=True)
    result_id: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(16), nullable=False)  # SET_VALUE | SET_UNIT | ACK | APPROVE | REJECT
    target: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    previous: Mapped[Any] = mapped_column(JSON, nullable=True)
    new: Mapped[Any] = mapped_column(JSON, nullable=True)
    reviewer: Mapped[str] = mapped_column(String(128), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class FieldOverride(Base):
    """Human-provided value for one field of one result row (never overwrites OCR evidence)."""

    __tablename__ = "field_overrides"
    override_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    result_id: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    field: Mapped[str] = mapped_column(String(64), nullable=False)  # metric:<name> | party:<id> | unit
    value: Mapped[str] = mapped_column(String(128), nullable=False)
    decimals: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    reviewer: Mapped[str] = mapped_column(String(128), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    __table_args__ = (UniqueConstraint("result_id", "field"),)


class Acknowledgement(Base):
    """Reviewer confirmation that a specific issue code was checked against the source."""

    __tablename__ = "acknowledgements"
    ack_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    result_id: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    code: Mapped[str] = mapped_column(String(64), nullable=False)
    reviewer: Mapped[str] = mapped_column(String(128), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    __table_args__ = (UniqueConstraint("result_id", "code"),)


# ---------------------------------------------------------------- published
class PublishedResult(Base):
    __tablename__ = "published_results"
    published_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    result_id: Mapped[str] = mapped_column(String(200), ForeignKey("result_rows.result_id"), nullable=False, unique=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    document_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    election_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    unit_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    unit_name: Mapped[str] = mapped_column(String(256), nullable=False)
    unit_level: Mapped[str] = mapped_column(String(32), nullable=False)
    vote_type: Mapped[str] = mapped_column(String(16), nullable=False)
    eligible_voters: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    voters: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    valid_votes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    invalid_votes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    turnout_percent: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    provenance: Mapped[Any] = mapped_column(JSON, nullable=False)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    superseded_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class PublishedPartyResult(Base):
    __tablename__ = "published_party_results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    published_id: Mapped[int] = mapped_column(Integer, ForeignKey("published_results.published_id"), nullable=False, index=True)
    party_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    votes: Mapped[int] = mapped_column(Integer, nullable=False)
    cell_id: Mapped[Optional[str]] = mapped_column(String(160), nullable=True)
    source: Mapped[str] = mapped_column(String(16), nullable=False)


def make_engine(database_url: str) -> Engine:
    engine = create_engine(database_url, future=True)
    if database_url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _fk_on(dbapi_conn, _):  # pragma: no cover - driver hook
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

    Base.metadata.create_all(engine)
    _add_missing_columns(engine)
    return engine


def _add_missing_columns(engine: Engine) -> None:
    """Minimal forward migration: add columns that exist in the model but not in the DB.

    Only nullable columns are added (existing evidence rows keep NULL there, which
    every reader treats as 'not recorded'). Anything else must be migrated by hand.
    """
    from sqlalchemy import inspect, text

    insp = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not insp.has_table(table.name):
                continue
            existing = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name in existing:
                    continue
                if not col.nullable:
                    raise RuntimeError(f"cannot auto-migrate NOT NULL column {table.name}.{col.name}")
                ddl = col.type.compile(dialect=engine.dialect)
                conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {ddl}'))


def make_session(engine: Engine) -> Session:
    return sessionmaker(bind=engine, expire_on_commit=False)()
