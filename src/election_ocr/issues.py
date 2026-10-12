"""
Machine-readable issue codes.

Every reason why a value is not published is expressed as an ``Issue`` with a
stable ``IssueCode`` (never free log text), so review queues can be filtered
and error frequencies measured.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Mapping, Optional

from pydantic import BaseModel, Field


class Severity(str, Enum):
    INFO = "INFO"  # recorded, never blocks
    WARNING = "WARNING"  # recorded, never blocks (candidate for later calibration)
    BLOCKING = "BLOCKING"  # prevents publication until resolved by a human


class IssueScope(str, Enum):
    DOCUMENT = "DOCUMENT"
    PAGE = "PAGE"
    TABLE = "TABLE"
    SECTION = "SECTION"
    ROW = "ROW"
    CELL = "CELL"


class IssueCode(str, Enum):
    # --- ingestion / reference -------------------------------------------------
    REFERENCE_UNVERIFIED = "REFERENCE_UNVERIFIED"
    REFERENCE_DATA_MISSING = "REFERENCE_DATA_MISSING"
    # --- layout / structure ----------------------------------------------------
    TABLE_NOT_DETECTED = "TABLE_NOT_DETECTED"
    HEADER_UNRESOLVED = "HEADER_UNRESOLVED"
    HEADER_COLUMN_INFERRED = "HEADER_COLUMN_INFERRED"
    COLUMN_COUNT_MISMATCH = "COLUMN_COUNT_MISMATCH"
    TABLE_STRUCTURE_DISAGREEMENT = "TABLE_STRUCTURE_DISAGREEMENT"
    PAGE_SKEW_UNCORRECTED = "PAGE_SKEW_UNCORRECTED"
    LINE_ANCHOR_MISSING = "LINE_ANCHOR_MISSING"
    TOKEN_UNASSIGNED = "TOKEN_UNASSIGNED"
    NOISE_TOKEN_IGNORED = "NOISE_TOKEN_IGNORED"
    DETACHED_TEXT = "DETACHED_TEXT"
    CELL_BOUNDARY_AMBIGUOUS = "CELL_BOUNDARY_AMBIGUOUS"
    ROW_GROUP_INVALID = "ROW_GROUP_INVALID"
    MARKER_UNVERIFIED = "MARKER_UNVERIFIED"
    UNEXPECTED_CELL_CONTENT = "UNEXPECTED_CELL_CONTENT"
    INDEX_MISMATCH = "INDEX_MISMATCH"
    ML_STRUCTURE_DISAGREEMENT = "ML_STRUCTURE_DISAGREEMENT"
    ROW_UNINTERPRETED = "ROW_UNINTERPRETED"
    SECTION_LABEL_UNRESOLVED = "SECTION_LABEL_UNRESOLVED"
    SECTION_CONTEXT_MISSING = "SECTION_CONTEXT_MISSING"
    # --- OCR -------------------------------------------------------------------
    OCR_ENGINE_DISAGREEMENT = "OCR_ENGINE_DISAGREEMENT"
    OCR_SINGLE_SOURCE = "OCR_SINGLE_SOURCE"
    OCR_NO_READING = "OCR_NO_READING"
    MISSING_VALUE = "MISSING_VALUE"
    LABEL_DISAGREEMENT = "LABEL_DISAGREEMENT"
    GLYPH_SHAPE_CONFLICT = "GLYPH_SHAPE_CONFLICT"
    BLANK_AS_ZERO = "BLANK_AS_ZERO"
    # --- entities --------------------------------------------------------------
    ENTITY_NOT_FOUND = "ENTITY_NOT_FOUND"
    ENTITY_MULTIPLE_MATCHES = "ENTITY_MULTIPLE_MATCHES"
    ENTITY_FUZZY_ONLY = "ENTITY_FUZZY_ONLY"
    ENTITY_OUT_OF_CONTEXT = "ENTITY_OUT_OF_CONTEXT"
    ENTITY_NOT_VALID_FOR_DATE = "ENTITY_NOT_VALID_FOR_DATE"
    PARTY_NOT_ALLOWED_FOR_ELECTION = "PARTY_NOT_ALLOWED_FOR_ELECTION"
    # --- arithmetic ------------------------------------------------------------
    VOTERS_EXCEED_ELIGIBLE = "VOTERS_EXCEED_ELIGIBLE"
    VALID_EXCEED_VOTERS = "VALID_EXCEED_VOTERS"
    VOTERS_NE_VALID_PLUS_INVALID = "VOTERS_NE_VALID_PLUS_INVALID"
    PARTY_SUM_MISMATCH = "PARTY_SUM_MISMATCH"
    TURNOUT_MISMATCH = "TURNOUT_MISMATCH"
    DEFINITION_DIFFERS = "DEFINITION_DIFFERS"
    PERCENT_ROUNDING_ADJUSTED = "PERCENT_ROUNDING_ADJUSTED"
    VALUE_OUT_OF_RANGE = "VALUE_OUT_OF_RANGE"
    ARITHMETIC_CHECK_NOT_EXECUTABLE = "ARITHMETIC_CHECK_NOT_EXECUTABLE"
    CROSS_ROLE_MISMATCH = "CROSS_ROLE_MISMATCH"
    PERCENT_MISMATCH = "PERCENT_MISMATCH"
    PERCENT_CHECK_NOT_EXECUTABLE = "PERCENT_CHECK_NOT_EXECUTABLE"
    DERIVED_VALUE_MISMATCH = "DERIVED_VALUE_MISMATCH"
    DERIVED_CHECK_NOT_EXECUTABLE = "DERIVED_CHECK_NOT_EXECUTABLE"
    # --- hierarchy / cross-page -----------------------------------------------
    HIERARCHICAL_SUM_MISMATCH = "HIERARCHICAL_SUM_MISMATCH"
    HIERARCHY_UNVERIFIED = "HIERARCHY_UNVERIFIED"
    PAGE_STITCH_MISMATCH = "PAGE_STITCH_MISMATCH"
    DUPLICATE_UNIT_IN_SECTION = "DUPLICATE_UNIT_IN_SECTION"
    # --- review ----------------------------------------------------------------
    HUMAN_CORRECTED = "HUMAN_CORRECTED"


_DEFAULT_SEVERITY: Dict[IssueCode, Severity] = {code: Severity.BLOCKING for code in IssueCode}
_DEFAULT_SEVERITY.update(
    {
        IssueCode.HUMAN_CORRECTED: Severity.INFO,
        IssueCode.NOISE_TOKEN_IGNORED: Severity.INFO,
        IssueCode.DETACHED_TEXT: Severity.INFO,
        IssueCode.PAGE_SKEW_UNCORRECTED: Severity.WARNING,
        IssueCode.BLANK_AS_ZERO: Severity.INFO,
        # printed value follows another definition (decided on the run's data, see DefinitionProbe)
        IssueCode.DEFINITION_DIFFERS: Severity.INFO,
        IssueCode.PERCENT_ROUNDING_ADJUSTED: Severity.INFO,
        # a single unreadable header confirmed by its confirmed neighbours (see semantics/header.py)
        IssueCode.HEADER_COLUMN_INFERRED: Severity.INFO,
        # TATR is recorded as evidence only (docs/decisions.md, D4); the column
        # structure is accepted on two deterministic evidences.
        IssueCode.ML_STRUCTURE_DISAGREEMENT: Severity.INFO,
    }
)


def default_severity(code: IssueCode, overrides: Optional[Mapping[str, str]] = None) -> Severity:
    if overrides and code.value in overrides:
        return Severity(overrides[code.value])
    return _DEFAULT_SEVERITY[code]


class Issue(BaseModel):
    code: IssueCode
    severity: Severity
    scope: IssueScope
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    field: Optional[str] = Field(default=None, description="Result field the issue refers to, if any.")

    @property
    def blocking(self) -> bool:
        return self.severity == Severity.BLOCKING


class IssueFactory:
    """Creates issues with the configured severity policy."""

    def __init__(self, overrides: Optional[Mapping[str, str]] = None):
        self.overrides = dict(overrides or {})

    def make(
        self,
        code: IssueCode,
        scope: IssueScope,
        message: str,
        field: Optional[str] = None,
        **details: Any,
    ) -> Issue:
        return Issue(
            code=code,
            severity=default_severity(code, self.overrides),
            scope=scope,
            message=message,
            field=field,
            details=details,
        )
