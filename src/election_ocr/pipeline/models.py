"""
Data models flowing between pipeline stages.

Hierarchy:  PageRecord -> TableRecord -> CellRecord
            TableRecord -> RowRecord (one per unit x row role)
Everything keeps its evidence (boxes, candidates, issues); nothing is
overwritten when values are normalized or resolved.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from election_ocr.issues import Issue
from election_ocr.ocr.resolver import CellResolution


class HeaderMatch(BaseModel):
    column_index: int
    column_key: str
    layers: Dict[str, List[str]] = Field(default_factory=dict, description="engine -> header text layers")
    best_score: float = 0.0
    best_engine: Optional[str] = None
    competing_column: Optional[str] = None
    competing_score: Optional[float] = None
    matched: bool = False
    confirmed_by: str = ""
    contradicted: bool = Field(default=False, description="another column's alias or a neighbour's number was read here")


class LabelReading(BaseModel):
    """One engine's reading of a label (municipality / section / total row)."""

    engine: str
    independence_group: str
    text: str
    confidence: Optional[float] = None


class CellRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    cell_id: str
    table_id: str
    column_index: int
    column_key: str
    value_type: str
    line_index: Optional[int]
    box: List[float]
    box_pdf: List[float]
    assignment_status: str
    crosses_boundary: bool = False
    line_distance_ratio: Optional[float] = None
    resolution: CellResolution
    labels: List[LabelReading] = Field(default_factory=list, description="Only for label cells.")
    shape: Optional[str] = Field(default=None, description="Geometric token class (dash | narrow | wide).")
    shape_conflict: Optional[str] = None
    format_id: Optional[str] = Field(default=None, description="Number format used for the final resolution.")
    readings: List[Any] = Field(default_factory=list, exclude=True, description="Raw engine readings (not persisted).")
    between_lines: Optional[Tuple[int, int]] = Field(
        default=None, exclude=True, description="AMBIGUOUS cell centred between these two lines (not persisted)."
    )


class RowRecord(BaseModel):
    """A candidate result row: one unit (or section total) x one row role."""

    result_key: str
    table_id: str
    page_number: int
    group_index: int
    line_indices: List[int]
    row_kind: str = Field(description="DATA | SECTION_TOTAL")
    marker: Optional[str] = None
    role_id: Optional[str] = None
    role_kind: str = "absolute"
    percent_of_role: Optional[str] = None
    vote_type: str
    election_id: str
    publishable_role: bool = True
    label_readings: List[LabelReading] = Field(default_factory=list)
    label_cell_ids: List[str] = Field(default_factory=list)
    unit_name_readings: Dict[str, str] = Field(default_factory=dict, description="group -> name part of the label")
    unit_index: Optional[str] = Field(default=None, description="Printed unit index / key (agreed).")
    unit_index_cell_id: Optional[str] = None
    unit_level: str
    section_key: Optional[str] = None
    section_kind: Optional[str] = None
    section_name: Optional[str] = None
    section_name_readings: Dict[str, str] = Field(default_factory=dict)
    section_level: Optional[str] = None
    context: Dict[str, str] = Field(default_factory=dict)
    state: str
    field_cells: Dict[str, Optional[str]] = Field(default_factory=dict, description="metric -> cell_id")
    party_cells: Dict[str, Optional[str]] = Field(default_factory=dict, description="party_id -> cell_id")
    marker_cell_id: Optional[str] = None
    related_table_ids: List[str] = Field(default_factory=list, description="Further tables (page spread).")
    issues: List[Issue] = Field(default_factory=list)


class SectionHeaderRecord(BaseModel):
    table_id: str
    page_number: int
    y: float
    readings: List[LabelReading]
    kind: Optional[str] = None
    name: Optional[str] = None
    name_readings: Dict[str, str] = Field(default_factory=dict)
    continued: bool = False
    heading: bool = False
    index: Optional[str] = None
    agreed: bool = False
    cell_ids: List[str] = Field(default_factory=list)
    issues: List[Issue] = Field(default_factory=list)


class TableRecord(BaseModel):
    table_id: str
    page_number: int
    table_profile_id: str
    variant_id: str
    detection_score: float
    region: List[float]
    header_bottom_y: Optional[float]
    body: List[float]
    columns: List[List[float]]
    ml_columns: List[List[float]]
    column_agreement: Dict[str, object] = Field(default_factory=dict)
    header_matches: List[HeaderMatch] = Field(default_factory=list)
    line_pitch: Optional[float] = None
    glyph_height: Optional[float] = None
    line_count: int = 0
    cells: List[CellRecord] = Field(default_factory=list)
    rows: List[RowRecord] = Field(default_factory=list)
    section_headers: List[SectionHeaderRecord] = Field(default_factory=list)
    issues: List[Issue] = Field(default_factory=list)

    def cell(self, cell_id: Optional[str]) -> Optional[CellRecord]:
        if cell_id is None:
            return None
        return next((c for c in self.cells if c.cell_id == cell_id), None)


class PageRecord(BaseModel):
    page_id: str
    page_number: int
    width_px: int
    height_px: int
    dpi: int
    image_path: str
    image_sha256: str
    deskew_angle: float
    deskew_applied: bool
    rotation_matrix: Optional[List[List[float]]] = None
    has_text_layer: bool
    text_layer_word_count: int = 0
    quality: Dict[str, float] = Field(default_factory=dict)
    tables: List[TableRecord] = Field(default_factory=list)
    issues: List[Issue] = Field(default_factory=list)
