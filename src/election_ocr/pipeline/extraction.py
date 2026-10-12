"""
Extraction of ONE physical page: rendered image -> table record with cells and OCR evidence.

Stages (each a method of ``TableExtractor``)
--------------------------------------------
1. table region: long horizontal rules or the page ink (strategy), TATR detection
   only as a recorded fallback                                                      CV (+ ML)
2. header end: full-width rule or printed column-number row                        CV
3. column partition (``layout.column_strategies``); TATR structure recorded only    CV (+ ML evidence)
4. glyph tokens + anchor lines, token shape classes                                CV
5. header verification against the profile                                         OCR + fuzzy compare
6. OCR of every token by all engines (``ocr.token_reader``)                         Tesseract / EasyOCR / text layer
7. token-level geometry issues (noise, detached text, unassigned tokens)           deterministic

Extraction does not depend on any other page, so pages can be extracted in
parallel worker processes; only the interpretation (section context) is
sequential (see ``pipeline.page_processor``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from election_ocr.config import PipelineConfig
from election_ocr.geometry import Box
from election_ocr.ingestion.render import RenderedPage
from election_ocr.issues import Issue, IssueCode, IssueFactory, IssueScope
from election_ocr.layout.column_strategies import ColumnPartition, PartitionContext, RegionResult, mm_to_px, strategy_for
from election_ocr.layout.structure import (
    ColumnInterval,
    body_bottom,
    compare_columns,
    find_full_width_hlines,
    header_bottom,
    merge_duplicate_ml_columns,
    select_tables,
)
from election_ocr.layout.table_structure import column_number_boxes, find_column_number_row
from election_ocr.layout.tatr import TableDetector, TableStructureRecognizer, pil_from_gray
from election_ocr.layout.tokens import TokenLayout, build_token_layout, connected_components
from election_ocr.ocr.engines import Engines, OCRRequest
from election_ocr.ocr.token_reader import TokenReader
from election_ocr.pipeline.models import CellRecord, PageRecord, TableRecord
from election_ocr.preprocessing.image_ops import LineMasks, ink_mask, separate_ruling_lines
from election_ocr.reference.models import ColumnSpec, TableProfile, TableVariant
from election_ocr.semantics.header import match_headers

_REGION_PAD_MM = 3.0


@dataclass
class PageColumns:
    """The profile columns printed on ONE physical page (left page or continuation page)."""

    specs: List[ColumnSpec]
    anchor_key: str
    confirm_key: Optional[str]
    offset: int  # index of specs[0] in variant.all_columns()


def left_columns(tp: TableProfile, variant: TableVariant) -> PageColumns:
    return PageColumns(list(variant.columns), tp.line_anchor_column, tp.anchor_confirmation_column, 0)


def continuation_columns(variant: TableVariant) -> PageColumns:
    cont = variant.continuation
    if cont is None:
        raise ValueError(f"variant {variant.variant_id} has no continuation page")
    return PageColumns(list(cont.columns), cont.line_anchor_column, cont.anchor_confirmation_column, len(variant.columns))


@dataclass
class ExtractedTable:
    """Result of extracting one physical page. ``layout`` is None when extraction stopped early."""

    table: TableRecord
    layout: Optional[TokenLayout] = None
    token_issue_lines: List[Tuple[Issue, List[int]]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.layout is not None

    def compact(self) -> ExtractedTable:
        """Drop the page-sized component label image (not needed after OCR; keeps IPC small)."""
        if self.layout is not None:
            self.layout.labels = np.zeros((0, 0), dtype=np.int32)
        return self


class ExtractionStopped(Exception):
    """A structural failure after which no cell of the page can be trusted."""

    def __init__(self, issue: Issue):
        super().__init__(issue.message)
        self.issue = issue


@dataclass
class _HeaderEnd:
    bottom: int
    body: Box
    number_boxes: List[Box]
    glyph_height: float


class TableExtractor:
    def __init__(
        self,
        config: PipelineConfig,
        detector: TableDetector,
        structure: TableStructureRecognizer,
        engines: Engines,
        issues: IssueFactory,
    ):
        self.cfg = config
        self.detector = detector
        self.structure = structure
        self.engines = engines
        self.issues = issues
        self.reader = TokenReader(config, engines)

    def extract(
        self, page: PageRecord, rendered: RenderedPage, tp: TableProfile, variant: TableVariant, pc: PageColumns
    ) -> ExtractedTable:
        gray = rendered.gray
        ink = ink_mask(gray)
        masks = separate_ruling_lines(ink, rendered.dpi)
        strategy = strategy_for(tp.structure, self.cfg.structure)
        region = strategy.locate_region(masks, rendered.dpi)
        pil = pil_from_gray(gray)
        table, region_box = self._new_table(page, tp, variant, region, pil)
        issues: List[Issue] = []
        try:
            if region_box is None:
                raise ExtractionStopped(self.issues.make(IssueCode.TABLE_NOT_DETECTED, IssueScope.PAGE, "no table region found"))
            ml = self.structure.recognize(pil, region_box)
            ml_cols = merge_duplicate_ml_columns(ml.columns, ml.scores.get("table column"))
            table.ml_columns = [b.as_list() for b in ml_cols]
            header = self._header_end(masks, region_box, tp, pc, table)
            ctx = PartitionContext(
                masks=masks,
                region=region_box,
                body=header.body,
                header_bottom=header.bottom,
                long_rules=region.long_rules,
                ml_columns=ml_cols,
                number_boxes=header.number_boxes,
                glyph_height=header.glyph_height,
                dpi=rendered.dpi,
                specs=pc.specs,
            )
            partition = strategy.partition(ctx)
            comps_and_labels = self._record_partition(partition, header.body, pc, table, issues)
            self._record_ml_agreement(partition.columns, ml_cols, comps_and_labels[1], pc, table, issues)
            layout = self._token_layout(table.table_id, partition, comps_and_labels, masks, header.body, pc)
            table.line_pitch, table.glyph_height, table.line_count = layout.pitch, layout.char_height, len(layout.lines)
            self._verify_header(gray, masks, region_box, header, partition.columns, pc, layout, table, issues)
            table.cells = self.reader.read_cells(
                table.table_id, rendered, ink, partition.text_mask, layout, tp, variant, pc.specs
            )
            token_issue_lines = self._token_geometry_issues(table, layout, pc, issues)
        except ExtractionStopped as stop:
            issues.append(stop.issue)
            table.issues = issues
            return ExtractedTable(table)
        table.issues = issues
        return ExtractedTable(table, layout, token_issue_lines)

    # ------------------------------------------------------------- 1. region
    def _new_table(
        self, page: PageRecord, tp: TableProfile, variant: TableVariant, region: RegionResult, pil
    ) -> Tuple[TableRecord, Optional[Box]]:
        box, source, detection_score = region.box, region.source, 0.0
        if box is None:
            detections = select_tables(self.detector.detect(pil))
            if detections:
                det = max(detections, key=lambda d: d.box.width * d.box.height)
                h, w = pil.size[1], pil.size[0]
                box = det.box.pad(self.cfg.table_models.detection_padding_px, w, h)
                detection_score = det.score
                source = "ml_detection"
        table = TableRecord(
            table_id=f"{page.page_id}_t0",
            page_number=page.page_number,
            table_profile_id=tp.table_profile_id,
            variant_id=variant.variant_id,
            detection_score=detection_score,
            region=box.as_list() if box else [],
            header_bottom_y=None,
            body=[],
            columns=[],
            ml_columns=[],
            column_agreement={"region_source": source, "strategy": tp.structure.columns},
        )
        return table, box

    # ------------------------------------------------------------- 2. header end
    def _header_end(self, masks: LineMasks, region: Box, tp: TableProfile, pc: PageColumns, table: TableRecord) -> _HeaderEnd:
        char_h = glyph_height(masks.text, region)
        number_boxes: List[Box] = []
        hlines = find_full_width_hlines(masks.horizontal, region)
        if tp.structure.header_end == "rule":
            hb = header_bottom(hlines, region)
            if hb is None:
                raise ExtractionStopped(
                    self.issues.make(IssueCode.HEADER_UNRESOLVED, IssueScope.TABLE, "no full-width rule closing the header found")
                )
        else:
            expected = sum(1 for c in pc.specs if c.column_number is not None)
            found = find_column_number_row(masks.text, region, expected, char_h)
            if found is None:
                raise ExtractionStopped(
                    self.issues.make(
                        IssueCode.HEADER_UNRESOLVED,
                        IssueScope.TABLE,
                        f"printed column-number row with {expected} numbers not found",
                    )
                )
            (_, band_y1), number_boxes = found
            hb = band_y1 + 2
        table.header_bottom_y = hb
        bb = body_bottom(hlines, region, hb) if tp.structure.header_end == "rule" else int(region.y1)
        body = Box(x0=region.x0, y0=hb + 3, x1=region.x1, y1=bb - 2)
        table.body = body.as_list()
        return _HeaderEnd(bottom=hb, body=body, number_boxes=number_boxes, glyph_height=char_h)

    # ------------------------------------------------------------- 3. columns
    def _record_partition(self, partition: ColumnPartition, body: Box, pc: PageColumns, table: TableRecord, issues: List[Issue]):
        table.column_agreement.update(partition.evidence)
        labels, comps, dropped = connected_components(partition.text_mask, body, self.cfg.structure.min_component_area_px)
        columns = partition.columns
        table.columns = [[round(c.x0, 1), round(c.x1, 1)] for c in columns]
        table.column_agreement["separators"] = [
            {"x": line.x, "coverage": line.coverage, "evidence": line.evidence} for line in partition.separators
        ]
        for f in partition.findings:
            issues.append(self.issues.make(IssueCode(f.code), IssueScope.TABLE, f.message, **f.details))
        if len(columns) != len(pc.specs):
            raise ExtractionStopped(
                self.issues.make(
                    IssueCode.COLUMN_COUNT_MISMATCH,
                    IssueScope.TABLE,
                    f"{len(columns)} columns found ({table.column_agreement['strategy']}), profile expects {len(pc.specs)}",
                )
            )
        return labels, comps, dropped

    def _record_ml_agreement(self, columns, ml_cols, comps, pc: PageColumns, table: TableRecord, issues: List[Issue]) -> None:
        """TATR's column structure is evidence only, never decisive (see docs/decisions.md)."""
        agreement = compare_columns(
            columns,
            ml_cols,
            [c.value_type for c in pc.specs],
            [c.box for c in comps],
            self.cfg.structure.column_boundary_max_distance_ratio,
        )
        table.column_agreement["ml"] = {
            "agree": agreement.agree,
            "ml_count": agreement.ml_count,
            "max_deviation_ratio": agreement.max_center_deviation_ratio,
            "reason": agreement.reason,
            "details": agreement.details,
        }
        if not agreement.agree:
            issues.append(
                self.issues.make(
                    IssueCode.ML_STRUCTURE_DISAGREEMENT,
                    IssueScope.TABLE,
                    agreement.reason or "ML column structure differs (recorded only)",
                )
            )

    # ------------------------------------------------------------- 4. tokens
    def _token_layout(
        self, table_id: str, partition: ColumnPartition, comps_and_labels, masks: LineMasks, body: Box, pc: PageColumns
    ) -> TokenLayout:
        labels, comps, dropped = comps_and_labels
        s = self.cfg.structure
        keys = [c.key for c in pc.specs]
        label_idx = next((i for i, c in enumerate(pc.specs) if c.role == "unit_label"), None)
        layout = build_token_layout(
            labels,
            comps,
            partition.columns,
            keys.index(pc.anchor_key),
            token_prefix=table_id,
            min_component_height=s.min_component_height_px,
            max_offset_ratio=s.token_line_max_offset_ratio,
            min_margin_ratio=s.token_line_min_margin_ratio,
            dropped_specks=dropped,
            confirm_column=keys.index(pc.confirm_key) if pc.confirm_key else None,
            label_column=label_idx,
            # headings / footnotes running through the anchor column are no data lines;
            # glyphs separated by a printed rule are never one word
            text_run_max_gap_ratio=0.6,
            rules=partition.token_rules(),
            run_components=connected_components(masks.text, body, s.min_component_area_px)[1],
        )
        return layout

    # ------------------------------------------------------------- 5. header verification
    def _verify_header(
        self,
        gray: np.ndarray,
        masks: LineMasks,
        region: Box,
        header: _HeaderEnd,
        columns: List[ColumnInterval],
        pc: PageColumns,
        layout: TokenLayout,
        table: TableRecord,
        issues: List[Issue],
    ) -> None:
        header_top, hb = int(region.y0), header.bottom
        layers_by_column: List[Dict[str, List[str]]] = []
        for col in columns:
            x0, x1 = int(col.x0) + 2, int(col.x1) - 2
            crop = gray[header_top:hb, x0:x1].copy()
            crop[masks.vertical[header_top:hb, x0:x1] | masks.horizontal[header_top:hb, x0:x1]] = 255
            layers: Dict[str, List[str]] = {}
            if crop.size and self.engines.tesseract is not None:
                layers["tesseract"] = self.engines.tesseract.read_lines(crop, layout.char_height)
            if crop.size and self.engines.easyocr is not None:
                layers["easyocr"] = self.engines.easyocr.read_lines(crop, layout.char_height)
            layers_by_column.append(layers)
        self._add_column_number_readings(gray, masks, region, header, columns, pc, layers_by_column, table)
        table.header_matches = match_headers(pc.specs, layers_by_column, self.cfg.resolution.header_match_min_score)
        inferred = [m.column_key for m in table.header_matches if m.confirmed_by == "neighbours"]
        if inferred:
            issues.append(
                self.issues.make(
                    IssueCode.HEADER_COLUMN_INFERRED,
                    IssueScope.TABLE,
                    f"header of {inferred} unreadable; column confirmed by its confirmed neighbours and exact count",
                    columns=inferred,
                )
            )
        unmatched = [m.column_key for m in table.header_matches if not m.matched]
        if unmatched:
            issues.append(
                self.issues.make(
                    IssueCode.HEADER_UNRESOLVED,
                    IssueScope.TABLE,
                    f"header text does not confirm profile columns {unmatched}",
                    columns=unmatched,
                )
            )

    def _add_column_number_readings(
        self,
        gray: np.ndarray,
        masks: LineMasks,
        region: Box,
        header: _HeaderEnd,
        columns: List[ColumnInterval],
        pc: PageColumns,
        layers_by_column: List[Dict[str, List[str]]],
        table: TableRecord,
    ) -> None:
        """Printed column numbers ('1 ... 25') read one by one from tight crops (digits only).

        Inside the whole header crop the small numbers drown in the header text; their glyph
        groups are located geometrically (same band test as the column_numbers strategy) and
        each reading is added as one more header layer of the column containing it. The usual
        number confirmation then applies (expected number read, no other valid number read)."""
        expected = sum(1 for c in pc.specs if c.column_number is not None)
        if not expected:
            return
        located = self._locate_column_numbers(masks, region, header, columns, pc)
        if located is None:
            return
        char_h = header.glyph_height
        pad = max(3, int(0.3 * char_h))
        requests: List[OCRRequest] = []
        owner: Dict[str, int] = {}
        for i, (ci, b) in enumerate(located):
            if ci is None:
                continue
            x0, y0, x1, y1 = (int(v) for v in (b.x0 - pad, b.y0 - pad, b.x1 + pad, b.y1 + pad))
            crop = gray[y0:y1, x0:x1].copy()
            crop[masks.vertical[y0:y1, x0:x1] | masks.horizontal[y0:y1, x0:x1]] = 255
            tid = f"colnum_{i}"
            owner[tid] = ci
            shape = "narrow" if b.width < 1.2 * b.height else "wide"
            # header numbers are often printed smaller than the body: scale by their own height
            requests.append(OCRRequest(tid, crop, "integer", "0123456789", glyph_height=float(b.height), shape=shape))
        read: Dict[str, Dict[str, str]] = {}
        for engine in (self.engines.tesseract, self.engines.easyocr):
            if engine is None or not requests:
                continue
            for r in engine.read(requests):
                if r.variant.startswith("free"):
                    continue  # unrestricted Tesseract variant: letters are no column-number evidence
                read.setdefault(r.token_id, {})[r.engine] = r.raw_text
                layers_by_column[owner[r.token_id]].setdefault(r.engine, []).append(r.raw_text.strip())
        table.column_agreement["column_number_readings"] = {str(columns[owner[t]].index): v for t, v in read.items()}

    @staticmethod
    def _locate_column_numbers(
        masks: LineMasks, region: Box, header: _HeaderEnd, columns: List[ColumnInterval], pc: PageColumns
    ) -> Optional[List[Tuple[Optional[int], Box]]]:
        if header.number_boxes:
            return [(next((k for k, c in enumerate(columns) if c.x0 <= b.cx < c.x1), None), b) for b in header.number_boxes]
        band = Box(x0=region.x0, y0=int(region.y0), x1=region.x1, y1=header.bottom)
        found = column_number_boxes(masks.text, band, columns, [c.column_number for c in pc.specs], header.glyph_height)
        if found is None:
            return None
        return [(k, b) for k, b in sorted(found.items())]

    # ------------------------------------------------------------- 7. geometry issues
    def _token_geometry_issues(
        self, table: TableRecord, layout: TokenLayout, pc: PageColumns, issues: List[Issue]
    ) -> List[Tuple[Issue, List[int]]]:
        """Classify cells that could not be assigned to a data line (noise, detached text, real problems)."""
        s = self.cfg.structure
        label_idx = next((i for i, c in enumerate(pc.specs) if c.role == "unit_label"), None)
        noise_limit = s.noise_token_max_size_ratio * layout.char_height
        lines_cy = np.array([line.cy for line in layout.lines]) if layout.lines else np.array([])
        single_glyph = {t.token_id for t in layout.tokens if len(t.component_labels) == 1}
        token_issue_lines: List[Tuple[Issue, List[int]]] = []
        for cell in table.cells:
            if label_idx is not None and cell.column_index == label_idx:
                self._label_column_mark(cell, layout, noise_limit, single_glyph, issues)
                continue
            if cell.assignment_status == "MARK":
                issues.append(
                    self.issues.make(
                        IssueCode.NOISE_TOKEN_IGNORED,
                        IssueScope.TABLE,
                        "print mark (star/underline) ignored",
                        cell_id=cell.cell_id,
                    )
                )
                continue
            if cell.assignment_status in ("UNASSIGNED", "AMBIGUOUS"):
                found = self._loose_cell_issue(cell, layout, lines_cy, noise_limit, issues)
                if found is not None:
                    token_issue_lines.append(found)
        return token_issue_lines

    def _label_column_mark(
        self, cell: CellRecord, layout: TokenLayout, noise_limit: float, single_glyph, issues: List[Issue]
    ) -> None:
        bw, bh = cell.box[2] - cell.box[0], cell.box[3] - cell.box[1]
        tiny = bw < noise_limit and bh < noise_limit
        # one isolated small blob (no word: headings and names have several glyphs)
        lone = cell.cell_id in single_glyph and max(bw, bh) <= 0.7 * layout.char_height
        if cell.assignment_status in ("UNASSIGNED", "AMBIGUOUS") and (tiny or lone):
            # a speck or mark between rows (read as '1', 'c' or "'") must not open a section header
            cell.assignment_status = "MARK"
            issues.append(
                self.issues.make(
                    IssueCode.NOISE_TOKEN_IGNORED, IssueScope.TABLE, "tiny label-column mark ignored", cell_id=cell.cell_id
                )
            )

    def _loose_cell_issue(
        self, cell: CellRecord, layout: TokenLayout, lines_cy: np.ndarray, noise_limit: float, issues: List[Issue]
    ) -> Optional[Tuple[Issue, List[int]]]:
        bw, bh = cell.box[2] - cell.box[0], cell.box[3] - cell.box[1]
        cy = (cell.box[1] + cell.box[3]) / 2.0
        near = [int(i) for i in np.argsort(np.abs(lines_cy - cy))[:2]] if lines_cy.size else []
        if bw < noise_limit and bh < noise_limit:
            issues.append(
                self.issues.make(
                    IssueCode.NOISE_TOKEN_IGNORED, IssueScope.TABLE, "tiny unassigned mark ignored", cell_id=cell.cell_id
                )
            )
            return None
        if (
            lines_cy.size
            and layout.pitch
            and float(np.min(np.abs(lines_cy - cy))) > self.cfg.structure.detached_token_min_pitch * layout.pitch
        ):
            # a heading/caption between blocks: it cannot belong to any data line
            issues.append(
                self.issues.make(
                    IssueCode.DETACHED_TEXT,
                    IssueScope.TABLE,
                    f"text in column {cell.column_key} away from every data line",
                    cell_id=cell.cell_id,
                )
            )
            return None
        if cell.assignment_status != "UNASSIGNED":
            return None
        issue = self.issues.make(
            IssueCode.TOKEN_UNASSIGNED,
            IssueScope.ROW,
            f"glyph token in column {cell.column_key} not assignable to a line",
            cell_id=cell.cell_id,
        )
        return issue, near


def glyph_height(text_mask: np.ndarray, region: Box) -> float:
    """Median height of the text components inside the region (20 px if there are none)."""
    x0, y0, x1, y1 = region.as_int_tuple()
    n, _, stats, _ = cv2.connectedComponentsWithStats(text_mask[y0:y1, x0:x1].astype(np.uint8), connectivity=8)
    heights = [int(stats[i][cv2.CC_STAT_HEIGHT]) for i in range(1, n) if stats[i][cv2.CC_STAT_HEIGHT] >= 6]
    return float(median(heights)) if heights else 20.0


__all__ = [
    "ExtractedTable",
    "PageColumns",
    "TableExtractor",
    "continuation_columns",
    "glyph_height",
    "left_columns",
    "mm_to_px",
]
