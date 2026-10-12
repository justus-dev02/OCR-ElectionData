"""
Page processing: rendered page(s) -> table -> cells (with OCR evidence) -> candidate rows.

The work is split into two phases with different execution constraints:

``extract`` / ``extract_spread``   (``pipeline.extraction``)
    Image analysis and OCR of one physical page or one page spread. Depends on
    nothing but the page itself and the profile, so it may run in parallel worker
    processes (see ``pipeline.runner``).

``interpret``                       (``pipeline.interpretation``)
    Line groups, sections, roles and number resolution. Sequential: the section
    context (e.g. the county) is carried from one page to the next.

``process`` / ``process_spread`` run both phases for one page (single-page callers, tests).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from election_ocr.config import OCRSettings, PipelineConfig
from election_ocr.ingestion.render import RenderedPage
from election_ocr.issues import Issue, IssueCode, IssueFactory, IssueScope
from election_ocr.layout.column_strategies import has_unnumbered_trailing_columns
from election_ocr.layout.tatr import TableDetector, TableStructureRecognizer
from election_ocr.ocr.engines import Engines
from election_ocr.pipeline.extraction import (
    ExtractedTable,
    PageColumns,
    TableExtractor,
    continuation_columns,
    left_columns,
)
from election_ocr.pipeline.interpretation import interpret_tables, pair_spread_lines
from election_ocr.pipeline.models import PageRecord
from election_ocr.reference.models import TableProfile, TableVariant
from election_ocr.semantics.rows import SectionContext

__all__ = [
    "Engines",
    "ExtractedTable",
    "PageColumns",
    "PageExtraction",
    "PageProcessor",
    "has_unnumbered_trailing_columns",
    "interpret_extraction",
]


@dataclass
class PageExtraction:
    """Extraction result of one page (``parts=[left]``) or one spread (``parts=[left, right]``).

    Picklable: this is what a worker process sends back to the runner.
    """

    pages: Tuple[PageRecord, ...]
    parts: List[ExtractedTable]
    pairing: Dict[int, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(p.ok for p in self.parts)

    def failure_issue(self) -> Issue:
        failed = next(p for p in self.parts if not p.ok)
        return failed.table.issues[-1]


class PageProcessor:
    def __init__(
        self,
        config: PipelineConfig,
        detector: TableDetector,
        structure: TableStructureRecognizer,
        engines: Engines,
        issues: IssueFactory,
    ):
        self.cfg = config
        self.issues = issues
        self.extractor = TableExtractor(config, detector, structure, engines, issues)

    # ------------------------------------------------------------ phase 1: extraction
    def extract(self, page_id: str, rendered: RenderedPage, tp: TableProfile, variant: TableVariant) -> PageExtraction:
        page = self._page_record(page_id, rendered)
        ext = self.extractor.extract(page, rendered, tp, variant, left_columns(tp, variant))
        page.tables.append(ext.table)
        return PageExtraction(pages=(page,), parts=[ext.compact()])

    def extract_spread(
        self, page_ids: Tuple[str, str], rendered: Tuple[RenderedPage, RenderedPage], tp: TableProfile, variant: TableVariant
    ) -> PageExtraction:
        """Left page + continuation page printed as one table (same lines, further columns)."""
        left_page = self._page_record(page_ids[0], rendered[0])
        right_page = self._page_record(page_ids[1], rendered[1])
        left = self.extractor.extract(left_page, rendered[0], tp, variant, left_columns(tp, variant))
        right = self.extractor.extract(right_page, rendered[1], tp, variant, continuation_columns(variant))
        left_page.tables.append(left.table)
        right_page.tables.append(right.table)
        extraction = PageExtraction(pages=(left_page, right_page), parts=[left.compact(), right.compact()])
        if extraction.ok:
            assert left.layout is not None and right.layout is not None
            extraction.pairing, stitch_issues = pair_spread_lines(left.layout, right.layout, self.issues)
            left.table.issues.extend(stitch_issues)
            right.table.issues.extend(stitch_issues)
        return extraction

    # ------------------------------------------------------------ phase 2: interpretation
    def interpret(
        self, extraction: PageExtraction, tp: TableProfile, variant: TableVariant, carry: SectionContext
    ) -> SectionContext:
        return interpret_extraction(extraction, tp, variant, carry, self.cfg.ocr, self.issues)

    # ------------------------------------------------------------ both phases
    def process(
        self, page_id: str, rendered: RenderedPage, tp: TableProfile, variant: TableVariant, carry: SectionContext
    ) -> Tuple[PageRecord, SectionContext]:
        extraction = self.extract(page_id, rendered, tp, variant)
        return extraction.pages[0], self.interpret(extraction, tp, variant, carry)

    def process_spread(
        self,
        page_ids: Tuple[str, str],
        rendered: Tuple[RenderedPage, RenderedPage],
        tp: TableProfile,
        variant: TableVariant,
        carry: SectionContext,
    ) -> Tuple[Tuple[PageRecord, PageRecord], SectionContext]:
        extraction = self.extract_spread(page_ids, rendered, tp, variant)
        left, right = extraction.pages
        return (left, right), self.interpret(extraction, tp, variant, carry)

    # ------------------------------------------------------------ helpers
    def _page_record(self, page_id: str, rendered: RenderedPage) -> PageRecord:
        h, w = rendered.gray.shape
        page = PageRecord(
            page_id=page_id,
            page_number=rendered.page_number,
            width_px=w,
            height_px=h,
            dpi=rendered.dpi,
            image_path=str(rendered.image_path),
            image_sha256=rendered.image_sha256,
            deskew_angle=rendered.deskew_angle,
            deskew_applied=rendered.deskew_applied,
            rotation_matrix=rendered.rotation_matrix.tolist() if rendered.rotation_matrix is not None else None,
            has_text_layer=bool(rendered.words),
            text_layer_word_count=len(rendered.words),
            quality=rendered.quality,
        )
        if abs(rendered.deskew_angle) >= self.cfg.render.min_deskew_degrees and not rendered.deskew_applied:
            page.issues.append(
                self.issues.make(
                    IssueCode.PAGE_SKEW_UNCORRECTED, IssueScope.PAGE, f"skew {rendered.deskew_angle} deg not corrected"
                )
            )
        return page


def interpret_extraction(
    extraction: PageExtraction,
    tp: TableProfile,
    variant: TableVariant,
    carry: SectionContext,
    ocr: OCRSettings,
    issues: IssueFactory,
) -> SectionContext:
    """Phase 2 for one extracted page/spread; returns the section context for the next page."""
    if not extraction.ok:
        # after an unreadable page the section context is no longer trustworthy
        return SectionContext(
            key=carry.key,
            kind=carry.kind,
            name=carry.name,
            level=carry.level,
            agreed=False,
            issues=[*carry.issues, extraction.failure_issue()],
        )
    return interpret_tables(tp, variant, extraction.parts, extraction.pairing, carry, ocr, issues)
