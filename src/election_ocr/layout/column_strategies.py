"""
Column partition strategies (Strategy pattern, selected by the table profile's
``structure.columns``).

Every strategy answers two questions for one physical page, purely from the
image geometry (no OCR):

1. Where is the table? (``locate_region``)
2. Which x-intervals are the profile's columns? (``partition``)

``body_rules``      vertical ruling lines crossing the body; faint rules need a
                    second evidence (clean whitespace gutter or an ML boundary)
``header_rules``    rules only in the header band; the body must not cross them
``column_numbers``  rule-less line-printer tables: printed column numbers +
                    body ink runs (see ``table_structure.columns_from_numbers``)

The strategies only reorganise the former monolithic extraction step; the
algorithms themselves live in ``layout.structure`` and ``layout.table_structure``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from election_ocr.config import StructureSettings
from election_ocr.geometry import Box
from election_ocr.layout.structure import (
    ColumnInterval,
    VerticalLine,
    add_ml_confirmed_weak_lines,
    deterministic_columns,
    erase_near_vertical_lines,
    find_vertical_lines,
)
from election_ocr.layout.table_structure import (
    HLine,
    StructureFinding,
    add_gutter_confirmed_weak_lines,
    body_ink_runs,
    columns_from_numbers,
    columns_from_separators,
    find_long_hlines,
    header_rule_separators,
    ink_region,
    region_from_rules,
)
from election_ocr.layout.tokens import connected_components
from election_ocr.preprocessing.image_ops import LineMasks
from election_ocr.reference.models import ColumnSpec, StructureSpec

# Region padding around the long rules / page ink. Labels of an unruled outer column
# may start a few millimetres outside the rules (1961: ~3 mm), hence the wider sides.
REGION_PAD_MM = 3.0
REGION_SIDE_PAD_MM = 6.0


def mm_to_px(mm: float, dpi: int) -> float:
    return dpi * mm / 25.4


def has_unnumbered_trailing_columns(specs: Sequence[ColumnSpec]) -> bool:
    """True if a column without printed number follows the last numbered column."""
    last = max((i for i, c in enumerate(specs) if c.column_number is not None), default=-1)
    return any(c.column_number is None for c in specs[last + 1 :])


@dataclass
class RegionResult:
    box: Optional[Box]
    source: Optional[str]  # page_ink | long_rules (ML fallback is the caller's)
    long_rules: List[HLine] = field(default_factory=list)


@dataclass
class PartitionContext:
    """Everything a strategy may look at (all in page pixels)."""

    masks: LineMasks
    region: Box
    body: Box
    header_bottom: int
    long_rules: List[HLine]
    ml_columns: List[Box]
    number_boxes: List[Box]
    glyph_height: float
    dpi: int
    specs: List[ColumnSpec]


@dataclass
class ColumnPartition:
    columns: List[ColumnInterval]
    text_mask: np.ndarray  # mask the glyph tokens are built from
    separators: List[VerticalLine] = field(default_factory=list)
    findings: List[StructureFinding] = field(default_factory=list)
    evidence: Dict[str, object] = field(default_factory=dict)  # recorded in table.column_agreement

    def token_rules(self) -> List[Tuple[float, float, float]]:
        """Vertical rules (x, y0, y1) that separate words during anchor-line building."""
        return []


@dataclass
class _RuledPartition(ColumnPartition):
    """Body rules separate columns line by line: they are passed on to the token layout."""

    def token_rules(self) -> List[Tuple[float, float, float]]:
        return [(line.x, line.y0, line.y1) for line in self.separators]


class ColumnStrategy:
    """Base class: region + column partition for one physical page."""

    name: str = ""

    def __init__(self, spec: StructureSpec, settings: StructureSettings):
        self.spec = spec
        self.settings = settings

    def locate_region(self, masks: LineMasks, dpi: int) -> RegionResult:
        raise NotImplementedError

    def partition(self, ctx: PartitionContext) -> ColumnPartition:
        raise NotImplementedError

    # helpers shared by the ruled strategies
    def _rules_region(self, masks: LineMasks, dpi: int) -> RegionResult:
        h, w = masks.horizontal.shape
        pad = int(round(mm_to_px(REGION_PAD_MM, dpi)))
        long_rules = find_long_hlines(masks.horizontal, int(self.spec.region_rule_min_width_ratio * w))
        box = region_from_rules(long_rules, h, pad, int(round(mm_to_px(REGION_SIDE_PAD_MM, dpi))))
        return RegionResult(box=box, source="long_rules", long_rules=long_rules)

    def _erase_px(self, dpi: int) -> int:
        return max(1, int(round(mm_to_px(self.settings.ruling_line_erase_mm, dpi))))

    def _bridge_px(self, dpi: int) -> int:
        return int(round(mm_to_px(self.settings.ruling_line_bridge_mm, dpi)))


class BodyRulesStrategy(ColumnStrategy):
    name = "body_rules"

    def locate_region(self, masks: LineMasks, dpi: int) -> RegionResult:
        return self._rules_region(masks, dpi)

    def partition(self, ctx: PartitionContext) -> ColumnPartition:
        s = self.settings
        bridge = self._bridge_px(ctx.dpi)
        erase_px = self._erase_px(ctx.dpi)
        strong = find_vertical_lines(ctx.masks.vertical, ctx.region, s.min_ruling_line_height_ratio, bridge)
        weak = find_vertical_lines(ctx.masks.vertical, ctx.region, s.weak_ruling_line_height_ratio, bridge)
        pre_mask = erase_near_vertical_lines(ctx.masks.text, weak, ctx.body, erase_px)
        _, pre_comps, _ = connected_components(pre_mask, ctx.body, s.min_component_area_px)
        pre_boxes = [c.box for c in pre_comps]
        max_off = mm_to_px(s.weak_line_ml_max_offset_mm, ctx.dpi)
        # faint rules need a second evidence: a clean body gutter (deterministic) ...
        separators = add_gutter_confirmed_weak_lines(strong, weak, pre_boxes, max_off, tolerance_px=1.0)
        # ... or, recorded as such, an ML column boundary
        separators = add_ml_confirmed_weak_lines(separators, weak, ctx.ml_columns, pre_boxes, max_offset_px=max_off)
        text_mask = erase_near_vertical_lines(ctx.masks.text, separators, ctx.body, erase_px)
        columns = deterministic_columns(text_mask, separators, ctx.body)
        return _RuledPartition(columns=columns, text_mask=text_mask, separators=separators)


class HeaderRulesStrategy(ColumnStrategy):
    name = "header_rules"

    def locate_region(self, masks: LineMasks, dpi: int) -> RegionResult:
        return self._rules_region(masks, dpi)

    def partition(self, ctx: PartitionContext) -> ColumnPartition:
        hb = ctx.header_bottom
        header_band = Box(x0=ctx.region.x0, y0=ctx.region.y0, x1=ctx.region.x1, y1=hb + 3)
        # separators below spanning header cells cover only the lower part of the header:
        # accept rules covering >= 1/4 of the header that reach the header's bottom rule
        separators = [
            line
            for line in header_rule_separators(ctx.masks.vertical, header_band, 0.25, self._bridge_px(ctx.dpi))
            if line.y1 >= hb - mm_to_px(1.5, ctx.dpi)
        ]
        text_mask = ctx.masks.text
        _, pre_comps, _ = connected_components(text_mask, ctx.body, self.settings.min_component_area_px)
        boxes = [c.box for c in pre_comps]
        # open outer columns: the header box (long rules) gives the table width even when the
        # outermost column is empty on this page; body ink reaching further out extends it
        rule_left = min((line.x0 for line in ctx.long_rules), default=ctx.body.x0)
        rule_right = max((line.x1 for line in ctx.long_rules), default=ctx.body.x1)
        ink_left = min([b.x0 for b in boxes] + [rule_left])
        ink_right = max([b.x1 for b in boxes] + [rule_right])
        columns = columns_from_separators([line.x for line in separators], ctx.body, ink_left, ink_right, edge_margin=12)
        return ColumnPartition(columns=columns, text_mask=text_mask, separators=separators)


class ColumnNumbersStrategy(ColumnStrategy):
    name = "column_numbers"

    def locate_region(self, masks: LineMasks, dpi: int) -> RegionResult:
        pad = int(round(mm_to_px(REGION_PAD_MM, dpi)))
        return RegionResult(box=ink_region(masks.text, pad), source="page_ink")

    def partition(self, ctx: PartitionContext) -> ColumnPartition:
        char_h = ctx.glyph_height
        text_mask = ctx.masks.text
        _, pre_comps, _ = connected_components(text_mask, ctx.body, self.settings.min_component_area_px)
        # leader dots and decimal commas sit on the printer's character grid and would split
        # columns into dot-wide runs: only glyph-sized components define the ink runs
        glyphs = [c.box for c in pre_comps if c.box.height >= 0.6 * char_h]
        # monospace print leaves a blank strip between character cells on every line (a
        # narrow '1' centred in its cell leaves up to ~0.5 glyph heights); column gutters
        # are at least one blank cell (~0.85 glyph heights) wide
        runs = body_ink_runs(glyphs, ctx.body, 1.5 * char_h, max_noise_lines=2, merge_gap_px=0.6 * char_h)
        findings: List[StructureFinding] = []
        # the table ends with its last printed column number: ink starting more than one
        # column pitch further right (scan edge, binding shadow) is outside the table
        numbers = ctx.number_boxes
        pitches = [b.cx - a.cx for a, b in zip(numbers, numbers[1:])]
        right_limit = numbers[-1].x1 + (float(median(pitches)) if pitches else 0.0)
        outside = [] if has_unnumbered_trailing_columns(ctx.specs) else [r for r in runs if r[0] > right_limit]
        if outside:
            findings.append(
                StructureFinding(
                    "DETACHED_TEXT",
                    f"{len(outside)} ink run(s) right of the table (beyond the last column number) ignored",
                    {"runs": [[round(a, 1), round(b, 1)] for a, b in outside]},
                )
            )
            runs = [r for r in runs if r not in outside]
        columns, partition_findings = columns_from_numbers(
            runs,
            numbers,
            [c.column_number for c in ctx.specs],
            ctx.body,
            elastic=[c.role == "unit_label" for c in ctx.specs],
            min_run_width=0.25 * char_h,
        )
        return ColumnPartition(
            columns=columns or [],
            text_mask=text_mask,
            findings=findings + partition_findings,
            evidence={
                "body_runs": [[round(a, 1), round(b, 1)] for a, b in runs],
                "column_numbers": [b.as_list() for b in numbers],
            },
        )


_STRATEGIES = {s.name: s for s in (BodyRulesStrategy, HeaderRulesStrategy, ColumnNumbersStrategy)}


def strategy_for(spec: StructureSpec, settings: StructureSettings) -> ColumnStrategy:
    return _STRATEGIES[spec.columns](spec, settings)
