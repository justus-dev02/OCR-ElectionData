"""
Ruling-line geometry and the (recorded-only) agreement check against TATR.

* ``find_vertical_lines`` / ``find_full_width_hlines``: ruling lines from the line masks
* ``deterministic_columns``: column intervals between vertical rules (body_rules strategy)
* ``compare_columns``: does the ML column structure (Table Transformer) support the
  deterministic partition? The result is stored as evidence (INFO issue
  ``ML_STRUCTURE_DISAGREEMENT``); it never decides the structure, because TATR
  disagreed on 18-36 % of the pages and fails on the rule-less 1961-1969 layouts
  (docs/decisions.md, D4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import numpy as np

from election_ocr.geometry import Box


@dataclass
class VerticalLine:
    x: int
    y0: int
    y1: int
    coverage: float = 1.0
    evidence: str = "strong"  # strong | weak+ml

    @property
    def length(self) -> int:
        return self.y1 - self.y0


@dataclass
class ColumnInterval:
    index: int
    x0: float
    x1: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def width(self) -> float:
        return self.x1 - self.x0


@dataclass
class ColumnAgreement:
    agree: bool
    expected: int
    deterministic_count: int
    ml_count: int
    max_center_deviation_ratio: Optional[float]
    details: List[dict] = field(default_factory=list)
    reason: Optional[str] = None


def _runs(values: np.ndarray) -> List[Tuple[int, int]]:
    """Consecutive index runs where ``values`` is True -> [(start, end_exclusive)]."""
    runs: List[Tuple[int, int]] = []
    start = None
    for i, v in enumerate(values):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(values)))
    return runs


def find_vertical_lines(
    vertical_mask: np.ndarray, region: Box, min_height_ratio: float, bridge_gap_px: int = 0
) -> List[VerticalLine]:
    """Vertical ruling lines covering at least ``min_height_ratio`` of the region height.

    Faintly printed rules break into segments; gaps up to ``bridge_gap_px`` are
    closed morphologically before measuring coverage (measurement only - the
    text mask is not modified). Spurious extra lines are harmless for precision:
    they change the column count and are caught by the profile/ML checks.
    """
    x0, y0, x1, y1 = region.as_int_tuple()
    sub = vertical_mask[y0:y1, x0:x1]
    if sub.size == 0:
        return []
    if bridge_gap_px > 0:
        import cv2

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, bridge_gap_px))
        sub = cv2.morphologyEx(sub.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
    col_counts = sub.sum(axis=0)
    min_len = min_height_ratio * (y1 - y0)
    lines: List[VerticalLine] = []
    for a, b in _runs(col_counts >= min_len):
        xs = range(a, b)
        best = max(xs, key=lambda x: col_counts[x])
        ys = np.where(sub[:, best])[0]
        lines.append(
            VerticalLine(
                x=int(round((a + b - 1) / 2.0)) + x0,
                y0=int(ys.min()) + y0,
                y1=int(ys.max()) + y0,
                coverage=round(float(col_counts[best]) / (y1 - y0), 4),
            )
        )
    return lines


def add_ml_confirmed_weak_lines(
    strong: List[VerticalLine],
    weak_candidates: List[VerticalLine],
    ml_columns: List[Box],
    glyph_boxes: List[Box],
    max_offset_px: float,
) -> List[VerticalLine]:
    """Accept a faint ruling line only where the ML structure also puts a column boundary.

    Conditions: no strong line within ``max_offset_px`` of the ML boundary, a weak
    line within ``max_offset_px`` of it, and no glyph centre between the two.
    The accepted line is flagged ``evidence='weak+ml'`` for the audit trail.
    """
    ml_sorted = sorted(ml_columns, key=lambda b: b.cx)
    bounds = [(a.x1 + b.x0) / 2.0 for a, b in zip(ml_sorted, ml_sorted[1:])]
    heights = sorted(b.height for b in glyph_boxes if b.height > 3)
    char_h = heights[len(heights) // 2] if heights else 20.0
    out = list(strong)
    for b in bounds:
        if any(abs(l.x - b) <= max_offset_px for l in out):
            continue
        near = [w for w in weak_candidates if abs(w.x - b) <= max_offset_px]
        if not near:
            continue
        w = max(near, key=lambda l: l.coverage)
        lo, hi = sorted((w.x, b))
        if any(lo < gb.cx < hi and not _crosses_rule(gb, w.x, glyph_boxes, char_h) for gb in glyph_boxes):
            continue
        out.append(VerticalLine(x=w.x, y0=w.y0, y1=w.y1, coverage=w.coverage, evidence="weak+ml"))
    return sorted(out, key=lambda l: l.x)


def find_full_width_hlines(horizontal_mask: np.ndarray, region: Box, min_width_ratio: float = 0.85) -> List[int]:
    """Horizontal rules spanning ``min_width_ratio`` of the region width.

    Measured per BAND of rule rows, widened by a few rows, on the union across
    the band (small gaps bridged): a rule printed or scanned with a slight tilt
    covers the width only across several pixel rows.
    """
    from election_ocr.layout.table_structure import rule_extent

    x0, y0, x1, y1 = region.as_int_tuple()
    sub = horizontal_mask[y0:y1, x0:x1]
    if sub.size == 0:
        return []
    width = x1 - x0
    row_counts = sub.sum(axis=1)
    out: List[int] = []
    for a, b in _runs(row_counts >= 0.2 * width):
        a2, b2 = max(0, a - 6), min(sub.shape[0], b + 6)
        s0, s1 = rule_extent(sub[a2:b2], bridge_px=10)
        if s1 - s0 >= min_width_ratio * width:
            out.append(int(round((a + b - 1) / 2.0)) + y0)
    return out


def header_bottom(hlines: List[int], region: Box, max_header_ratio: float = 0.3) -> Optional[int]:
    """The lowest full-width horizontal rule within the top part of the table closes the header."""
    limit = region.y0 + max_header_ratio * region.height
    candidates = [y for y in hlines if region.y0 + 5 < y <= limit]
    return max(candidates) if candidates else None


def body_bottom(hlines: List[int], region: Box, header_y: int) -> int:
    """Last full-width rule clearly below the header (table bottom border), else region bottom."""
    below = [y for y in hlines if y > header_y + 0.2 * region.height]
    return max(below) if below else int(region.y1)


def deterministic_columns(
    text_mask: np.ndarray,
    vlines: List[VerticalLine],
    body: Box,
    edge_margin_px: int = 12,
    min_row_ink_ratio: float = 0.05,
) -> List[ColumnInterval]:
    """Column intervals between consecutive vertical ruling lines inside the body.

    Outer columns without a border line are closed by the horizontal extent of
    text ink in the body. Intervals without any text ink are dropped.
    """
    bx0, by0, bx1, by1 = body.as_int_tuple()
    sub = text_mask[by0:by1, bx0:bx1]
    ink_cols = np.where(sub.sum(axis=0) > 0)[0]
    if ink_cols.size == 0:
        return []
    ink_left, ink_right = bx0 + int(ink_cols.min()), bx0 + int(ink_cols.max()) + 1
    xs = sorted(v.x for v in vlines if bx0 - edge_margin_px <= v.x <= bx1 + edge_margin_px)
    bounds: List[float] = []
    if not xs or ink_left < xs[0] - edge_margin_px:
        bounds.append(float(ink_left))
    bounds.extend(float(x) for x in xs)
    if not xs or ink_right > xs[-1] + edge_margin_px:
        bounds.append(float(ink_right))
    intervals: List[ColumnInterval] = []
    body_rows = max(1, sub.shape[0])
    for a, b in zip(bounds[:-1], bounds[1:]):
        ia, ib = int(a) - bx0, int(b) - bx0
        ia, ib = max(0, ia), min(sub.shape[1], ib)
        if ib - ia <= 2:
            continue
        band = sub[:, ia + 2 : max(ia + 3, ib - 2)]
        # A real column has text on a noticeable share of the body rows; scan specks do not.
        if band.any(axis=1).sum() / body_rows < min_row_ink_ratio:
            continue
        intervals.append(ColumnInterval(index=len(intervals), x0=a, x1=b))
    return intervals


def merge_duplicate_ml_columns(ml_columns: List[Box], scores: Optional[List[float]] = None, x_iou: float = 0.5) -> List[Box]:
    """Collapse ML column boxes that cover (almost) the same x-range (keep higher score)."""
    items = list(zip(ml_columns, scores or [1.0] * len(ml_columns)))
    items.sort(key=lambda t: -t[1])
    kept: List[Box] = []
    for box, _ in items:
        dup = False
        for k in kept:
            inter = box.horizontal_overlap(k)
            union = max(box.x1, k.x1) - min(box.x0, k.x0)
            if union > 0 and inter / union > x_iou:
                dup = True
                break
        if not dup:
            kept.append(box)
    return sorted(kept, key=lambda b: b.cx)


def _crosses_rule(glyph: Box, rule_x: float, glyphs: List[Box], char_h: float) -> bool:
    """True if the glyph is part of a text run continuing across the rule (e.g. a heading)."""
    left = glyph.cx < rule_x
    for other in glyphs:
        if other is glyph or (other.cx < rule_x) == left:
            continue
        if other.vertical_overlap(glyph) < 0.5 * min(other.height, glyph.height):
            continue
        gap = other.x0 - glyph.x1 if left else glyph.x0 - other.x1
        if gap <= 1.5 * char_h:
            return True
    return False


def compare_columns(
    deterministic: List[ColumnInterval],
    ml_columns: List[Box],
    column_value_types: List[str],
    glyph_boxes: List[Box],
    max_distance_ratio: float,
) -> ColumnAgreement:
    """Check that the ML column structure supports the ruling-line structure.

    Rules:
    * the deterministic column count must equal the profile's column count;
    * every deterministic separator must be matched (one-to-one) by an ML column
      boundary that induces the SAME partition of glyphs, i.e. no glyph centre
      lies between the two boundaries (sanity bound: distance <=
      ``max_distance_ratio`` x the narrower adjacent column width). Glyphs of
      text runs that continue across the rule (headings) are not counted -
      numbers never cross a rule;
    * additional ML boundaries are tolerated only inside ``label`` columns
      (e.g. index number vs. name) - inside numeric/marker columns they signal
      a possibly missing printed rule and make the table disagree.
    """
    expected = len(column_value_types)
    ml_sorted = sorted(ml_columns, key=lambda b: b.cx)
    if len(deterministic) != expected:
        return ColumnAgreement(
            agree=False,
            expected=expected,
            deterministic_count=len(deterministic),
            ml_count=len(ml_sorted),
            max_center_deviation_ratio=None,
            reason="ruling-line column count differs from profile",
        )
    if len(ml_sorted) < 2:
        return ColumnAgreement(False, expected, len(deterministic), len(ml_sorted), None, reason="no ML columns")
    ml_bounds = [(a.x1 + b.x0) / 2.0 for a, b in zip(ml_sorted, ml_sorted[1:])]
    heights = sorted(b.height for b in glyph_boxes if b.height > 3)
    char_h = heights[len(heights) // 2] if heights else 20.0
    used: Set[int] = set()
    details: List[dict] = []
    worst = 0.0
    ok = True
    for left, right in zip(deterministic, deterministic[1:]):
        detail, ratio, matched_index = _match_separator(left, right, ml_bounds, used, glyph_boxes, char_h, max_distance_ratio)
        if ratio is not None:
            worst = max(worst, ratio)
        if matched_index is not None:
            used.add(matched_index)
        ok = ok and matched_index is not None
        details.append(detail)
    extra = _extra_ml_boundaries(ml_bounds, used, deterministic, column_value_types)
    ok = ok and all(e["tolerated"] for e in extra)
    return ColumnAgreement(
        agree=ok,
        expected=expected,
        deterministic_count=len(deterministic),
        ml_count=len(ml_sorted),
        max_center_deviation_ratio=round(worst, 4),
        details=[*details, {"extra_ml_boundaries": extra}],
        reason=None if ok else "ML column boundaries do not support the ruling-line separators",
    )


def _match_separator(
    left: ColumnInterval,
    right: ColumnInterval,
    ml_bounds: List[float],
    used: Set[int],
    glyph_boxes: List[Box],
    char_h: float,
    max_distance_ratio: float,
) -> Tuple[dict, Optional[float], Optional[int]]:
    """Nearest unused ML boundary to the separator between ``left`` and ``right``; it matches when
    it is close enough and no glyph (other than text running across the rule) lies in between.
    Returns (evidence, deviation ratio, matched ML boundary index or None)."""
    sep = right.x0
    allowed = max_distance_ratio * min(left.width, right.width)
    best: Optional[Tuple[int, float]] = None
    for i, b in enumerate(ml_bounds):
        if i in used:
            continue
        dev = abs(b - sep)
        if best is None or dev < best[1]:
            best = (i, dev)
    glyphs_between = 0
    spanning_ignored = 0
    if best is not None:
        lo, hi = sorted((sep, ml_bounds[best[0]]))
        for gb in glyph_boxes:
            if lo < gb.cx < hi:
                if _crosses_rule(gb, sep, glyph_boxes, char_h):
                    spanning_ignored += 1
                else:
                    glyphs_between += 1
    matched = best is not None and best[1] <= allowed and glyphs_between == 0
    ratio = (best[1] / max(min(left.width, right.width), 1.0)) if best else None
    detail = {
        "separator_x": round(sep, 1),
        "ml_boundary_x": round(ml_bounds[best[0]], 1) if best else None,
        "deviation_ratio": round(ratio, 4) if ratio is not None else None,
        "glyphs_between": glyphs_between,
        "spanning_text_glyphs_ignored": spanning_ignored,
        "matched": matched,
    }
    return detail, ratio, best[0] if matched and best is not None else None


def _extra_ml_boundaries(
    ml_bounds: List[float], used: Set[int], deterministic: List[ColumnInterval], column_value_types: List[str]
) -> List[dict]:
    """ML boundaries without a separator: tolerated only inside label columns (index vs. name)."""
    extra = []
    for i, b in enumerate(ml_bounds):
        if i in used:
            continue
        col = next((c for c in deterministic if c.x0 <= b < c.x1), None)
        vtype = column_value_types[col.index] if col is not None else None
        extra.append({"ml_boundary_x": round(b, 1), "inside_column": col.index if col else None, "tolerated": vtype == "label"})
    return extra


def erase_near_vertical_lines(text_mask: np.ndarray, vlines: List[VerticalLine], body: Box, halfwidth: int) -> np.ndarray:
    """Remove short ruling-line fragments that survived the morphological opening."""
    out = text_mask.copy()
    y0, y1 = int(body.y0), int(body.y1)
    for v in vlines:
        out[y0:y1, max(0, v.x - halfwidth) : v.x + halfwidth + 1] = False
    return out


def select_tables(detections, iou_threshold: float = 0.5):
    """Non-maximum suppression on detected table boxes (highest score first)."""
    kept: list = []
    for det in detections:
        if all(_iou(det.box, k.box) < iou_threshold for k in kept):
            kept.append(det)
    return kept


def _iou(a: Box, b: Box) -> float:
    inter = a.horizontal_overlap(b) * a.vertical_overlap(b)
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0
