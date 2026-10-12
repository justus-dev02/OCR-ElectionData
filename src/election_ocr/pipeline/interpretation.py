"""
Interpretation of extracted tables: line groups, sections, roles and the final
number resolution.

This stage is sequential across pages: the section context (e.g. the county a
municipality belongs to) is carried from one page to the next.

* ``pair_spread_lines``  pairs the anchor lines of a left page and its continuation page
* ``interpret_tables``   runs the profile-driven ``TableInterpreter`` on one table or spread
* ``finalize_numbers``   resolves number cells with the format bound to their (column, slot)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from election_ocr.config import OCRSettings
from election_ocr.issues import Issue, IssueCode, IssueFactory, IssueScope
from election_ocr.layout.tokens import AnchorLine, TokenLayout
from election_ocr.ocr.resolver import resolve_cell, shape_conflict
from election_ocr.pipeline.extraction import ExtractedTable
from election_ocr.pipeline.models import CellRecord
from election_ocr.reference.models import NumberFormat, TableProfile, TableVariant
from election_ocr.semantics.rows import SectionContext, TableInterpreter

# Lines that may be dropped at the top of ONE side of a spread (headings printed on one page only).
_SPREAD_DROP_OPTIONS = [(0, 0), (1, 0), (2, 0), (0, 1), (0, 2)]


def pair_spread_lines(left: TokenLayout, right: TokenLayout, issues: IssueFactory) -> Tuple[Dict[int, int], List[Issue]]:
    """Pair the anchor lines of a left page and its continuation page (right line -> left line).

    Both pages print the same lines in the same order. Every paired line must sit at the
    same relative position (right page scaled to the left page's span, so scan scale
    differences cancel out) within half the inner line spacing. The unit index check
    per row (label index == index column) is the second, row-level evidence.

    Headings printed across the spread may create a line on one page only (e.g. a
    letter-spaced 'N o r d w u e r t t e m b e r g' reaching the anchor column), so up
    to two surplus lines are allowed at the top of ONE side and at the end; every
    other line must pair. Surplus left lines get no right half (rows stay blocked);
    surplus right lines are recorded; a real lost data line also breaks the section sums.
    """
    found: List[Issue] = []
    ll, rl = left.lines, right.lines
    if not ll or not rl or not left.pitch or not right.pitch:
        found.append(issues.make(IssueCode.PAGE_STITCH_MISMATCH, IssueScope.TABLE, "no anchor lines on one page of the spread"))
        return {}, found
    best: Optional[Tuple[int, float, int, int]] = None
    for drop_left, drop_right in _SPREAD_DROP_OPTIONS:
        fit = _pairing_fit(ll[drop_left:], rl[drop_right:], left.pitch)
        if fit is None:
            continue
        n, worst = fit
        if best is None or n > best[0] or (n == best[0] and worst < best[1]):
            best = (n, worst, drop_left, drop_right)
    if best is None:
        found.append(
            issues.make(
                IssueCode.PAGE_STITCH_MISMATCH,
                IssueScope.TABLE,
                f"lines of the spread pages cannot be paired ({len(ll)} left, {len(rl)} right)",
                left_lines=len(ll),
                right_lines=len(rl),
            )
        )
        return {}, found
    n, worst, drop_left, drop_right = best
    pairing = {b.index: a.index for a, b in zip(ll[drop_left : drop_left + n], rl[drop_right : drop_right + n])}
    paired_left = set(pairing.values())
    unpaired_left = [line.index for line in ll if line.index not in paired_left]
    unpaired_right = [line.index for line in rl if line.index not in pairing]
    if unpaired_left or unpaired_right:
        found.append(
            issues.make(
                IssueCode.DETACHED_TEXT,
                IssueScope.TABLE,
                f"spread lines paired with {len(unpaired_left)} surplus left / {len(unpaired_right)} surplus right line(s)",
                unpaired_left=unpaired_left,
                unpaired_right=unpaired_right,
                max_deviation_pitch=round(worst, 3),
            )
        )
    return pairing, found


def _pairing_fit(a_lines: List[AnchorLine], b_lines: List[AnchorLine], left_pitch: float) -> Optional[Tuple[int, float]]:
    """(paired line count, worst deviation in inner line spacings) or None if this alignment fails."""
    n = min(len(a_lines), len(b_lines))
    if n == 0 or len(a_lines) - n > 2 or len(b_lines) - n > 2:
        return None
    # positions relative to the first paired line; the right page is scaled by the ratio of
    # the paired spans (scan scale), deviations measured in the left page's INNER line spacing
    span_a = a_lines[n - 1].cy - a_lines[0].cy
    span_b = b_lines[n - 1].cy - b_lines[0].cy
    if n > 1 and (span_a <= 0 or span_b <= 0):
        return None
    scale = span_a / span_b if n > 1 else 1.0
    if abs(scale - 1.0) > 0.05:
        return None  # facing pages of one print differ in scale by a few percent at most
    diffs = sorted(y.cy - x.cy for x, y in zip(a_lines[: n - 1], a_lines[1:n]))
    inner = diffs[len(diffs) // 4] if diffs else left_pitch
    worst = max(abs((a.cy - a_lines[0].cy) - (b.cy - b_lines[0].cy) * scale) / inner for a, b in zip(a_lines[:n], b_lines[:n]))
    return (n, worst) if worst <= 0.5 else None


def _working_cells(
    variant: TableVariant, parts: List[ExtractedTable], pairing: Dict[int, int]
) -> Tuple[List[CellRecord], Dict[str, CellRecord]]:
    """Cells of all parts with column/line indices referring to variant.all_columns() and the
    left page's lines. Cells of the continuation page are copies; their resolutions are
    written back to the stored originals afterwards."""
    working: List[CellRecord] = []
    originals: Dict[str, CellRecord] = {}
    for pi, part in enumerate(parts):
        offset = 0 if pi == 0 else len(variant.columns)
        for c in part.table.cells:
            originals[c.cell_id] = c
            if pi == 0:
                working.append(c)
                continue
            line = pairing.get(c.line_index) if c.line_index is not None else None
            copy = c.model_copy()
            copy.column_index = c.column_index + offset
            copy.line_index = line
            if line is None and c.assignment_status == "ASSIGNED":
                copy.assignment_status = "UNASSIGNED"
            copy.readings = c.readings
            if c.between_lines is not None:
                a, b = (pairing.get(i) for i in c.between_lines)
                copy.between_lines = (a, b) if a is not None and b is not None else None
            working.append(copy)
    return working, originals


def interpret_tables(
    tp: TableProfile,
    variant: TableVariant,
    parts: List[ExtractedTable],
    pairing: Dict[int, int],
    carry: SectionContext,
    ocr: OCRSettings,
    issues: IssueFactory,
) -> SectionContext:
    """Interpret one table (``parts=[left]``) or a spread (``parts=[left, right]``); returns the new carry."""
    left = parts[0]
    table = left.table
    if left.layout is None:
        raise ValueError("cannot interpret a table whose extraction stopped")
    spread = len(parts) > 1
    stitched = not spread or bool(pairing)
    working, originals = _working_cells(variant, parts, pairing)
    interpreter = TableInterpreter(tp, variant, ocr.min_independent_sources, issues)
    result = interpreter.interpret(
        table.table_id,
        table.page_number,
        left.layout.lines,
        left.layout.pitch,
        working if stitched else list(left.table.cells),
        carry,
        related_table_ids=[p.table.table_id for p in parts[1:]],
    )
    # numbers are resolved with the format bound to their (column, slot)
    finalize_numbers(working, tp, variant, result.line_slots, ocr)
    for c in working:
        o = originals[c.cell_id]
        if o is not c:
            o.resolution, o.value_type, o.format_id, o.shape_conflict = c.resolution, c.value_type, c.format_id, c.shape_conflict
    table.issues.extend(result.issues)
    for pi, part in enumerate(parts):
        for issue, near in part.token_issue_lines:
            near_left = near if pi == 0 else [pairing[n] for n in near if n in pairing]
            for r in result.rows:
                if set(r.line_indices) & set(near_left):
                    r.issues.append(issue)
    if spread and not stitched:
        for r in result.rows:
            r.issues.extend(i for i in table.issues if i.code == IssueCode.PAGE_STITCH_MISMATCH)
    table.rows = result.rows
    table.section_headers = result.headers
    return result.carry


def finalize_numbers(
    cells: List[CellRecord], tp: TableProfile, variant: TableVariant, line_slots: Dict[int, int], ocr: OCRSettings
) -> None:
    cols = variant.all_columns()
    formats = tp.cell_formats(variant)
    named = dict(tp.formats)
    for c in cells:
        spec = cols[c.column_index]
        if spec.value_type not in ("integer", "decimal"):
            continue
        slot = line_slots.get(c.line_index) if c.line_index is not None else None
        fmt: Optional[NumberFormat] = formats.get((spec.key, slot)) if slot is not None else None
        fmt = fmt or spec.own_format()
        if fmt is None:  # pragma: no cover - integer/decimal columns always have their own format
            continue
        c.resolution = resolve_cell(
            c.readings,
            fmt.value_type,
            ocr,
            integer_format=fmt.integer_format,
            decimal_format=fmt.decimal_format,
        )
        c.value_type = fmt.value_type
        c.format_id = next((k for k, f in named.items() if f == fmt), "column")
        c.shape_conflict = shape_conflict(c.resolution, c.shape)
