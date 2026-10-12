"""
Deterministic table structure (region, header end, column partition).

A column structure is accepted only on two deterministic evidences; the ML
table model (TATR) is recorded, never decisive (docs/decisions.md, D4).
Three strategies, selected by the table profile (``structure.columns``):

``body_rules``      Vertical ruling lines crossing the table body define the
                    separators (faint rules only with a second confirmation:
                    a clean gutter or an ML boundary). Second evidence: the
                    printed header text of every column (``semantics.header``)
                    and no glyph crossing a separator.
``header_rules``    Rules exist only in the header band (body printed without
                    rules). The header rules define the separators; the body
                    must confirm them: no glyph may cross a separator.
``column_numbers``  No rules at all (line printer). The printed column-number
                    row below the header gives the position of every numbered
                    column (also of columns that are empty on this page); the
                    body's ink runs give the unnumbered columns. Every numbered
                    column must own at most one ink run containing its number,
                    runs never span two numbers, and the unnumbered columns
                    between two numbers must match the profile exactly.

The header end is a full-width rule (``rule``) or the column-number row
(``column_numbers``). Nothing here reads text: all decisions are geometric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from election_ocr.geometry import Box
from election_ocr.layout.structure import ColumnInterval, VerticalLine, _runs, find_vertical_lines


@dataclass
class HLine:
    y: int
    x0: int
    x1: int

    @property
    def length(self) -> int:
        return self.x1 - self.x0


@dataclass
class StructureFinding:
    """A reason why the structure is not accepted (converted to an Issue by the caller)."""

    code: str
    message: str
    details: dict = field(default_factory=dict)


def rule_extent(mask_band: np.ndarray, bridge_px: int) -> Tuple[int, int]:
    """Longest horizontal extent of a rule band (union over its rows, gaps <= bridge_px closed)."""
    union = mask_band.any(axis=0)
    runs = _runs(union)
    merged: List[List[int]] = []
    for a, b in runs:
        if merged and a - merged[-1][1] <= bridge_px:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    if not merged:
        return 0, 0
    a, b = max(merged, key=lambda r: r[1] - r[0])
    return int(a), int(b)


def find_long_hlines(horizontal_mask: np.ndarray, min_length_px: int, tilt_rows: int = 6, bridge_px: int = 10) -> List[HLine]:
    """Horizontal ruling lines of at least ``min_length_px`` (one entry per printed rule).

    A slightly tilted rule reaches the full length only across several pixel rows,
    so the extent is measured on the union of the band widened by ``tilt_rows``.
    Bands touching the top or bottom image border are scan edges, not rules.
    """
    h, w = horizontal_mask.shape
    out: List[HLine] = []
    row_counts = horizontal_mask.sum(axis=1)
    for a, b in _runs(row_counts >= 0.2 * min_length_px):
        if a == 0 or b >= h:
            continue  # band touching the image border: scan edge / binding shadow, never a printed rule
        a2, b2 = max(0, a - tilt_rows), min(h, b + tilt_rows)
        x0, x1 = rule_extent(horizontal_mask[a2:b2], bridge_px)
        if x1 - x0 >= min_length_px:
            out.append(HLine(y=int(round((a + b - 1) / 2.0)), x0=x0, x1=x1))
    return out


def region_from_rules(hlines: List[HLine], page_h: int, pad_px: int, side_pad_px: Optional[int] = None) -> Optional[Box]:
    """Table region spanned by the long horizontal rules: from the top rule down to the page bottom.

    Labels of open (unruled) outer columns may start slightly outside the rules, hence
    the wider ``side_pad_px``. The body is closed later by the last anchor line
    (tokens further below are OUTSIDE).
    """
    if not hlines:
        return None
    side = pad_px if side_pad_px is None else side_pad_px
    x0 = min(l.x0 for l in hlines)
    x1 = max(l.x1 for l in hlines)
    y0 = min(l.y for l in hlines)
    return Box(x0=max(0, x0 - side), y0=max(0, y0 - pad_px), x1=x1 + side, y1=page_h - 1)


def ink_region(text_mask: np.ndarray, pad_px: int) -> Optional[Box]:
    ys, xs = np.where(text_mask)
    if ys.size == 0:
        return None
    h, w = text_mask.shape
    return Box(
        x0=max(0, int(xs.min()) - pad_px),
        y0=max(0, int(ys.min()) - pad_px),
        x1=min(w - 1, int(xs.max()) + pad_px),
        y1=min(h - 1, int(ys.max()) + pad_px),
    )


def header_rule_separators(vertical_mask: np.ndarray, header: Box, min_coverage: float, bridge_gap_px: int) -> List[VerticalLine]:
    """Vertical rules inside the header band (``header_rules`` strategy)."""
    return find_vertical_lines(vertical_mask, header, min_coverage, bridge_gap_px)


# ----------------------------------------------------------- text line bands
def text_bands(text_mask: np.ndarray, region: Box, min_rows: int = 3) -> List[Tuple[int, int]]:
    """Horizontal bands of text (y ranges with ink) inside the region."""
    x0, y0, x1, y1 = region.as_int_tuple()
    prof = text_mask[y0:y1, x0:x1].any(axis=1)
    return [(a + y0, b + y0) for a, b in _runs(prof) if b - a >= min_rows]


def glyph_groups(text_mask: np.ndarray, band: Tuple[int, int], region: Box, gap_px: float) -> List[Box]:
    """Ink groups of one text band, split where the horizontal gap exceeds ``gap_px``."""
    x0, _, x1, _ = region.as_int_tuple()
    y0, y1 = band
    cols = text_mask[y0:y1, x0:x1].any(axis=0)
    runs = _runs(cols)
    groups: List[List[Tuple[int, int]]] = []
    for r in runs:
        if groups and r[0] - groups[-1][-1][1] <= gap_px:
            groups[-1].append(r)
        else:
            groups.append([r])
    out: List[Box] = []
    for g in groups:
        gx0, gx1 = g[0][0] + x0, g[-1][1] + x0
        sub = text_mask[y0:y1, gx0:gx1]
        ys = np.where(sub.any(axis=1))[0]
        out.append(Box(x0=gx0, y0=y0 + int(ys.min()), x1=gx1, y1=y0 + int(ys.max()) + 1))
    return out


def find_column_number_row(
    text_mask: np.ndarray,
    region: Box,
    expected_numbers: int,
    char_h: float,
    max_rel_y: float = 0.4,
) -> Optional[Tuple[Tuple[int, int], List[Box]]]:
    """The printed column-number row ('1  2  3 ... 11') closing a rule-less header.

    Criteria (all geometric): within the top ``max_rel_y`` of the region, a text
    band consisting of exactly ``expected_numbers`` ink groups, each at most two
    glyphs wide and of glyph height. If several bands qualify, the lowest one is
    used only if they are identical in group count - otherwise None (ambiguous).
    """
    limit = region.y0 + max_rel_y * region.height
    hits = []
    for band in text_bands(text_mask, region):
        if band[0] > limit:
            break
        groups = glyph_groups(text_mask, band, region, gap_px=1.2 * char_h)
        # scan specks (both dimensions far below a glyph) are no printed numbers; printed column
        # numbers share one font size, so a mark of clearly different height is no number either
        groups = [g for g in groups if g.height >= 0.4 * char_h or g.width >= 0.4 * char_h]
        if len(groups) > expected_numbers:
            med_h = float(np.median([g.height for g in groups]))
            groups = [g for g in groups if abs(g.height - med_h) <= 0.25 * med_h]
        if len(groups) != expected_numbers:
            continue
        if all(g.width <= 2.6 * char_h and 0.6 * char_h <= g.height <= 1.6 * char_h for g in groups):
            hits.append((band, groups))
    if not hits:
        return None
    return hits[-1]


def column_number_boxes(
    text_mask: np.ndarray,
    header: Box,
    columns: List[ColumnInterval],
    numbered: List[Optional[int]],
    char_h: float,
) -> Optional[Dict[int, Box]]:
    """Printed column numbers inside a header whose columns are already known (header rules).

    The lowest header text band in which EVERY numbered column holds exactly one glyph group of
    number size (specks ignored) is the number row; returns column index -> number box, else None.
    Unnumbered columns may hold anything (rule remnants, wrapped header words)."""
    for band in reversed(text_bands(text_mask, header)):
        groups = [
            g
            for g in glyph_groups(text_mask, band, header, gap_px=0.6 * char_h)
            if g.height >= 0.4 * char_h or g.width >= 0.4 * char_h
        ]
        out: Dict[int, Box] = {}
        ok = True
        for col, n in zip(columns, numbered):
            if n is None:
                continue
            inside = [g for g in groups if col.x0 <= g.cx < col.x1]
            if len(inside) != 1 or not (0.4 * char_h <= inside[0].height <= 1.6 * char_h and inside[0].width <= 2.6 * char_h):
                ok = False
                break
            out[col.index] = inside[0]
        if ok and out:
            return out
    return None


# ----------------------------------------------------------------- body runs
def body_ink_runs(
    comp_boxes: Sequence[Box], body: Box, line_height: float, max_noise_lines: int, merge_gap_px: float = 0.0
) -> List[Tuple[float, float]]:
    """x-intervals occupied by body ink on more than ``max_noise_lines`` text lines.

    Occupancy is counted per text line (components grouped into lines of
    ``line_height``), so a single heading crossing a gutter does not close it.
    Runs separated by less than ``merge_gap_px`` are joined (character-cell gaps
    of monospace print).
    """
    if not comp_boxes:
        return []
    width = int(body.x1 - body.x0) + 1
    rows = max(1, int((body.y1 - body.y0) / max(line_height, 1.0)) + 1)
    occ = np.zeros((rows, width), dtype=bool)
    for b in comp_boxes:
        r = min(rows - 1, max(0, int((b.cy - body.y0) / max(line_height, 1.0))))
        a, e = int(b.x0 - body.x0), int(b.x1 - body.x0)
        occ[r, max(0, a) : min(width, e + 1)] = True
    counts = occ.sum(axis=0)
    runs: List[List[float]] = []
    for start, end in _runs(counts > max_noise_lines):
        if runs and start - runs[-1][1] < merge_gap_px:
            runs[-1][1] = end
        else:
            runs.append([start, end])
    return [(a + body.x0, b + body.x0) for a, b in runs]


def columns_from_numbers(
    runs: List[Tuple[float, float]],
    number_boxes: List[Box],
    numbered: List[Optional[int]],
    body: Box,
    elastic: Optional[List[bool]] = None,
    min_run_width: float = 0.0,
) -> Tuple[Optional[List[ColumnInterval]], List[StructureFinding]]:
    """Assign body ink runs and printed column numbers to the profile's column sequence.

    ``numbered`` lists, per profile column (left to right), its printed column
    number or None; ``number_boxes`` are the number glyph groups left to right
    (their count was checked by ``find_column_number_row``).

    Rules: a run may belong to at most one printed number; between two numbered
    columns (and before the first / after the last) the unnumbered profile columns
    take the unowned runs in order, one run each - except ONE ``elastic`` column
    (the free-text label column, whose words and leader dots leave gaps on few
    lines), which absorbs all surplus runs between its fixed neighbours. Any other
    surplus or missing run rejects the structure.
    """
    elastic = elastic or [False] * len(numbered)
    runs = [r for r in runs if r[1] - r[0] >= min_run_width]
    numbers_in_order = [n for n in numbered if n is not None]
    if numbers_in_order != sorted(numbers_in_order) or len(numbers_in_order) != len(number_boxes):
        return None, [StructureFinding("TABLE_STRUCTURE_DISAGREEMENT", "profile column numbers not strictly ordered")]
    num_box = dict(zip(numbers_in_order, number_boxes))
    owner, finding = _run_owners(runs, num_box)
    if finding is not None:
        return None, [finding]
    _adopt_adjacent_runs(runs, owner, numbered, num_box)
    extents = _numbered_extents(runs, owner, numbered, num_box)
    finding = _fill_unnumbered_segments(runs, owner, numbered, elastic, extents)
    if finding is None:
        finding = _check_runs_used(runs, owner, extents)
    if finding is not None:
        return None, [finding]
    complete = [e for e in extents if e is not None]
    return _column_bounds(complete, body), []


def _run_owners(
    runs: List[Tuple[float, float]], num_box: Dict[int, Box]
) -> Tuple[List[Optional[int]], Optional[StructureFinding]]:
    """1) a run containing one printed number belongs to that numbered column (two numbers: reject)."""
    owner: List[Optional[int]] = []
    for a, b in runs:
        owners = [n for n, box in num_box.items() if a - 0.5 * box.width <= box.cx <= b + 0.5 * box.width]
        if len(owners) > 1:
            return owner, StructureFinding(
                "TABLE_STRUCTURE_DISAGREEMENT",
                f"body ink run {int(a)}-{int(b)} spans printed column numbers {owners}",
                {"run": [a, b], "numbers": owners},
            )
        owner.append(owners[0] if owners else None)
    return owner, None


def _adopt_adjacent_runs(
    runs: List[Tuple[float, float]], owner: List[Optional[int]], numbered: List[Optional[int]], num_box: Dict[int, Box]
) -> None:
    """1b) short right-aligned values ('466' below a centred '8') may not reach their number: a run
    between two ADJACENT numbered columns (no unnumbered profile column in between) belongs to the
    neighbour without a run of its own - only if exactly one neighbour qualifies."""
    numbers_in_order = [n for n in numbered if n is not None]
    pos_of = {n: p for p, n in enumerate(numbered) if n is not None}
    for i, (a, b) in enumerate(runs):
        if owner[i] is not None:
            continue
        cx = (a + b) / 2.0
        left_n = max((n for n in numbers_in_order if num_box[n].cx < cx), default=None)
        right_n = min((n for n in numbers_in_order if num_box[n].cx > cx), default=None)
        lo = pos_of[left_n] if left_n is not None else -1
        hi = pos_of[right_n] if right_n is not None else len(numbered)
        if any(numbered[p] is None for p in range(lo + 1, hi)):
            continue  # unnumbered columns in between: decided by the segment rule
        candidates = [n for n in (left_n, right_n) if n is not None and n not in owner]
        if len(candidates) == 1:
            owner[i] = candidates[0]


def _numbered_extents(
    runs: List[Tuple[float, float]], owner: List[Optional[int]], numbered: List[Optional[int]], num_box: Dict[int, Box]
) -> List[Optional[Tuple[float, float]]]:
    """x-extent of every numbered column: its run plus its printed number (number only if empty on this page)."""
    owned = {n: runs[i] for i, n in enumerate(owner) if n is not None}
    extents: List[Optional[Tuple[float, float]]] = [None] * len(numbered)
    for pos, n in enumerate(numbered):
        if n is not None:
            box = num_box[n]
            a, b = owned.get(n, (box.x0, box.x1))
            extents[pos] = (min(a, box.x0), max(b, box.x1))
    return extents


def _fill_unnumbered_segments(
    runs: List[Tuple[float, float]],
    owner: List[Optional[int]],
    numbered: List[Optional[int]],
    elastic: List[bool],
    extents: List[Optional[Tuple[float, float]]],
) -> Optional[StructureFinding]:
    """2) the unnumbered columns between two numbered ones take the unowned runs there, in order."""
    pos = 0
    while pos < len(numbered):
        if numbered[pos] is not None:
            pos += 1
            continue
        seg_start = pos
        while pos < len(numbered) and numbered[pos] is None:
            pos += 1
        seg = list(range(seg_start, pos))
        left_ext = extents[seg_start - 1] if seg_start > 0 else None
        right_ext = extents[pos] if pos < len(numbered) else None
        left_x = left_ext[1] if left_ext is not None else float("-inf")
        right_x = right_ext[0] if right_ext is not None else float("inf")
        seg_runs = [r for r, o in zip(runs, owner) if o is None and left_x <= (r[0] + r[1]) / 2 <= right_x]
        flexible = [p for p in seg if elastic[p]]
        if len(seg_runs) < len(seg) or (len(seg_runs) > len(seg) and len(flexible) != 1):
            return StructureFinding(
                "TABLE_STRUCTURE_DISAGREEMENT",
                f"{len(seg_runs)} ink run(s) for the {len(seg)} unnumbered column(s) at positions {seg}",
                {"runs": seg_runs, "positions": seg},
            )
        if len(seg_runs) == len(seg):
            for p, r in zip(seg, seg_runs):
                extents[p] = r
            continue
        k = seg.index(flexible[0])
        before, after = seg[:k], seg[k + 1 :]
        for p, r in zip(before, seg_runs[: len(before)]):
            extents[p] = r
        for p, r in zip(after, seg_runs[len(seg_runs) - len(after) :]):
            extents[p] = r
        middle = seg_runs[len(before) : len(seg_runs) - len(after)]
        extents[flexible[0]] = (middle[0][0], middle[-1][1])
    return None


def _check_runs_used(
    runs: List[Tuple[float, float]], owner: List[Optional[int]], extents: List[Optional[Tuple[float, float]]]
) -> Optional[StructureFinding]:
    """3) every run must be used and adjacent columns must not overlap."""
    used = sum(1 for o in owner if o is not None) + sum(
        1 for r, o in zip(runs, owner) if o is None and any(e is not None and e[0] <= r[0] and r[1] <= e[1] for e in extents)
    )
    if used != len(runs):
        return StructureFinding("TABLE_STRUCTURE_DISAGREEMENT", f"{len(runs) - used} ink run(s) outside every profile column")
    complete = [e for e in extents if e is not None]
    for (a0, a1), (b0, b1) in zip(complete, complete[1:]):
        if b0 <= a1:
            return StructureFinding(
                "TABLE_STRUCTURE_DISAGREEMENT", "adjacent columns overlap", {"left": [a0, a1], "right": [b0, b1]}
            )
    return None


def _column_bounds(extents: List[Tuple[float, float]], body: Box) -> List[ColumnInterval]:
    """Column boundaries in the middle of the gutters; outer columns end half a typical gutter
    beyond their ink (not at the page edge, where scan artefacts would fall into them)."""
    gaps = sorted(b0 - a1 for (_, a1), (b0, _) in zip(extents, extents[1:]))
    half_gap = gaps[len(gaps) // 2] / 2.0 if gaps else 0.0
    left = max(float(body.x0), extents[0][0] - half_gap)
    right = min(float(body.x1), extents[-1][1] + half_gap)
    bounds = [left] + [(a1 + b0) / 2.0 for (_, a1), (b0, _) in zip(extents, extents[1:])] + [right]
    return [ColumnInterval(index=i, x0=a, x1=b) for i, (a, b) in enumerate(zip(bounds, bounds[1:]))]


def columns_from_separators(
    separators: List[float], body: Box, ink_left: float, ink_right: float, edge_margin: float
) -> List[ColumnInterval]:
    """Columns between separator x positions; open outer columns are closed by the ink extent."""
    xs = sorted(x for x in separators if body.x0 - edge_margin <= x <= body.x1 + edge_margin)
    bounds: List[float] = []
    if not xs or ink_left < xs[0] - edge_margin:
        bounds.append(float(ink_left))
    bounds.extend(float(x) for x in xs)
    if not xs or ink_right > xs[-1] + edge_margin:
        bounds.append(float(ink_right))
    return [ColumnInterval(index=i, x0=a, x1=b) for i, (a, b) in enumerate(zip(bounds, bounds[1:])) if b - a > 2]


def clean_gutter(x: float, comp_boxes: Sequence[Box], tolerance_px: float) -> bool:
    """True if no glyph of the body overlaps the vertical line at ``x`` (+- tolerance)."""
    return not any(b.x0 - tolerance_px <= x <= b.x1 + tolerance_px for b in comp_boxes)


def add_gutter_confirmed_weak_lines(
    lines: List[VerticalLine],
    weak_candidates: List[VerticalLine],
    glyph_boxes: Sequence[Box],
    min_distance_px: float,
    tolerance_px: float,
) -> List[VerticalLine]:
    """Accept a faint ruling line that runs through a clean body gutter (second evidence).

    A faint rule candidate (vertical ink covering part of the table height) is
    accepted when no glyph of the body touches it: printed rule + whitespace
    alignment of the text are independent evidences for the same separator.
    """
    out = list(lines)
    for w in sorted(weak_candidates, key=lambda l: -l.coverage):
        if any(abs(l.x - w.x) <= min_distance_px for l in out):
            continue
        if clean_gutter(w.x, glyph_boxes, tolerance_px):
            out.append(VerticalLine(x=w.x, y0=w.y0, y1=w.y1, coverage=w.coverage, evidence="weak+gutter"))
    return sorted(out, key=lambda l: l.x)
