"""
Glyph tokens and line assignment.

A *token* is the set of ink components inside one column that belong to one
printed text line (e.g. all digits of ``5 950``). Tokens are built purely from
the image (connected components of the text mask with ruling lines removed),
independent of any OCR engine. OCR is then run per token.

Algorithm (robust for dense tables with only a few pixels between lines):

1. Connected components of the text mask inside the table body.
2. Anchor lines: components of the profile's ``line_anchor_column`` are
   clustered by *vertical overlap* (glyphs of one line overlap, glyphs of
   neighbouring lines never do). Each cluster of normal glyph height = 1 line.
3. Every other component is assigned individually to its nearest anchor line.
   The assignment is rejected when the offset exceeds ``max_offset_ratio`` of
   the line pitch or when the nearest line is not clearly closer than the
   second nearest (``min_margin_ratio``).
4. Tokens = assigned components grouped by (column, line). Components that
   could not be assigned are clustered per column into separate tokens that
   carry their status (UNASSIGNED / AMBIGUOUS / OUTSIDE) - they are reported,
   never silently dropped or guessed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from election_ocr.geometry import Box
from election_ocr.layout.structure import ColumnInterval


@dataclass
class Component:
    label: int
    box: Box
    area: int
    column_index: Optional[int] = None
    crosses_boundary: bool = False
    line_index: Optional[int] = None
    status: str = "UNASSIGNED"
    distance_ratio: Optional[float] = None
    between: Optional[Tuple[int, int]] = None  # AMBIGUOUS: the two nearest lines it sits between


@dataclass
class Token:
    token_id: str
    column_index: int
    box: Box
    component_labels: List[int]
    crosses_boundary: bool = False
    line_index: Optional[int] = None
    line_distance_ratio: Optional[float] = None
    assignment_status: str = "UNASSIGNED"  # ASSIGNED | UNASSIGNED | AMBIGUOUS | OUTSIDE | MARK
    between_lines: Optional[Tuple[int, int]] = None  # AMBIGUOUS token centred between these two lines


@dataclass
class AnchorLine:
    index: int
    cy: float
    y0: float
    y1: float
    box: Box


@dataclass
class TokenLayout:
    labels: np.ndarray
    tokens: List[Token]
    lines: List[AnchorLine]
    pitch: Optional[float]
    char_height: float
    dropped_specks: int = 0
    notes: Dict[str, int] = field(default_factory=dict)


def connected_components(text_mask: np.ndarray, region: Box, min_area: int) -> Tuple[np.ndarray, List[Component], int]:
    x0, y0, x1, y1 = region.as_int_tuple()
    labels_full = np.zeros(text_mask.shape, dtype=np.int32)
    sub = text_mask[y0:y1, x0:x1].astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
    comps: List[Component] = []
    dropped = 0
    keep = np.zeros(n, dtype=bool)
    for i in range(1, n):
        cx, cy, w, h, area = (int(v) for v in stats[i])
        if area < min_area:
            dropped += 1
            continue
        keep[i] = True
        comps.append(Component(label=i, box=Box(x0=cx + x0, y0=cy + y0, x1=cx + w + x0, y1=cy + h + y0), area=area))
    labels[~keep[labels]] = 0
    labels_full[y0:y1, x0:x1] = labels
    return labels_full, comps, dropped


def _overlap_clusters(comps: List[Component], min_overlap_ratio: float = 0.3) -> List[List[Component]]:
    """Cluster components whose vertical extents overlap substantially."""
    clusters: List[List[Component]] = []
    for comp in sorted(comps, key=lambda c: c.box.cy):
        placed = False
        if clusters:
            cb = _box_of(clusters[-1])
            ov = comp.box.vertical_overlap(cb)
            if ov >= min_overlap_ratio * min(comp.box.height, cb.height):
                clusters[-1].append(comp)
                placed = True
        if not placed:
            clusters.append([comp])
    return clusters


def _box_of(comps: List[Component]) -> Box:
    box = comps[0].box
    for c in comps[1:]:
        box = box.union(c.box)
    return box


def assign_columns(components: List[Component], columns: List[ColumnInterval], char_height: float) -> None:
    tol = max(2.0, 0.15 * char_height)
    for comp in components:
        for c in columns:
            if c.x0 <= comp.box.cx < c.x1:
                comp.column_index = c.index
                break
        comp.crosses_boundary = any(comp.box.x0 < c.x0 - tol and comp.box.x1 > c.x0 + tol for c in columns[1:])
    _attach_trailing_signs(components, char_height)


def _attach_trailing_signs(components: List[Component], char_height: float) -> None:
    """A dash glued to the right of a digit ('5973-', trailing minus of a right-aligned number)
    belongs to that number's column even where it protrudes past the column boundary. A dash
    standing alone (printed zero) has no such neighbour and keeps its column."""
    glyphs = [c for c in components if c.column_index is not None and c.box.height >= 0.6 * char_height]
    dashes = [
        c
        for c in components
        if c.column_index is not None
        and c.box.height <= 0.35 * char_height
        and 0.3 * char_height <= c.box.width <= 1.2 * char_height
        and c.box.width >= 1.5 * c.box.height
    ]
    if not glyphs or not dashes:
        return
    gb = np.array([[g.box.x0, g.box.y0, g.box.x1, g.box.y1] for g in glyphs], dtype=float)
    max_gap = 0.5 * char_height
    for d in dashes:
        on_row = (gb[:, 1] <= d.box.cy) & (d.box.cy <= gb[:, 3])
        gap_left = d.box.x0 - gb[:, 2]
        gap_right = gb[:, 0] - d.box.x1
        left = np.where(on_row & (gap_left >= -1) & (gap_left <= max_gap))[0]
        right = np.where(on_row & (gap_right >= -1) & (gap_right <= max_gap))[0]
        if len(left) == 0 or len(right) > 0:
            continue  # nothing to attach to, or glued on both sides: leave it
        owner = glyphs[int(left[np.argmax(gb[left, 2])])]
        if owner.column_index != d.column_index:
            d.column_index = owner.column_index
            d.crosses_boundary = False


def estimate_char_height(components: List[Component], min_component_height: int) -> float:
    heights = [c.box.height for c in components if c.box.height >= min_component_height]
    return float(median(heights)) if heights else float(min_component_height)


def _part_of_text_run(
    cluster: List[Component],
    components: List[Component],
    max_gap: float,
    rules: Sequence[Tuple[float, float, float]] = (),
) -> bool:
    """True if a cluster glyph continues a word into a neighbouring column (heading, footnote).

    In data lines the anchor token stands alone between column gutters or rules; letters
    of a heading or footnote printed across columns are only a few pixels apart. In a
    ruled table (``rules`` = vertical rules (x, y0, y1)) a line next to a rule that covers it
    is inside the table; only text outside the rules (footnotes) is checked.
    """
    ref_h = _box_of(cluster).height

    def glyph(c: Component) -> bool:
        # letters/digits only: specks, and short box rules (taller than any text line) are no word evidence
        return 0.5 * ref_h <= c.box.height <= 1.6 * ref_h and c.box.width >= 0.15 * ref_h

    components = [c for c in components if glyph(c) or c in cluster]
    ids = {id(c) for c in cluster}
    probes = cluster
    if rules:
        b = _box_of(cluster)
        reach = max(1.5 * b.width, 3.0 * ref_h)  # narrow anchor columns (markers) need a wider window
        near = [r for r in rules if b.x0 - reach <= r[0] <= b.x1 + reach]
        if near and min(y0 for _, y0, _ in near) <= b.cy <= max(y1 for _, _, y1 in near):
            # within the vertical span of the rules next to the anchor (faint rules may have gaps):
            # inside the ruled table, columns are separated by the rules
            return False
        # outside the ruled area (footnotes): any word of the line crossing a column boundary
        band = _box_of(cluster)
        probes = [c for c in components if c.box.vertical_overlap(band) >= 0.5 * min(c.box.height, band.height)]
        ids = set()

    def close(a: Component, b: Component) -> bool:
        if b.box.vertical_overlap(a.box) < 0.5 * min(a.box.height, b.box.height):
            return False
        return max(b.box.x0 - a.box.x1, a.box.x0 - b.box.x1) <= max_gap

    if rules:
        for g in probes:
            for o in components:
                if id(o) not in ids and o.column_index != g.column_index and close(g, o):
                    return True
        return False
    # Without rules: a heading/footnote is ONE continuous chain of glyphs over several columns;
    # two large numbers almost touching across one gutter ('35250 1813310') are data.
    band = _box_of(cluster)
    row = [c for c in components if c.box.vertical_overlap(band) > 0]
    chain = list(cluster)
    seen = {id(c) for c in chain}
    i = 0
    while i < len(chain):
        g = chain[i]
        i += 1
        for o in row:
            if id(o) not in seen and close(g, o):
                seen.add(id(o))
                chain.append(o)
    return len({c.column_index for c in chain if c.column_index is not None}) >= 3


def build_anchor_lines(
    components: List[Component],
    anchor_column: int,
    char_height: float,
    confirm_column: Optional[int] = None,
    text_run_max_gap: Optional[float] = None,
    rules: Sequence[Tuple[float, float, float]] = (),
    run_components: Optional[List[Component]] = None,
) -> Tuple[List[AnchorLine], Optional[float], int]:
    """Anchor lines from the anchor column; returns (lines, pitch, rejected_count).

    With ``confirm_column`` an anchor cluster is only accepted if that column has
    ink overlapping the same line (a spilled-over header word is not a data line).
    With ``text_run_max_gap`` a cluster whose glyphs continue a word into a neighbouring
    column is rejected (centred headings, footnotes). ``run_components`` are the glyphs
    of the text mask BEFORE rule-strip erasure (letters crossing a column position are
    intact there); their column indices must be assigned.
    """
    anchor_comps = [c for c in components if c.column_index == anchor_column]
    confirm_boxes = [c.box for c in components if confirm_column is not None and c.column_index == confirm_column]
    lines: List[AnchorLine] = []
    rejected = 0
    for cl in _overlap_clusters(anchor_comps):
        box = _box_of(cl)
        # Specks / isolated punctuation must not create a text line.
        if box.height < 0.5 * char_height:
            continue
        if confirm_column is not None and not any(
            b.vertical_overlap(box) >= 0.3 * min(b.height, box.height) for b in confirm_boxes
        ):
            rejected += 1
            continue
        if text_run_max_gap is not None and _part_of_text_run(
            cl, run_components if run_components is not None else components, text_run_max_gap, rules
        ):
            rejected += 1
            continue
        lines.append(AnchorLine(index=len(lines), cy=box.cy, y0=box.y0, y1=box.y1, box=box))
    diffs = [b.cy - a.cy for a, b in zip(lines, lines[1:])]
    pitch = float(median(diffs)) if diffs else None
    return lines, pitch, rejected


def is_mark(comp: Component, line_height: float) -> bool:
    """Non-glyph print marks in data columns: winner stars ('***') and underlines.

    * small mark: both dimensions below ~half a glyph (stars, stray dots)
    * underline: one flat component much wider than a dash (>= 1.5 glyph heights)
    """
    w, h = comp.box.width, comp.box.height
    return (h < 0.45 * line_height and w < 0.6 * line_height) or (h <= 0.25 * line_height and w >= 1.5 * line_height)


def is_star_mark(comp: Component, line_height: float) -> bool:
    """A larger print mark (line-printer winner star, 0.45-0.75 glyph heights, not dash-shaped).
    Only taken as a mark where it touches no line band (checked by the caller): a broken digit
    fragment on a text line must never be dropped."""
    w, h = comp.box.width, comp.box.height
    return 0.45 * line_height <= h < 0.75 * line_height and w < 0.8 * line_height and w < 2.0 * h


def _star_triples(
    components: List[Component], lines: List[AnchorLine], line_height: float, label_column: Optional[int]
) -> Set[int]:
    """Labels of printed '***' groups (bold winner stars, almost glyph-sized): exactly three solid,
    roughly square blobs side by side on one row, touching no line band. Anything else between the
    lines stays a token (UNASSIGNED -> flagged), e.g. a number printed half a line off."""
    cand = [
        c
        for c in components
        if c.column_index is not None
        and c.column_index != label_column
        and 0.45 * line_height <= c.box.height < 0.95 * line_height
        and 0.7 <= c.box.width / max(c.box.height, 1.0) <= 1.4
        and c.area >= 0.6 * c.box.width * c.box.height
        and not any(c.box.vertical_overlap(l.box) > 0 for l in lines)
    ]
    cand.sort(key=lambda c: (c.column_index, c.box.x0))
    out: Set[int] = set()
    used: Set[int] = set()
    for i, c in enumerate(cand):
        if c.label in used:
            continue
        group = [c]
        for d in cand[i + 1 :]:
            last = group[-1]
            if (
                d.column_index == c.column_index
                and abs(d.box.cy - last.box.cy) <= 0.25 * line_height
                and 0 <= d.box.x0 - last.box.x1 <= 0.6 * line_height
            ):
                group.append(d)
        used.update(g.label for g in group)
        if len(group) == 3:
            out.update(g.label for g in group)
    return out


def assign_components_to_lines(
    components: List[Component],
    lines: List[AnchorLine],
    pitch: float,
    max_offset_ratio: float,
    min_margin_ratio: float,
    label_column: Optional[int] = None,
) -> None:
    if not lines:
        return
    centers = np.array([l.cy for l in lines])
    last_cy = lines[-1].cy
    line_height = float(median(l.box.height for l in lines))
    star_triples = _star_triples(components, lines, line_height, label_column)
    for comp in components:
        if comp.label in star_triples:
            comp.status = "MARK"
            continue
        if comp.column_index is None:
            comp.status = "OUTSIDE"
            continue
        if comp.column_index != label_column and is_mark(comp, line_height):
            overlapping = [i for i, l in enumerate(lines) if comp.box.vertical_overlap(l.box) > 0]
            flat = comp.box.height <= 0.25 * line_height and comp.box.width >= 1.5 * line_height
            if len(overlapping) == 1 and not flat:
                comp.status, comp.line_index = "ASSIGNED", overlapping[0]  # decimal comma / point of its line
            else:
                comp.status = "MARK"  # star or underline: never part of a value
            continue
        if comp.column_index != label_column and is_star_mark(comp, line_height):
            if not any(comp.box.vertical_overlap(l.box) > 0 for l in lines):
                comp.status = "MARK"
                continue
        if comp.box.cy > last_cy + 1.5 * pitch:
            comp.status = "OUTSIDE"
            continue
        d = np.abs(centers - comp.box.cy)
        order = np.argsort(d)
        d1 = float(d[order[0]])
        d2 = float(d[order[1]]) if len(order) > 1 else float("inf")
        comp.distance_ratio = round(d1 / pitch, 4)
        # Punctuation (decimal commas) hangs below the baseline, so its centre lies between
        # two lines; it still overlaps exactly one line band -> that line is unambiguous.
        # Only small marks: letters of a wrapped label line must not be pulled in by descenders.
        overlapping = (
            [i for i, l in enumerate(lines) if comp.box.vertical_overlap(l.box) > 0]
            if comp.box.height <= 0.6 * line_height
            else []
        )
        if len(overlapping) == 1:
            comp.status = "ASSIGNED"
            comp.line_index = overlapping[0]
            continue
        if d1 > max_offset_ratio * pitch:
            comp.status = "UNASSIGNED"
        elif (d2 - d1) < min_margin_ratio * pitch:
            comp.status = "AMBIGUOUS"
            comp.between = tuple(sorted((int(order[0]), int(order[1]))))  # type: ignore[assignment]
        else:
            comp.status = "ASSIGNED"
            comp.line_index = int(order[0])
    _enforce_word_coherence(components, line_height)


def _enforce_word_coherence(components: List[Component], line_height: float) -> None:
    """The glyphs of one printed word share one line: if their individual assignments differ (a
    number printed half-way between two lines: '1' and '3' ambiguous, '6' just inside one line),
    the whole word is AMBIGUOUS - a word fragment must never be read as a value of its own."""
    glyphs = [
        c
        for c in components
        if c.column_index is not None and c.status in ("ASSIGNED", "AMBIGUOUS") and c.box.height > 0.6 * line_height
    ]
    by_col: Dict[int, List[Component]] = {}
    for c in glyphs:
        if c.column_index is not None:
            by_col.setdefault(c.column_index, []).append(c)
    for comps in by_col.values():
        comps.sort(key=lambda c: c.box.x0)
        words: List[List[Component]] = []
        for c in comps:
            for w in words:
                last = w[-1]
                if c.box.x0 - last.box.x1 <= 0.6 * line_height and c.box.vertical_overlap(last.box) >= 0.5 * min(
                    c.box.height, last.box.height
                ):
                    w.append(c)
                    break
            else:
                words.append([c])
        for w in words:
            states = {(c.status, c.line_index) for c in w}
            if len(states) < 2:
                continue
            between = {c.between for c in w if c.between is not None}
            near = {c.line_index for c in w if c.line_index is not None}
            pair = next(iter(between)) if len(between) == 1 else None
            if pair is not None and not near <= set(pair):
                pair = None  # inconsistent neighbourhood: no 'between' evidence
            for c in w:
                c.status, c.line_index, c.between = "AMBIGUOUS", None, pair


def build_token_layout(
    labels: np.ndarray,
    components: List[Component],
    columns: List[ColumnInterval],
    anchor_column: int,
    token_prefix: str,
    min_component_height: int,
    max_offset_ratio: float,
    min_margin_ratio: float,
    dropped_specks: int = 0,
    confirm_column: Optional[int] = None,
    label_column: Optional[int] = None,
    text_run_max_gap_ratio: Optional[float] = None,
    rules: Sequence[Tuple[float, float, float]] = (),
    run_components: Optional[List[Component]] = None,
) -> TokenLayout:
    char_h = estimate_char_height(components, min_component_height)
    assign_columns(components, columns, char_h)
    if run_components is not None:
        assign_columns(run_components, columns, char_h)
    lines, pitch, rejected = build_anchor_lines(
        components,
        anchor_column,
        char_h,
        confirm_column,
        text_run_max_gap=text_run_max_gap_ratio * char_h if text_run_max_gap_ratio is not None else None,
        rules=rules,
        run_components=run_components,
    )
    if pitch:
        assign_components_to_lines(components, lines, pitch, max_offset_ratio, min_margin_ratio, label_column)
    grouped: Dict[Tuple[int, int], List[Component]] = {}
    loose: Dict[Tuple[int, str], List[Component]] = {}
    for comp in components:
        if comp.column_index is None:
            continue
        if comp.status == "ASSIGNED" and comp.line_index is not None:
            grouped.setdefault((comp.column_index, comp.line_index), []).append(comp)
        else:
            loose.setdefault((comp.column_index, comp.status), []).append(comp)
    tokens = _line_tokens(grouped, token_prefix)
    tokens += _loose_tokens(loose, token_prefix, {t.token_id for t in tokens})
    merged_spans = 0
    if label_column is not None:
        tokens, merged_spans = merge_spanning_labels(tokens, label_column, char_h)
    tokens.sort(key=lambda t: (t.box.cy, t.column_index))
    return TokenLayout(
        labels=labels,
        tokens=tokens,
        lines=lines,
        pitch=pitch,
        char_height=char_h,
        dropped_specks=dropped_specks,
        notes={"rejected_anchor_clusters": rejected, "merged_spanning_labels": merged_spans},
    )


def _line_tokens(grouped: Dict[Tuple[int, int], List[Component]], token_prefix: str) -> List[Token]:
    """One token per (column, line) of assigned components."""
    tokens: List[Token] = []
    for (col, line), comps in sorted(grouped.items()):
        tokens.append(
            Token(
                token_id=f"{token_prefix}_c{col:02d}_l{line:03d}",
                column_index=col,
                box=_box_of(comps),
                component_labels=[c.label for c in comps],
                crosses_boundary=any(c.crosses_boundary for c in comps),
                line_index=line,
                line_distance_ratio=max(c.distance_ratio or 0.0 for c in comps),
                assignment_status="ASSIGNED",
            )
        )
    return tokens


def _loose_tokens(loose: Dict[Tuple[int, str], List[Component]], token_prefix: str, used_ids: Set[str]) -> List[Token]:
    """Components without a line: clustered per column and status into separate (reported) tokens."""
    tokens: List[Token] = []
    for (col, status), comps in sorted(loose.items()):
        for cl in _overlap_clusters(comps):
            box = _box_of(cl)
            base = f"{token_prefix}_c{col:02d}_{status[0].lower()}{int(box.y0):05d}x{int(box.x0):05d}"
            token_id, n = base, 1
            while token_id in used_ids:  # ids must be unique per table (DB primary key)
                token_id, n = f"{base}_{n}", n + 1
            used_ids.add(token_id)
            tokens.append(
                Token(
                    token_id=token_id,
                    column_index=col,
                    box=box,
                    component_labels=[c.label for c in cl],
                    crosses_boundary=any(c.crosses_boundary for c in cl),
                    line_index=None,
                    line_distance_ratio=min((c.distance_ratio for c in cl if c.distance_ratio is not None), default=None),
                    assignment_status=status,
                    between_lines=_common_between(cl),
                )
            )
    return tokens


def _common_between(comps: List[Component]) -> Optional[Tuple[int, int]]:
    pairs = {c.between for c in comps}
    return pairs.pop() if len(pairs) == 1 else None


TOKEN_SHAPES = ("dash", "narrow", "wide")


def classify_token_shape(
    own_mask: np.ndarray,
    glyph_height: float,
    dash_max_height_ratio: float,
    dash_min_aspect: float,
    narrow_max_width_ratio: float,
    min_component_area: int,
) -> str:
    """Geometric class of a token, determined from its own ink BEFORE any OCR reading.

    * ``dash``   - exactly one component, flat (height <= ``dash_max_height_ratio`` x glyph
                   height) and elongated (width >= ``dash_min_aspect`` x its height)
    * ``narrow`` - total ink width < ``narrow_max_width_ratio`` x glyph height (one glyph)
    * ``wide``   - everything else (multi-digit numbers, labels)

    The class selects engine preprocessing (calibrated per class) and is stored as
    evidence for the dash/shape consistency check; it never produces a value itself.
    """
    mask = own_mask.astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps = [stats[i] for i in range(1, n) if stats[i][cv2.CC_STAT_AREA] >= min_component_area]
    if not comps:
        return "narrow"
    if len(comps) == 1:
        w, h = int(comps[0][cv2.CC_STAT_WIDTH]), int(comps[0][cv2.CC_STAT_HEIGHT])
        if h <= dash_max_height_ratio * glyph_height and w >= dash_min_aspect * h:
            return "dash"
    x0 = min(int(c[cv2.CC_STAT_LEFT]) for c in comps)
    x1 = max(int(c[cv2.CC_STAT_LEFT] + c[cv2.CC_STAT_WIDTH]) for c in comps)
    return "narrow" if (x1 - x0) < narrow_max_width_ratio * glyph_height else "wide"


def merge_spanning_labels(tokens: List[Token], label_column: int, char_height: float) -> Tuple[List[Token], int]:
    """Merge a loose label token with loose tokens continuing it across column rules.

    Section headings are often wider than the label column; the part beyond the
    rule would otherwise appear as a stray token in a data column.
    """
    loose = [t for t in tokens if t.assignment_status != "ASSIGNED"]
    consumed = set()
    merged_count = 0
    for head in sorted((t for t in loose if t.column_index == label_column), key=lambda t: t.box.x0):
        if head.token_id in consumed:
            continue
        current = head
        while True:
            nxt = None
            for t in loose:
                if t.token_id in consumed or t.token_id == head.token_id or t.column_index <= current.column_index:
                    continue
                if t.box.vertical_overlap(current.box) < 0.5 * min(t.box.height, current.box.height):
                    continue
                if t.box.x0 - current.box.x1 > 2.0 * char_height:
                    continue
                if nxt is None or t.box.x0 < nxt.box.x0:
                    nxt = t
            if nxt is None:
                break
            consumed.add(nxt.token_id)
            head.box = head.box.union(nxt.box)
            head.component_labels = head.component_labels + nxt.component_labels
            current = nxt
            merged_count += 1
    return [t for t in tokens if t.token_id not in consumed], merged_count
