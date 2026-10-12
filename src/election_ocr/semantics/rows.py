"""
Row / section interpretation of a reconstructed table.

Turns anchor lines + cells into candidate result rows according to the table
profile and tracks the section context (e.g. county) across lines and pages.

Line groups and roles
---------------------
A *line group* is what the source prints for one unit: one line
(``single_line``) or several lines (``marker_group``, e.g. Erststimmen /
Zweitstimmen / percentages). Groups are formed by vertical gaps
(``group_by: gap``) or start at every line carrying a label or index
(``group_by: label_start``); a group must have exactly the profile's slot
count, otherwise it is reported and not interpreted.

Every *role* of the profile produces one result row per group. Its fields are
read from explicit bindings (column + slot), so values printed once for the
whole group (e.g. eligible voters) or columns belonging to different vote
types on the same line (page spreads) are expressed in the profile, not in
code. Marker columns are verified per slot (value / ink required / must be
empty), and any ink inside the group that no binding consumes is reported -
a value in an unexpected place means the group is not what the profile says.

Label semantics (section header / heading / total row / unit row) are decided
by independent classification of every OCR reading (see ``labels.py``) and
accepted only when enough independence groups agree without contradiction.
Section identity is order based: a new header opens a new section, a
'continued' header or a header-less page start continues the current one.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Set, Tuple

from election_ocr.issues import Issue, IssueCode, IssueFactory, IssueScope
from election_ocr.layout.tokens import AnchorLine
from election_ocr.ocr.resolver import CellStatus
from election_ocr.parsing.text import exact_key, normalize_unicode
from election_ocr.pipeline.models import CellRecord, LabelReading, RowRecord, SectionHeaderRecord
from election_ocr.reference.models import PARTY_FIELD_PREFIX, TableProfile, TableVariant
from election_ocr.semantics.labels import AgreedClass, LabelClassifier, agree


@dataclass
class SectionContext:
    key: Optional[str]
    kind: Optional[str]
    name: Optional[str]
    level: Optional[str]
    agreed: bool
    issues: List[Issue] = field(default_factory=list)
    name_readings: Dict[str, str] = field(default_factory=dict)
    grouping: bool = False  # structural heading ('Stadtkreise'): no entity, no printed total
    unit_level: Optional[str] = None  # level of unit rows inside a grouping
    closed: bool = False  # a footer total row has closed the section
    index: Optional[str] = None  # printed section key ('131 LKR. AALEN')
    implicit: bool = False  # block without header, closed by its total row


def empty_context() -> SectionContext:
    return SectionContext(key=None, kind=None, name=None, level=None, agreed=False)


@dataclass
class Interpretation:
    rows: List[RowRecord]
    headers: List[SectionHeaderRecord]
    carry: SectionContext
    issues: List[Issue]
    line_slots: Dict[int, int]  # line index -> slot inside its group (grouped lines only)


def group_texts(cells: Sequence[CellRecord]) -> Dict[str, LabelReading]:
    """Join the label readings of several cells per independence group (reading order)."""
    ordered = sorted(cells, key=lambda c: (c.box[1], c.box[0]))
    parts: Dict[str, List[str]] = {}
    engine_of: Dict[str, str] = {}
    confs: Dict[str, List[float]] = {}
    for c in ordered:
        for r in c.labels:
            if r.text:
                parts.setdefault(r.independence_group, []).append(r.text)
            engine_of[r.independence_group] = r.engine
            if r.confidence is not None:
                confs.setdefault(r.independence_group, []).append(r.confidence)
    out: Dict[str, LabelReading] = {}
    for g, eng in engine_of.items():
        cs = confs.get(g)
        out[g] = LabelReading(
            engine=eng,
            independence_group=g,
            text=normalize_unicode(" ".join(parts.get(g, []))),
            confidence=round(sum(cs) / len(cs), 4) if cs else None,
        )
    return out


def agreed_name(names: Dict[str, str], min_sources: int) -> Optional[str]:
    keys = {exact_key(v) for v in names.values()}
    if len(keys) == 1 and len(names) >= min_sources:
        return next(iter(names.values()))
    return None


class TableInterpreter:
    def __init__(self, profile: TableProfile, variant: TableVariant, min_sources: int, issues: IssueFactory):
        self.profile = profile
        self.variant = variant
        self.min_sources = min_sources
        self._centred_index: Dict[Tuple[int, int], CellRecord] = {}
        self.issues = issues
        headings = profile.structural_headings
        self.classifier = LabelClassifier(
            label_pattern=profile.label_pattern,
            section_kinds=dict(profile.section_kinds),
            continued_prefixes=profile.continued_prefixes,
            total_keywords=profile.total_keywords,
            min_similarity=profile.keyword_min_similarity,
            headings=[k for k, h in headings.items() if not h.with_name],
            named_headings=[k for k, h in headings.items() if h.with_name],
            section_index_allowed=profile.section_index_allowed,
            keyword_precedence=profile.keyword_precedence,
        )
        # Column indices refer to variant.all_columns(): page spreads interpret the
        # left page's columns followed by the continuation page's columns.
        cols = variant.all_columns()
        self.col_idx = {c.key: i for i, c in enumerate(cols)}
        self.label_col = next(i for i, c in enumerate(cols) if c.role == "unit_label")
        self.index_cols = [i for i, c in enumerate(cols) if c.role == "unit_index"]
        self.index_col = self.index_cols[0] if self.index_cols else None
        self.marker_expect = {self.col_idx[k]: v for k, v in profile.effective_marker_columns().items()}
        self.primary_marker_col = self.col_idx.get(profile.marker_column) if profile.marker_column else None
        self.roles = profile.effective_roles()
        self.bindings = profile.bindings(variant)

    def classify(self, readings: Dict[str, LabelReading]) -> AgreedClass:
        return agree({g: self.classifier.classify(r.text) for g, r in readings.items()}, self.min_sources)

    # ------------------------------------------------------------------ public
    def interpret(
        self,
        table_id: str,
        page_number: int,
        lines: List[AnchorLine],
        pitch: Optional[float],
        cells: List[CellRecord],
        carry: SectionContext,
        related_table_ids: Sequence[str] = (),
    ) -> Interpretation:
        table_issues: List[Issue] = []
        if not lines or not pitch:
            table_issues.append(
                self.issues.make(IssueCode.LINE_ANCHOR_MISSING, IssueScope.TABLE, "no anchor lines found in table body")
            )
            return Interpretation([], [], carry, table_issues, {})

        by_pos: Dict[Tuple[int, int], CellRecord] = {}
        loose_labels: List[CellRecord] = []
        for c in cells:
            if c.line_index is not None and c.assignment_status == "ASSIGNED":
                by_pos[(c.column_index, c.line_index)] = c
            elif c.column_index == self.label_col and c.assignment_status != "MARK":
                loose_labels.append(c)

        groups, group_issues = self._groups(lines, pitch, by_pos)
        table_issues.extend(group_issues)
        line_slots = {li: slot for g in groups for slot, li in enumerate(g)}
        line_by_idx = {line.index: line for line in lines}
        self._line_by_idx = line_by_idx
        self._centred_index = self._centred_indices(cells, groups)
        group_labels: Dict[int, List[CellRecord]] = {gi: [] for gi in range(len(groups))}
        for gi, g in enumerate(groups):
            for li in g:
                label_cell = by_pos.get((self.label_col, li))
                if label_cell is not None:
                    group_labels[gi].append(label_cell)
        header_cells = self._place_loose_labels(loose_labels, groups, group_labels, line_by_idx, pitch, table_issues)

        headers = self._section_headers(table_id, page_number, header_cells)
        events: List[Tuple[float, str, int]] = [(h.y, "H", i) for i, h in enumerate(headers)]
        events += [(line_by_idx[g[0]].y0, "G", gi) for gi, g in enumerate(groups)]
        events.sort()

        ctx = carry
        rows: List[RowRecord] = []
        for _, kind, idx in events:
            if kind == "H":
                ctx = self._apply_header(table_id, headers[idx], ctx)
                continue
            new_rows, ctx = self._rows_for_group(
                table_id, page_number, idx, groups[idx], group_labels[idx], by_pos, ctx, related_table_ids
            )
            rows.extend(new_rows)
        return Interpretation(rows, headers, ctx, table_issues, line_slots)

    def _centred_indices(self, cells: List[CellRecord], groups: List[List[int]]) -> Dict[Tuple[int, int], CellRecord]:
        """An index printed once per group, vertically centred between two of its lines (a spread's
        right-hand 'Nr.' column), is AMBIGUOUS between exactly those lines -> it is the group's index."""
        first_of = {li: g[0] for g in groups if len(g) > 1 for li in g}
        out: Dict[Tuple[int, int], CellRecord] = {}
        for c in cells:
            if c.column_index not in self.index_cols or c.assignment_status != "AMBIGUOUS" or not c.between_lines:
                continue
            a, b = c.between_lines
            if a in first_of and first_of.get(b) == first_of[a]:
                out[(c.column_index, first_of[a])] = c
        return out

    def _place_loose_labels(
        self,
        loose_labels: List[CellRecord],
        groups: List[List[int]],
        group_labels: Dict[int, List[CellRecord]],
        line_by_idx: Dict[int, AnchorLine],
        pitch: float,
        table_issues: List[Issue],
    ) -> List[CellRecord]:
        """Attach wrapped label parts to their group; returns the label cells that are section headers
        (or unresolved header candidates - never silently ignored)."""
        header_cells: List[CellRecord] = []
        for c in loose_labels:
            readings = group_texts([c])
            if not any(any(ch.isalnum() for ch in r.text) for r in readings.values()):
                # dashes, underline remnants, specks: cannot be a heading or a name
                table_issues.append(
                    self.issues.make(
                        IssueCode.NOISE_TOKEN_IGNORED,
                        IssueScope.TABLE,
                        "label-column mark without letters or digits ignored",
                        cell_id=c.cell_id,
                        readings={g: r.text for g, r in readings.items()},
                    )
                )
                continue
            cls = self.classify(readings)
            if c.assignment_status == "OUTSIDE" and cls.cls not in ("SECTION", "HEADING"):
                # below the last data line (page foot): only an agreed heading may affect the
                # section context; other marks are recorded (a following page without a new
                # header is caught by the closed section / hierarchy checks)
                table_issues.append(
                    self.issues.make(
                        IssueCode.DETACHED_TEXT,
                        IssueScope.TABLE,
                        "label-column text below the last data line",
                        cell_id=c.cell_id,
                        readings={g: r.text for g, r in readings.items()},
                    )
                )
                continue
            if cls.cls in ("SECTION", "HEADING") or cls.conflict:
                header_cells.append(c)
                continue
            cy = (c.box[1] + c.box[3]) / 2.0
            target = self._wrapped_target(cy, groups, line_by_idx, pitch)
            if target is None and cls.cls == "UNIT":
                target = self._nearest_group(cy, groups, line_by_idx, pitch)
            if target is None:
                header_cells.append(c)  # reported as unresolved header, never ignored
            else:
                group_labels[target].append(c)
        return header_cells

    def _apply_header(self, table_id: str, h: SectionHeaderRecord, ctx: SectionContext) -> SectionContext:
        if not h.agreed:
            return SectionContext(
                key=f"{table_id}_s{int(h.y):05d}_unresolved",
                kind=None,
                name=None,
                level=None,
                agreed=False,
                issues=list(h.issues),
            )
        if h.heading:
            return self._open_heading(table_id, h, ctx)
        return self._open_section(table_id, h.y, h.kind, h.name_readings, h.continued, ctx, h.index)

    # ----------------------------------------------------------------- grouping
    def _groups(self, lines, pitch, by_pos) -> Tuple[List[List[int]], List[Issue]]:
        if self.profile.row_mode == "single_line":
            return [[l.index] for l in lines], []
        if self.profile.group_by == "label_start":
            return self._label_start_groups(lines, by_pos)
        return self._gap_groups(lines, pitch, by_pos)

    def _invalid_group(self, run: List[int], why: str) -> Issue:
        return self.issues.make(IssueCode.ROW_GROUP_INVALID, IssueScope.TABLE, why, lines=run)

    def _gap_groups(self, lines, pitch, by_pos) -> Tuple[List[List[int]], List[Issue]]:
        # The spacing INSIDE a group is the smallest regular spacing; with two-line
        # groups the median of all spacings would mix it with the gaps between groups.
        diffs = sorted(b.cy - a.cy for a, b in zip(lines, lines[1:]))
        inner = diffs[len(diffs) // 4] if diffs else pitch
        runs: List[List[int]] = []
        for l in lines:
            if runs and l.cy - lines[runs[-1][-1]].cy <= self.profile.group_gap_ratio * inner:
                runs[-1].append(l.index)
            else:
                runs.append([l.index])
        n = self.profile.slot_count
        seq = self.profile.marker_sequence
        groups: List[List[int]] = []
        issues: List[Issue] = []
        for run in runs:
            if len(run) > n and len(run) % n == 0:
                # Groups printed without vertical gap: split only if every marker agrees.
                values = [self._marker_value(by_pos, li) for li in run]
                if values == seq * (len(run) // n):
                    groups.extend(run[k : k + n] for k in range(0, len(run), n))
                    continue
            if len(run) != n:
                issues.append(self._invalid_group(run, f"line group of {len(run)} lines does not match marker sequence {seq}"))
                continue
            groups.append(run)
        return groups, issues

    @staticmethod
    def _substantial(cell: Optional[CellRecord], line: AnchorLine) -> bool:
        """A mark of at least half the line's glyph height/width (specks are no content)."""
        if cell is None:
            return False
        w, h = cell.box[2] - cell.box[0], cell.box[3] - cell.box[1]
        lh = line.box.height
        # glyph-like: tall enough (a '1') or wide enough (a dash), never a 1-px scan streak
        return (h >= 0.5 * lh and w >= 0.15 * lh) or w >= 0.5 * lh

    def _label_start_groups(self, lines, by_pos) -> Tuple[List[List[int]], List[Issue]]:
        n = self.profile.slot_count
        groups: List[List[int]] = []
        issues: List[Issue] = []
        current: List[int] = []
        started = False
        for l in lines:
            is_start = self._substantial(by_pos.get((self.label_col, l.index)), l) or (
                self.index_col is not None and self._substantial(by_pos.get((self.index_col, l.index)), l)
            )
            if is_start and current:
                if started:
                    groups.append(current)
                else:
                    issues.append(
                        self._invalid_group(current, "lines before the first labelled line (group split across pages?)")
                    )
                current = []
            started = started or is_start
            current.append(l.index)
        if current:
            if started:
                groups.append(current)
            else:
                issues.append(self._invalid_group(current, "lines without any labelled line"))
        valid: List[List[int]] = []
        for g in groups:
            if len(g) != n:
                issues.append(self._invalid_group(g, f"group of {len(g)} lines, profile expects {n} per unit"))
            else:
                valid.append(g)
        return valid, issues

    def _marker_value(self, by_pos, line_index) -> Optional[str]:
        """Structural marker of a line: the single vocabulary value read by the engines (see
        ``_group_structure_issues``); None if nobody reads a marker or readings contradict."""
        if self.primary_marker_col is None:
            return None
        c = by_pos.get((self.primary_marker_col, line_index))
        if c is None:
            return None
        read = {x.parse.value for x in c.resolution.candidates if x.parse.value is not None}
        return next(iter(read)) if len(read) == 1 else None

    def _wrapped_target(self, cy: float, groups, line_by_idx, pitch) -> Optional[int]:
        mode = self.profile.wrapped_label_attach
        best = None
        for gi, g in enumerate(groups):
            first, last = line_by_idx[g[0]], line_by_idx[g[-1]]
            tol = self.profile.wrapped_label_max_pitch * pitch
            if mode == "group" and first.y0 - tol <= cy <= last.y1 + tol:
                return gi
            if mode == "next" and cy < first.cy and first.cy - cy <= 1.5 * pitch:
                if best is None or first.cy < line_by_idx[groups[best][0]].cy:
                    best = gi
            if mode == "previous" and cy > last.cy and cy - last.cy <= 1.5 * pitch:
                if best is None or last.cy > line_by_idx[groups[best][-1]].cy:
                    best = gi
        return best

    @staticmethod
    def _nearest_group(cy: float, groups, line_by_idx, pitch) -> Optional[int]:
        """Group whose line span is closest to ``cy`` (within 1.5 line pitches), unambiguous only."""
        dists = []
        for gi, g in enumerate(groups):
            top, bottom = line_by_idx[g[0]].y0, line_by_idx[g[-1]].y1
            d = 0.0 if top <= cy <= bottom else min(abs(cy - top), abs(cy - bottom))
            dists.append((d, gi))
        dists.sort()
        if not dists or dists[0][0] > 1.5 * pitch:
            return None
        if len(dists) > 1 and dists[1][0] - dists[0][0] < 0.5 * pitch:
            return None
        return dists[0][1]

    # ----------------------------------------------------------------- sections
    def _section_headers(self, table_id, page_number, header_cells: List[CellRecord]) -> List[SectionHeaderRecord]:
        header_cells = sorted(header_cells, key=lambda c: c.box[1])
        merged: List[List[CellRecord]] = []
        for c in header_cells:
            if merged and c.box[1] <= max(x.box[3] for x in merged[-1]):
                merged[-1].append(c)
            else:
                merged.append([c])
        out: List[SectionHeaderRecord] = []
        for cl in merged:
            readings = group_texts(cl)
            cls = self.classify(readings)
            rec = SectionHeaderRecord(
                table_id=table_id,
                page_number=page_number,
                y=min(c.box[1] for c in cl),
                readings=list(readings.values()),
                cell_ids=[c.cell_id for c in cl],
            )
            if cls.cls in ("SECTION", "HEADING"):
                rec.kind = cls.kind
                rec.heading = cls.cls == "HEADING"
                rec.continued = bool(cls.continued)
                rec.name_readings = cls.names
                rec.name = agreed_name(cls.names, self.min_sources)
                rec.index = agreed_name(cls.indices, self.min_sources)
                rec.agreed = True
            else:
                rec.issues.append(
                    self.issues.make(
                        IssueCode.SECTION_LABEL_UNRESOLVED,
                        IssueScope.SECTION,
                        "label outside any row is not an agreed section header",
                        readings={r.independence_group: r.text for r in readings.values()},
                        votes=cls.votes,
                    )
                )
            out.append(rec)
        return out

    def _same_section_name(self, footer: Dict[str, str], header: Dict[str, str]) -> bool:
        """Structural check (not a value): at least ``min_sources`` engine groups read exactly the
        same name in the section's header and in its total line (garbled readings abstain)."""
        same = [g for g, n in footer.items() if g in header and exact_key(n) == exact_key(header[g])]
        return len(same) >= self.min_sources

    def _open_heading(self, table_id: str, h: SectionHeaderRecord, carry: SectionContext) -> SectionContext:
        spec = self.profile.structural_headings[h.kind or ""]
        issues: List[Issue] = []
        if spec.context_key:
            expected = self.variant.context.get(spec.context_key)
            if expected is None or h.name is None or exact_key(h.name) != exact_key(expected):
                issues.append(
                    self.issues.make(
                        IssueCode.SECTION_LABEL_UNRESOLVED,
                        IssueScope.SECTION,
                        f"heading '{h.kind}' names '{h.name}', profile context {spec.context_key}='{expected}'",
                        readings=h.name_readings,
                    )
                )
        return SectionContext(
            key=f"{table_id}_h{int(h.y):05d}",
            kind=h.kind,
            name=h.name,
            level=None,
            agreed=True,
            issues=issues,
            name_readings=dict(h.name_readings),
            grouping=True,
            unit_level=spec.unit_level,
        )

    def _open_section(
        self,
        table_id: str,
        y: float,
        kind: Optional[str],
        names: Dict[str, str],
        continued: bool,
        carry: SectionContext,
        index: Optional[str] = None,
    ) -> SectionContext:
        if continued and carry.key and not carry.grouping:
            continued_issues = list(carry.issues)
            name = agreed_name(names, self.min_sources)
            if name and carry.name and exact_key(name) != exact_key(carry.name):
                # 'noch: Landkreis X' must continue X - otherwise pages are out of order or a header was missed
                continued_issues.append(
                    self.issues.make(
                        IssueCode.SECTION_LABEL_UNRESOLVED,
                        IssueScope.SECTION,
                        f"continued section '{name}' does not match the open section '{carry.name}'",
                    )
                )
            merged = dict(carry.name_readings)
            for g, n in names.items():
                merged.setdefault(g, n)
            return SectionContext(
                key=carry.key,
                kind=carry.kind or kind,
                name=carry.name or name,
                level=carry.level,
                agreed=carry.agreed,
                issues=continued_issues,
                name_readings=merged,
                index=carry.index,
            )
        issues: List[Issue] = []
        if continued:
            issues.append(
                self.issues.make(
                    IssueCode.SECTION_CONTEXT_MISSING,
                    IssueScope.SECTION,
                    "header marks a continued section but no preceding section is known",
                )
            )
        return SectionContext(
            key=f"{table_id}_s{int(y):05d}",
            kind=kind,
            name=agreed_name(names, self.min_sources),
            level=self.profile.section_kinds.get(kind or ""),
            agreed=True,
            issues=issues,
            name_readings=dict(names),
            index=index,
        )

    # --------------------------------------------------------------------- rows
    def _group_structure_issues(
        self,
        group: List[int],
        by_pos: Dict[Tuple[int, int], CellRecord],
        line_by_idx: Dict[int, AnchorLine],
        row_kind: str = "DATA",
    ) -> List[Issue]:
        """Marker expectations per slot + ink that no binding consumes (specks are ignored)."""
        out: List[Issue] = []
        for ci, expect in self.marker_expect.items():
            key = self.variant.all_columns()[ci].key
            for slot, li in enumerate(group):
                e, cell = expect[slot], by_pos.get((ci, li))
                if e is None:
                    if self._substantial(cell, line_by_idx[li]):
                        out.append(
                            self.issues.make(
                                IssueCode.ROW_GROUP_INVALID, IssueScope.ROW, f"unexpected mark in '{key}' on slot {slot}"
                            )
                        )
                elif e == "?":
                    continue  # ink allowed, not required (e.g. the seat-winning party of a constituency)
                elif cell is None:
                    out.append(
                        self.issues.make(
                            IssueCode.MISSING_VALUE,
                            IssueScope.ROW,
                            f"marker '{key}' missing on slot {slot}",
                            field=f"marker:{key}",
                        )
                    )
                elif e != "*":
                    # Structural marker (not a data value): verified when at least one
                    # engine group reads the expected marker and none reads another
                    # vocabulary marker. Unparseable readings abstain.
                    read = {c.parse.value for c in cell.resolution.candidates if c.parse.value is not None}
                    if read - {e}:
                        out.append(
                            self.issues.make(
                                IssueCode.ROW_GROUP_INVALID,
                                IssueScope.ROW,
                                f"marker read as {sorted(read)} in '{key}' on slot {slot}, profile expects '{e}'",
                                cell_id=cell.cell_id,
                            )
                        )
                    elif not read:
                        out.append(
                            self.issues.make(
                                IssueCode.MARKER_UNVERIFIED,
                                IssueScope.ROW,
                                f"marker '{key}' on slot {slot} not read by any engine",
                                field=f"marker:{key}",
                                cell_id=cell.cell_id,
                            )
                        )
        consumed: Set[Tuple[int, int]] = set()
        for items in self.bindings.values():
            for b in items:
                consumed.add((self.col_idx[b.column], group[b.slot]))
        structural = {self.label_col, *self.marker_expect}
        for (ci, li), cell in by_pos.items():
            if li not in group or ci in structural or (ci, li) in consumed:
                continue
            if not self._substantial(cell, line_by_idx[li]):
                continue
            if ci in self.index_cols and (li == group[0] or row_kind != "DATA"):
                continue  # unit index on the first line; totals may print an index range ('163-' / '195')
            out.append(
                self.issues.make(
                    IssueCode.UNEXPECTED_CELL_CONTENT,
                    IssueScope.ROW,
                    f"ink in column '{self.variant.all_columns()[ci].key}' on slot {group.index(li)}, "
                    "where the profile expects no value",
                    cell_id=cell.cell_id,
                )
            )
        return out

    def _index(self, cls: AgreedClass, group: List[int], by_pos) -> Tuple[Optional[str], Optional[str], List[Issue]]:
        """Printed unit index: from the label pattern and/or index columns (page spreads repeat it on
        the continuation page); every agreed reading must be the same number."""
        issues: List[Issue] = []
        label_index = agreed_name(cls.indices, self.min_sources) if cls.cls == "UNIT" else None
        values: List[Tuple[str, str]] = [("label", label_index)] if label_index is not None else []
        first_cell = None
        for ci in self.index_cols:
            cell = by_pos.get((ci, group[0]))
            centred = self._centred_index.get((ci, group[0]))
            if cell is not None and centred is not None:
                issues.append(
                    self.issues.make(
                        IssueCode.INDEX_MISMATCH,
                        IssueScope.ROW,
                        f"two printed indices in '{cell.column_key}' (on the first line and between lines)",
                        cell_id=centred.cell_id,
                    )
                )
            cell = cell or centred
            if cell is None:
                continue
            first_cell = first_cell or cell
            if cell.resolution.status == CellStatus.AGREED and cell.resolution.value is not None:
                values.append((cell.column_key, cell.resolution.value))
            else:
                issues.append(
                    self.issues.make(
                        IssueCode.INDEX_MISMATCH,
                        IssueScope.ROW,
                        f"printed index in '{cell.column_key}' not agreed ({cell.resolution.status.value})",
                        cell_id=cell.cell_id,
                    )
                )
        if len({int(v) for _, v in values}) > 1:
            issues.append(self.issues.make(IssueCode.INDEX_MISMATCH, IssueScope.ROW, f"printed indices differ: {dict(values)}"))
        unit_index = values[0][1] if values else None
        return unit_index, first_cell.cell_id if first_cell is not None else None, issues

    def _has_index(self, readings: Dict[str, LabelReading], group: List[int], by_pos) -> bool:
        """Any printed index at all (presence, not agreement): an index cell or a label starting with digits."""
        line = self._line_by_idx[group[0]]
        if any(self._substantial(by_pos.get((ci, group[0])), line) for ci in self.index_cols):
            return True
        if any((ci, group[0]) in self._centred_index for ci in self.index_cols):
            return True
        # an index embedded in the label: >= min_sources engine groups read a number as first word
        numbered = sum(1 for r in readings.values() if r.text.split(" ", 1)[0].isdigit())
        return numbered >= self.min_sources

    def _strip_slot_label_markers(self, group: List[int], label_cells: List[CellRecord]) -> Tuple[List[CellRecord], List[Issue]]:
        """Remove the marker printed at the end of the label column on marker slots ('vH').

        The remainder (a wrapped part of the name) stays label text. At least one engine
        group must read the marker exactly (case-insensitive); tokens of up to three
        characters in that position are taken as (misread) markers."""
        markers = self.profile.slot_label_markers
        if not markers:
            return label_cells, []
        out: List[CellRecord] = []
        issues: List[Issue] = []
        for c in label_cells:
            slot = group.index(c.line_index) if c.line_index in group else None
            if slot not in markers:
                out.append(c)
                continue
            marker = markers[slot].casefold()
            read_ok = False
            labels = []
            for r in c.labels:
                tokens = r.text.split()
                if tokens and (tokens[-1].casefold() == marker or len(tokens[-1]) <= 3):
                    read_ok = read_ok or tokens[-1].casefold() == marker
                    tokens = tokens[:-1]
                labels.append(r.model_copy(update={"text": " ".join(tokens)}))
            if not read_ok:
                issues.append(
                    self.issues.make(
                        IssueCode.MARKER_UNVERIFIED,
                        IssueScope.ROW,
                        f"label marker '{markers[slot]}' on slot {slot} not read",
                        cell_id=c.cell_id,
                    )
                )
            if any(l.text for l in labels):
                out.append(c.model_copy(update={"labels": labels}))
        return out, issues

    def _rows_for_group(
        self,
        table_id: str,
        page_number: int,
        group_index: int,
        group: List[int],
        label_cells: List[CellRecord],
        by_pos: Dict[Tuple[int, int], CellRecord],
        ctx: SectionContext,
        related_table_ids: Sequence[str],
    ) -> Tuple[List[RowRecord], SectionContext]:
        """One result row per profile role for one printed unit (line group); returns the new context."""
        label_cells, row_issues = self._strip_slot_label_markers(group, label_cells)
        readings = group_texts(label_cells)
        cls = self._group_class(readings, group, by_pos)
        decision = self._row_kind(table_id, cls, readings, label_cells, ctx, row_issues)
        ctx = decision.ctx
        if self.profile.implicit_sections:
            ctx = self._implicit_section(table_id, group, decision, ctx)
        if not label_cells:
            row_issues.append(self.issues.make(IssueCode.MISSING_VALUE, IssueScope.ROW, "row has no label", field="unit_label"))
        row_issues.extend(self._context_issues(decision.row_kind, ctx))
        row_issues.extend(self._group_structure_issues(group, by_pos, self._line_by_idx, decision.row_kind))
        unit_index, index_cell_id = self._row_index(decision.row_kind, cls, group, by_pos, row_issues)
        rows = [
            self._role_row(
                role,
                table_id=table_id,
                page_number=page_number,
                group_index=group_index,
                group=group,
                by_pos=by_pos,
                ctx=ctx,
                row_kind=decision.row_kind,
                readings=readings,
                label_cells=label_cells,
                unit_names=decision.unit_names,
                unit_index=unit_index,
                index_cell_id=index_cell_id,
                row_issues=row_issues,
                related_table_ids=related_table_ids,
            )
            for role in self.roles
        ]
        if decision.close_after:
            ctx = replace(ctx, closed=True)
        return rows, ctx

    def _group_class(self, readings: Dict[str, LabelReading], group: List[int], by_pos) -> AgreedClass:
        cls = self.classify(readings)
        if (
            self.profile.total_when_unindexed
            and cls.cls not in ("SECTION", "TOTAL", "HEADING")  # class votes may conflict on garbled keywords
            and readings
            and not self._has_index(readings, group, by_pos)
            and sum(1 for r in readings.values() if r.text[:1].isalpha()) >= self.min_sources
        ):
            # a labelled row WITHOUT any printed index is the total of its block (unit rows always
            # carry one); decided on the absence of the index, not on the garbled keyword
            cls = replace(cls, cls="TOTAL", conflict=False, names={g: r.text for g, r in readings.items() if r.text})
        return cls

    def _row_kind(
        self,
        table_id: str,
        cls: AgreedClass,
        readings: Dict[str, LabelReading],
        label_cells: List[CellRecord],
        ctx: SectionContext,
        row_issues: List[Issue],
    ) -> _RowDecision:
        """DATA / SECTION_TOTAL row, the unit names it carries and its effect on the section context."""
        if cls.cls == "SECTION" and self.profile.section_header_has_totals:
            top = min(c.box[1] for c in label_cells)
            ctx = self._open_section(table_id, top, cls.kind, cls.names, False, ctx, agreed_name(cls.indices, self.min_sources))
            return _RowDecision("SECTION_TOTAL", {}, False, ctx)
        if cls.cls == "SECTION" and self.profile.section_footer_has_totals:
            return _RowDecision("SECTION_TOTAL", self._section_footer(cls, readings, ctx, row_issues), True, ctx)
        if cls.cls == "TOTAL":
            return _RowDecision("SECTION_TOTAL", cls.names if self.profile.implicit_sections else {}, True, ctx)
        if cls.cls == "UNIT":
            return _RowDecision("DATA", cls.names, False, ctx)
        code = IssueCode.LABEL_DISAGREEMENT if cls.conflict else IssueCode.ROW_UNINTERPRETED
        row_issues.append(
            self.issues.make(
                code,
                IssueScope.ROW,
                "row label class not agreed by enough OCR engines",
                readings={g: r.text for g, r in readings.items()},
                votes=cls.votes,
            )
        )
        return _RowDecision("DATA", cls.names, False, ctx)

    def _section_footer(
        self, cls: AgreedClass, readings: Dict[str, LabelReading], ctx: SectionContext, row_issues: List[Issue]
    ) -> Dict[str, str]:
        """A section total printed at the END of its section: must match the open section."""
        name = agreed_name(cls.names, self.min_sources)
        if ctx.implicit and not ctx.closed:
            return cls.names  # names the block it closes; there was no header to compare with
        if self.profile.implicit_sections and (not ctx.key or ctx.closed):
            return cls.names  # grand total (decided by the caller)
        if not ctx.key or ctx.grouping or ctx.closed:
            row_issues.append(
                self.issues.make(IssueCode.SECTION_CONTEXT_MISSING, IssueScope.ROW, "section total without an open section")
            )
        elif self.profile.section_kinds.get(cls.kind or "") != ctx.level or not self._same_section_name(
            cls.names, ctx.name_readings
        ):
            row_issues.append(
                self.issues.make(
                    IssueCode.SECTION_LABEL_UNRESOLVED,
                    IssueScope.ROW,
                    f"section total '{cls.kind} {name}' does not match the open section '{ctx.kind} {ctx.name}'",
                    readings={g: r.text for g, r in readings.items()},
                )
            )
        return {}

    def _implicit_section(self, table_id: str, group: List[int], decision: _RowDecision, ctx: SectionContext) -> SectionContext:
        """Blocks without a header: a data row after a total opens a new block; a total right after a
        total is the grand total over everything."""
        if decision.row_kind == "DATA" and (not ctx.key or ctx.closed):
            y = self._line_by_idx[group[0]].y0
            return SectionContext(key=f"{table_id}_i{int(y):05d}", kind=None, name=None, level=None, agreed=True, implicit=True)
        if decision.row_kind == "SECTION_TOTAL" and (not ctx.key or ctx.closed):
            decision.row_kind = "GRAND_TOTAL"
        return ctx

    def _context_issues(self, row_kind: str, ctx: SectionContext) -> List[Issue]:
        if row_kind == "GRAND_TOTAL":
            return []
        if not ctx.key:
            return [self.issues.make(IssueCode.SECTION_CONTEXT_MISSING, IssueScope.ROW, "no section context before this row")]
        if ctx.closed and row_kind == "DATA":
            return [self.issues.make(IssueCode.SECTION_CONTEXT_MISSING, IssueScope.ROW, "row after the total of its section")]
        return []

    def _row_index(
        self, row_kind: str, cls: AgreedClass, group: List[int], by_pos, row_issues: List[Issue]
    ) -> Tuple[Optional[str], Optional[str]]:
        """(unit index, its cell id) of a data row; totals must not carry a single unit index."""
        if row_kind == "DATA":
            unit_index, index_cell_id, index_issues = self._index(cls, group, by_pos)
            row_issues.extend(index_issues)
            return unit_index, index_cell_id
        agreed_idx = [
            c
            for ci in self.index_cols
            if (c := by_pos.get((ci, group[0])) or self._centred_index.get((ci, group[0]))) is not None
            and c.resolution.status == CellStatus.AGREED
        ]
        if agreed_idx:  # an index RANGE ('163-195') does not parse as one number and is fine
            row_issues.append(self.issues.make(IssueCode.INDEX_MISMATCH, IssueScope.ROW, "total row carries a unit index"))
        return None, None

    def _row_level(self, row_kind: str, ctx: SectionContext) -> str:
        if row_kind == "GRAND_TOTAL":
            return self.profile.grand_total_level
        if row_kind == "SECTION_TOTAL" and ctx.implicit:
            return self.profile.section_total_level or self.profile.unit_level_default
        if row_kind == "SECTION_TOTAL":
            return ctx.level or self.profile.unit_level_default
        return (ctx.unit_level if ctx.grouping and ctx.unit_level else None) or self.profile.unit_level_default

    def _row_context(self, ctx: SectionContext, unit_index: Optional[str]) -> Dict[str, str]:
        context = dict(self.variant.context)
        if ctx.grouping:
            context["_section_grouping"] = "true"
        if ctx.implicit:
            context["_section_implicit"] = "true"
        if unit_index is not None:
            context["_unit_index"] = unit_index
        return context

    def _role_row(
        self,
        role,
        *,
        table_id: str,
        page_number: int,
        group_index: int,
        group: List[int],
        by_pos: Dict[Tuple[int, int], CellRecord],
        ctx: SectionContext,
        row_kind: str,
        readings: Dict[str, LabelReading],
        label_cells: List[CellRecord],
        unit_names: Dict[str, str],
        unit_index: Optional[str],
        index_cell_id: Optional[str],
        row_issues: List[Issue],
        related_table_ids: Sequence[str],
    ) -> RowRecord:
        field_cells: Dict[str, Optional[str]] = {}
        party_cells: Dict[str, Optional[str]] = {}
        for b in self.bindings[role.role_id]:
            cell = by_pos.get((self.col_idx[b.column], group[b.slot]))
            cid = cell.cell_id if cell else None
            if b.field.startswith(PARTY_FIELD_PREFIX):
                party_cells[b.field[len(PARTY_FIELD_PREFIX) :]] = cid
            else:
                field_cells[b.field] = cid
        li = group[role.slot]
        marker_cell = by_pos.get((self.primary_marker_col, li)) if self.primary_marker_col is not None else None
        return RowRecord(
            result_key=f"{table_id}_g{group_index:03d}_{role.role_id}",
            table_id=table_id,
            page_number=page_number,
            group_index=group_index,
            line_indices=list(group),
            row_kind=row_kind,
            marker=self.profile.marker_sequence[role.slot] if self.profile.row_mode == "marker_group" else None,
            role_id=role.role_id,
            role_kind=role.kind,
            percent_of_role=role.percent_of,
            vote_type=role.vote_type.value,
            election_id=role.election_id or self.profile.election_id,
            publishable_role=role.publish,
            label_readings=list(readings.values()),
            label_cell_ids=[c.cell_id for c in label_cells],
            unit_name_readings=unit_names,
            unit_index=unit_index,
            unit_index_cell_id=index_cell_id,
            unit_level=self._row_level(row_kind, ctx),
            section_key=None if row_kind == "GRAND_TOTAL" else ctx.key,
            section_kind=ctx.kind,
            section_name=ctx.name,
            section_name_readings=dict(ctx.name_readings),
            section_level=ctx.level,
            context=self._row_context(ctx, unit_index),
            state=self.variant.state,
            field_cells=field_cells,
            party_cells=party_cells,
            marker_cell_id=marker_cell.cell_id if marker_cell else None,
            related_table_ids=list(related_table_ids),
            issues=list(row_issues) + list(ctx.issues),
        )


@dataclass
class _RowDecision:
    row_kind: str  # DATA | SECTION_TOTAL | GRAND_TOTAL
    unit_names: Dict[str, str]  # group -> unit name read on the label
    close_after: bool  # the row closes its section (footer total)
    ctx: SectionContext
