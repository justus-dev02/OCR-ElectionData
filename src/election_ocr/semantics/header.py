"""
Header verification.

The column binding comes from the document profile (position of each column
between ruling lines). The printed header text is used to *verify* that
binding: every column's header layers (text lines inside the header cell,
top to bottom) are compared with the profile's ``header_aliases``.

A column is verified when either
* its best header-text similarity (over all engines and all contiguous
  combinations of header lines) reaches ``header_match_min_score`` and no other
  column's alias scores higher (otherwise the column order may be shifted), or
* the printed column number below the header (profile ``column_number``) is
  read in that column and no engine reads a different column number there.

A column that is confirmed by neither may be *inferred from its position*
(docs/decisions.md, D5) when the column count is exact (checked by the
caller), it is not contradicted - no other column's alias matches its header
better, no engine reads the column number of a NEIGHBOURING column there -
and every neighbour is itself confirmed directly (at the table edge: the one
inner neighbour). Two adjacent unconfirmed columns are never inferred: only
then could a partition error shift values without any contradiction.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from rapidfuzz import fuzz

from election_ocr.parsing.text import header_key
from election_ocr.pipeline.models import HeaderMatch
from election_ocr.reference.models import ColumnSpec


def _spans(layers: List[str]) -> List[Tuple[Tuple[int, int], str]]:
    """((first, last_exclusive) layer span, header key) for every contiguous layer combination."""
    out = []
    for i in range(len(layers)):
        for j in range(i + 1, len(layers) + 1):
            k = header_key(" ".join(layers[i:j]))
            if k:
                out.append(((i, j), k))
    return out


def best_alias_span(layers: List[str], aliases: List[str]) -> Tuple[float, Optional[Tuple[int, int]]]:
    """Best similarity of any alias to any contiguous layer combination, and that span."""
    best, span = 0.0, None
    for alias in aliases:
        ak = header_key(alias)
        if not ak:
            continue
        for sp, v in _spans(layers):
            sc = fuzz.ratio(ak, v) / 100.0
            if sc > best:
                best, span = sc, sp
    return round(best, 4), span


def column_number_seen(layers: List[str], number: int) -> bool:
    """True if one header layer consists exactly of the expected printed column number."""
    return any(re.fullmatch(r"\s*%d\s*" % number, ln or "") is not None for ln in layers)


def match_headers(
    columns: List[ColumnSpec],
    layers_by_column: List[Dict[str, List[str]]],
    min_score: float,
    min_number_sources: int = 1,
) -> List[HeaderMatch]:
    """Verify each profile column against the printed header.

    A column is confirmed by (a) its header text: best alias similarity over all
    engines >= ``min_score`` and no other column's alias scoring higher, or
    (b) the printed column number (if the profile defines ``column_number``)
    read by ``min_number_sources`` engine(s) while no engine reads a different
    valid column number in that column (a shifted column would show one).
    """
    matches: List[HeaderMatch] = []
    for idx, (spec, layers_by_engine) in enumerate(zip(columns, layers_by_column)):
        best_score, best_engine = 0.0, None
        competitor: Dict[str, float] = {}
        for engine, layers in layers_by_engine.items():
            own, own_span = best_alias_span(layers, spec.header_aliases)
            if own > best_score:
                best_score, best_engine = own, engine
            for other in columns:
                if other.key == spec.key:
                    continue
                sc, span = best_alias_span(layers, other.header_aliases)
                # A competitor matching only a part of the lines that carry the column's own
                # alias ('Wahl-' inside 'Wahl- berechtigte') is no contradiction.
                if (
                    own >= min_score
                    and span is not None
                    and own_span is not None
                    and own_span[0] <= span[0]
                    and span[1] <= own_span[1]
                    and span != own_span
                ):
                    continue
                competitor[other.key] = max(competitor.get(other.key, 0.0), sc)
        comp_key, comp_score = max(competitor.items(), key=lambda kv: kv[1], default=(None, 0.0))
        competing = comp_key is not None and comp_score > best_score and comp_score >= min_score
        text_ok = best_score >= min_score and not competing
        number_ok = False
        neighbour_number_seen = False
        if spec.column_number is not None:
            seen = [e for e, layers in layers_by_engine.items() if column_number_seen(layers, spec.column_number)]
            valid_numbers = {c.column_number for c in columns if c.column_number is not None}
            conflicting = [
                e
                for e, layers in layers_by_engine.items()
                if any(n != spec.column_number and column_number_seen(layers, n) for n in valid_numbers)
            ]
            number_ok = len(seen) >= min_number_sources and not conflicting
            neighbours = [
                c.column_number for c in columns[max(0, idx - 1) : idx + 2] if c.key != spec.key and c.column_number is not None
            ]
            neighbour_number_seen = any(column_number_seen(layers, n) for layers in layers_by_engine.values() for n in neighbours)
        matches.append(
            HeaderMatch(
                column_index=idx,
                column_key=spec.key,
                layers=layers_by_engine,
                best_score=best_score,
                best_engine=best_engine,
                competing_column=comp_key if competing else None,
                competing_score=comp_score if competing else None,
                matched=text_ok or number_ok,
                confirmed_by=("text" if text_ok else "") + ("+number" if number_ok else ""),
                contradicted=competing or neighbour_number_seen,
            )
        )
    direct = [m.matched for m in matches]
    for i, m in enumerate(matches):
        if m.matched or m.contradicted:
            continue
        neighbours = [j for j in (i - 1, i + 1) if 0 <= j < len(matches)]
        if neighbours and all(direct[j] for j in neighbours):
            m.matched = True
            m.confirmed_by = "neighbours"
    return matches
