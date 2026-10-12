"""
Candidate resolver for a single cell.

No majority vote and no "best confidence wins". A value is AGREED only if

* at least ``min_independent_sources`` distinct independence groups read
  exactly the same value under the strict grammar, and
* no group dissents (reads a different value, or an unparseable reading whose
  digits differ from the agreed value).

Everything else is reported with an explicit status so that downstream
validation can route the row to human review.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Dict, List, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, Field

from election_ocr.config import OCRSettings
from election_ocr.ocr.engines import OCRReading
from election_ocr.parsing.numbers import (
    DecimalFormat,
    IntegerFormat,
    ParseResult,
    ParseStatus,
    parse_decimal,
    parse_integer,
    parse_marker,
)


class CellStatus(str, Enum):
    AGREED = "AGREED"
    DISAGREED = "DISAGREED"
    SINGLE_SOURCE = "SINGLE_SOURCE"
    NO_READING = "NO_READING"  # ink present, but no engine produced a usable reading
    EMPTY = "EMPTY"  # no ink in the cell at all
    HUMAN = "HUMAN"  # value set by a reviewer


class Candidate(BaseModel):
    engine: str
    engine_version: str
    variant: str
    independence_group: str
    raw_text: str
    confidence: Optional[float]
    parse: ParseResult
    role: str = Field(description="support | dissent | abstain | noise")
    note: Optional[str] = None
    meta: Dict[str, object] = Field(default_factory=dict)


class CellResolution(BaseModel):
    status: CellStatus
    value: Optional[str] = None
    decimals: Optional[int] = None
    supporting_groups: List[str] = Field(default_factory=list)
    dissenting_groups: List[str] = Field(default_factory=list)
    candidates: List[Candidate] = Field(default_factory=list)
    low_confidence_groups: List[str] = Field(default_factory=list)

    def as_int(self) -> Optional[int]:
        return int(self.value) if self.value is not None and not self.decimals else None


def allowlist_for(value_type: str, integer_format: Optional[IntegerFormat], vocabulary: Sequence[str]) -> Optional[str]:
    if value_type == "integer":
        chars = "0123456789-"
        return chars
    if value_type == "decimal":
        return "0123456789,.-"
    if value_type == "marker":
        chars = "".join(sorted({ch for tok in vocabulary for ch in tok}))
        return chars or None
    return None


def parse_reading(
    raw: str,
    value_type: str,
    integer_format: Optional[IntegerFormat],
    decimal_format: Optional[DecimalFormat],
    vocabulary: Sequence[str],
) -> ParseResult:
    if value_type == "integer":
        return parse_integer(raw, integer_format or IntegerFormat())
    if value_type == "decimal":
        return parse_decimal(raw, decimal_format or DecimalFormat())
    if value_type == "marker":
        return parse_marker(raw, vocabulary)
    raise ValueError(f"parse_reading() does not handle value_type={value_type}")


def shape_conflict(resolution: CellResolution, shape: Optional[str]) -> Optional[str]:
    """Consistency of an AGREED numeric value with the token's geometric shape.

    The shape is measured on the ink before OCR (``classify_token_shape``) and is
    therefore independent evidence:

    * a dash-shaped token can only mean a printed dash, i.e. value 0 (engines that
      read it as the digit 0 reach the same value) - any other value is a conflict;
    * a value supported by a DASH reading requires a dash-shaped token.

    Returns a reason string or None. It never changes the value; the caller turns
    a conflict into a blocking issue.
    """
    if resolution.status != CellStatus.AGREED or shape is None:
        return None
    supporting = [c for c in resolution.candidates if c.role == "support" and c.parse.value == resolution.value]
    dash_support = any(c.parse.status == ParseStatus.DASH for c in supporting)
    if shape == "dash" and resolution.value != "0":
        return f"dash-shaped token agreed as '{resolution.value}'"
    if shape != "dash" and dash_support:
        return f"agreed via dash reading, but token shape is '{shape}'"
    return None


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s or "")


def resolve_cell(
    readings: Sequence[OCRReading],
    value_type: str,
    settings: OCRSettings,
    integer_format: Optional[IntegerFormat] = None,
    decimal_format: Optional[DecimalFormat] = None,
    vocabulary: Sequence[str] = (),
) -> CellResolution:
    if not readings:
        return CellResolution(status=CellStatus.EMPTY)
    parsed = [_candidate(r, value_type, integer_format, decimal_format, vocabulary) for r in readings]
    low_conf_groups = _screen_candidates(parsed, settings)
    groups: Dict[str, List[Candidate]] = {}
    for c in parsed:
        groups.setdefault(c.independence_group, []).append(c)
    group_value, conflicted = _group_values(groups)
    values = {v for v in group_value.values() if v is not None}
    consensus: Optional[str] = next(iter(values)) if len(values) == 1 else None
    dissent_groups = _dissenting_groups(groups, group_value, conflicted, consensus)
    supporting = sorted(g for g, v in group_value.items() if consensus is not None and v == consensus)
    decimals = next((c.parse.decimals for c in parsed if c.role == "support" and c.parse.value == consensus), None)

    if dissent_groups or len(values) > 1:
        status, value = CellStatus.DISAGREED, None
    elif consensus is None:
        status, value = CellStatus.NO_READING, None
    elif len(supporting) >= settings.min_independent_sources:
        status, value = CellStatus.AGREED, consensus
    else:
        status, value = CellStatus.SINGLE_SOURCE, None
    return CellResolution(
        status=status,
        value=value,
        decimals=decimals if value is not None else None,
        supporting_groups=supporting,
        dissenting_groups=sorted(dissent_groups),
        candidates=parsed,
        low_confidence_groups=sorted(set(low_conf_groups)),
    )


def _candidate(
    r: OCRReading,
    value_type: str,
    integer_format: Optional[IntegerFormat],
    decimal_format: Optional[DecimalFormat],
    vocabulary: Sequence[str],
) -> Candidate:
    return Candidate(
        engine=r.engine,
        engine_version=r.engine_version,
        variant=r.variant,
        independence_group=r.independence_group,
        raw_text=r.raw_text,
        confidence=r.confidence,
        parse=parse_reading(r.raw_text, value_type, integer_format, decimal_format, vocabulary),
        role="abstain",
        meta=dict(r.meta),
    )


def _screen_candidates(parsed: List[Candidate], settings: OCRSettings) -> List[str]:
    """1) per candidate: can it count at all? Sets ``role``; returns the low-confidence groups."""
    floors = settings.min_candidate_confidence
    low_conf_groups: List[str] = []
    for c in parsed:
        floor = floors.get(c.engine)
        if floor is not None and c.confidence is not None and c.confidence < floor:
            c.note = f"confidence {c.confidence} below floor {floor}"
            low_conf_groups.append(c.independence_group)
            continue
        if c.meta.get("word_overflow"):
            c.note = "text-layer word extends beyond token"
            continue
        if c.parse.status == ParseStatus.EMPTY:
            c.note = "empty reading"
            continue
        if c.parse.used_confusion_map and not settings.allow_confusion_normalization_for_agreement:
            c.note = "value only obtainable via OCR confusion substitution"
            c.role = "noise"
            continue
        # text present but not a valid value: provisional dissent, refined per group below
        c.role = "support" if c.parse.value is not None else "dissent"
    return low_conf_groups


def _group_values(groups: Dict[str, List[Candidate]]) -> Tuple[Dict[str, Optional[str]], List[str]]:
    """2) per independence group: one value, a conflict between its variants, or nothing."""
    group_value: Dict[str, Optional[str]] = {}
    conflicted: List[str] = []
    for g, cands in groups.items():
        values = {c.parse.value for c in cands if c.role == "support"}
        if len(values) > 1:
            conflicted.append(g)
            group_value[g] = None
        elif len(values) == 1:
            value = values.pop() or ""
            group_value[g] = value
            # invalid variants inside a group that has a valid reading are variant noise
            for c in cands:
                if c.role == "dissent":
                    c.role = "noise" if _digits(c.raw_text) in ("", _digits(value)) else "dissent"
        else:
            group_value[g] = None
    return group_value, conflicted


def _dissenting_groups(
    groups: Dict[str, List[Candidate]], group_value: Dict[str, Optional[str]], conflicted: List[str], consensus: Optional[str]
) -> Set[str]:
    """3) groups that contradict: internal conflicts, different values, or unparseable text with other digits."""
    dissent = set(conflicted)
    if len({v for v in group_value.values() if v is not None}) > 1:
        dissent.update(g for g, v in group_value.items() if v is not None)
    for g, cands in groups.items():
        if group_value[g] is not None or g in conflicted:
            continue
        for c in cands:
            if c.role != "dissent":
                continue
            if consensus is not None and _digits(c.raw_text) == _digits(consensus) and _digits(c.raw_text):
                c.role = "noise"
                c.note = "unparseable, but same digits as consensus"
            else:
                dissent.add(g)
    return dissent
