"""
Strict numeric parsers.

Design rules
------------
* The raw OCR string is never modified in place; every result carries the raw
  input, the normalized string and the list of normalization steps applied.
* Each value type has its own grammar (integer / decimal / marker). There is
  no "strip every non-digit and hope" parsing.
* OCR confusion substitutions (``O -> 0`` ...) are only attempted when the
  strict parse fails, and the result is flagged ``used_confusion_map`` so the
  candidate resolver can decide whether such readings may count as evidence.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Dict, List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from election_ocr.parsing.text import normalize_unicode

# Unicode dash / minus characters (typography, not domain knowledge).
DASH_CHARS = frozenset({"-", "‐", "‑", "‒", "–", "—", "−", "―", "_"})

# Generic OCR glyph confusions for digit fields. Applied only as a fallback and
# always recorded as a normalization step.
CONFUSION_MAP: Dict[str, str] = {
    "O": "0",
    "o": "0",
    "D": "0",
    "Q": "0",
    "l": "1",
    "I": "1",
    "|": "1",
    "i": "1",
    "S": "5",
    "s": "5",
    "B": "8",
    "Z": "2",
    "z": "2",
    "G": "6",
}


class ParseStatus(str, Enum):
    OK = "OK"  # value parsed under the strict grammar
    EMPTY = "EMPTY"  # nothing readable
    DASH = "DASH"  # explicit dash (meaning depends on profile)
    FAILED = "FAILED"  # text present, but it does not fit the grammar


class ParseResult(BaseModel):
    raw: str
    normalized: str
    status: ParseStatus
    value: Optional[str] = Field(
        default=None, description="Canonical string form of the value (int as digits, decimal as '12.3')."
    )
    decimals: Optional[int] = None
    steps: List[str] = Field(default_factory=list)
    used_confusion_map: bool = False
    reason: Optional[str] = None

    def as_int(self) -> Optional[int]:
        return int(self.value) if self.value is not None and self.decimals in (None, 0) else None

    def as_decimal(self) -> Optional[Decimal]:
        return Decimal(self.value) if self.value is not None else None


SignMode = Literal["none", "trailing"]


class IntegerFormat(BaseModel):
    thousands_separators: List[str] = Field(default_factory=lambda: [" "])
    dash_means_zero: bool = False
    min_value: int = Field(default=0, description="Lower bound of the magnitude.")
    max_value: Optional[int] = None
    sign: SignMode = Field(default="none", description="'trailing': negative numbers printed as '5973-' (change rows).")


class DecimalFormat(BaseModel):
    decimal_separator: str = ","
    decimals: Optional[int] = Field(default=None, description="Exact number of decimals expected, if fixed.")
    dash_means_zero: bool = Field(default=False, description="A printed dash stands for 0 (e.g. vH of a party without votes).")
    min_value: Optional[Decimal] = Field(default=None, description="Lower bound of the magnitude.")
    max_value: Optional[Decimal] = None
    sign: SignMode = Field(default="none", description="'trailing': negative numbers printed as '1,6-'.")


def _split_trailing_sign(s: str, mode: str, steps: List[str]) -> Tuple[str, bool]:
    """'5973-' -> ('5973', True) when the format prints negatives with a trailing minus."""
    if mode == "trailing" and len(s) > 1 and s[-1] in DASH_CHARS and any(ch.isdigit() for ch in s[:-1]):
        steps.append("TRAILING_MINUS")
        return s[:-1].rstrip(), True
    return s, False


def _negate(result: ParseResult, negative: bool) -> ParseResult:
    if negative and result.status == ParseStatus.OK and result.value is not None and Decimal(result.value) != 0:
        result.value = "-" + result.value
    return result


def _base_normalize(raw: str, steps: List[str]) -> str:
    s = normalize_unicode(raw)
    if s != raw:
        steps.append("UNICODE_WHITESPACE_NORMALIZED")
    return s


def _is_dash(s: str) -> bool:
    return len(s) > 0 and all(ch in DASH_CHARS for ch in s.replace(" ", ""))


def _integer_regex(seps: Sequence[str]) -> re.Pattern:
    if seps:
        sep_class = "[" + "".join(re.escape(s) for s in seps) + "]"
        return re.compile(rf"^(?:\d+|\d{{1,3}}(?:{sep_class}\d{{3}})+)$")
    return re.compile(r"^\d+$")


def _apply_confusions(s: str) -> Optional[str]:
    if not any(ch in CONFUSION_MAP for ch in s):
        return None
    return "".join(CONFUSION_MAP.get(ch, ch) for ch in s)


def parse_integer(raw: str, fmt: IntegerFormat) -> ParseResult:
    steps: List[str] = []
    s = _base_normalize(raw, steps)
    if not s:
        return ParseResult(raw=raw, normalized=s, status=ParseStatus.EMPTY, steps=steps)
    if _is_dash(s):
        if fmt.dash_means_zero:
            steps.append("DASH_AS_ZERO")
            return ParseResult(raw=raw, normalized=s, status=ParseStatus.DASH, value="0", decimals=0, steps=steps)
        return ParseResult(raw=raw, normalized=s, status=ParseStatus.DASH, steps=steps, reason="dash not allowed as zero here")

    s, negative = _split_trailing_sign(s, fmt.sign, steps)
    rx = _integer_regex(fmt.thousands_separators)
    result = _try_integer(raw, s, rx, fmt, steps, used_confusion=False)
    if result.status == ParseStatus.OK:
        return _negate(result, negative)
    substituted = _apply_confusions(s)
    if substituted is not None:
        steps2 = steps + [f"OCR_CONFUSION_MAP:{s}->{substituted}"]
        alt = _try_integer(raw, substituted, rx, fmt, steps2, used_confusion=True)
        if alt.status == ParseStatus.OK:
            return _negate(alt, negative)
    return result


def _try_integer(raw: str, s: str, rx: re.Pattern, fmt: IntegerFormat, steps: List[str], used_confusion: bool) -> ParseResult:
    if not rx.match(s):
        return ParseResult(
            raw=raw,
            normalized=s,
            status=ParseStatus.FAILED,
            steps=list(steps),
            used_confusion_map=used_confusion,
            reason="does not match integer grammar",
        )
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits != s:
        steps = steps + ["THOUSANDS_SEPARATOR_REMOVED"]
    value = int(digits)
    if value < fmt.min_value or (fmt.max_value is not None and value > fmt.max_value):
        return ParseResult(
            raw=raw,
            normalized=s,
            status=ParseStatus.FAILED,
            steps=steps,
            used_confusion_map=used_confusion,
            reason=f"value {value} outside [{fmt.min_value}, {fmt.max_value}]",
        )
    return ParseResult(
        raw=raw,
        normalized=s,
        status=ParseStatus.OK,
        value=str(value),
        decimals=0,
        steps=steps,
        used_confusion_map=used_confusion,
    )


def parse_decimal(raw: str, fmt: DecimalFormat) -> ParseResult:
    steps: List[str] = []
    s = _base_normalize(raw, steps).replace(" ", "")
    if not s:
        return ParseResult(raw=raw, normalized=s, status=ParseStatus.EMPTY, steps=steps)
    if _is_dash(s):
        if fmt.dash_means_zero:
            steps.append("DASH_AS_ZERO")
            return ParseResult(raw=raw, normalized=s, status=ParseStatus.DASH, value="0", decimals=fmt.decimals or 0, steps=steps)
        return ParseResult(raw=raw, normalized=s, status=ParseStatus.DASH, steps=steps, reason="dash in decimal field")
    s, negative = _split_trailing_sign(s, fmt.sign, steps)
    result = _try_decimal(raw, s, fmt, steps, used_confusion=False)
    if result.status == ParseStatus.OK:
        return _negate(result, negative)
    substituted = _apply_confusions(s)
    other_sep = "." if fmt.decimal_separator == "," else ","
    candidates = []
    if substituted is not None:
        candidates.append((substituted, [f"OCR_CONFUSION_MAP:{s}->{substituted}"]))
    if other_sep in s:
        swapped = s.replace(other_sep, fmt.decimal_separator)
        candidates.append((swapped, [f"DECIMAL_SEPARATOR_SWAPPED:{other_sep}->{fmt.decimal_separator}"]))
    for cand, extra in candidates:
        alt = _try_decimal(raw, cand, fmt, steps + extra, used_confusion=True)
        if alt.status == ParseStatus.OK:
            return _negate(alt, negative)
    return result


def _try_decimal(raw: str, s: str, fmt: DecimalFormat, steps: List[str], used_confusion: bool) -> ParseResult:
    sep = re.escape(fmt.decimal_separator)
    if fmt.decimals is not None:
        rx = re.compile(rf"^\d+{sep}\d{{{fmt.decimals}}}$") if fmt.decimals > 0 else re.compile(r"^\d+$")
    else:
        rx = re.compile(rf"^\d+(?:{sep}\d+)?$")
    if not rx.match(s):
        return ParseResult(
            raw=raw,
            normalized=s,
            status=ParseStatus.FAILED,
            steps=list(steps),
            used_confusion_map=used_confusion,
            reason="does not match decimal grammar",
        )
    canonical = s.replace(fmt.decimal_separator, ".")
    try:
        value = Decimal(canonical)
    except InvalidOperation:  # pragma: no cover - regex guarantees validity
        return ParseResult(raw=raw, normalized=s, status=ParseStatus.FAILED, steps=steps, reason="invalid decimal")
    if (fmt.min_value is not None and value < fmt.min_value) or (fmt.max_value is not None and value > fmt.max_value):
        return ParseResult(
            raw=raw,
            normalized=s,
            status=ParseStatus.FAILED,
            steps=steps,
            used_confusion_map=used_confusion,
            reason=f"value {value} outside [{fmt.min_value}, {fmt.max_value}]",
        )
    decimals = len(canonical.split(".")[1]) if "." in canonical else 0
    return ParseResult(
        raw=raw,
        normalized=s,
        status=ParseStatus.OK,
        value=canonical,
        decimals=decimals,
        steps=steps,
        used_confusion_map=used_confusion,
    )


def parse_marker(raw: str, vocabulary: Sequence[str]) -> ParseResult:
    """Exact (case-insensitive) match against a closed vocabulary, e.g. vote markers."""
    steps: List[str] = []
    s = _base_normalize(raw, steps).replace(" ", "")
    if not s:
        return ParseResult(raw=raw, normalized=s, status=ParseStatus.EMPTY, steps=steps)
    for token in vocabulary:
        if s.casefold() == token.casefold():
            if s != token:
                steps.append("MARKER_CASE_NORMALIZED")
            return ParseResult(raw=raw, normalized=s, status=ParseStatus.OK, value=token, steps=steps)
    return ParseResult(raw=raw, normalized=s, status=ParseStatus.FAILED, steps=steps, reason="not in marker vocabulary")
