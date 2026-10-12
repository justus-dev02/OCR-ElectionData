"""
Deterministic validation framework.

Works on ``RowValues`` - a neutral view of one candidate result row whose
field values come from agreed OCR cells or from human review decisions. The
same code therefore validates fresh pipeline output and re-validates rows
after review.

Rule families
-------------
* cell     : OCR agreement status, missing required values, token geometry
* row      : arithmetic identities / inequalities (profile ``required_checks``)
* group    : fields that must be identical across row roles (e.g. eligible
             voters of Erst- and Zweitstimmen lines)
* section  : hierarchical sums - members must add up to the printed total
* percent  : printed shares (percent roles) must equal the shares computed
             from the absolute values of the same group (blocking)
* blanks   : an empty party cell is 0 only if the party sum then equals the
             valid votes exactly (profile flag, docs/decisions.md D6)

A passing arithmetic check is *evidence*, never proof: it never overrides an
OCR disagreement (a transposition error can satisfy a sum by accident).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from election_ocr.issues import Issue, IssueCode, IssueFactory, IssueScope
from election_ocr.ocr.resolver import CellStatus
from election_ocr.reference.models import PercentDefinition


@dataclass
class FieldValue:
    name: str
    cell_id: Optional[str]
    status: str  # CellStatus value or "ABSENT"
    value: Optional[Decimal] = None
    decimals: Optional[int] = None
    source: str = "ocr"  # ocr | human
    required: bool = True
    assignment_status: Optional[str] = None
    crosses_boundary: bool = False
    shape_conflict: Optional[str] = None

    @property
    def usable(self) -> bool:
        return self.value is not None and self.status in (CellStatus.AGREED.value, CellStatus.HUMAN.value, BLANK_ZERO)


BLANK = "BLANK"  # no ink, profile allows a zero proof (see apply_blank_zero)
BLANK_ZERO = "BLANK_ZERO"  # no ink, zero proven by the party sum


@dataclass
class RowValues:
    result_key: str
    row_kind: str
    marker: Optional[str]
    vote_type: str
    group_key: str
    section_key: Optional[str]
    publishable_role: bool
    role_id: Optional[str] = None
    role_kind: str = "absolute"
    percent_of_role: Optional[str] = None
    section_grouping: bool = False
    metrics: Dict[str, FieldValue] = field(default_factory=dict)
    parties: Dict[str, FieldValue] = field(default_factory=dict)
    acknowledged: Dict[str, str] = field(default_factory=dict)  # IssueCode -> reviewer reason

    def value(self, metric: str) -> Optional[Decimal]:
        fv = self.metrics.get(metric)
        return fv.value if fv is not None and fv.usable else None


def _usable_values(fields: Sequence[Optional[FieldValue]]) -> Optional[List[Decimal]]:
    """The values of the fields if ALL of them are usable, else None."""
    out: List[Decimal] = []
    for fv in fields:
        if fv is None or not fv.usable or fv.value is None:
            return None
        out.append(fv.value)
    return out


def cell_issues(fv: FieldValue, factory: IssueFactory) -> List[Issue]:
    out: List[Issue] = []
    fname = fv.name
    if fv.source == "human":
        out.append(factory.make(IssueCode.HUMAN_CORRECTED, IssueScope.CELL, f"{fname} set by reviewer", field=fname))
        return out
    if fv.status in (BLANK, BLANK_ZERO):
        return out  # decided by apply_blank_zero / percent checks
    if fv.status == "ABSENT":
        if fv.required:
            out.append(factory.make(IssueCode.MISSING_VALUE, IssueScope.CELL, f"no ink found for {fname}", field=fname))
        return out
    status_map = {
        CellStatus.DISAGREED.value: (IssueCode.OCR_ENGINE_DISAGREEMENT, "OCR engines disagree"),
        CellStatus.SINGLE_SOURCE.value: (IssueCode.OCR_SINGLE_SOURCE, "value read by only one engine group"),
        CellStatus.NO_READING.value: (IssueCode.OCR_NO_READING, "ink present but no usable reading"),
    }
    if fv.status in status_map:
        code, msg = status_map[fv.status]
        out.append(factory.make(code, IssueScope.CELL, f"{fname}: {msg}", field=fname, cell_id=fv.cell_id))
    if fv.assignment_status == "AMBIGUOUS":
        out.append(
            factory.make(IssueCode.CELL_BOUNDARY_AMBIGUOUS, IssueScope.CELL, f"{fname}: ambiguous line assignment", field=fname)
        )
    if fv.crosses_boundary:
        out.append(
            factory.make(IssueCode.CELL_BOUNDARY_AMBIGUOUS, IssueScope.CELL, f"{fname}: glyph crosses a column rule", field=fname)
        )
    if fv.shape_conflict:
        out.append(
            factory.make(
                IssueCode.GLYPH_SHAPE_CONFLICT, IssueScope.CELL, f"{fname}: {fv.shape_conflict}", field=fname, cell_id=fv.cell_id
            )
        )
    return out


def apply_blank_zero(row: RowValues, factory: IssueFactory) -> List[Issue]:
    """Decide empty party cells of an absolute row (profile ``blank_party_cells_zero_with_sum_proof``).

    Zero is accepted only if every other party value and the valid votes are usable
    and the party sum with the blanks as 0 equals the valid votes EXACTLY; then the
    blanks become 0 (status BLANK_ZERO, INFO issue). Otherwise every blank is a
    blocking MISSING_VALUE. Percent rows are decided by the percent checks.
    """
    out: List[Issue] = []
    blanks = [fv for fv in row.parties.values() if fv.status == BLANK]
    if not blanks or row.role_kind != "absolute":
        return out
    valid = row.value("valid_votes")
    others = [fv for fv in row.parties.values() if fv.status != BLANK]
    other_values = _usable_values(others)
    proven = valid is not None and other_values is not None and sum(other_values, Decimal(0)) == valid
    for fv in blanks:
        if proven:
            fv.value, fv.status, fv.decimals, fv.source = Decimal(0), BLANK_ZERO, 0, "blank_zero"
            out.append(
                factory.make(
                    IssueCode.BLANK_AS_ZERO, IssueScope.CELL, f"{fv.name}: empty cell = 0 (party sum proven)", field=fv.name
                )
            )
        else:
            out.append(
                factory.make(
                    IssueCode.MISSING_VALUE,
                    IssueScope.CELL,
                    f"{fv.name}: empty cell, zero not proven by the party sum",
                    field=fv.name,
                )
            )
    return out


def _fmt(v: Optional[Decimal]) -> Optional[str]:
    return None if v is None else str(v)


def computed_turnout(row: RowValues) -> Optional[Decimal]:
    """voters / eligible x 100, rounded half up to the printed decimals (None if not computable)."""
    ev, vo = row.value("eligible_voters"), row.value("voters")
    fv = row.metrics.get("turnout_percent")
    if ev is None or vo is None or ev == 0 or fv is None or not fv.usable:
        return None
    return (vo * Decimal(100) / ev).quantize(Decimal(1).scaleb(-(fv.decimals or 0)), rounding=ROUND_HALF_UP)


def turnout_match(row: RowValues) -> Optional[bool]:
    """Does the printed turnout follow voters/eligible? None if not decidable for this row."""
    computed = computed_turnout(row)
    return None if computed is None else computed == row.metrics["turnout_percent"].value


def row_checks(row: RowValues, required_checks: Iterable[str], factory: IssueFactory) -> List[Issue]:
    """Run the profile's arithmetic row checks; a check without agreed operands is itself blocking."""
    out: List[Issue] = []
    for check in required_checks:
        fn = ROW_CHECKS.get(check)
        if fn is None:  # pragma: no cover - guarded by the profile schema
            raise ValueError(f"unknown check {check}")
        out.extend(fn(row, factory))
    return out


def _operands(row: RowValues, check: str, factory: IssueFactory, *metrics: str) -> Tuple[Optional[List[Decimal]], List[Issue]]:
    """Usable values of ``metrics`` in order, or None plus an ARITHMETIC_CHECK_NOT_EXECUTABLE issue."""
    values = [row.value(m) for m in metrics]
    missing = [m for m, v in zip(metrics, values) if v is None]
    if missing:
        return None, [_not_executable(factory, check, missing)]
    return [v for v in values if v is not None], []


def _not_executable(factory: IssueFactory, check: str, missing: List[str]) -> Issue:
    return factory.make(
        IssueCode.ARITHMETIC_CHECK_NOT_EXECUTABLE,
        IssueScope.ROW,
        f"check '{check}' cannot run: missing agreed values for {missing}",
        check=check,
        missing=missing,
    )


def _check_voters_le_eligible(row: RowValues, factory: IssueFactory) -> List[Issue]:
    vals, out = _operands(row, "voters_le_eligible", factory, "eligible_voters", "voters")
    if vals is None:
        return out
    ev, vo = vals
    return [factory.make(IssueCode.VOTERS_EXCEED_ELIGIBLE, IssueScope.ROW, f"voters {vo} > eligible {ev}")] if vo > ev else []


def _check_valid_le_voters(row: RowValues, factory: IssueFactory) -> List[Issue]:
    vals, out = _operands(row, "valid_le_voters", factory, "valid_votes", "voters")
    if vals is None:
        return out
    va, vo = vals
    return [factory.make(IssueCode.VALID_EXCEED_VOTERS, IssueScope.ROW, f"valid {va} > voters {vo}")] if va > vo else []


def _check_voters_eq_valid_plus_invalid(row: RowValues, factory: IssueFactory) -> List[Issue]:
    vals, out = _operands(row, "voters_eq_valid_plus_invalid", factory, "voters", "valid_votes", "invalid_votes")
    if vals is None:
        return out
    vo, va, inv = vals
    if vo == va + inv:
        return []
    return [factory.make(IssueCode.VOTERS_NE_VALID_PLUS_INVALID, IssueScope.ROW, f"voters {vo} != valid {va} + invalid {inv}")]


def _check_valid_plus_invalid_le_eligible(row: RowValues, factory: IssueFactory) -> List[Issue]:
    vals, out = _operands(row, "valid_plus_invalid_le_eligible", factory, "eligible_voters", "valid_votes", "invalid_votes")
    if vals is None:
        return out
    ev, va, inv = vals
    if va + inv <= ev:
        return []
    return [factory.make(IssueCode.VOTERS_EXCEED_ELIGIBLE, IssueScope.ROW, f"valid {va} + invalid {inv} > eligible {ev}")]


def _check_party_sum_eq_valid(row: RowValues, factory: IssueFactory) -> List[Issue]:
    va = row.value("valid_votes")
    missing = [p for p, fv in row.parties.items() if not fv.usable]
    party_values = _usable_values(list(row.parties.values()))
    if va is None or missing or party_values is None:
        return [
            _not_executable(
                factory, "party_sum_eq_valid", (["valid_votes"] if va is None else []) + [f"party:{p}" for p in missing]
            )
        ]
    total = sum(party_values, Decimal(0))
    if total == va:
        return []
    return [
        factory.make(
            IssueCode.PARTY_SUM_MISMATCH,
            IssueScope.ROW,
            f"sum of party votes {total} != valid votes {va}",
            party_sum=str(total),
            valid_votes=str(va),
            difference=str(total - va),
        )
    ]


def _check_turnout_consistent(row: RowValues, factory: IssueFactory) -> List[Issue]:
    vals, out = _operands(row, "turnout_consistent", factory, "eligible_voters", "voters", "turnout_percent")
    if vals is None:
        return out
    ev, _, reported = vals
    if ev == 0:
        return [factory.make(IssueCode.VALUE_OUT_OF_RANGE, IssueScope.ROW, "eligible voters is zero")]
    computed = computed_turnout(row)
    if computed == reported:
        return []
    turnout_fv = row.metrics.get("turnout_percent")
    decimals = (turnout_fv.decimals if turnout_fv else 0) or 0
    return [
        factory.make(
            IssueCode.TURNOUT_MISMATCH,
            IssueScope.ROW,
            f"reported turnout {reported} != computed {computed} (round half up, {decimals} decimals)",
            reported=_fmt(reported),
            computed=_fmt(computed),
        )
    ]


ROW_CHECKS: Dict[str, Callable[[RowValues, IssueFactory], List[Issue]]] = {
    "voters_le_eligible": _check_voters_le_eligible,
    "valid_le_voters": _check_valid_le_voters,
    "voters_eq_valid_plus_invalid": _check_voters_eq_valid_plus_invalid,
    "valid_plus_invalid_le_eligible": _check_valid_plus_invalid_le_eligible,
    "party_sum_eq_valid": _check_party_sum_eq_valid,
    "turnout_consistent": _check_turnout_consistent,
}


def cross_role_checks(
    rows: List[RowValues], consistent_fields: Dict[str, List[str]], factory: IssueFactory
) -> Dict[str, List[Issue]]:
    """Fields that must be identical across markers of one group (e.g. eligible voters of rows 1 and 2)."""
    out: Dict[str, List[Issue]] = {}
    for metric, markers in consistent_fields.items():
        members = [r for r in rows if r.marker in markers]
        values = {r.result_key: r.value(metric) for r in members}
        known = {v for v in values.values() if v is not None}
        if len(known) > 1:
            for r in members:
                out.setdefault(r.result_key, []).append(
                    factory.make(
                        IssueCode.CROSS_ROLE_MISMATCH,
                        IssueScope.ROW,
                        f"{metric} differs between row roles {markers}: {sorted(map(str, known))}",
                        field=metric,
                    )
                )
    return out


def hierarchy_checks(
    rows: List[RowValues],
    totals_expected: bool,
    require_consistency: bool,
    factory: IssueFactory,
) -> Dict[str, List[Issue]]:
    """Members of a section must sum to the section's printed total (per row role)."""
    out: Dict[str, List[Issue]] = {}
    sections: Dict[tuple, Dict[str, List[RowValues]]] = {}
    for r in rows:
        if r.section_key is None or r.role_kind != "absolute" or r.section_grouping:
            continue  # shares/differences are no sums; groupings ('Stadtkreise') print no total
        if r.row_kind == "GRAND_TOTAL":
            continue  # checked against all rows below
        key = (r.section_key, r.role_id or r.marker or r.vote_type)
        bucket = sections.setdefault(key, {"DATA": [], "SECTION_TOTAL": []})
        bucket.setdefault(r.row_kind, []).append(r)

    for (section, role), bucket in sections.items():
        members, totals = bucket["DATA"], bucket["SECTION_TOTAL"]
        affected = members + totals
        if len(totals) > 1:
            for r in affected:
                out.setdefault(r.result_key, []).append(
                    factory.make(
                        IssueCode.HIERARCHY_UNVERIFIED, IssueScope.SECTION, f"section {section} has {len(totals)} total rows"
                    )
                )
            continue
        if not totals:
            if totals_expected and require_consistency and len(members) > 1:
                for r in affected:
                    out.setdefault(r.result_key, []).append(
                        factory.make(
                            IssueCode.HIERARCHY_UNVERIFIED,
                            IssueScope.SECTION,
                            f"no total row found for section {section} (role {role})",
                        )
                    )
            continue
        total = totals[0]
        if not members:
            continue  # e.g. an independent city printed as a single total line
        mismatches, unverifiable = sum_mismatches(total, members)
        for r in affected:
            if mismatches:
                out.setdefault(r.result_key, []).append(
                    factory.make(
                        IssueCode.HIERARCHICAL_SUM_MISMATCH,
                        IssueScope.SECTION,
                        f"members of section {section} (role {role}) do not add up to the total",
                        mismatches=mismatches,
                    )
                )
            if unverifiable and require_consistency:
                out.setdefault(r.result_key, []).append(
                    factory.make(
                        IssueCode.HIERARCHY_UNVERIFIED,
                        IssueScope.SECTION,
                        f"section sum not verifiable for {unverifiable} (unresolved member or total values)",
                        fields=unverifiable,
                    )
                )
    return out


def sum_mismatches(total: RowValues, members: List[RowValues]) -> Tuple[List[dict], List[str]]:
    """Compare every absolute field of a total row with the sum over its members.

    Returns (mismatches, unverifiable field names); turnout is a ratio and never summed."""
    fields = [("metric", n) for n in total.metrics if n != "turnout_percent"] + [("party", p) for p in total.parties]
    mismatches: List[dict] = []
    unverifiable: List[str] = []
    for kind, name in fields:
        tv = total.metrics.get(name) if kind == "metric" else total.parties.get(name)
        member_values = _usable_values([(m.metrics.get(name) if kind == "metric" else m.parties.get(name)) for m in members])
        if tv is None or not tv.usable or tv.value is None or member_values is None:
            unverifiable.append(name)
            continue
        s = sum(member_values, Decimal(0))
        if s != tv.value:
            mismatches.append({"field": name, "sum_members": str(s), "total": str(tv.value), "difference": str(s - tv.value)})
    return mismatches, unverifiable


def grand_total_checks(rows: List[RowValues], complete: bool, factory: IssueFactory) -> Dict[str, List[Issue]]:
    """A grand total (e.g. 'Land Baden-Wuerttemberg') must equal the sum of ALL unit rows of its role.

    Only verifiable when the run covers every profiled page (``complete``)."""
    out: Dict[str, List[Issue]] = {}
    for gt in [r for r in rows if r.row_kind == "GRAND_TOTAL" and r.role_kind == "absolute"]:
        members = [r for r in rows if r.row_kind == "DATA" and r.role_id == gt.role_id and r.role_kind == "absolute"]
        if not complete:
            out.setdefault(gt.result_key, []).append(
                factory.make(IssueCode.HIERARCHY_UNVERIFIED, IssueScope.SECTION, "grand total: run does not cover all pages")
            )
            continue
        mismatches, unverifiable = sum_mismatches(gt, members)
        if mismatches:
            out.setdefault(gt.result_key, []).append(
                factory.make(
                    IssueCode.HIERARCHICAL_SUM_MISMATCH,
                    IssueScope.SECTION,
                    "grand total differs from the sum of all units",
                    mismatches=mismatches,
                )
            )
        if unverifiable:
            out.setdefault(gt.result_key, []).append(
                factory.make(
                    IssueCode.HIERARCHY_UNVERIFIED,
                    IssueScope.SECTION,
                    f"grand total not verifiable for {unverifiable}",
                    fields=unverifiable,
                )
            )
    return out


def duplicate_unit_checks(
    rows: List[RowValues], unit_of: Dict[str, Optional[str]], factory: IssueFactory
) -> Dict[str, List[Issue]]:
    out: Dict[str, List[Issue]] = {}
    seen: Dict[tuple, List[str]] = {}
    for r in rows:
        uid = unit_of.get(r.result_key)
        if uid is None:
            continue
        seen.setdefault((uid, r.role_id or r.marker or r.vote_type, r.row_kind), []).append(r.result_key)
    for (uid, _, _), keys in seen.items():
        if len(keys) > 1:
            for k in keys:
                out.setdefault(k, []).append(
                    factory.make(IssueCode.DUPLICATE_UNIT_IN_SECTION, IssueScope.ROW, f"unit {uid} appears {len(keys)} times")
                )
    return out


def _denominator(absolute: RowValues, d: PercentDefinition) -> Optional[Decimal]:
    """Sum of the denominator fields of the absolute row (None if any is not usable)."""
    dens = [_abs_value(absolute, x) for x in d.denominator]
    known = [x for x in dens if x is not None]
    return sum(known, Decimal(0)) if len(known) == len(dens) else None


def _abs_value(row: RowValues, name: str) -> Optional[Decimal]:
    fv = row.parties.get(name[len("party:") :]) if name.startswith("party:") else row.metrics.get(name)
    return fv.value if fv is not None and fv.usable else None


def _party_shares(
    pct: RowValues, absolute: RowValues, definitions: Mapping[str, PercentDefinition]
) -> Optional[Dict[str, Tuple[Decimal, Decimal, Optional[int]]]]:
    """(printed, exact, decimals) per party share defined over the valid votes; None if any is not usable."""
    d = definitions.get("party:*")
    if d is None or not pct.parties:
        return None
    den = _denominator(absolute, d)
    if den is None or den == 0:
        return None
    out: Dict[str, Tuple[Decimal, Decimal, Optional[int]]] = {}
    for k, fv in pct.parties.items():
        num = _abs_value(absolute, "party:" + k)
        if fv.status in (BLANK, "ABSENT"):
            if num is None:
                return None
            out[k] = (Decimal(0), num * Decimal(100) / den, None)
            continue
        if not fv.usable or fv.value is None or num is None:
            return None
        out[k] = (fv.value, num * Decimal(100) / den, fv.decimals or 0)
    # every party of the absolute row must have its share (else the sum cannot be 100)
    if set(absolute.parties) - set(out):
        return None
    return out


def share_row_rounding(pct: RowValues, absolute: RowValues, definitions: Mapping[str, PercentDefinition]) -> Optional[str]:
    """How the party shares of one percent row are rounded (None: not decidable).

    'half_up'  every share equals its exact value rounded half up
    'sum100'   not all half up, but every share is the exact value rounded down or up AND the
               printed shares add up to exactly 100 (rounding adjusted so the row sums to 100)
    'other'    neither (misreading, printing error or another definition)
    """
    shares = _party_shares(pct, absolute, definitions)
    if shares is None:
        return None
    decimals = max((dec for _, _, dec in shares.values() if dec is not None), default=0)
    q = Decimal(1).scaleb(-decimals)
    half_up = all(p == e.quantize(q, rounding=ROUND_HALF_UP) for p, e, _ in shares.values())
    if half_up:
        return "half_up"
    bounded = all(
        e.quantize(q, rounding=ROUND_FLOOR) <= p <= e.quantize(q, rounding=ROUND_CEILING) for p, e, _ in shares.values()
    )
    if bounded and sum(p for p, _, _ in shares.values()) == 100:
        return "sum100"
    return "other"


def follows_sum100(pct: RowValues, absolute: RowValues, definitions: Mapping[str, PercentDefinition]) -> Optional[bool]:
    """Probe evidence: is this row consistent with sum-to-100 rounding? (None: not decidable)

    Consistent: shares adjusted to 100 ('sum100'), or plain half-up shares that happen to add
    up to exactly 100. Inconsistent: half-up shares adding up to something else, or 'other'."""
    kind = share_row_rounding(pct, absolute, definitions)
    if kind is None:
        return None
    if kind == "half_up":
        shares = _party_shares(pct, absolute, definitions) or {}
        return sum(p for p, _, _ in shares.values()) == 100
    return kind == "sum100"


def percent_field_results(
    pct: RowValues, absolute: RowValues, definitions: Mapping[str, PercentDefinition], sum100: bool = False
) -> Dict[str, Tuple[Optional[bool], dict]]:
    """Per printed share of a percent row: (matches?, evidence). matches=None: not decidable.

    computed = numerator / sum(denominator) x 100, rounded half up to the printed number
    of decimals. An empty percent cell stands for 0 and requires the absolute numerator
    to be exactly 0. With ``sum100`` (decided per run by a probe: the document adjusts the
    rounding so that the party shares of a row add up to 100) a party share is also accepted
    when it lies between the exact value rounded down and up - only in rows whose printed
    party shares add up to exactly 100.
    """
    out: Dict[str, Tuple[Optional[bool], dict]] = {}
    row_kind = share_row_rounding(pct, absolute, definitions) if sum100 else None
    adjusted_row = row_kind == "sum100"
    fields = [(f"{k}", fv) for k, fv in pct.metrics.items()] + [(f"party:{k}", fv) for k, fv in pct.parties.items()]
    for name, fv in fields:
        d = definitions.get(name) or (definitions.get("party:*") if name.startswith("party:") else None)
        if d is None:
            out[name] = (None, {"reason": "no percent definition"})
            continue
        numerator = name if d.numerator == "party:*" else d.numerator
        num = _abs_value(absolute, numerator)
        den = _denominator(absolute, d)
        if num is None or den is None:
            out[name] = (None, {"reason": "absolute values not usable"})
            continue
        if fv.status in (BLANK, "ABSENT"):
            out[name] = (num == 0, {"field": name, "printed": "(empty)", "computed_from": str(num)})
            continue
        if not fv.usable or fv.value is None:
            out[name] = (None, {"reason": "printed share not usable"})
            continue
        decimals = fv.decimals or 0
        if den == 0:
            out[name] = (num == 0 and fv.value == 0, {"field": name, "printed": str(fv.value), "computed": None})
            continue
        computed = (num * Decimal(100) / den).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)
        if computed != fv.value and sum100 and name.startswith("party:"):
            if adjusted_row:
                # share_row_rounding already proved floor <= printed <= ceil for every party share
                out[name] = (True, {"field": name, "printed": str(fv.value), "computed": str(computed), "rounding": "sum100"})
                continue
            q = Decimal(1).scaleb(-decimals)
            exact = num * Decimal(100) / den
            if row_kind is None and exact.quantize(q, rounding=ROUND_FLOOR) <= fv.value <= exact.quantize(
                q, rounding=ROUND_CEILING
            ):
                # possibly adjusted to 100, but the row is incomplete: cannot be verified either way
                out[name] = (None, {"reason": "sum-to-100 rounding not verifiable (row incomplete)"})
                continue
        out[name] = (computed == fv.value, {"field": name, "printed": str(fv.value), "computed": str(computed)})
    return out


def percent_checks(
    pct: RowValues,
    absolute: Optional[RowValues],
    definitions: Mapping[str, PercentDefinition],
    factory: IssueFactory,
    skip_fields: Iterable[str] = (),
    sum100: bool = False,
) -> List[Issue]:
    """Every printed share of a percent row must equal the share computed from the absolute row.

    Fields in ``skip_fields`` follow another printed definition (decided by a definition
    probe over the whole run) and are not checked. Returned issues apply to BOTH rows.
    """
    out: List[Issue] = []
    if absolute is None:
        out.append(
            factory.make(
                IssueCode.PERCENT_CHECK_NOT_EXECUTABLE, IssueScope.ROW, f"no absolute row for percent role {pct.role_id}"
            )
        )
        return out
    skip = set(skip_fields)
    mismatches, missing, adjusted = [], [], []
    for name, (ok, evidence) in percent_field_results(pct, absolute, definitions, sum100=sum100).items():
        if name in skip:
            continue
        if ok is None:
            missing.append(name)
        elif not ok:
            mismatches.append(evidence)
        elif evidence.get("rounding") == "sum100":
            adjusted.append(evidence)
    if adjusted:
        out.append(
            factory.make(
                IssueCode.PERCENT_ROUNDING_ADJUSTED,
                IssueScope.ROW,
                f"{len(adjusted)} printed share(s) of {pct.role_id} rounded so that the row adds up to 100 "
                "(accepted: run-level rounding probe holds)",
                adjusted=adjusted,
            )
        )
    if mismatches:
        out.append(
            factory.make(
                IssueCode.PERCENT_MISMATCH,
                IssueScope.ROW,
                f"printed shares of {pct.role_id} do not match the absolute values of {absolute.role_id}",
                mismatches=mismatches,
            )
        )
    if missing:
        out.append(
            factory.make(
                IssueCode.PERCENT_CHECK_NOT_EXECUTABLE,
                IssueScope.ROW,
                f"percent check {pct.role_id} not executable for {missing}",
                fields=missing,
            )
        )
    return out


def _row_fields(row: RowValues):
    return [(k, fv) for k, fv in row.metrics.items()] + [(f"party:{k}", fv) for k, fv in row.parties.items()]


def difference_field_results(diff: RowValues, a: RowValues, b: RowValues) -> Dict[str, Tuple[Optional[bool], dict]]:
    """Printed difference row: value == a - b for every field (empty cell = 0)."""
    out: Dict[str, Tuple[Optional[bool], dict]] = {}
    for name, fv in _row_fields(diff):
        va, vb = _abs_value(a, name), _abs_value(b, name)
        if va is None or vb is None:
            out[name] = (None, {"reason": "operands not usable"})
            continue
        computed = va - vb
        if fv.status in (BLANK, "ABSENT"):
            out[name] = (computed == 0, {"field": name, "printed": "(empty)", "computed": str(computed)})
        elif not fv.usable:
            out[name] = (None, {"reason": "printed difference not usable"})
        else:
            out[name] = (fv.value == computed, {"field": name, "printed": str(fv.value), "computed": str(computed)})
    return out


def change_field_results(change: RowValues, diff: RowValues, base: RowValues) -> Dict[str, Tuple[Optional[bool], dict]]:
    """Printed relative change: value == difference / base x 100 (rounded half up to the printed
    decimals); an empty cell is only consistent where the base or the difference is 0."""
    out: Dict[str, Tuple[Optional[bool], dict]] = {}
    for name, fv in _row_fields(change):
        d, b = _abs_value(diff, name), _abs_value(base, name)
        if d is None or b is None:
            out[name] = (None, {"reason": "operands not usable"})
            continue
        if fv.status in (BLANK, "ABSENT"):
            out[name] = (b == 0 or d == 0, {"field": name, "printed": "(empty)", "difference": str(d), "base": str(b)})
            continue
        if not fv.usable:
            out[name] = (None, {"reason": "printed change not usable"})
            continue
        if b == 0:
            out[name] = (False, {"field": name, "printed": str(fv.value), "base": "0"})
            continue
        q = Decimal(1).scaleb(-(fv.decimals or 0))
        computed = (d * Decimal(100) / b).quantize(q, rounding=ROUND_HALF_UP)
        out[name] = (computed == fv.value, {"field": name, "printed": str(fv.value), "computed": str(computed)})
    return out


def derived_checks(
    row: RowValues,
    results: Dict[str, Tuple[Optional[bool], dict]],
    factory: IssueFactory,
    skip_fields: Iterable[str] = (),
) -> List[Issue]:
    """Issues for a difference/change row from its per-field results (fields in skip_fields follow
    another printed definition, decided by a definition probe)."""
    skip = set(skip_fields)
    mismatches = [ev for n, (ok, ev) in results.items() if n not in skip and ok is False]
    missing = [n for n, (ok, _) in results.items() if n not in skip and ok is None]
    out: List[Issue] = []
    if mismatches:
        out.append(
            factory.make(
                IssueCode.DERIVED_VALUE_MISMATCH,
                IssueScope.ROW,
                f"printed {row.role_kind} values of {row.role_id} do not match the rows they derive from",
                mismatches=mismatches,
            )
        )
    if missing:
        out.append(
            factory.make(
                IssueCode.DERIVED_CHECK_NOT_EXECUTABLE,
                IssueScope.ROW,
                f"{row.role_kind} check {row.role_id} not executable for {missing}",
                fields=missing,
            )
        )
    return out


# ------------------------------------------------------------ definition probes
HOLDS, DIFFERS, UNDECIDED = "HOLDS", "DIFFERS", "UNDECIDED"


@dataclass
class DefinitionProbe:
    """Does a printed quantity follow the definition the check assumes - decided on the data.

    Printed statistics differ between publications (e.g. a 'Wahlbeteiligung' that includes
    postal voters counted outside the municipality). Instead of switching checks per
    document, every run measures the share of decidable rows in which the printed value
    equals its computation:

    * share >= ``holds_min_share``  -> the definition holds; a deviating row is an error (blocking)
    * share <= ``differs_max_share`` -> the publication uses another definition; the check is
                                         skipped and the value is stored with a note
    * otherwise, or fewer than ``min_rows`` decidable rows -> UNDECIDED; checks stay blocking
    """

    key: str
    matches: int = 0
    total: int = 0
    decision: str = UNDECIDED

    def add(self, result: Optional[bool]) -> None:
        if result is None:
            return
        self.total += 1
        self.matches += int(result)

    def decide(self, min_rows: int, holds_min_share: float, differs_max_share: float) -> str:
        if self.total < min_rows:
            self.decision = UNDECIDED
        else:
            share = self.matches / self.total
            self.decision = HOLDS if share >= holds_min_share else DIFFERS if share <= differs_max_share else UNDECIDED
        return self.decision

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "matches": self.matches,
            "decidable_rows": self.total,
            "share": round(self.matches / self.total, 4) if self.total else None,
            "decision": self.decision,
        }
