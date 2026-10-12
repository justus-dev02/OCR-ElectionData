"""
Domain reference data models.

These describe *what is allowed to exist* (elections, parties per election,
administrative units valid at a date, document/table profiles). They are loaded
from ``reference/`` - never defined in code - and every record carries a
``verified`` flag: unverified reference data blocks publication.
"""

from __future__ import annotations

import re
from datetime import date
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

from election_ocr.parsing.numbers import DecimalFormat, IntegerFormat


class VoteType(str, Enum):
    ERSTSTIMME = "ERSTSTIMME"
    ZWEITSTIMME = "ZWEITSTIMME"
    SINGLE = "SINGLE"  # one-vote system
    COMPARISON = "COMPARISON"  # comparison figures of another election printed alongside


METRIC_FIELDS = ("eligible_voters", "voters", "valid_votes", "invalid_votes", "turnout_percent")
MetricName = Literal["eligible_voters", "voters", "valid_votes", "invalid_votes", "turnout_percent"]

CHECK_NAMES = (
    "voters_le_eligible",
    "valid_le_voters",
    "voters_eq_valid_plus_invalid",
    "valid_plus_invalid_le_eligible",
    "party_sum_eq_valid",
    "turnout_consistent",
)
CheckName = Literal[
    "voters_le_eligible",
    "valid_le_voters",
    "voters_eq_valid_plus_invalid",
    "valid_plus_invalid_le_eligible",
    "party_sum_eq_valid",
    "turnout_consistent",
]


class Party(BaseModel):
    party_id: str
    canonical_name: str
    full_name: Optional[str] = None
    is_aggregate: bool = Field(default=False, description="True for residual categories like 'others'.")
    verified: bool = False
    source: Optional[str] = None


class ElectionPartyEntry(BaseModel):
    party_id: str
    display_name: str
    aliases: List[str] = Field(default_factory=list)
    states: Optional[List[str]] = Field(default=None, description="States where the party stood; None = all.")
    note: Optional[str] = None


class Election(BaseModel):
    election_id: str
    name: str
    election_type: str
    election_date: date
    vote_types: List[VoteType]
    parties: List[ElectionPartyEntry]
    verified: bool = False
    sources: List[str] = Field(default_factory=list)

    def party_entry(self, party_id: str) -> Optional[ElectionPartyEntry]:
        return next((p for p in self.parties if p.party_id == party_id), None)

    def party_allowed(self, party_id: str, state: Optional[str]) -> bool:
        entry = self.party_entry(party_id)
        if entry is None:
            return False
        return entry.states is None or (state is not None and state in entry.states)


class AdministrativeUnit(BaseModel):
    unit_id: str
    name: str
    level: str
    parent_unit_id: Optional[str] = None
    official_key: Optional[str] = None
    valid_from: Optional[date] = None
    valid_to: Optional[date] = None
    aliases: List[str] = Field(default_factory=list)
    verified: bool = False
    source: Optional[str] = None

    def valid_on(self, d: date) -> bool:
        if self.valid_from and d < self.valid_from:
            return False
        return not (self.valid_to and d > self.valid_to)


# ---------------------------------------------------------------------------
# Document / table profiles
# ---------------------------------------------------------------------------

ColumnRole = Literal["unit_label", "unit_index", "row_marker", "metric", "party", "ignore"]
ValueType = Literal["label", "marker", "integer", "decimal"]

PARTY_FIELD_PREFIX = "party:"


def is_valid_field(name: str) -> bool:
    return name in METRIC_FIELDS or (name.startswith(PARTY_FIELD_PREFIX) and len(name) > len(PARTY_FIELD_PREFIX))


class NumberFormat(BaseModel):
    """A named number grammar (profiles reference it from field bindings)."""

    value_type: Literal["integer", "decimal"]
    integer_format: Optional[IntegerFormat] = None
    decimal_format: Optional[DecimalFormat] = None

    @model_validator(mode="after")
    def _check(self) -> NumberFormat:
        if self.value_type == "integer" and self.integer_format is None:
            raise ValueError("integer format requires 'integer_format'")
        if self.value_type == "decimal" and self.decimal_format is None:
            raise ValueError("decimal format requires 'decimal_format'")
        return self


class ColumnSpec(BaseModel):
    key: str
    role: ColumnRole
    value_type: ValueType
    header_aliases: List[str] = Field(min_length=1)
    metric: Optional[MetricName] = None
    party_id: Optional[str] = None
    required: bool = True
    column_number: Optional[int] = Field(default=None, description="Printed column number below the header, if any.")
    integer_format: Optional[IntegerFormat] = None
    decimal_format: Optional[DecimalFormat] = None
    roles: Optional[List[str]] = Field(
        default=None,
        description="Result roles this column feeds by default binding (None = every role). "
        "Used e.g. when one printed line carries Erst- AND Zweitstimmen in different columns.",
    )

    @model_validator(mode="after")
    def _check_role(self) -> ColumnSpec:
        if self.role == "metric" and not self.metric:
            raise ValueError(f"column {self.key}: role=metric requires 'metric'")
        if self.role == "party" and not self.party_id:
            raise ValueError(f"column {self.key}: role=party requires 'party_id'")
        if self.value_type == "integer" and self.integer_format is None:
            raise ValueError(f"column {self.key}: integer column requires 'integer_format'")
        if self.value_type == "decimal" and self.decimal_format is None:
            raise ValueError(f"column {self.key}: decimal column requires 'decimal_format'")
        if self.role == "unit_index" and self.value_type != "integer":
            raise ValueError(f"column {self.key}: unit_index column must be an integer column")
        return self

    @property
    def field(self) -> Optional[str]:
        """Result field fed by this column (metric name or 'party:<id>')."""
        if self.role == "metric":
            return self.metric
        if self.role == "party":
            return PARTY_FIELD_PREFIX + str(self.party_id)
        return None

    def own_format(self) -> Optional[NumberFormat]:
        if self.value_type == "integer":
            return NumberFormat(value_type="integer", integer_format=self.integer_format)
        if self.value_type == "decimal":
            return NumberFormat(value_type="decimal", decimal_format=self.decimal_format)
        return None


class RowRole(BaseModel):
    vote_type: VoteType
    publish: bool = True
    election_id: Optional[str] = Field(default=None, description="Override, e.g. for comparison figures.")


class FieldBinding(BaseModel):
    """Where one result field of a role is printed: column (+ line slot) and number grammar."""

    field: str = Field(description="metric name or 'party:<party_id>'")
    column: str
    slot: Optional[int] = Field(default=None, description="Line slot inside the group (default: the role's slot).")
    format: Optional[str] = Field(default=None, description="Key of TableProfile.formats (default: column format).")

    @field_validator("field")
    @classmethod
    def _field_ok(cls, v: str) -> str:
        if not is_valid_field(v):
            raise ValueError(f"unknown result field '{v}'")
        return v


class RoleSpec(BaseModel):
    """One result row produced per line group (e.g. absolute Erststimmen, percentages of Zweitstimmen)."""

    role_id: str
    vote_type: VoteType
    slot: int = Field(default=0, ge=0, description="Line slot of the group this role is printed on.")
    publish: bool = True
    election_id: Optional[str] = None
    kind: Literal["absolute", "percent", "difference", "change"] = "absolute"
    percent_of: Optional[str] = Field(default=None, description="Percent roles: role_id of the absolute role.")
    difference_of: Optional[List[str]] = Field(
        default=None, description="Difference roles: [minuend role, subtrahend role]; printed value = a - b."
    )
    change_of: Optional[List[str]] = Field(
        default=None,
        description="Change roles: [difference role, base role]; printed value = difference / base x 100 "
        "(relative change in percent, empty where the base is 0).",
    )
    fields: Optional[List[FieldBinding]] = Field(
        default=None, description="Explicit bindings; None = every metric/party column on the role's slot."
    )
    required_checks: Optional[List[CheckName]] = Field(
        default=None, description="Override of the table's required_checks (absolute roles only)."
    )

    @model_validator(mode="after")
    def _check(self) -> RoleSpec:
        needs = {"percent": "percent_of", "difference": "difference_of", "change": "change_of"}
        for kind, attr in needs.items():
            if (self.kind == kind) != (getattr(self, attr) is not None):
                raise ValueError(f"role {self.role_id}: '{attr}' is required for and only allowed with kind={kind}")
        for attr in ("difference_of", "change_of"):
            v = getattr(self, attr)
            if v is not None and len(v) != 2:
                raise ValueError(f"role {self.role_id}: '{attr}' needs exactly two roles")
        if self.kind != "absolute" and self.publish:
            raise ValueError(f"role {self.role_id}: {self.kind} roles are cross-checks and must not be published")
        return self

    @property
    def referenced_roles(self) -> List[str]:
        return [r for r in ([self.percent_of] if self.percent_of else []) + (self.difference_of or []) + (self.change_of or [])]


class PercentDefinition(BaseModel):
    """Printed share = numerator / sum(denominator) x 100 (fields of the absolute role).

    ``party:*`` as numerator means 'the same party as the percent field'.
    """

    numerator: str
    denominator: List[str] = Field(min_length=1)


class StructuralHeading(BaseModel):
    """A printed heading that is not an administrative unit (e.g. 'Stadtkreise', 'Regierungsbezirk X')."""

    with_name: bool = Field(
        default=False,
        description="Printed as keyword + name ('Regierungsbezirk Nordbaden'); otherwise the keyword stands alone.",
    )
    unit_level: Optional[str] = Field(
        default=None,
        description="Administrative level of the unit rows that follow (until the next section header). "
        "Those rows form a grouping without a printed total.",
    )
    context_key: Optional[str] = Field(
        default=None,
        description="The heading names a context value (e.g. 'regierungsbezirk') that must equal the variant context.",
    )


class StructureSpec(BaseModel):
    """How the column structure is recognized (see layout/table_structure.py)."""

    columns: Literal["body_rules", "header_rules", "column_numbers"] = "body_rules"
    header_end: Literal["rule", "column_numbers"] = "rule"
    region_rule_min_width_ratio: float = Field(
        default=0.45, description="Horizontal rules at least this fraction of the page width delimit the table."
    )

    @model_validator(mode="after")
    def _check(self) -> StructureSpec:
        if (self.columns == "column_numbers") != (self.header_end == "column_numbers"):
            raise ValueError("columns=column_numbers and header_end=column_numbers go together")
        return self


class ResolvedBinding(BaseModel):
    field: str
    column: str
    slot: int
    number_format: NumberFormat


def parse_page_ranges(spec: str) -> List[Tuple[int, int, int]]:
    """'10-32,40,45-81/2' -> [(10, 32, 1), (40, 40, 1), (45, 81, 2)] (step 2 = every second page)."""
    ranges: List[Tuple[int, int, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        step = 1
        if "/" in part:
            part, step_s = part.split("/", 1)
            step = int(step_s)
            if step < 1:
                raise ValueError(f"invalid step in page range '{part}/{step_s}'")
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
        else:
            lo = hi = int(part)
        if lo < 1 or hi < lo:
            raise ValueError(f"invalid page range '{part}'")
        if step > 1 and (hi - lo) % step:
            raise ValueError(f"page range '{part}/{step}' does not end on a step")
        ranges.append((lo, hi, step))
    if not ranges:
        raise ValueError("empty page range specification")
    return ranges


def expand_page_ranges(spec: str) -> List[int]:
    return sorted({p for lo, hi, step in parse_page_ranges(spec) for p in range(lo, hi + 1, step)})


class SpreadContinuation(BaseModel):
    """The table continues on the facing page (same printed lines, further columns)."""

    page_offset: int = Field(default=1, description="Continuation page = page + offset.")
    columns: List[ColumnSpec] = Field(min_length=1)
    line_anchor_column: str
    anchor_confirmation_column: Optional[str] = None
    index_column: Optional[str] = Field(
        default=None, description="unit_index column on the continuation page that must equal the unit's index."
    )

    @model_validator(mode="after")
    def _check(self) -> SpreadContinuation:
        keys = [c.key for c in self.columns]
        for k in (self.line_anchor_column, self.anchor_confirmation_column, self.index_column):
            if k and k not in keys:
                raise ValueError(f"continuation: column '{k}' not defined")
        if any(c.role == "unit_label" for c in self.columns):
            raise ValueError("continuation page must not define a unit_label column")
        return self


class TableVariant(BaseModel):
    variant_id: str
    pages: str
    state: str
    context: Dict[str, str] = Field(default_factory=dict, description="Known context, e.g. region name.")
    columns: List[ColumnSpec]
    continuation: Optional[SpreadContinuation] = None

    @field_validator("pages")
    @classmethod
    def _pages_valid(cls, v: str) -> str:
        parse_page_ranges(v)
        return v

    def covers_page(self, page_number: int) -> bool:
        return page_number in expand_page_ranges(self.pages)

    def continuation_page(self, page_number: int) -> Optional[int]:
        return page_number + self.continuation.page_offset if self.continuation else None

    def all_columns(self) -> List[ColumnSpec]:
        """Columns of the (left) page followed by the continuation page's columns."""
        return list(self.columns) + (list(self.continuation.columns) if self.continuation else [])

    def column(self, key: str) -> ColumnSpec:
        for c in self.all_columns():
            if c.key == key:
                return c
        raise KeyError(key)


class TableProfile(BaseModel):
    table_profile_id: str
    election_id: str
    description: str = ""
    row_mode: Literal["single_line", "marker_group"]
    line_anchor_column: str
    anchor_confirmation_column: Optional[str] = Field(
        default=None,
        description="A line anchor is only accepted if this column has ink on the same line "
        "(prevents headers spilling into the anchor column from creating data lines).",
    )
    marker_column: Optional[str] = None
    marker_sequence: List[str] = Field(
        default_factory=list, description="Expected marker per line slot (with marker_column), or slot names."
    )
    row_roles: Dict[str, RowRole] = Field(default_factory=dict)
    single_line_vote_type: Optional[VoteType] = None
    roles: List[RoleSpec] = Field(default_factory=list, description="Explicit result roles (see RoleSpec).")
    marker_columns: Dict[str, List[Optional[str]]] = Field(
        default_factory=dict,
        description="Extra marker/designator columns: column -> expectation per slot "
        "('<value>' must be read, '*' ink required, '?' ink allowed, null must be empty).",
    )
    group_by: Literal["gap", "label_start"] = Field(
        default="gap",
        description="gap: consecutive lines closer than group_gap_ratio pitches form a group; "
        "label_start: a group starts at every line with ink in the label or index column.",
    )
    group_gap_ratio: float = 1.6
    formats: Dict[str, NumberFormat] = Field(default_factory=dict)
    label_pattern: str = Field(description="Regex for unit rows; needs (?P<name>...) and optionally (?P<index>...).")
    section_kinds: Dict[str, str] = Field(
        description="Section keyword as printed (e.g. 'Landkreis') -> administrative level of the section."
    )
    keyword_precedence: bool = Field(
        default=False,
        description="Check section/total/heading keywords BEFORE label_pattern (needed when unit labels carry no "
        "index, e.g. line-printer tables where 'NOCH LKR. AALEN' would otherwise match the unit pattern).",
    )
    section_index_allowed: bool = Field(
        default=False, description="Section headers may start with a printed index/key (e.g. '131 LKR. AALEN')."
    )
    continued_prefixes: List[str] = Field(default_factory=list, description="e.g. ['noch'] for continued sections")
    section_header_has_totals: bool = False
    section_footer_has_totals: bool = Field(
        default=False,
        description="A section label printed on a data line CLOSES the current section as its total row "
        "(kind and name must match the open section).",
    )
    total_keywords: List[str] = Field(default_factory=list, description="Labels of section total rows, e.g. ['Zusammen']")
    structural_headings: Dict[str, StructuralHeading] = Field(default_factory=dict)
    keyword_min_similarity: float = Field(
        default=0.75,
        description="Similarity threshold for the closed keyword vocabulary above (structure detection only; "
        "names are always resolved exactly).",
    )
    wrapped_label_attach: Literal["group", "next", "previous"] = "group"
    wrapped_label_max_pitch: float = Field(
        default=0.5, description="'group' mode: a loose label line may lie this many pitches outside the group."
    )
    slot_label_markers: Dict[int, str] = Field(
        default_factory=dict,
        description="Slot -> marker printed at the END of the label column on that line (e.g. {1: 'vH'}); it is "
        "verified structurally and is no part of the unit name.",
    )
    implicit_sections: bool = Field(
        default=False,
        description="Sections are not opened by headers: rows belong to the current block until a total row "
        "closes it; a total directly after a total is the grand total over all rows of the run.",
    )
    section_total_level: Optional[str] = Field(
        default=None, description="implicit_sections: administrative level of a block total (e.g. REGIERUNGSBEZIRK)."
    )
    grand_total_level: str = Field(default="LAND", description="Administrative level of a grand total row.")
    total_when_unindexed: bool = Field(
        default=False,
        description="A labelled row whose label carries no index (unit rows always do) is a total row.",
    )
    required_checks: List[CheckName] = Field(default_factory=list)
    consistent_fields_across_markers: Dict[MetricName, List[str]] = Field(
        default_factory=dict,
        description="Metric -> markers whose values must be identical (e.g. eligible voters of rows 1 and 2).",
    )
    percent_definitions: Dict[str, PercentDefinition] = Field(default_factory=dict)
    blank_party_cells_zero_with_sum_proof: bool = Field(
        default=False,
        description="An EMPTY party cell counts as 0 only if the party sum then equals the valid votes exactly "
        "(docs/decisions.md, D6); otherwise the row goes to review.",
    )
    name_matching: Literal["exact", "ascii_printout"] = Field(
        default="exact",
        description="ascii_printout: names printed without umlauts/punctuation (line printer); gazetteer names are "
        "transliterated the same way (ae/oe/ue/ss, punctuation removed) before the still exact comparison.",
    )
    unit_level_default: str = Field(description="Administrative level of ordinary data rows.")
    structure: StructureSpec = Field(default_factory=StructureSpec)
    field_notes: Dict[str, str] = Field(
        default_factory=dict,
        description="Notes attached to published fields, e.g. a printed turnout with a different definition.",
    )
    variants: List[TableVariant]

    # ------------------------------------------------------------------ derived
    @property
    def slot_count(self) -> int:
        return len(self.marker_sequence) if self.row_mode == "marker_group" else 1

    def effective_marker_columns(self) -> Dict[str, List[Optional[str]]]:
        out: Dict[str, List[Optional[str]]] = {}
        if self.row_mode == "marker_group" and self.marker_column:
            out[self.marker_column] = list(self.marker_sequence)
        out.update(self.marker_columns)
        return out

    def effective_roles(self) -> List[RoleSpec]:
        if self.roles:
            return list(self.roles)
        if self.row_mode == "single_line":
            return [RoleSpec(role_id="row", vote_type=self.single_line_vote_type, slot=0)]
        return [
            RoleSpec(
                role_id=m,
                vote_type=self.row_roles[m].vote_type,
                slot=i,
                publish=self.row_roles[m].publish,
                election_id=self.row_roles[m].election_id,
            )
            for i, m in enumerate(self.marker_sequence)
        ]

    def marker_vocabulary(self, column_key: str) -> List[str]:
        expect = self.effective_marker_columns().get(column_key, [])
        return sorted({e for e in expect if e is not None and e not in ("*", "?")})

    def role(self, role_id: str) -> RoleSpec:
        return next(r for r in self.effective_roles() if r.role_id == role_id)

    def bindings(self, variant: TableVariant) -> Dict[str, List[ResolvedBinding]]:
        """role_id -> resolved field bindings (column, slot, number format)."""
        cols = {c.key: c for c in variant.all_columns()}
        out: Dict[str, List[ResolvedBinding]] = {}
        for role in self.effective_roles():
            items: List[ResolvedBinding] = []
            if role.fields is None:
                for c in variant.all_columns():
                    if c.field is None or (c.roles is not None and role.role_id not in c.roles):
                        continue
                    items.append(ResolvedBinding(field=c.field, column=c.key, slot=role.slot, number_format=c.own_format()))
            else:
                for b in role.fields:
                    bound = cols.get(b.column)
                    if bound is None:
                        raise ValueError(
                            f"{self.table_profile_id}/{variant.variant_id}: role {role.role_id} binds unknown column '{b.column}'"
                        )
                    fmt: Optional[NumberFormat]
                    if b.format is not None:
                        if b.format not in self.formats:
                            raise ValueError(f"{self.table_profile_id}: unknown format '{b.format}'")
                        fmt = self.formats[b.format]
                    else:
                        fmt = bound.own_format()
                    if fmt is None:
                        raise ValueError(f"{self.table_profile_id}: column '{bound.key}' has no number format")
                    items.append(
                        ResolvedBinding(
                            field=b.field, column=b.column, slot=role.slot if b.slot is None else b.slot, number_format=fmt
                        )
                    )
            seen = [i.field for i in items]
            dup = sorted({f for f in seen if seen.count(f) > 1})
            if dup:
                raise ValueError(f"{self.table_profile_id}/{variant.variant_id}: role {role.role_id} binds {dup} twice")
            out[role.role_id] = items
        return out

    def cell_formats(self, variant: TableVariant) -> Dict[Tuple[str, int], NumberFormat]:
        """(column key, slot) -> number format used to resolve that cell (conflicts are profile errors)."""
        out: Dict[Tuple[str, int], NumberFormat] = {}
        for items in self.bindings(variant).values():
            for b in items:
                key = (b.column, b.slot)
                if key in out and out[key] != b.number_format:
                    raise ValueError(
                        f"{self.table_profile_id}/{variant.variant_id}: column '{b.column}' slot {b.slot} "
                        "bound with two different number formats"
                    )
                out[key] = b.number_format
        return out

    @model_validator(mode="after")
    def _check(self) -> TableProfile:
        if "name" not in re.compile(self.label_pattern).groupindex:
            raise ValueError(f"{self.table_profile_id}: label_pattern needs a (?P<name>...) group")
        if not self.section_kinds and not self.implicit_sections:
            raise ValueError(f"{self.table_profile_id}: section_kinds must not be empty (or use implicit_sections)")
        if self.section_header_has_totals and self.section_footer_has_totals:
            raise ValueError(f"{self.table_profile_id}: section totals are either in the header or the footer line")
        if self.row_mode == "marker_group":
            if not self.marker_sequence:
                raise ValueError(f"{self.table_profile_id}: marker_group mode needs marker_sequence (one entry per line slot)")
            if not self.marker_column and not self.slot_label_markers and self.group_by == "gap" and not self.roles:
                raise ValueError(f"{self.table_profile_id}: marker_group without marker column needs explicit roles")
            if not self.roles:
                if len(set(self.marker_sequence)) != len(self.marker_sequence):
                    raise ValueError(f"{self.table_profile_id}: repeated markers need explicit 'roles'")
                missing = set(self.marker_sequence) - set(self.row_roles)
                if missing:
                    raise ValueError(f"{self.table_profile_id}: row_roles missing markers {sorted(missing)}")
        else:
            if self.single_line_vote_type is None and not self.roles:
                raise ValueError(f"{self.table_profile_id}: single_line mode needs single_line_vote_type or roles")
        roles = self.effective_roles()
        ids = [r.role_id for r in roles]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{self.table_profile_id}: duplicate role ids")
        by_id = {x.role_id: x for x in roles}
        for r in roles:
            if r.slot >= self.slot_count:
                raise ValueError(f"{self.table_profile_id}: role {r.role_id} slot {r.slot} >= slot count {self.slot_count}")
            for ref in r.referenced_roles:
                if ref not in by_id:
                    raise ValueError(f"{self.table_profile_id}: role {r.role_id} references unknown role '{ref}'")
            if r.kind == "percent" and by_id[r.percent_of or ""].kind != "absolute":
                raise ValueError(f"{self.table_profile_id}: role {r.role_id} percent_of must name an absolute role")
            if r.kind == "difference" and any(by_id[x].kind != "absolute" for x in r.difference_of or []):
                raise ValueError(f"{self.table_profile_id}: role {r.role_id} difference_of must name absolute roles")
            change_of = r.change_of or ["", ""]
            if r.kind == "change" and (by_id[change_of[0]].kind != "difference" or by_id[change_of[1]].kind != "absolute"):
                raise ValueError(f"{self.table_profile_id}: role {r.role_id} change_of must be [difference role, absolute role]")
        if any(r.kind == "percent" for r in roles) and not self.percent_definitions:
            raise ValueError(f"{self.table_profile_id}: percent roles need percent_definitions")
        for key in self.percent_definitions:
            if key != PARTY_FIELD_PREFIX + "*" and not is_valid_field(key):
                raise ValueError(f"{self.table_profile_id}: percent definition for unknown field '{key}'")
        for slot in self.slot_label_markers:
            if not 0 < slot < self.slot_count:
                raise ValueError(f"{self.table_profile_id}: slot_label_markers slot {slot} out of range")
        for col, expect in self.effective_marker_columns().items():
            if len(expect) != self.slot_count:
                raise ValueError(f"{self.table_profile_id}: marker column '{col}' needs {self.slot_count} slot entries")
        for v in self.variants:
            cols = v.all_columns()
            keys = [c.key for c in cols]
            if len(keys) != len(set(keys)):
                raise ValueError(f"{self.table_profile_id}/{v.variant_id}: duplicate column keys")
            if sum(1 for c in v.columns if c.role == "unit_label") != 1:
                raise ValueError(f"{self.table_profile_id}/{v.variant_id}: exactly one unit_label column required")
            if sum(1 for c in v.columns if c.role == "unit_index") > 1:
                raise ValueError(f"{self.table_profile_id}/{v.variant_id}: at most one unit_index column")
            if self.line_anchor_column not in [c.key for c in v.columns]:
                raise ValueError(f"{self.table_profile_id}/{v.variant_id}: anchor column missing")
            if self.anchor_confirmation_column and self.anchor_confirmation_column not in keys:
                raise ValueError(f"{self.table_profile_id}/{v.variant_id}: anchor confirmation column missing")
            for col in self.effective_marker_columns():
                if col not in [c.key for c in v.columns]:
                    raise ValueError(f"{self.table_profile_id}/{v.variant_id}: marker column '{col}' missing")
            if self.structure.columns == "column_numbers":
                numbered = [c.column_number for c in v.columns if c.column_number is not None]
                if not numbered or numbered != sorted(numbered) or len(set(numbered)) != len(numbered):
                    raise ValueError(
                        f"{self.table_profile_id}/{v.variant_id}: column_numbers strategy needs strictly increasing column numbers"
                    )
            self.bindings(v)  # raises on unknown columns / duplicate fields
            self.cell_formats(v)  # raises on conflicting formats
        return self

    def variant_for_page(self, page_number: int) -> List[TableVariant]:
        return [v for v in self.variants if v.covers_page(page_number)]


class DocumentProfile(BaseModel):
    document_profile_id: str
    description: str = ""
    sha256: List[str] = Field(min_length=1)
    tables: List[TableProfile]
    verified: bool = False
    sources: List[str] = Field(default_factory=list)

    def table_profiles_for_page(self, page_number: int) -> List[Tuple[TableProfile, TableVariant]]:
        out = []
        for t in self.tables:
            for v in t.variant_for_page(page_number):
                out.append((t, v))
        return out

    def continuation_pages(self) -> Dict[int, int]:
        """continuation page -> its left (primary) page."""
        out: Dict[int, int] = {}
        for t in self.tables:
            for v in t.variants:
                if v.continuation:
                    for p in expand_page_ranges(v.pages):
                        out[p + v.continuation.page_offset] = p
        return out
