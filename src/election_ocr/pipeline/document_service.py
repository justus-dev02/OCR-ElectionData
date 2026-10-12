"""
Document-level validation, entity resolution and the publication gate.

``validate_run`` (re)computes every issue of every candidate row of a run from
the stored evidence plus human review decisions, and moves rows to VALIDATED
or NEEDS_REVIEW through the state machine. ``publish_run`` copies VALIDATED
rows into the published tables - and re-checks the gate conditions itself, so
a row can never reach the published dataset through another path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC
from decimal import Decimal
from typing import Dict, List, Mapping, Optional, Set, Tuple

from sqlalchemy import select

from election_ocr.config import PipelineConfig
from election_ocr.issues import Issue, IssueCode, IssueFactory, IssueScope, Severity
from election_ocr.ocr.resolver import CellStatus
from election_ocr.persistence.db import (
    Acknowledgement,
    Cell,
    FieldOverride,
    Page,
    PublishedPartyResult,
    PublishedResult,
    ResultRow,
    TableRow,
)
from election_ocr.persistence.repository import Repository
from election_ocr.pipeline.models import LabelReading
from election_ocr.pipeline.state_machine import ResultStatus
from election_ocr.reference.models import DocumentProfile, TableProfile, TableVariant, expand_page_ranges
from election_ocr.reference.repository import ReferenceRepository
from election_ocr.semantics.entities import UnitResolver
from election_ocr.validation.rules import (
    BLANK,
    DIFFERS,
    HOLDS,
    DefinitionProbe,
    FieldValue,
    RowValues,
    apply_blank_zero,
    cell_issues,
    change_field_results,
    cross_role_checks,
    derived_checks,
    difference_field_results,
    duplicate_unit_checks,
    follows_sum100,
    grand_total_checks,
    hierarchy_checks,
    percent_checks,
    percent_field_results,
    row_checks,
    turnout_match,
)

TERMINAL = {ResultStatus.PUBLISHED.value, ResultStatus.SUPERSEDED.value, ResultStatus.REJECTED.value}


# run-level probe: party shares rounded so that each row adds up to 100 (see rules.share_row_rounding)
ROUNDING_PROBE = "rounding:sum100"


@dataclass
class ValidationSummary:
    rows: int
    validated: int
    needs_review: int
    non_publishable_roles: int
    issue_counts: Dict[str, int]


@dataclass
class _RunState:
    """Everything one validation pass works on (loaded once per run)."""

    profile: DocumentProfile
    rows: List[ResultRow]
    cells: Dict[str, Cell]
    tables: Dict[str, TableRow]
    pages: Dict[str, Page]
    overrides: Dict[str, Dict[str, FieldOverride]]
    acks: Dict[str, Dict[str, Acknowledgement]]
    values: Dict[str, RowValues] = field(default_factory=dict)
    specs: Dict[str, Tuple[TableProfile, TableVariant]] = field(default_factory=dict)
    issues: Dict[str, List[Issue]] = field(default_factory=dict)
    groups: Dict[str, List[RowValues]] = field(default_factory=dict)
    probes: Dict[Tuple[str, str], DefinitionProbe] = field(default_factory=dict)
    differs: Set[Tuple[str, str]] = field(default_factory=set)


class DocumentService:
    def __init__(self, repo: Repository, reference: ReferenceRepository, config: PipelineConfig, issues: IssueFactory):
        self.repo = repo
        self.ref = reference
        self.cfg = config
        self.issues = issues
        self.units = UnitResolver(
            reference,
            issues,
            min_sources=config.ocr.min_independent_sources,
            fuzzy_min_score=config.resolution.fuzzy_candidate_min_score,
            max_candidates=config.resolution.max_candidates,
            require_verified=config.publication.require_verified_reference_data,
        )

    # ----------------------------------------------------------- lookups
    def profile_of_run(self, run_id: str) -> DocumentProfile:
        run = self.repo.get_run(run_id)
        if run is None or run.document_profile_id not in self.ref.profiles:
            raise ValueError(f"run {run_id} has no known document profile")
        return self.ref.profiles[run.document_profile_id]

    @staticmethod
    def table_spec(profile: DocumentProfile, table_profile_id: str, variant_id: str) -> Tuple[TableProfile, TableVariant]:
        tp = next(t for t in profile.tables if t.table_profile_id == table_profile_id)
        return tp, next(v for v in tp.variants if v.variant_id == variant_id)

    # ----------------------------------------------------------- row values
    @staticmethod
    def role_id_of(row: ResultRow) -> str:
        """Role of a stored row (rows of runs before roles existed: derived role = marker / 'row')."""
        return row.role_id or row.marker or "row"

    def build_values(
        self,
        row: ResultRow,
        cells: Dict[str, Cell],
        tp: TableProfile,
        variant: TableVariant,
        overrides: Dict[str, FieldOverride],
        acks: Mapping[str, Acknowledgement],
    ) -> RowValues:
        role_id = self.role_id_of(row)
        cols = {c.key: c for c in variant.all_columns()}
        bound = {b.field: cols[b.column] for b in tp.bindings(variant).get(role_id, [])}
        role_kind = row.role_kind or "absolute"

        def required(key: str) -> bool:
            spec = bound.get(key[len("metric:") :] if key.startswith("metric:") else key)
            return spec.required if spec is not None else True

        def fv(name: str, cell_id: Optional[str], key: str) -> FieldValue:
            ov = overrides.get(key)
            if ov is not None:
                return FieldValue(
                    name=name,
                    cell_id=cell_id,
                    status=CellStatus.HUMAN.value,
                    value=Decimal(ov.value),
                    decimals=ov.decimals,
                    source="human",
                    required=required(key),
                )
            if cell_id is None or cell_id not in cells:
                # derived rows (shares, differences, changes): an empty cell is judged by their checks;
                # absolute rows: party cells only, with the party-sum proof (profile flag)
                blank_allowed = role_kind != "absolute" or (key.startswith("party:") and tp.blank_party_cells_zero_with_sum_proof)
                return FieldValue(name=name, cell_id=None, status=BLANK if blank_allowed else "ABSENT", required=required(key))
            c = cells[cell_id]
            return FieldValue(
                name=name,
                cell_id=cell_id,
                status=c.status,
                value=Decimal(c.value) if c.value is not None and c.status == CellStatus.AGREED.value else None,
                decimals=c.decimals,
                required=required(key),
                assignment_status=c.assignment_status,
                crosses_boundary=c.crosses_boundary,
                shape_conflict=c.shape_conflict,
            )

        return RowValues(
            result_key=row.result_id,
            row_kind=row.row_kind,
            marker=row.marker,
            vote_type=row.vote_type,
            group_key=f"{row.table_id}_g{row.group_index}",
            section_key=row.section_key,
            publishable_role=row.publishable_role,
            role_id=role_id,
            role_kind=role_kind,
            percent_of_role=row.percent_of_role,
            section_grouping=(row.context or {}).get("_section_grouping") == "true",
            metrics={m: fv(m, cid, f"metric:{m}") for m, cid in row.field_cells.items()},
            parties={p: fv(f"party:{p}", cid, f"party:{p}") for p, cid in row.party_cells.items()},
            acknowledged={k: getattr(v, "reason", "") for k, v in acks.items()},
        )

    # ----------------------------------------------------------- validation
    def validate_run(self, run_id: str, actor: str = "pipeline") -> ValidationSummary:
        """Recompute every issue of every non-terminal row of the run and move rows through the state machine."""
        st = self._load_state(run_id)
        self._inherited_issues(st)
        self._cell_issues(st)
        st.probes = self._definition_probes(st.rows, st.values, st.specs, st.groups)
        st.differs = {k for k, p in st.probes.items() if p.decision == DIFFERS and k[1] != ROUNDING_PROBE}
        self._row_arithmetic(st)
        self._group_checks(st)
        unit_of = self._entity_checks(st)
        self._run_level_checks(st, unit_of, run_id)
        table_cases = self._table_cases(st, run_id)
        return self._persist_validation(st, run_id, actor, table_cases)

    def _load_state(self, run_id: str) -> _RunState:
        profile = self.profile_of_run(run_id)
        st = _RunState(
            profile=profile,
            rows=[r for r in self.repo.rows(run_id) if r.status not in TERMINAL],
            cells=self.repo.cells(run_id),
            tables=self.repo.tables(run_id),
            pages=self.repo.pages(run_id),
            overrides=self.repo.overrides(run_id),
            acks=self.repo.acknowledgements(run_id),
        )
        for r in st.rows:
            t = st.tables[r.table_id]
            tp, variant = st.specs[r.result_id] = self.table_spec(profile, t.table_profile_id, t.variant_id)
            st.values[r.result_id] = self.build_values(
                r, st.cells, tp, variant, st.overrides.get(r.result_id, {}), st.acks.get(r.result_id, {})
            )
            st.issues[r.result_id] = []
        # rows grouped by printed unit (row group) for the group checks
        for r in st.rows:
            v = st.values[r.result_id]
            st.groups.setdefault(v.group_key, []).append(v)
        return st

    def _inherited_issues(self, st: _RunState) -> None:
        """1) structural issues from interpretation + blocking page/table issues
        (a page spread inherits the issues of BOTH pages)."""
        for r in st.rows:
            found = st.issues[r.result_id]
            found.extend(Issue(**i) for i in (r.row_issues or []))
            for tid in [r.table_id, *(r.related_table_ids or [])]:
                t = st.tables.get(tid)
                if t is None:
                    found.append(self.issues.make(IssueCode.PAGE_STITCH_MISMATCH, IssueScope.ROW, f"related table {tid} missing"))
                    continue
                page = st.pages[t.page_id]
                for inherited in list(page.issues or []) + list(t.issues or []):
                    iss = Issue(**inherited)
                    if iss.severity == Severity.BLOCKING:
                        found.append(iss)

    def _cell_issues(self, st: _RunState) -> None:
        """2) cell issues and empty cells (blank = 0 only with the party-sum proof)."""
        for r in st.rows:
            v = st.values[r.result_id]
            for fvx in list(v.metrics.values()) + list(v.parties.values()):
                st.issues[r.result_id].extend(cell_issues(fvx, self.issues))
            st.issues[r.result_id].extend(apply_blank_zero(v, self.issues))

    def _row_arithmetic(self, st: _RunState) -> None:
        """2b) row arithmetic; a check whose printed definition differs (probe) is replaced by a note."""
        for r in st.rows:
            v = st.values[r.result_id]
            tp, _ = st.specs[r.result_id]
            if v.role_kind != "absolute":
                continue
            checks = self._required_checks(tp, v)
            probe_key = (tp.table_profile_id, "turnout_consistent")
            if "turnout_consistent" in checks and probe_key in st.differs:
                checks = [c for c in checks if c != "turnout_consistent"]
                st.issues[r.result_id].append(self._definition_note(st.probes[probe_key], "turnout_percent"))
            st.issues[r.result_id].extend(row_checks(v, checks, self.issues))

    def _group_checks(self, st: _RunState) -> None:
        """3) identical fields across roles, printed differences/changes and percentages."""
        for members in st.groups.values():
            tp = st.specs[members[0].result_key][0]
            consistent = {str(k): v for k, v in tp.consistent_fields_across_markers.items()}
            for rid, extra in cross_role_checks(members, consistent, self.issues).items():
                st.issues[rid].extend(extra)
            by_role = {m.role_id: m for m in members}
            for m in members:
                if m.role_kind in ("difference", "change"):
                    self._derived_role_checks(st, tp, m, by_role)
                elif m.role_kind == "percent":
                    self._percent_role_checks(st, tp, m, by_role)

    def _derived_role_checks(
        self, st: _RunState, tp: TableProfile, m: RowValues, by_role: Mapping[Optional[str], RowValues]
    ) -> None:
        results = self._derived_results(tp, m, by_role)
        if results is None:
            found = [
                self.issues.make(
                    IssueCode.DERIVED_CHECK_NOT_EXECUTABLE, IssueScope.ROW, f"rows referenced by {m.role_id} missing"
                )
            ]
        else:
            prefix = m.role_kind + ":"
            skip_fields = {k.split(":", 1)[1] for (tpid, k) in st.differs if tpid == tp.table_profile_id and k.startswith(prefix)}
            found = derived_checks(m, results, self.issues, skip_fields=skip_fields)
            for f in sorted(skip_fields):
                found.append(self._definition_note(st.probes[(tp.table_profile_id, f"{m.role_kind}:{f}")], f))
        st.issues[m.result_key].extend(found)
        for ref in tp.role(str(m.role_id)).referenced_roles:
            if ref in by_role and by_role[ref].role_kind == "absolute":
                st.issues[by_role[ref].result_key].extend(found)

    def _percent_role_checks(
        self, st: _RunState, tp: TableProfile, m: RowValues, by_role: Mapping[Optional[str], RowValues]
    ) -> None:
        absolute = by_role.get(m.percent_of_role)
        skip = {f for (tpid, f) in st.differs if tpid == tp.table_profile_id and f.startswith("percent:")}
        skip_fields = {f[len("percent:") :] for f in skip}
        found = percent_checks(
            m, absolute, tp.percent_definitions, self.issues, skip_fields=skip_fields, sum100=self._sum100(st.probes, tp)
        )
        for f in sorted(skip_fields):
            found.append(self._definition_note(st.probes[(tp.table_profile_id, "percent:" + f)], f))
        st.issues[m.result_key].extend(found)
        if absolute is not None:
            st.issues[absolute.result_key].extend(found)

    def _entity_checks(self, st: _RunState) -> Dict[str, Optional[str]]:
        """4) entities (sections first, then units inside their section) and reference data."""
        unit_of: Dict[str, Optional[str]] = {}
        section_unit: Dict[str, Tuple[Optional[str], List[Issue]]] = {}
        for r in st.rows:
            if r.section_key and r.section_key not in section_unit:
                section_unit[r.section_key] = self._resolve_section(r, st.specs[r.result_id][0].name_matching)
        for r in st.rows:
            election = self.ref.elections.get(r.election_id)
            unit_id, unit_name, ent_issues, resolution = self._resolve_row_unit(
                r, section_unit, st.overrides.get(r.result_id, {}), st.specs[r.result_id][0].name_matching
            )
            r.unit_id, r.unit_name, r.unit_resolution = unit_id, unit_name, resolution
            unit_of[r.result_id] = unit_id
            st.issues[r.result_id].extend(ent_issues)
            st.issues[r.result_id].extend(self._reference_issues(r, st.profile, st.specs[r.result_id]))
            if election is None:
                st.issues[r.result_id].append(
                    self.issues.make(IssueCode.REFERENCE_DATA_MISSING, IssueScope.ROW, f"unknown election {r.election_id}")
                )
        return unit_of

    def _run_level_checks(self, st: _RunState, unit_of: Dict[str, Optional[str]], run_id: str) -> None:
        """5) hierarchy sums, duplicate units and grand totals (per row role across the whole run)."""
        all_values = list(st.values.values())
        tp0 = st.profile.tables[0]
        totals_expected = bool(tp0.total_keywords) or tp0.section_header_has_totals or tp0.section_footer_has_totals
        require = self.cfg.publication.require_hierarchical_consistency
        for rid, extra in hierarchy_checks(all_values, totals_expected, require, self.issues).items():
            st.issues[rid].extend(extra)
        for rid, extra in duplicate_unit_checks(all_values, unit_of, self.issues).items():
            st.issues[rid].extend(extra)
        run = self.repo.get_run(run_id)
        run_pages = set((run.pages_requested if run is not None else None) or [])
        for t in st.profile.tables:
            pages_t = {p for v in t.variants for p in expand_page_ranges(v.pages)}
            rows_t = [st.values[r.result_id] for r in st.rows if st.specs[r.result_id][0].table_profile_id == t.table_profile_id]
            for rid, extra in grand_total_checks(rows_t, pages_t <= run_pages, self.issues).items():
                st.issues[rid].extend(extra)

    def _table_cases(self, st: _RunState, run_id: str) -> int:
        """6) tables/pages that produced no rows at all must still reach a human."""
        rows_per_table: Dict[str, int] = {}
        for r in self.repo.rows(run_id):
            rows_per_table[r.table_id] = rows_per_table.get(r.table_id, 0) + 1
        table_cases = 0
        for t in st.tables.values():
            if rows_per_table.get(t.table_id):
                continue
            inherited = [Issue(**i) for i in list(st.pages[t.page_id].issues or []) + list(t.issues or [])]
            codes = sorted({i.code.value for i in inherited if i.blocking} | {"ROW_UNINTERPRETED"})
            self.repo.upsert_table_case(run_id, t.table_id, codes)
            table_cases += 1
        return table_cases

    def _persist_validation(self, st: _RunState, run_id: str, actor: str, table_cases: int) -> ValidationSummary:
        """7) persist issues, statuses and review cases."""
        counts: Dict[str, int] = {}
        validated = needs_review = non_pub = 0
        for r in st.rows:
            acked = set(st.acks.get(r.result_id, {}).keys())
            row_issues = _dedupe(st.issues[r.result_id])
            self.repo.replace_issues(run_id, r.result_id, row_issues, acked)
            blocking = [i for i in row_issues if i.blocking and i.code.value not in acked]
            for i in blocking:
                counts[i.code.value] = counts.get(i.code.value, 0) + 1
            current = ResultStatus(r.status)
            if blocking:
                if current in (ResultStatus.EXTRACTED, ResultStatus.REVIEWED, ResultStatus.VALIDATED):
                    self.repo.set_status(
                        r, ResultStatus.NEEDS_REVIEW, actor, reason=",".join(sorted({i.code.value for i in blocking}))
                    )
                needs_review += 1
                if r.publishable_role:
                    self.repo.upsert_case(r, sorted({i.code.value for i in blocking}))
            else:
                if current in (ResultStatus.EXTRACTED, ResultStatus.REVIEWED, ResultStatus.NEEDS_REVIEW):
                    reason = (
                        "no blocking issue"
                        if current != ResultStatus.NEEDS_REVIEW
                        else "re-validation: all blocking issues resolved"
                    )
                    self.repo.set_status(r, ResultStatus.VALIDATED, actor, reason=reason)
                if ResultStatus(r.status) == ResultStatus.VALIDATED:
                    validated += 1
                    self.repo.close_case(r, "RESOLVED")
                else:
                    needs_review += 1
            if not r.publishable_role:
                non_pub += 1
        if table_cases:
            counts["TABLE_WITHOUT_ROWS"] = table_cases
        run = self.repo.get_run(run_id)
        if run is not None:
            run.stats = {**(run.stats or {}), "definition_probes": [p.as_dict() for p in st.probes.values()]}
        self.repo.commit()
        return ValidationSummary(len(st.rows), validated, needs_review, non_pub, counts)

    # ----------------------------------------------------------- definitions
    @staticmethod
    def _required_checks(tp: TableProfile, v: RowValues) -> List[str]:
        role = tp.role(v.role_id) if v.role_id in {x.role_id for x in tp.effective_roles()} else None
        return list(role.required_checks if role is not None and role.required_checks is not None else tp.required_checks)

    def _definition_probes(self, rows, values, specs, groups) -> Dict[Tuple[str, str], DefinitionProbe]:
        """Measure per table profile how often printed turnout / shares equal their computation."""
        probes: Dict[Tuple[str, str], DefinitionProbe] = {}

        def probe(tp: TableProfile, key: str) -> DefinitionProbe:
            k = (tp.table_profile_id, key)
            if k not in probes:
                probes[k] = DefinitionProbe(key=f"{tp.table_profile_id}:{key}")
            return probes[k]

        for r in rows:
            v = values[r.result_id]
            tp, _ = specs[r.result_id]
            if (
                v.role_kind == "absolute"
                and "turnout_consistent" in self._required_checks(tp, v)
                and "turnout_percent" in v.metrics
            ):
                probe(tp, "turnout_consistent").add(turnout_match(v))
        # share rounding first: the per-field share probes below depend on it
        for members in groups.values():
            tp = specs[members[0].result_key][0]
            by_role = {m.role_id: m for m in members}
            for m in members:
                absolute = by_role.get(m.percent_of_role) if m.role_kind == "percent" else None
                if absolute is not None:
                    probe(tp, ROUNDING_PROBE).add(follows_sum100(m, absolute, tp.percent_definitions))
        s = self.cfg.definition_probe
        for (_, key), p in probes.items():
            if key == ROUNDING_PROBE:
                p.decide(s.rounding_min_rows, s.holds_min_share, s.differs_max_share)
        for members in groups.values():
            tp = specs[members[0].result_key][0]
            by_role = {m.role_id: m for m in members}
            sum100 = self._sum100(probes, tp)
            for m in members:
                absolute = by_role.get(m.percent_of_role) if m.role_kind == "percent" else None
                if absolute is None:
                    continue
                for name, (ok, _) in percent_field_results(m, absolute, tp.percent_definitions, sum100=sum100).items():
                    probe(tp, "percent:" + name).add(ok)
            for m in members:
                if m.role_kind in ("difference", "change"):
                    for name, (ok, _) in (self._derived_results(tp, m, by_role) or {}).items():
                        probe(tp, f"{m.role_kind}:{name}").add(ok)
        for (_, key), p in probes.items():
            if key != ROUNDING_PROBE:
                p.decide(s.min_rows, s.holds_min_share, s.differs_max_share)
        return probes

    @staticmethod
    def _sum100(probes, tp: TableProfile) -> bool:
        """The document rounds party shares so that each row adds up to 100 (probe holds)."""
        p = probes.get((tp.table_profile_id, ROUNDING_PROBE))
        return p is not None and p.decision == HOLDS

    @staticmethod
    def _derived_results(
        tp: TableProfile, m: RowValues, by_role: Mapping[Optional[str], RowValues]
    ) -> Optional[Dict[str, Tuple[Optional[bool], dict]]]:
        """Per-field results of a difference/change row (None if a referenced row is missing)."""
        role = tp.role(m.role_id or "")
        if m.role_kind == "difference":
            a, b = (by_role.get(x) for x in role.difference_of or [])
            return None if a is None or b is None else difference_field_results(m, a, b)
        d, base = (by_role.get(x) for x in role.change_of or [])
        return None if d is None or base is None else change_field_results(m, d, base)

    def _definition_note(self, probe: DefinitionProbe, field: str) -> Issue:
        return self.issues.make(
            IssueCode.DEFINITION_DIFFERS,
            IssueScope.ROW,
            f"printed {field} follows another definition in this document: only {probe.matches} of {probe.total} rows "
            "match the computation - stored as printed, not checked",
            field=field,
            probe=probe.as_dict(),
        )

    # ----------------------------------------------------------- entities
    def _section_readings(self, r: ResultRow) -> List[LabelReading]:
        names = (r.context or {}).get("_section_name_readings", {}) or {}
        return [LabelReading(engine=g, independence_group=g, text=t) for g, t in names.items()]

    def _resolve_section(self, r: ResultRow, key_mode: str = "exact") -> Tuple[Optional[str], List[Issue]]:
        if (r.context or {}).get("_section_grouping") == "true" or (r.context or {}).get("_section_implicit") == "true":
            return None, []  # structural heading ('Stadtkreise'): not an administrative unit
        if not r.section_level:
            return None, [
                self.issues.make(IssueCode.SECTION_CONTEXT_MISSING, IssueScope.SECTION, "section level unknown", field="section")
            ]
        election = self.ref.elections.get(r.election_id)
        on = election.election_date if election else None
        ctx = {k: v for k, v in (r.context or {}).items() if not k.startswith("_")}
        res = self.units.resolve(
            self._section_readings(r), None, r.section_level, on, r.state_name, ctx, scope=IssueScope.SECTION, key_mode=key_mode
        )
        for i in res.issues:
            i.field = "section"
        return res.unit_id, res.issues

    def _resolve_row_unit(
        self,
        r: ResultRow,
        section_unit: Dict[str, Tuple[Optional[str], List[Issue]]],
        overrides: Dict[str, FieldOverride],
        key_mode: str = "exact",
    ) -> Tuple[Optional[str], Optional[str], List[Issue], dict]:
        sec_id, sec_issues = section_unit.get(r.section_key or "", (None, []))
        out_issues: List[Issue] = list(sec_issues)
        ov = overrides.get("unit")
        if ov is not None:
            unit = self.ref.units.get(ov.value)
            if unit is None:
                return (
                    None,
                    None,
                    out_issues
                    + [
                        self.issues.make(
                            IssueCode.ENTITY_NOT_FOUND, IssueScope.ROW, f"override unit {ov.value} unknown", field="unit_id"
                        )
                    ],
                    {"method": "human", "unit_id": ov.value},
                )
            extra = []
            if self.cfg.publication.require_verified_reference_data and not unit.verified:
                extra.append(
                    self.issues.make(
                        IssueCode.REFERENCE_UNVERIFIED, IssueScope.ROW, f"unit {unit.unit_id} not verified", field="unit_id"
                    )
                )
            return unit.unit_id, unit.name, extra, {"method": "human", "unit_id": unit.unit_id, "reviewer": ov.reviewer}
        if r.row_kind == "GRAND_TOTAL" or (
            r.row_kind == "SECTION_TOTAL" and (r.context or {}).get("_section_implicit") == "true"
        ):
            pass  # no header names these blocks: resolved by their own label below (level set by the profile)
        elif r.row_kind == "SECTION_TOTAL":
            if sec_id is None:
                return None, None, out_issues, {"method": "section"}
            unit = self.ref.units[sec_id]
            return unit.unit_id, unit.name, out_issues, {"method": "section", "unit_id": unit.unit_id}
        election = self.ref.elections.get(r.election_id)
        names = (r.context or {}).get("_unit_name_readings", {}) or {}
        readings = [LabelReading(engine=g, independence_group=g, text=t) for g, t in names.items()]
        ctx = {k: v for k, v in (r.context or {}).items() if not k.startswith("_")}
        res = self.units.resolve(
            readings,
            None,
            r.unit_level,
            election.election_date if election else None,
            r.state_name,
            ctx,
            parent_unit_id=sec_id,
            key_mode=key_mode,
        )
        resolution = {
            "method": res.method,
            "readings": res.readings,
            "supporting_groups": res.supporting_groups,
            "candidates": [c.__dict__ for c in res.candidates],
            "margin": res.margin,
            "section_unit_id": sec_id,
        }
        return res.unit_id, res.unit_name, out_issues + res.issues, resolution

    def _reference_issues(self, r: ResultRow, profile: DocumentProfile, spec: Tuple[TableProfile, TableVariant]) -> List[Issue]:
        if not self.cfg.publication.require_verified_reference_data or not r.publishable_role:
            return []
        out: List[Issue] = []
        unverified = []
        election = self.ref.elections.get(r.election_id)
        if election is not None and not election.verified:
            unverified.append(f"election:{election.election_id}")
        if not profile.verified:
            unverified.append(f"profile:{profile.document_profile_id}")
        for pid in r.party_cells:
            party = self.ref.parties.get(pid)
            if party is None or not party.verified:
                unverified.append(f"party:{pid}")
            if election is not None and not election.party_allowed(pid, r.state_name):
                out.append(
                    self.issues.make(
                        IssueCode.PARTY_NOT_ALLOWED_FOR_ELECTION, IssueScope.ROW, f"party {pid} not allowed", field=f"party:{pid}"
                    )
                )
        if unverified:
            out.append(
                self.issues.make(
                    IssueCode.REFERENCE_UNVERIFIED,
                    IssueScope.ROW,
                    "reference data not verified: " + ", ".join(unverified),
                    items=unverified,
                )
            )
        return out

    # ----------------------------------------------------------- publication
    def publish_run(self, run_id: str, actor: str = "pipeline") -> int:
        profile = self.profile_of_run(run_id)
        run = self.repo.get_run(run_id)
        cells = self.repo.cells(run_id)
        tables = self.repo.tables(run_id)
        pages = self.repo.pages(run_id)
        overrides = self.repo.overrides(run_id)
        acks = self.repo.acknowledgements(run_id)
        published = 0
        for r in self.repo.rows(run_id):
            if r.status != ResultStatus.VALIDATED.value:
                continue
            if not r.publishable_role and not self.cfg.publication.publish_row_roles_marked_non_publishable:
                continue
            # --- gate re-check (defence in depth) ---
            open_blocking = [
                i for i in self.repo.issues_for(r.result_id) if i.severity == Severity.BLOCKING.value and not i.acknowledged
            ]
            if open_blocking or r.unit_id is None:
                raise RuntimeError(f"publication gate violated for {r.result_id}: validation state inconsistent")
            t = tables[r.table_id]
            tp, variant = self.table_spec(profile, t.table_profile_id, t.variant_id)
            v = self.build_values(r, cells, tp, variant, overrides.get(r.result_id, {}), acks.get(r.result_id, {}))
            apply_blank_zero(v, self.issues)
            if any(not f.usable for f in list(v.metrics.values()) + list(v.parties.values()) if f.status != "ABSENT"):
                raise RuntimeError(f"publication gate violated for {r.result_id}: unusable field value")
            self._supersede_previous(r)
            prov = self._provenance(r, v, run, t, pages[t.page_id], tp)
            prov["definition_notes"] = [
                {"field": i.field, "note": i.message}
                for i in self.repo.issues_for(r.result_id)
                if i.code == IssueCode.DEFINITION_DIFFERS.value
            ]
            pub = PublishedResult(
                result_id=r.result_id,
                run_id=run_id,
                document_id=r.document_id,
                election_id=r.election_id,
                unit_id=r.unit_id,
                unit_name=r.unit_name,
                unit_level=r.unit_level,
                vote_type=r.vote_type,
                eligible_voters=_int(v.value("eligible_voters")),
                voters=_int(v.value("voters")),
                valid_votes=_int(v.value("valid_votes")),
                invalid_votes=_int(v.value("invalid_votes")),
                turnout_percent=str(v.value("turnout_percent")) if v.value("turnout_percent") is not None else None,
                provenance=prov,
            )
            self.repo.s.add(pub)
            self.repo.s.flush()
            for pid, f in v.parties.items():
                if f.value is None:
                    continue
                self.repo.s.add(
                    PublishedPartyResult(
                        published_id=pub.published_id, party_id=pid, votes=int(f.value), cell_id=f.cell_id, source=f.source
                    )
                )
            self.repo.set_status(r, ResultStatus.PUBLISHED, actor, reason="publication gate passed")
            published += 1
        self.repo.commit()
        return published

    def _supersede_previous(self, r: ResultRow) -> None:
        q = select(PublishedResult).where(
            PublishedResult.document_id == r.document_id,
            PublishedResult.election_id == r.election_id,
            PublishedResult.unit_id == r.unit_id,
            PublishedResult.vote_type == r.vote_type,
            PublishedResult.superseded_at.is_(None),
        )
        from datetime import datetime

        for old in self.repo.s.execute(q).scalars():
            old.superseded_at = datetime.now(UTC)
            old_row = self.repo.row(old.result_id)
            if old_row is not None and old_row.status == ResultStatus.PUBLISHED.value:
                self.repo.set_status(old_row, ResultStatus.SUPERSEDED, "pipeline", reason=f"superseded by {r.result_id}")

    def _provenance(self, r: ResultRow, v: RowValues, run, table, page, tp: TableProfile) -> dict:
        fields = {}
        for f in list(v.metrics.values()) + list(v.parties.values()):
            fields[f.name] = {
                "value": None if f.value is None else str(f.value),
                "source": f.source,
                "cell_id": f.cell_id,
                "status": f.status,
            }
        return {
            "run_id": run.run_id,
            "document_id": r.document_id,
            "document_profile_id": run.document_profile_id,
            "config_fingerprint": run.config_fingerprint,
            "reference_snapshot": run.reference_snapshot,
            "engine_versions": run.engine_versions,
            "model_info": run.model_info,
            "page_number": r.page_number,
            "page_image": page.image_path,
            "page_image_sha256": page.image_sha256,
            "table_id": table.table_id,
            "result_key": r.result_key,
            "label_cell_ids": r.label_cell_ids,
            "unit_resolution": r.unit_resolution,
            "role_id": v.role_id,
            "related_table_ids": r.related_table_ids or [],
            "unit_index": (r.context or {}).get("_unit_index"),
            "field_notes": {k: n for k, n in tp.field_notes.items() if k in fields},
            "fields": fields,
        }


def _int(v: Optional[Decimal]) -> Optional[int]:
    return int(v) if v is not None else None


def _dedupe(issues: List[Issue]) -> List[Issue]:
    seen = set()
    out = []
    for i in issues:
        key = (i.code, i.field, i.message)
        if key not in seen:
            seen.add(key)
            out.append(i)
    return out
