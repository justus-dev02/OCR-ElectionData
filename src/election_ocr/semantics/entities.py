"""
Administrative unit resolution (election-date and context aware).

Decision procedure for one label (several independent OCR readings):

1. Candidate pool: units of the expected level, valid on the election date,
   inside the known context (state, region, section/county) - via the
   gazetteer hierarchy, never via string heuristics.
2. For every reading: EXACT match of the normalized name against unit names
   and registered aliases (``exact_key``; umlauts are NOT folded).
3. Accept a unit only if at least ``min_sources`` independence groups match it
   exactly, no group matches a different unit exactly, and the pool contains
   exactly one unit with that name.
4. Otherwise report the reason and list fuzzy candidates (score + margin to
   the runner-up) for the reviewer. Unknown names never create new units.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from rapidfuzz import fuzz

from election_ocr.issues import Issue, IssueCode, IssueFactory, IssueScope
from election_ocr.parsing.text import exact_key, fuzzy_key, printout_key
from election_ocr.pipeline.models import LabelReading
from election_ocr.reference.models import AdministrativeUnit
from election_ocr.reference.repository import ReferenceRepository

KeyFunction = Callable[[str], str]
KEY_FUNCTIONS: Dict[str, KeyFunction] = {"exact": exact_key, "ascii_printout": printout_key}


@dataclass
class FuzzyCandidate:
    unit_id: str
    name: str
    score: float


@dataclass
class UnitResolution:
    unit_id: Optional[str]
    unit_name: Optional[str]
    supporting_groups: List[str] = field(default_factory=list)
    readings: Dict[str, str] = field(default_factory=dict)
    candidates: List[FuzzyCandidate] = field(default_factory=list)
    margin: Optional[float] = None
    issues: List[Issue] = field(default_factory=list)
    method: str = "none"


class UnitResolver:
    def __init__(
        self,
        reference: ReferenceRepository,
        issues: IssueFactory,
        min_sources: int,
        fuzzy_min_score: float,
        max_candidates: int,
        require_verified: bool,
    ):
        self.ref = reference
        self.issues = issues
        self.min_sources = min_sources
        self.fuzzy_min_score = fuzzy_min_score
        self.max_candidates = max_candidates
        self.require_verified = require_verified
        self._pool_cache: Dict[tuple, List[AdministrativeUnit]] = {}

    # ----------------------------------------------------------------- context
    def _in_state(self, unit: AdministrativeUnit, state: Optional[str]) -> bool:
        if not state:
            return True
        chain = [unit] + self.ref.ancestors(unit.unit_id)
        lands = [u for u in chain if u.level.upper() == "LAND"]
        return any(exact_key(u.name) == exact_key(state) or exact_key(state) in map(exact_key, u.aliases) for u in lands)

    def _context_ok(self, unit: AdministrativeUnit, context: Dict[str, str]) -> bool:
        chain = self.ref.ancestors(unit.unit_id)
        for level, name in context.items():
            matches = [u for u in chain if u.level.lower() == level.lower()]
            if matches and not any(
                exact_key(u.name) == exact_key(name) or exact_key(name) in map(exact_key, u.aliases) for u in matches
            ):
                return False
        return True

    def pool(
        self,
        level: str,
        on: date,
        state: Optional[str],
        context: Dict[str, str],
        parent_unit_id: Optional[str] = None,
    ) -> List[AdministrativeUnit]:
        key = (level.upper(), on, state, tuple(sorted(context.items())), parent_unit_id)
        if key not in self._pool_cache:
            self._pool_cache[key] = self._pool(level, on, state, context, parent_unit_id)
        return self._pool_cache[key]

    def _pool(
        self,
        level: str,
        on: date,
        state: Optional[str],
        context: Dict[str, str],
        parent_unit_id: Optional[str] = None,
    ) -> List[AdministrativeUnit]:
        units = [u for u in self.ref.units.values() if u.level.upper() == level.upper()]
        units = [u for u in units if self._in_state(u, state) and self._context_ok(u, context)]
        if parent_unit_id:
            units = [u for u in units if parent_unit_id in {a.unit_id for a in self.ref.ancestors(u.unit_id)}]
        return [u for u in units if u.valid_on(on)]

    # --------------------------------------------------------------- resolving
    def resolve(
        self,
        readings: Sequence[LabelReading],
        name_pattern: Optional[re.Pattern],
        level: str,
        on: Optional[date],
        state: Optional[str],
        context: Dict[str, str],
        parent_unit_id: Optional[str] = None,
        scope: IssueScope = IssueScope.ROW,
        key_mode: str = "exact",
    ) -> UnitResolution:
        """``key_mode`` (profile ``name_matching``) selects the normalization of BOTH sides of
        the exact comparison; 'ascii_printout' transliterates umlauts and drops punctuation."""
        key = KEY_FUNCTIONS[key_mode]
        names = _names(readings, name_pattern)
        res = UnitResolution(unit_id=None, unit_name=None, readings=names)
        if not self.ref.units:
            res.issues.append(
                self.issues.make(
                    IssueCode.REFERENCE_DATA_MISSING,
                    scope,
                    "gazetteer is empty - supply reference/administrative_units/*.csv",
                    field="unit_id",
                )
            )
            return res
        if not names:
            res.issues.append(self.issues.make(IssueCode.ENTITY_NOT_FOUND, scope, "no usable label reading", field="unit_id"))
            return res
        if on is None:
            res.issues.append(self.issues.make(IssueCode.REFERENCE_DATA_MISSING, scope, "election date unknown", field="unit_id"))
            return res

        pool = self.pool(level, on, state, context, parent_unit_id)
        matched = _exact_matches(names, pool, key)
        res.candidates = self._fuzzy(names.values(), pool)
        if len(res.candidates) >= 2:
            res.margin = round(res.candidates[0].score - res.candidates[1].score, 4)
        if not matched:
            res.issues.append(self._no_match_issue(res, names, level, on, key, scope))
            return res
        problem = self._match_problem(matched, scope)
        if problem is not None:
            res.issues.append(problem)
            return res
        unit = next(iter(matched.values()))[0]
        res.supporting_groups = sorted(matched)
        if len(res.supporting_groups) < self.min_sources:
            res.issues.append(
                self.issues.make(
                    IssueCode.OCR_SINGLE_SOURCE,
                    scope,
                    f"label matched exactly by {len(res.supporting_groups)} engine group(s) only",
                    field="unit_id",
                    unit_id=unit.unit_id,
                )
            )
            return res
        if self.require_verified and not unit.verified:
            res.issues.append(
                self.issues.make(IssueCode.REFERENCE_UNVERIFIED, scope, f"unit {unit.unit_id} is not verified", field="unit_id")
            )
        res.unit_id, res.unit_name, res.method = unit.unit_id, unit.name, "exact"
        return res

    def _no_match_issue(
        self, res: UnitResolution, names: Dict[str, str], level: str, on: date, key: KeyFunction, scope: IssueScope
    ) -> Issue:
        """Why nothing matched: only fuzzy candidates, a unit of another context or date, or nothing at all."""
        code = IssueCode.ENTITY_FUZZY_ONLY if res.candidates else IssueCode.ENTITY_NOT_FOUND
        if not res.candidates and self._exists_outside(names.values(), level, on, key):
            code = IssueCode.ENTITY_OUT_OF_CONTEXT
        elif not res.candidates and self._exists_other_date(names.values(), level, on, key):
            code = IssueCode.ENTITY_NOT_VALID_FOR_DATE
        return self.issues.make(
            code,
            scope,
            "no exact reference match for the label",
            field="unit_id",
            readings=names,
            candidates=[c.__dict__ for c in res.candidates],
        )

    def _match_problem(self, matched: Dict[str, List[AdministrativeUnit]], scope: IssueScope) -> Optional[Issue]:
        """Exact matches must be unique per reading and the same unit for every engine group."""
        if any(len(units) > 1 for units in matched.values()):
            return self.issues.make(
                IssueCode.ENTITY_MULTIPLE_MATCHES,
                scope,
                "label matches several reference units in the same context",
                field="unit_id",
                units=sorted({u.unit_id for units in matched.values() for u in units}),
            )
        if len({units[0].unit_id for units in matched.values()}) > 1:
            return self.issues.make(
                IssueCode.LABEL_DISAGREEMENT,
                scope,
                "OCR engines resolve the label to different units",
                field="unit_id",
                matches={g: units[0].unit_id for g, units in matched.items()},
            )
        return None

    def _fuzzy(self, texts, pool: List[AdministrativeUnit]) -> List[FuzzyCandidate]:
        scores: Dict[str, FuzzyCandidate] = {}
        for text in texts:
            fk = fuzzy_key(text)
            for u in pool:
                for n in [u.name] + list(u.aliases):
                    s = fuzz.ratio(fk, fuzzy_key(n)) / 100.0
                    if s >= self.fuzzy_min_score and (u.unit_id not in scores or s > scores[u.unit_id].score):
                        scores[u.unit_id] = FuzzyCandidate(unit_id=u.unit_id, name=u.name, score=round(s, 4))
        return sorted(scores.values(), key=lambda c: -c.score)[: self.max_candidates]

    def _exists_outside(self, texts, level: str, on: date, key=exact_key) -> bool:
        keys = {key(t) for t in texts}
        return any(
            u.level.upper() == level.upper() and u.valid_on(on) and (key(u.name) in keys or keys & set(map(key, u.aliases)))
            for u in self.ref.units.values()
        )

    def _exists_other_date(self, texts, level: str, on: date, key=exact_key) -> bool:
        keys = {key(t) for t in texts}
        return any(
            u.level.upper() == level.upper() and not u.valid_on(on) and (key(u.name) in keys or keys & set(map(key, u.aliases)))
            for u in self.ref.units.values()
        )


def _names(readings: Sequence[LabelReading], name_pattern: Optional[re.Pattern]) -> Dict[str, str]:
    """Name part of every reading per independence group (readings not matching the pattern abstain)."""
    names: Dict[str, str] = {}
    for r in readings:
        text = r.text
        if name_pattern is not None:
            m = name_pattern.match(text)
            if not m:
                continue
            text = m.group("name")
        if text and text.strip():
            names[r.independence_group] = text.strip()
    return names


def _exact_matches(
    names: Dict[str, str], pool: Iterable[AdministrativeUnit], key: KeyFunction
) -> Dict[str, List[AdministrativeUnit]]:
    """Units whose name or alias equals the reading after normalisation, per independence group."""
    index: Dict[str, List[AdministrativeUnit]] = {}
    for u in pool:
        for n in [u.name, *u.aliases]:
            index.setdefault(key(n), []).append(u)
    matched: Dict[str, List[AdministrativeUnit]] = {}
    for group, text in names.items():
        uniq = {u.unit_id: u for u in index.get(key(text), [])}
        if uniq:
            matched[group] = list(uniq.values())
    return matched


__all__ = ["KEY_FUNCTIONS", "FuzzyCandidate", "UnitResolution", "UnitResolver"]
