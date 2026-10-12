"""
External validation against independently digitised official data (GPWED state files).

Read-only comparison (``scripts/fetch_gpwed.sh`` downloads the file):

* **Hessen 1953, municipality level** (``BW_Hessen-1953.xlsx``, Hessian state statistics):
  every machine-agreed cell of a matched municipality row is compared with the
  official value. Rows are matched by NAMES only (never by the numbers being
  validated): the row's county must be known (agreed section name or a reading equal
  to an official county) and an engine reading of the municipality name must equal
  exactly one official municipality of that county (normalised with ``exact_key``).
Not used: the GENESIS municipality file of Baden-Wuerttemberg (``BTW-1949-1969.csv``, today's
boundaries). Its sums over all municipalities are NOT the printed state totals - postal votes
(from 1957) are not assigned to municipalities, and 1953 differs by up to 72 votes per party;
the printed totals were checked on the scans and match the pipeline (e.g. FDP 1953: 502,041).

The official file is an independent source (keyed by the statistical offices), so
an exact match is strong evidence and every mismatch is listed for inspection.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from election_ocr.parsing.text import exact_key
from election_ocr.persistence.db import Cell, ResultRow


# ---------------------------------------------------------------- Hessen 1953 (xlsx)
@dataclass(frozen=True)
class _VoteColumns:
    """0-based column positions of one vote type in sheet "Gemeinden" (header rows 2-4 of the file)."""

    invalid_votes: int
    valid_votes: int
    parties: Dict[str, int]


_HE_ELIGIBLE_COLUMN = 9  # "Wahlberechtigte insgesamt (A1+A2+A3)"
_HE_VOTE_COLUMNS = {
    "ERSTSTIMME": _VoteColumns(
        12, 13, {"spd": 14, "cdu": 15, "fdp": 16, "kpd": 17, "gb_bhe": 18, "gvp": 19, "dp": 20, "drp": 21, "dns": 22}
    ),
    "ZWEITSTIMME": _VoteColumns(23, 24, {"spd": 25, "cdu": 26, "fdp": 27, "kpd": 28, "gb_bhe": 29, "gvp": 30, "dp": 31}),
}
# parties printed as own columns in the 1953 Hessian publication; everything else is "Sonstige"
_HE_NAMED = ("spd", "cdu", "fdp", "kpd", "gb_bhe", "gvp", "dp")


@dataclass
class OfficialUnit:
    name: str
    county: Optional[str]
    values: Dict[str, Dict[str, int]]  # vote type -> field -> value


def _int(v) -> Optional[int]:
    if v is None or v == "":
        return None
    if isinstance(v, str):
        v = v.strip()
        if v in ("x", "-", "–"):
            return 0  # no candidate / no votes
        return int(v.replace(" ", "").replace(".", ""))
    return int(v)


def load_hessen_1953(path: Path) -> List[OfficialUnit]:
    """Municipalities of the official Hessian 1953 file (state and city totals excluded)."""
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["Gemeinden"]
    units: List[OfficialUnit] = []
    for row in ws.iter_rows(min_row=6, values_only=True):
        name, county = row[3], row[4]
        if not name or row[0] is None or row[2] is None:
            continue  # the state total has no municipality key
        values = {vote_type: _vote_values(row, cols) for vote_type, cols in _HE_VOTE_COLUMNS.items()}
        units.append(OfficialUnit(name=str(name).strip(), county=str(county).strip() if county else None, values=values))
    return units


def _vote_values(row: tuple, cols: _VoteColumns) -> Dict[str, int]:
    """Official values of one vote type, in the pipeline's field names ('Sonstige' = valid - named parties)."""
    parties = {p: _int(row[i]) or 0 for p, i in cols.parties.items()}
    fields: Dict[str, Optional[int]] = {
        "eligible_voters": _int(row[_HE_ELIGIBLE_COLUMN]),
        "invalid_votes": _int(row[cols.invalid_votes]),
        "valid_votes": _int(row[cols.valid_votes]),
        **{f"party:{p}": parties.get(p, 0) for p in _HE_NAMED},
    }
    valid = fields["valid_votes"]
    if valid is not None:
        fields["party:sonstige"] = valid - sum(parties.get(p, 0) for p in _HE_NAMED)
    return {k: v for k, v in fields.items() if v is not None}


@dataclass
class FieldComparison:
    rows_considered: int = 0
    rows_matched: int = 0
    rows_ambiguous: int = 0
    rows_unmatched: int = 0
    compared: Counter = field(default_factory=Counter)  # field -> compared cells
    equal: Counter = field(default_factory=Counter)  # field -> equal cells
    mismatches: List[dict] = field(default_factory=list)

    @property
    def total_compared(self) -> int:
        return sum(self.compared.values())

    @property
    def total_equal(self) -> int:
        return sum(self.equal.values())


def _cell_values(session: Session, run_id: str) -> Dict[str, Tuple[str, Optional[str]]]:
    return {
        cid: (st, val)
        for cid, st, val in session.execute(select(Cell.cell_id, Cell.status, Cell.value).where(Cell.run_id == run_id))
    }


def compare_hessen_1953(
    session: Session, run_id: str, official: List[OfficialUnit], row_ids: Optional[Iterable[str]] = None
) -> FieldComparison:
    """Compare every AGREED cell of matched municipality rows with the official values.

    ``row_ids`` restricts the comparison (e.g. to rows that passed all data checks)."""
    by_name: Dict[str, List[OfficialUnit]] = defaultdict(list)
    for u in official:
        by_name[exact_key(u.name)].append(u)
    counties = {exact_key(u.county) for u in official if u.county}
    cells = _cell_values(session, run_id)
    wanted = set(row_ids) if row_ids is not None else None
    out = FieldComparison()
    rows = session.execute(
        select(ResultRow).where(
            ResultRow.run_id == run_id, ResultRow.row_kind == "DATA", ResultRow.vote_type.in_(("ERSTSTIMME", "ZWEITSTIMME"))
        )
    ).scalars()
    for r in rows:
        if wanted is not None and r.result_id not in wanted:
            continue
        out.rows_considered += 1
        unit = _match_unit(r, by_name, counties)
        if unit is None:
            out.rows_unmatched += 1
            continue
        if unit == "ambiguous":
            out.rows_ambiguous += 1
            continue
        out.rows_matched += 1
        official_values = unit.values.get(r.vote_type, {})
        fields = {**{k: v for k, v in r.field_cells.items()}, **{f"party:{k}": v for k, v in r.party_cells.items()}}
        for name, cid in fields.items():
            if cid is None or name not in official_values:
                continue
            status, value = cells.get(cid, ("ABSENT", None))
            if status != "AGREED" or value is None:
                continue
            out.compared[name] += 1
            if int(value) == official_values[name]:
                out.equal[name] += 1
            else:
                out.mismatches.append(
                    {
                        "result_id": r.result_id,
                        "page": r.page_number,
                        "unit": unit.name,
                        "vote_type": r.vote_type,
                        "field": name,
                        "pipeline": value,
                        "official": official_values[name],
                        "cell_id": cid,
                    }
                )
    return out


def _match_unit(r: ResultRow, by_name: Dict[str, List[OfficialUnit]], counties: Set[str]):
    """The one official municipality of the row's county whose name equals an engine reading.

    Municipality names repeat across counties (e.g. 'Grebenau', 'Heringen'), so a row is only
    matched when its county is known: the agreed section name, or else section-name readings
    that equal exactly one official county name. Rows without a county stay unmatched."""
    county = _row_county(r, counties)
    if county is None:
        return None
    readings = (r.context or {}).get("_unit_name_readings") or {}
    keys = {exact_key(t) for t in readings.values() if t}
    hits = {id(u): u for k in keys for u in by_name.get(k, []) if u.county and exact_key(u.county) == county}
    if not hits:
        return None
    if len(hits) > 1:
        return "ambiguous"
    return next(iter(hits.values()))


def _row_county(r: ResultRow, counties: Set[str]) -> Optional[str]:
    if r.section_name and exact_key(r.section_name) in counties:
        return exact_key(r.section_name)
    readings = (r.context or {}).get("_section_name_readings") or {}
    found = {exact_key(t) for t in readings.values() if t and exact_key(t) in counties}
    return found.pop() if len(found) == 1 else None
