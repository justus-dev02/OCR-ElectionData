"""
Reference data repository.

Loads and cross-validates all domain reference data from the ``reference/``
directory:

    reference/parties.yaml                     canonical parties
    reference/elections/*.yaml                 elections + per-election party entries
    reference/administrative_units/*.csv       gazetteer (supplied by the user)
    reference/document_profiles/*.yaml         document/table profiles

Loading fails loudly (``ReferenceDataError``) on any inconsistency, e.g. a
profile referencing a party that is not allowed for its election/state.
"""

from __future__ import annotations

import csv
import hashlib
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

from election_ocr.reference.models import (
    PARTY_FIELD_PREFIX,
    AdministrativeUnit,
    DocumentProfile,
    Election,
    Party,
)

GAZETTEER_COLUMNS = [
    "unit_id",
    "name",
    "level",
    "parent_unit_id",
    "official_key",
    "valid_from",
    "valid_to",
    "aliases",
    "verified",
    "source",
]


class ReferenceDataError(ValueError):
    pass


class ReferenceRepository:
    def __init__(
        self,
        parties: Iterable[Party],
        elections: Iterable[Election],
        units: Iterable[AdministrativeUnit],
        profiles: Iterable[DocumentProfile],
        snapshot_hash: str = "in-memory",
    ):
        self.parties: Dict[str, Party] = {}
        for p in parties:
            if p.party_id in self.parties:
                raise ReferenceDataError(f"duplicate party_id {p.party_id}")
            self.parties[p.party_id] = p
        self.elections: Dict[str, Election] = {}
        for e in elections:
            if e.election_id in self.elections:
                raise ReferenceDataError(f"duplicate election_id {e.election_id}")
            self.elections[e.election_id] = e
        self.units: Dict[str, AdministrativeUnit] = {}
        for u in units:
            if u.unit_id in self.units:
                raise ReferenceDataError(f"duplicate unit_id {u.unit_id}")
            self.units[u.unit_id] = u
        self.profiles: Dict[str, DocumentProfile] = {}
        for pr in profiles:
            if pr.document_profile_id in self.profiles:
                raise ReferenceDataError(f"duplicate document_profile_id {pr.document_profile_id}")
            self.profiles[pr.document_profile_id] = pr
        self.snapshot_hash = snapshot_hash
        self._validate()

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, reference_dir: Path) -> ReferenceRepository:
        reference_dir = Path(reference_dir)
        if not reference_dir.exists():
            raise ReferenceDataError(f"reference directory not found: {reference_dir}")
        hasher = hashlib.sha256()

        def read_yaml(path: Path):
            data = path.read_bytes()
            hasher.update(path.name.encode())
            hasher.update(data)
            return yaml.safe_load(data) or {}

        parties_file = reference_dir / "parties.yaml"
        parties: List[Party] = []
        if parties_file.exists():
            parties = [Party(**p) for p in read_yaml(parties_file).get("parties", [])]

        elections = [Election(**read_yaml(p)) for p in sorted((reference_dir / "elections").glob("*.yaml"))]
        profiles = [DocumentProfile(**read_yaml(p)) for p in sorted((reference_dir / "document_profiles").glob("*.yaml"))]

        units: List[AdministrativeUnit] = []
        for csv_path in sorted((reference_dir / "administrative_units").glob("*.csv")):
            hasher.update(csv_path.name.encode())
            hasher.update(csv_path.read_bytes())
            units.extend(read_gazetteer_csv(csv_path))

        return cls(parties, elections, units, profiles, snapshot_hash=hasher.hexdigest()[:16])

    # --------------------------------------------------------------- validation
    def _validate(self) -> None:
        for e in self.elections.values():
            for entry in e.parties:
                if entry.party_id not in self.parties:
                    raise ReferenceDataError(f"election {e.election_id}: party_id '{entry.party_id}' not in parties.yaml")
        for u in self.units.values():
            if u.parent_unit_id and u.parent_unit_id not in self.units:
                raise ReferenceDataError(f"unit {u.unit_id}: unknown parent_unit_id {u.parent_unit_id}")
        for pr in self.profiles.values():
            for t in pr.tables:
                election = self.elections.get(t.election_id)
                if election is None:
                    raise ReferenceDataError(f"profile {pr.document_profile_id}: unknown election {t.election_id}")
                for role in t.effective_roles():
                    if role.election_id and role.election_id not in self.elections:
                        raise ReferenceDataError(
                            f"profile {pr.document_profile_id}: role {role.role_id} references unknown election "
                            f"{role.election_id}"
                        )
                for v in t.variants:
                    for role_id, items in t.bindings(v).items():
                        role = t.role(role_id)
                        if not role.publish:
                            # comparison/percent roles are never published; their columns follow the
                            # layout of the publishable roles (e.g. 1949 figures under 1953 party columns)
                            continue
                        role_election = self.elections[role.election_id] if role.election_id else election
                        for b in items:
                            if not b.field.startswith(PARTY_FIELD_PREFIX):
                                continue
                            party_id = b.field[len(PARTY_FIELD_PREFIX) :]
                            if not role_election.party_allowed(party_id, v.state):
                                raise ReferenceDataError(
                                    f"profile {pr.document_profile_id}/{v.variant_id}: party '{party_id}' "
                                    f"is not allowed in election {role_election.election_id} for state '{v.state}'"
                                )

    # ------------------------------------------------------------------ queries
    def profile_for_sha256(self, sha256: str) -> Optional[DocumentProfile]:
        matches = [p for p in self.profiles.values() if sha256 in p.sha256]
        if len(matches) > 1:
            raise ReferenceDataError(f"document {sha256[:12]} matches several profiles")
        return matches[0] if matches else None

    def units_valid_on(self, d: date) -> List[AdministrativeUnit]:
        return [u for u in self.units.values() if u.valid_on(d)]

    def children_of(self, unit_id: str, d: date) -> List[AdministrativeUnit]:
        return [u for u in self.units.values() if u.parent_unit_id == unit_id and u.valid_on(d)]

    def ancestors(self, unit_id: str) -> List[AdministrativeUnit]:
        out: List[AdministrativeUnit] = []
        seen = set()
        cur = self.units.get(unit_id)
        while cur and cur.parent_unit_id and cur.parent_unit_id not in seen:
            seen.add(cur.parent_unit_id)
            cur = self.units.get(cur.parent_unit_id)
            if cur:
                out.append(cur)
        return out


def _parse_date(value: str) -> Optional[date]:
    value = (value or "").strip()
    return date.fromisoformat(value) if value else None


def read_gazetteer_csv(path: Path) -> List[AdministrativeUnit]:
    """Read a user-supplied gazetteer CSV (UTF-8, ';' separated aliases)."""
    units: List[AdministrativeUnit] = []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in ("unit_id", "name", "level") if c not in (reader.fieldnames or [])]
        if missing:
            raise ReferenceDataError(f"{path.name}: missing required columns {missing}")
        unknown = [c for c in (reader.fieldnames or []) if c not in GAZETTEER_COLUMNS]
        if unknown:
            raise ReferenceDataError(f"{path.name}: unknown columns {unknown}")
        for line_no, row in enumerate(reader, start=2):
            try:
                units.append(
                    AdministrativeUnit(
                        unit_id=row["unit_id"].strip(),
                        name=row["name"].strip(),
                        level=row["level"].strip(),
                        parent_unit_id=(row.get("parent_unit_id") or "").strip() or None,
                        official_key=(row.get("official_key") or "").strip() or None,
                        valid_from=_parse_date(row.get("valid_from", "")),
                        valid_to=_parse_date(row.get("valid_to", "")),
                        aliases=[a.strip() for a in (row.get("aliases") or "").split(";") if a.strip()],
                        verified=(row.get("verified") or "").strip().lower() in ("1", "true", "yes", "ja"),
                        source=(row.get("source") or "").strip() or None,
                    )
                )
            except Exception as exc:
                raise ReferenceDataError(f"{path.name}:{line_no}: {exc}") from exc
    return units
