"""Shared fixtures and builders for the test-suite (no OCR/ML models needed)."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from election_ocr.config import PipelineConfig  # noqa: E402
from election_ocr.issues import IssueFactory  # noqa: E402
from election_ocr.ocr.engines import OCRReading  # noqa: E402
from election_ocr.reference.models import (  # noqa: E402
    AdministrativeUnit,
    DocumentProfile,
    Election,
    ElectionPartyEntry,
    Party,
)
from election_ocr.reference.repository import ReferenceRepository  # noqa: E402


def reading(group: str, text: str, conf=0.9, engine=None, variant="v", meta=None) -> OCRReading:
    return OCRReading(
        token_id="t",
        engine=engine or group,
        engine_version="test",
        variant=variant,
        independence_group=group,
        raw_text=text,
        confidence=conf,
        meta=meta or {},
    )


@pytest.fixture
def cfg(tmp_path) -> PipelineConfig:
    c = PipelineConfig()
    c.database_url = f"sqlite:///{tmp_path / 'test.db'}"
    c.paths.output_dir = tmp_path / "output"
    c.paths.gold_dir = tmp_path / "gold"
    c.ensure_directories()
    return c


@pytest.fixture
def issues() -> IssueFactory:
    return IssueFactory()


def profile_dict(verified: bool = True) -> dict:
    int_fmt = {"thousands_separators": [" "], "dash_means_zero": True}
    return {
        "document_profile_id": "test_profile",
        "sha256": ["a" * 64],
        "verified": verified,
        "tables": [
            {
                "table_profile_id": "t1",
                "election_id": "e1",
                "row_mode": "marker_group",
                "line_anchor_column": "marker",
                "marker_column": "marker",
                "marker_sequence": ["1", "2"],
                "row_roles": {"1": {"vote_type": "ERSTSTIMME"}, "2": {"vote_type": "ZWEITSTIMME"}},
                "label_pattern": r"^(?P<index>\d+)\s+(?P<name>\S.*)$",
                "section_kinds": {"Landkreis": "LANDKREIS"},
                "continued_prefixes": ["noch"],
                "total_keywords": ["Zusammen"],
                "unit_level_default": "GEMEINDE",
                "required_checks": ["party_sum_eq_valid", "valid_plus_invalid_le_eligible"],
                "consistent_fields_across_markers": {"eligible_voters": ["1", "2"]},
                "variants": [
                    {
                        "variant_id": "v1",
                        "pages": "1-10",
                        "state": "Land A",
                        "columns": [
                            {"key": "label", "role": "unit_label", "value_type": "label", "header_aliases": ["Gemeinde"]},
                            {"key": "marker", "role": "row_marker", "value_type": "marker", "header_aliases": ["Wahl"]},
                            {
                                "key": "eligible",
                                "role": "metric",
                                "metric": "eligible_voters",
                                "value_type": "integer",
                                "header_aliases": ["Wahlberechtigte"],
                                "integer_format": int_fmt,
                            },
                            {
                                "key": "invalid",
                                "role": "metric",
                                "metric": "invalid_votes",
                                "value_type": "integer",
                                "header_aliases": ["Ungültige"],
                                "integer_format": int_fmt,
                            },
                            {
                                "key": "valid",
                                "role": "metric",
                                "metric": "valid_votes",
                                "value_type": "integer",
                                "header_aliases": ["Gültige"],
                                "integer_format": int_fmt,
                            },
                            {
                                "key": "pa",
                                "role": "party",
                                "party_id": "pa",
                                "value_type": "integer",
                                "header_aliases": ["PA"],
                                "integer_format": int_fmt,
                            },
                            {
                                "key": "pb",
                                "role": "party",
                                "party_id": "pb",
                                "value_type": "integer",
                                "header_aliases": ["PB"],
                                "integer_format": int_fmt,
                            },
                        ],
                    }
                ],
            }
        ],
    }


def make_reference(verified: bool = True, with_units: bool = True) -> ReferenceRepository:
    parties = [
        Party(party_id="pa", canonical_name="PA", verified=verified),
        Party(party_id="pb", canonical_name="PB", verified=verified),
    ]
    elections = [
        Election(
            election_id="e1",
            name="Testwahl",
            election_type="Bundestagswahl",
            election_date=date(1953, 9, 6),
            vote_types=["ERSTSTIMME", "ZWEITSTIMME"],
            parties=[
                ElectionPartyEntry(party_id="pa", display_name="PA", states=["Land A"]),
                ElectionPartyEntry(party_id="pb", display_name="PB", states=["Land A"]),
            ],
            verified=verified,
        )
    ]
    units = []
    if with_units:
        units = [
            AdministrativeUnit(unit_id="land_a", name="Land A", level="LAND", verified=verified),
            AdministrativeUnit(unit_id="lk_x", name="Xstadt", level="LANDKREIS", parent_unit_id="land_a", verified=verified),
            AdministrativeUnit(unit_id="g_alpha", name="Alpha", level="GEMEINDE", parent_unit_id="lk_x", verified=verified),
            AdministrativeUnit(
                unit_id="g_beta", name="Beta, Stadt", level="GEMEINDE", parent_unit_id="lk_x", aliases=["Beta"], verified=verified
            ),
            AdministrativeUnit(
                unit_id="g_gamma_old",
                name="Gamma",
                level="GEMEINDE",
                parent_unit_id="lk_x",
                valid_to=date(1950, 1, 1),
                verified=verified,
            ),
        ]
    profiles = [DocumentProfile(**profile_dict(verified))]
    return ReferenceRepository(parties, elections, units, profiles)
