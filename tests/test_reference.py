import copy
from pathlib import Path

import pytest
from conftest import ROOT, make_reference, profile_dict

from election_ocr.reference.models import DocumentProfile
from election_ocr.reference.repository import ReferenceDataError, ReferenceRepository, read_gazetteer_csv


def test_project_reference_data_is_consistent():
    ref = ReferenceRepository.load(ROOT / "reference")
    assert {"btw_1949", "btw_1953"} <= set(ref.elections)
    assert ref.profile_for_sha256("dbe43f7d84a42b6b826500b8dfd4da8e042658a266816060b51d3f0d1c733eec") is not None
    # nothing in the shipped reference data is pre-verified: a human must sign off
    assert not any(p.verified for p in ref.parties.values())
    assert not any(e.verified for e in ref.elections.values())


def test_profile_with_party_not_allowed_is_rejected():
    ref = make_reference()
    bad = copy.deepcopy(profile_dict())
    bad["tables"][0]["variants"][0]["state"] = "Land B"
    with pytest.raises(ReferenceDataError):
        ReferenceRepository(ref.parties.values(), ref.elections.values(), ref.units.values(), [DocumentProfile(**bad)])


def test_profile_schema_validation():
    bad = copy.deepcopy(profile_dict())
    bad["tables"][0]["label_pattern"] = r"^\d+ .*$"  # no name group
    with pytest.raises(ValueError):
        DocumentProfile(**bad)
    bad = copy.deepcopy(profile_dict())
    del bad["tables"][0]["row_roles"]["2"]
    with pytest.raises(ValueError):
        DocumentProfile(**bad)


def test_unknown_parent_unit_is_rejected():
    ref = make_reference()
    units = list(ref.units.values())
    units[1] = units[1].model_copy(update={"parent_unit_id": "missing"})
    with pytest.raises(ReferenceDataError):
        ReferenceRepository(ref.parties.values(), ref.elections.values(), units, ref.profiles.values())


def test_gazetteer_csv(tmp_path: Path):
    p = tmp_path / "g.csv"
    p.write_text(
        "unit_id,name,level,parent_unit_id,valid_from,aliases,verified,source\n"
        "l,Land A,LAND,,,,true,test\n"
        'g1,"Beta, Stadt",GEMEINDE,l,1946-01-01,Beta;Beta Stadt,yes,test\n',
        encoding="utf-8",
    )
    units = read_gazetteer_csv(p)
    assert units[1].aliases == ["Beta", "Beta Stadt"] and units[1].verified and units[1].valid_from.year == 1946
    bad = tmp_path / "bad.csv"
    bad.write_text("unit_id,name,level,foo\nx,y,GEMEINDE,1\n", encoding="utf-8")
    with pytest.raises(ReferenceDataError):
        read_gazetteer_csv(bad)
