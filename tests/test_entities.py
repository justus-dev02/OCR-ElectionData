from datetime import date

from conftest import make_reference

from election_ocr.issues import IssueCode, IssueFactory
from election_ocr.pipeline.models import LabelReading
from election_ocr.semantics.entities import UnitResolver

ON = date(1953, 9, 6)


def resolver(ref=None, require_verified=True):
    return UnitResolver(ref or make_reference(), IssueFactory(), 2, 0.8, 5, require_verified)


def lr(group, text):
    return LabelReading(engine=group, independence_group=group, text=text)


def codes(res):
    return {i.code for i in res.issues}


def test_exact_match_by_two_groups_resolves():
    res = resolver().resolve([lr("a", "Alpha"), lr("b", "Alpha")], None, "GEMEINDE", ON, "Land A", {}, parent_unit_id="lk_x")
    assert res.unit_id == "g_alpha" and not res.issues


def test_alias_match_counts_as_exact():
    res = resolver().resolve([lr("a", "Beta"), lr("b", "Beta, Stadt")], None, "GEMEINDE", ON, "Land A", {}, "lk_x")
    assert res.unit_id == "g_beta"


def test_one_group_only_is_single_source():
    res = resolver().resolve([lr("a", "Alpha"), lr("b", "Aipha")], None, "GEMEINDE", ON, "Land A", {}, "lk_x")
    assert res.unit_id is None and IssueCode.OCR_SINGLE_SOURCE in codes(res)


def test_fuzzy_never_resolves_but_lists_candidates():
    res = resolver().resolve([lr("a", "Aipha"), lr("b", "Alpba")], None, "GEMEINDE", ON, "Land A", {}, "lk_x")
    assert res.unit_id is None and IssueCode.ENTITY_FUZZY_ONLY in codes(res)
    assert res.candidates and res.candidates[0].unit_id == "g_alpha"


def test_unknown_name_never_creates_a_unit():
    ref = make_reference()
    n = len(ref.units)
    res = resolver(ref).resolve([lr("a", "Omega"), lr("b", "Omega")], None, "GEMEINDE", ON, "Land A", {}, "lk_x")
    assert res.unit_id is None and IssueCode.ENTITY_NOT_FOUND in codes(res) and len(ref.units) == n


def test_unit_not_valid_on_election_date():
    res = resolver().resolve([lr("a", "Gamma"), lr("b", "Gamma")], None, "GEMEINDE", ON, "Land A", {}, "lk_x")
    assert res.unit_id is None and IssueCode.ENTITY_NOT_VALID_FOR_DATE in codes(res)


def test_wrong_state_context():
    res = resolver().resolve([lr("a", "Alpha"), lr("b", "Alpha")], None, "GEMEINDE", ON, "Land B", {}, None)
    assert res.unit_id is None and IssueCode.ENTITY_OUT_OF_CONTEXT in codes(res)


def test_unverified_unit_blocks():
    res = resolver(make_reference(verified=False)).resolve(
        [lr("a", "Alpha"), lr("b", "Alpha")], None, "GEMEINDE", ON, "Land A", {}, "lk_x"
    )
    assert res.unit_id == "g_alpha" and IssueCode.REFERENCE_UNVERIFIED in codes(res)


def test_empty_gazetteer_is_reported_explicitly():
    res = resolver(make_reference(with_units=False)).resolve([lr("a", "Alpha")], None, "GEMEINDE", ON, "Land A", {})
    assert IssueCode.REFERENCE_DATA_MISSING in codes(res)
