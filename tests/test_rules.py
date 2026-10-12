from decimal import Decimal

from election_ocr.issues import IssueCode, IssueFactory
from election_ocr.validation.rules import FieldValue, RowValues, cross_role_checks, hierarchy_checks, row_checks

F = IssueFactory()


def fv(name, v, status="AGREED", decimals=None):
    return FieldValue(name=name, cell_id=name, status=status, value=None if v is None else Decimal(str(v)), decimals=decimals)


def row(key="r", kind="DATA", marker="1", section="s", **metrics):
    parties = {k[2:]: fv(k, v) for k, v in metrics.items() if k.startswith("p_")}
    ms = {k: (v if isinstance(v, FieldValue) else fv(k, v)) for k, v in metrics.items() if not k.startswith("p_")}
    return RowValues(
        result_key=key,
        row_kind=kind,
        marker=marker,
        vote_type="X",
        group_key="g",
        section_key=section,
        publishable_role=True,
        metrics=ms,
        parties=parties,
    )


def codes(issues):
    return {i.code for i in issues}


def test_party_sum_ok_and_mismatch():
    ok = row(valid_votes=100, p_a=60, p_b=40)
    assert not row_checks(ok, ["party_sum_eq_valid"], F)
    bad = row(valid_votes=100, p_a=61, p_b=40)
    assert IssueCode.PARTY_SUM_MISMATCH in codes(row_checks(bad, ["party_sum_eq_valid"], F))


def test_check_not_executable_when_a_value_is_unresolved():
    r = row(valid_votes=100, p_a=60, p_b=None)
    r.parties["b"] = fv("p_b", None, status="DISAGREED")
    assert IssueCode.ARITHMETIC_CHECK_NOT_EXECUTABLE in codes(row_checks(r, ["party_sum_eq_valid"], F))


def test_voter_identities():
    r = row(eligible_voters=100, voters=101, valid_votes=90, invalid_votes=11)
    c = codes(row_checks(r, ["voters_le_eligible", "voters_eq_valid_plus_invalid", "valid_le_voters"], F))
    assert IssueCode.VOTERS_EXCEED_ELIGIBLE in c and IssueCode.VOTERS_NE_VALID_PLUS_INVALID not in c
    r2 = row(eligible_voters=100, valid_votes=95, invalid_votes=10)
    assert IssueCode.VOTERS_EXCEED_ELIGIBLE in codes(row_checks(r2, ["valid_plus_invalid_le_eligible"], F))


def test_turnout_uses_exact_rounding_of_printed_decimals():
    # 278 / 337 = 82.49...% -> 82.5
    good = row(eligible_voters=337, voters=278, turnout_percent=fv("turnout_percent", "82.5", decimals=1))
    assert not row_checks(good, ["turnout_consistent"], F)
    bad = row(eligible_voters=337, voters=278, turnout_percent=fv("turnout_percent", "82.4", decimals=1))
    assert IssueCode.TURNOUT_MISMATCH in codes(row_checks(bad, ["turnout_consistent"], F))


def test_cross_role_eligible_voters_must_match():
    a = row("a", marker="1", eligible_voters=440)
    b = row("b", marker="2", eligible_voters=441)
    out = cross_role_checks([a, b], {"eligible_voters": ["1", "2"]}, F)
    assert IssueCode.CROSS_ROLE_MISMATCH in codes(out["a"]) | codes(out["b"])


def test_hierarchy_sum_matches_and_mismatches():
    m1 = row("m1", valid_votes=10, p_a=6, p_b=4)
    m2 = row("m2", valid_votes=20, p_a=15, p_b=5)
    total = row("t", kind="SECTION_TOTAL", valid_votes=30, p_a=21, p_b=9)
    assert not hierarchy_checks([m1, m2, total], True, True, F)
    total_bad = row("t", kind="SECTION_TOTAL", valid_votes=30, p_a=22, p_b=9)
    out = hierarchy_checks([m1, m2, total_bad], True, True, F)
    assert all(IssueCode.HIERARCHICAL_SUM_MISMATCH in codes(out[k]) for k in ("m1", "m2", "t"))


def test_hierarchy_unverified_without_total_or_with_unresolved_values():
    m1 = row("m1", valid_votes=10, p_a=6, p_b=4)
    m2 = row("m2", valid_votes=20, p_a=15, p_b=5)
    out = hierarchy_checks([m1, m2], True, True, F)
    assert IssueCode.HIERARCHY_UNVERIFIED in codes(out["m1"])
    m2.parties["a"] = fv("p_a", None, status="DISAGREED")
    total = row("t", kind="SECTION_TOTAL", valid_votes=30, p_a=21, p_b=9)
    out = hierarchy_checks([m1, m2, total], True, True, F)
    assert IssueCode.HIERARCHY_UNVERIFIED in codes(out["t"])
