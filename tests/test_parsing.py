from decimal import Decimal

import pytest

from election_ocr.parsing.numbers import (
    DecimalFormat,
    IntegerFormat,
    ParseStatus,
    parse_decimal,
    parse_integer,
    parse_marker,
)
from election_ocr.parsing.text import exact_key, fuzzy_key, header_key, header_layer_variants

FMT = IntegerFormat(thousands_separators=[" "], dash_means_zero=True)


@pytest.mark.parametrize(
    "raw,value",
    [("1207", "1207"), ("1 207", "1207"), ("12 345 678", "12345678"), (" 5 950 ", "5950"), ("0", "0")],
)
def test_integer_strict_grammar_accepts(raw, value):
    r = parse_integer(raw, FMT)
    assert r.status == ParseStatus.OK and r.value == value and not r.used_confusion_map
    assert r.raw == raw  # raw evidence never overwritten


@pytest.mark.parametrize("raw", ["12 07", "1 2 3", "586,", "4:", "1.207", "12a", "1 2345"])
def test_integer_rejects_everything_else(raw):
    r = parse_integer(raw, FMT)
    assert r.status == ParseStatus.FAILED and r.value is None


def test_thousands_separator_step_recorded():
    r = parse_integer("1 207", FMT)
    assert "THOUSANDS_SEPARATOR_REMOVED" in r.steps


def test_dash_semantics_depend_on_profile():
    assert parse_integer("–", FMT).value == "0"
    no_zero = IntegerFormat(thousands_separators=[" "], dash_means_zero=False)
    r = parse_integer("-", no_zero)
    assert r.status == ParseStatus.DASH and r.value is None


def test_empty_is_empty():
    assert parse_integer("   ", FMT).status == ParseStatus.EMPTY


def test_confusion_map_is_fallback_and_flagged():
    r = parse_integer("1O7", FMT)
    assert r.status == ParseStatus.OK and r.value == "107" and r.used_confusion_map
    assert any(s.startswith("OCR_CONFUSION_MAP") for s in r.steps)


def test_integer_range_check():
    fmt = IntegerFormat(thousands_separators=[" "], max_value=100)
    assert parse_integer("101", fmt).status == ParseStatus.FAILED


def test_decimal_exact_decimals():
    fmt = DecimalFormat(decimal_separator=",", decimals=1, min_value=Decimal(0), max_value=Decimal(100))
    ok = parse_decimal("80,4", fmt)
    assert ok.status == ParseStatus.OK and ok.value == "80.4" and ok.decimals == 1
    assert parse_decimal("80,45", fmt).status == ParseStatus.FAILED
    assert parse_decimal("101,0", fmt).status == ParseStatus.FAILED


def test_decimal_separator_swap_is_flagged_as_confusion():
    fmt = DecimalFormat(decimal_separator=",", decimals=1)
    r = parse_decimal("65.7", fmt)
    assert r.value == "65.7" and r.used_confusion_map


def test_marker_closed_vocabulary():
    assert parse_marker("v", ["1", "2", "V"]).value == "V"
    assert parse_marker("Y", ["1", "2", "V"]).status == ParseStatus.FAILED


def test_exact_key_keeps_umlauts():
    assert exact_key("Spangenberg,Stadt") == exact_key("Spangenberg ,  Stadt")
    assert exact_key("Müllheim") != exact_key("Mullheim")
    assert fuzzy_key("Müllheim") == fuzzy_key("Mullheim")


def test_header_key_joins_hyphenated_lines():
    assert header_key("Wahl- berech- tigte") == "wahlberechtigte"
    assert "wahlberechtigte" in header_layer_variants(["Wahl-", "berech-", "tigte"])
