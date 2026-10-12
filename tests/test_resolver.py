from conftest import reading

from election_ocr.config import OCRSettings
from election_ocr.ocr.resolver import CellStatus, resolve_cell
from election_ocr.parsing.numbers import IntegerFormat

FMT = IntegerFormat(thousands_separators=[" "], dash_means_zero=True)
S = OCRSettings()


def resolve(*rs):
    return resolve_cell(list(rs), "integer", S, integer_format=FMT)


def test_two_independent_groups_agree():
    r = resolve(reading("tesseract", "1 207"), reading("easyocr", "1207"))
    assert r.status == CellStatus.AGREED and r.value == "1207"
    assert r.supporting_groups == ["easyocr", "tesseract"]


def test_any_dissent_is_disagreement_even_two_against_one():
    # no majority vote, not even 2 vs 1 (docs/decisions.md, D1)
    r = resolve(reading("tesseract", "337"), reading("easyocr", "337"), reading("pdf_text_layer", "357"))
    assert r.status == CellStatus.DISAGREED and r.value is None


def test_single_source_is_not_enough():
    r = resolve(reading("tesseract", "12"), reading("easyocr", ""))
    assert r.status == CellStatus.SINGLE_SOURCE and r.value is None


def test_variants_of_one_engine_do_not_count_twice():
    r = resolve(reading("tesseract", "12", variant="a"), reading("tesseract", "12", variant="b"))
    assert r.status == CellStatus.SINGLE_SOURCE


def test_conflicting_variants_inside_group_are_dissent():
    r = resolve(reading("tesseract", "12", variant="a"), reading("tesseract", "17", variant="b"), reading("easyocr", "12"))
    assert r.status == CellStatus.DISAGREED


def test_unparseable_noise_with_same_digits_is_not_dissent():
    r = resolve(reading("tesseract", "4", variant="w"), reading("tesseract", "4:", variant="f"), reading("easyocr", "4"))
    assert r.status == CellStatus.AGREED and r.value == "4"
    roles = {c.variant: c.role for c in r.candidates if c.engine == "tesseract"}
    assert roles["f"] == "noise"


def test_unparseable_reading_with_other_digits_is_dissent():
    r = resolve(reading("tesseract", "41:"), reading("easyocr", "4"))
    assert r.status == CellStatus.DISAGREED


def test_confusion_substitution_never_counts_by_default():
    r = resolve(reading("tesseract", "1O7"), reading("easyocr", "107"))
    assert r.status == CellStatus.SINGLE_SOURCE
    assert any(c.role == "noise" for c in r.candidates)


def test_no_reading_and_empty():
    assert resolve().status == CellStatus.EMPTY
    assert resolve(reading("tesseract", ""), reading("easyocr", "")).status == CellStatus.NO_READING


def test_text_layer_word_overflow_abstains():
    r = resolve(reading("tesseract", "5"), reading("easyocr", "5"), reading("pdf_text_layer", "57", meta={"word_overflow": True}))
    assert r.status == CellStatus.AGREED


def test_confidence_floor_is_optional_and_explicit():
    s = OCRSettings(min_candidate_confidence={"easyocr": 0.5})
    r = resolve_cell([reading("tesseract", "5"), reading("easyocr", "5", conf=0.1)], "integer", s, integer_format=FMT)
    assert r.status == CellStatus.SINGLE_SOURCE and r.low_confidence_groups == ["easyocr"]
