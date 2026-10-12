"""Regression tests on real source pages (slow: Tesseract + EasyOCR + TATR).

Run with:  pytest -m slow
Expected values were checked visually against the page images; the invariant
tested is "never AGREE on a different value" - abstaining (review) is allowed.
"""

from pathlib import Path

import pytest
from conftest import ROOT

pytestmark = pytest.mark.slow

HESSEN = ROOT / "data" / "Hessen_Bundestagswahl_Statistik1953.pdf"
BW49 = ROOT / "data" / "010_2611_49001.pdf"
BW53 = ROOT / "data" / "011_2611_53001.pdf"
BW61 = ROOT / "data" / "080_2611_61001.pdf"
BW65 = ROOT / "data" / "121_2611_65001.pdf"
BW69 = ROOT / "data" / "166_2611_69001.pdf"
STRUCTURE_CODES = {
    "TABLE_NOT_DETECTED",
    "HEADER_UNRESOLVED",
    "COLUMN_COUNT_MISMATCH",
    "TABLE_STRUCTURE_DISAGREEMENT",
    "PAGE_STITCH_MISMATCH",
}


def _processor():
    from election_ocr.config import load_config
    from election_ocr.issues import IssueFactory
    from election_ocr.layout.tatr import TATRStructureRecognizer, TATRTableDetector
    from election_ocr.pipeline.page_processor import PageProcessor
    from election_ocr.pipeline.runner import build_engines
    from election_ocr.reference import ReferenceRepository

    cfg = load_config()
    ref = ReferenceRepository.load(cfg.paths.reference_dir)
    pp = PageProcessor(
        cfg,
        TATRTableDetector(cfg.table_models),
        TATRStructureRecognizer(cfg.table_models),
        build_engines(cfg, "test"),
        IssueFactory(),
    )
    return cfg, ref, pp


def _structure_ok(table):
    blocking = {i.code.value for i in table.issues if i.blocking}
    return not (blocking & STRUCTURE_CODES) and all(m.matched for m in table.header_matches)


def _process(pdf: Path, page: int, profile_id: str, tmp_path):
    if not pdf.exists():
        pytest.skip("sample PDF not present")
    from election_ocr.config import load_config
    from election_ocr.ingestion.render import render_page
    from election_ocr.issues import IssueFactory
    from election_ocr.layout.tatr import TATRStructureRecognizer, TATRTableDetector
    from election_ocr.pipeline.page_processor import PageProcessor
    from election_ocr.pipeline.runner import build_engines
    from election_ocr.reference import ReferenceRepository
    from election_ocr.semantics.rows import empty_context

    cfg = load_config()
    ref = ReferenceRepository.load(cfg.paths.reference_dir)
    ((tp, var),) = ref.profiles[profile_id].table_profiles_for_page(page)
    engines = build_engines(cfg, "test")
    pp = PageProcessor(
        cfg, TATRTableDetector(cfg.table_models), TATRStructureRecognizer(cfg.table_models), engines, IssueFactory()
    )
    rendered = render_page(pdf, page, cfg.render, tmp_path)
    record, _ = pp.process("slow", rendered, tp, var, empty_context())
    return record.tables[0]


def _row(table, label_part, marker=None, role=None):
    return next(
        r
        for r in table.rows
        if (role is None and r.marker == marker or role is not None and r.role_id == role)
        and any(label_part.upper() in x.text.upper() for x in r.label_readings)
    )


def _check(table, row, expected, other=None):
    cells = {**row.field_cells, **row.party_cells}
    for key, want in expected.items():
        cell = table.cell(cells[key]) or (other.cell(cells[key]) if other is not None else None)
        assert cell is not None, key
        if cell.resolution.status.value == "AGREED":
            assert cell.resolution.value == want, (key, cell.resolution.value, want)


def test_hessen_page_120_structure_and_values(tmp_path):
    t = _process(HESSEN, 120, "hessen_btw1953_gemeinden", tmp_path)
    assert _structure_ok(t)
    r = _row(t, "49 ", "1")  # "49 Niedervorschütz", Erststimmen line
    _check(
        t,
        r,
        {
            "eligible_voters": "440",
            "invalid_votes": "22",
            "valid_votes": "369",
            "spd": "163",
            "fdp": "80",
            "cdu": "48",
            "kpd": "8",
            "gb_bhe": "58",
            "gvp": "12",
            "dp": "0",
            "sonstige": "0",
        },
    )
    agreed = [c for c in t.cells if c.value_type == "integer" and c.resolution.status.value == "AGREED"]
    assert len(agreed) > 600  # automation level regression guard


def test_bw1949_page_40_source_inconsistency_is_caught(tmp_path):
    t = _process(BW49, 40, "bw_btw1949_gemeinden", tmp_path)
    assert _structure_ok(t)
    r = _row(t, "Sattelbach")
    # printed exactly like this in the source; 235/257 = 91.4 % != 90.7 % -> must be flagged by validation
    _check(t, r, {"eligible_voters": "257", "voters": "235", "valid_votes": "228", "turnout_percent": "90.7"})
    from decimal import Decimal

    from election_ocr.issues import IssueCode, IssueFactory
    from election_ocr.validation.rules import FieldValue, RowValues, row_checks

    vals = {k: t.cell(v).resolution for k, v in r.field_cells.items()}
    rv = RowValues(
        result_key="x",
        row_kind="DATA",
        marker=None,
        vote_type="SINGLE",
        group_key="g",
        section_key="s",
        publishable_role=True,
        metrics={
            k: FieldValue(
                name=k, cell_id=None, status=v.status.value, value=Decimal(v.value) if v.value else None, decimals=v.decimals
            )
            for k, v in vals.items()
        },
    )
    assert IssueCode.TURNOUT_MISMATCH in {i.code for i in row_checks(rv, ["turnout_consistent"], IssueFactory())}


# --------------------------------------------- volumes added 2026-09-30 (values read on the page images)
def test_bw1953_spread_41_42(tmp_path):
    if not BW53.exists():
        pytest.skip("sample PDF not present")
    from election_ocr.ingestion.render import render_page
    from election_ocr.semantics.rows import empty_context

    cfg, ref, pp = _processor()
    ((tp, var),) = ref.profiles["bw_btw1953_gemeinden"].table_profiles_for_page(41)
    pages = (render_page(BW53, 41, cfg.render, tmp_path), render_page(BW53, 42, cfg.render, tmp_path))
    (left, right), _ = pp.process_spread(("l", "r"), pages, tp, var, empty_context())
    lt, rt = left.tables[0], right.tables[0]
    assert _structure_ok(lt) and _structure_ok(rt)
    r = _row(lt, "GAILDORF", role="erst")
    assert r.unit_index == "16" and not [i for i in r.issues if i.code.value in ("INDEX_MISMATCH", "PAGE_STITCH_MISMATCH")]
    _check(
        lt,
        r,
        {
            "eligible_voters": "2518",
            "voters": "2131",
            "turnout_percent": "84.6",
            "invalid_votes": "46",
            "valid_votes": "2085",
            "cdu": "927",
            "spd": "597",
            "fdp": "367",
            "kpd": "9",
            "gb_bhe": "137",
            "dp": "19",
        },
        rt,
    )


def test_bw1961_page_35(tmp_path):
    t = _process(BW61, 35, "bw_btw1961_gemeinden", tmp_path)
    assert _structure_ok(t)
    # located by the agreed printed index (the name is read 'Affalterbacn' by one engine)
    e, z = (next(r for r in t.rows if r.unit_index == "1" and r.role_id == role) for role in ("erst", "zweit"))
    _check(
        t,
        e,
        {"eligible_voters": "1024", "voters": "827", "invalid_votes": "42", "valid_votes": "785", "cdu": "304", "spd": "332"},
    )
    _check(
        t,
        z,
        {"eligible_voters": "1024", "voters": "827", "invalid_votes": "85", "valid_votes": "742", "cdu": "269", "spd": "322"},
    )
    assert z.field_cells["eligible_voters"] == e.field_cells["eligible_voters"]  # printed once on the E line


def test_bw1965_page_31_printout(tmp_path):
    t = _process(BW65, 31, "bw_btw1965_gemeinden", tmp_path)
    assert _structure_ok(t)
    e = _row(t, "DIRGENHEIM", role="abs_e")
    assert e.unit_index == "120"
    _check(
        t,
        e,
        {
            "eligible_voters": "165",
            "voters": "149",
            "turnout_percent": "91.9",
            "invalid_votes": "4",
            "valid_votes": "145",
            "cdu": "125",
            "spd": "12",
            "fdp": "8",
        },
    )
    _check(
        t,
        _row(t, "DIRGENHEIM", role="abs_z"),
        {"invalid_votes": "6", "valid_votes": "143", "cdu": "124", "spd": "11", "fdp": "8"},
    )
    _check(
        t,
        _row(t, "DIRGENHEIM", role="pct_e"),
        {"invalid_votes": "2.7", "valid_votes": "97.3", "cdu": "86.2", "spd": "8.3", "fdp": "5.5"},
    )


def test_bw1969_page_32_printout(tmp_path):
    t = _process(BW69, 32, "bw_btw1969_gemeinden", tmp_path)
    assert _structure_ok(t)
    e = _row(t, "DORFMERKINGEN", role="abs_e")
    _check(
        t,
        e,
        {
            "eligible_voters": "587",
            "voters": "508",
            "invalid_votes": "38",
            "valid_votes": "470",
            "cdu": "358",
            "spd": "76",
            "fdp": "15",
            "npd": "17",
            "einzelbewerber": "2",
        },
    )


# --------------------------------------------- Wahlkreis tables added 2026-10-01 (values read on the page images)
def _spread(pdf: Path, page: int, profile_id: str, tmp_path):
    if not pdf.exists():
        pytest.skip("sample PDF not present")
    from election_ocr.ingestion.render import render_page
    from election_ocr.semantics.rows import empty_context

    cfg, ref, pp = _processor()
    ((tp, var),) = (x for x in ref.profiles[profile_id].table_profiles_for_page(page) if "wahlkreis" in x[0].table_profile_id)
    pages = (render_page(pdf, page, cfg.render, tmp_path), render_page(pdf, page + 1, cfg.render, tmp_path))
    (left, right), _ = pp.process_spread(("l", "r"), pages, tp, var, empty_context())
    return left.tables[0], right.tables[0]


def test_bw1961_wahlkreise_spread_31_32(tmp_path):
    lt, rt = _spread(BW61, 31, "bw_btw1961_gemeinden", tmp_path)
    assert _structure_ok(lt) and _structure_ok(rt)
    assert lt.line_count == rt.line_count == 68  # 33 constituencies + Land total, 2 lines each
    e = next(r for r in lt.rows if r.unit_index == "163" and r.role_id == "erst")
    # the right-hand number is printed centred between the E and Z lines: it must not yield a fragment
    assert not [i for i in e.issues if i.code.value == "INDEX_MISMATCH" and "differ" in i.message]
    _check(
        lt,
        e,
        {
            "eligible_voters": "225514",
            "voters": "186695",
            "turnout_percent": "82.8",
            "invalid_votes": "4459",
            "valid_votes": "182236",
            "cdu": "58608",
            "spd": "68994",
            "fdp": "44795",
        },
        rt,
    )
    z = next(r for r in lt.rows if r.unit_index == "163" and r.role_id == "zweit")
    _check(lt, z, {"invalid_votes": "7884", "valid_votes": "178811", "cdu": "58338", "spd": "67740"}, rt)


def test_bw1965_wahlkreise_spread_20_21(tmp_path):
    lt, rt = _spread(BW65, 20, "bw_btw1965_gemeinden", tmp_path)
    assert _structure_ok(lt) and _structure_ok(rt)
    assert lt.line_count == rt.line_count == 54  # 9 constituencies x 6 lines (65/%/61/%/V/%)
    e = _row(lt, "STUTTGART I", role="e65")
    assert e.unit_index == "164"
    _check(
        lt,
        e,
        {
            "eligible_voters": "145275",
            "voters": "121213",
            "invalid_votes": "2092",
            "valid_votes": "119121",
            "cdu": "42749",
            "spd": "54574",
            "fdp": "15982",
            "aud": "245",
            "dfu": "3191",
            "npd": "2380",
        },
        rt,
    )
    _check(
        lt,
        _row(lt, "STUTTGART I", role="z65"),
        {"invalid_votes": "2408", "valid_votes": "118805", "cdu": "41688", "npd": "2651"},
        rt,
    )
    # V line: trailing minus ('2327-'), the minus protrudes into the next column on the scan
    _check(lt, _row(lt, "STUTTGART I", role="de"), {"eligible_voters": "-2327", "voters": "-5973", "invalid_votes": "-540"}, rt)
