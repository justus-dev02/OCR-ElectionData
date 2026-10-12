import numpy as np
from conftest import make_reference

from election_ocr.geometry import Box
from election_ocr.issues import IssueCode, IssueFactory
from election_ocr.layout.tokens import AnchorLine
from election_ocr.ocr.engines import OCRRequest, TextLayerEngine, TextLayerWord
from election_ocr.ocr.resolver import CellResolution, CellStatus
from election_ocr.pipeline.models import CellRecord, LabelReading
from election_ocr.reference.models import ColumnSpec
from election_ocr.semantics.header import match_headers
from election_ocr.semantics.labels import LabelClassifier, agree
from election_ocr.semantics.rows import TableInterpreter, empty_context


def spec(key, aliases, number=None, vtype="integer"):
    kw = {"integer_format": {"thousands_separators": [" "]}} if vtype == "integer" else {}
    return ColumnSpec(
        key=key,
        role="metric" if vtype == "integer" else "unit_label",
        metric="valid_votes" if vtype == "integer" else None,
        value_type=vtype,
        header_aliases=aliases,
        column_number=number,
        **kw,
    )


def test_header_text_match_and_competitor():
    cols = [spec("label", ["Gemeinde"], vtype="label"), spec("elig", ["Wahlberechtigte"]), spec("marker", ["Wahl"])]
    layers = [{"t": ["Gemeinde"]}, {"t": ["Wahl-", "berechtigte"], "e": ["Wahl-", "berechligte"]}, {"t": ["Wahl"]}]
    m = match_headers(cols, layers, 0.85)
    assert all(x.matched for x in m)  # 'Wahl' fragment does not beat the full match


def test_header_shift_is_detected():
    cols = [spec("a", ["SPD"]), spec("b", ["FDP"])]
    m = match_headers(cols, [{"t": ["FDP"]}, {"t": ["CDU"]}], 0.85)
    assert not m[0].matched and m[0].competing_column == "b" and not m[1].matched


def test_header_confirmed_by_printed_column_number():
    cols = [spec("fdp", ["FDP/DVP"], number=7), spec("x", ["KPD"], number=8)]
    ok = match_headers(cols, [{"t": ["FDDyp", "7"], "e": ["FDFhvP"]}, {"t": ["KPD", "8"]}], 0.85)
    assert ok[0].matched and ok[0].confirmed_by == "+number"
    shifted = match_headers(cols, [{"t": ["FDDyp", "8"], "e": ["7"]}, {"t": ["KPD"]}], 0.85)
    assert not shifted[0].matched  # conflicting column numbers


def test_label_classifier():
    c = LabelClassifier(
        r"^(?P<index>\d+)\s+(?P<name>\S.*)$",
        {"Landkreis": "LANDKREIS", "Kreis": "LANDKREIS", "Stadtkreis": "STADTKREIS"},
        ["noch"],
        ["Zusammen"],
        0.75,
    )
    assert c.classify("49 Wiedervorschütz").cls == "UNIT"
    s = c.classify("noch Lendkreis Melsungen")
    assert (s.cls, s.kind, s.continued, s.name) == ("SECTION", "Landkreis", True, "Melsungen")
    assert c.classify("Zuganmen").cls == "TOTAL"
    assert c.classify("5j Ostheim").cls == "UNKNOWN"
    assert c.classify("Kreuzwertheim").cls == "UNKNOWN"  # single token is never a section


def test_label_agreement_requires_two_groups_without_conflict():
    c = LabelClassifier(r"^(?P<index>\d+)\s+(?P<name>\S.*)$", {"Landkreis": "LANDKREIS"}, [], ["Zusammen"], 0.75)
    a = agree({"t": c.classify("Landkreis Alsfeld"), "e": c.classify("Landkreis Alsfeid")}, 2)
    assert a.cls == "SECTION" and a.names == {"t": "Alsfeld", "e": "Alsfeid"}
    b = agree({"t": c.classify("Landkreis Alsfeld"), "e": c.classify("12 Alsfeld")}, 2)
    assert b.cls is None and b.conflict


def _cell(cid, col, line, text=None, value=None, y=0.0, status=CellStatus.AGREED, assigned=True):
    labels = [LabelReading(engine=g, independence_group=g, text=text) for g in ("t", "e")] if text is not None else []
    return CellRecord(
        cell_id=cid,
        table_id="T",
        column_index=col,
        column_key=str(col),
        value_type="label" if text is not None else "integer",
        line_index=line if assigned else None,
        box=[10, y, 50, y + 20],
        box_pdf=[0, 0, 0, 0],
        assignment_status="ASSIGNED" if assigned else "UNASSIGNED",
        resolution=CellResolution(status=status, value=value),
        labels=labels,
    )


def test_table_interpreter_marker_groups_sections_and_totals():
    ref = make_reference()
    prof = ref.profiles["test_profile"]
    tp, var = prof.tables[0], prof.tables[0].variants[0]
    lines = [
        AnchorLine(index=i, cy=y + 10, y0=y, y1=y + 20, box=Box(x0=0, y0=y, x1=1, y1=y + 20))
        for i, y in enumerate([100, 128, 200, 228])
    ]
    cells = [
        _cell("h", 0, None, text="Landkreis Xstadt", y=60, assigned=False),
        _cell("l1", 0, 0, text="1 Alpha", y=100),
        _cell("m0", 1, 0, value="1", y=100),
        _cell("m1", 1, 1, value="2", y=128),
        _cell("l2", 0, 2, text="Zusammen", y=200),
        _cell("m2", 1, 2, value="1", y=200),
        _cell("m3", 1, 3, value="2", y=228),
    ]
    interp = TableInterpreter(tp, var, 2, IssueFactory()).interpret("T", 1, lines, 28.0, cells, empty_context())
    rows, headers = interp.rows, interp.headers
    assert [h.kind for h in headers] == ["Landkreis"] and headers[0].name == "Xstadt"
    assert [(r.row_kind, r.marker) for r in rows] == [
        ("DATA", "1"),
        ("DATA", "2"),
        ("SECTION_TOTAL", "1"),
        ("SECTION_TOTAL", "2"),
    ]
    assert len({r.section_key for r in rows}) == 1 and rows[2].unit_level == "LANDKREIS"
    assert rows[0].unit_name_readings == {"t": "Alpha", "e": "Alpha"}


def test_interpreter_flags_invalid_group_and_missing_context():
    ref = make_reference()
    tp, var = ref.profiles["test_profile"].tables[0], ref.profiles["test_profile"].tables[0].variants[0]
    lines = [
        AnchorLine(index=i, cy=y + 10, y0=y, y1=y + 20, box=Box(x0=0, y0=y, x1=1, y1=y + 20))
        for i, y in enumerate([100, 128, 156])
    ]
    cells = [_cell("l1", 0, 0, text="1 Alpha", y=100)] + [
        _cell(f"m{i}", 1, i, value=v, y=100 + 28 * i) for i, v in enumerate(["1", "2", "1"])
    ]
    interp = TableInterpreter(tp, var, 2, IssueFactory()).interpret("T", 1, lines, 28.0, cells, empty_context())
    rows, issues = interp.rows, interp.issues
    assert not rows and IssueCode.ROW_GROUP_INVALID in {i.code for i in issues}


def test_text_layer_registration_corrects_vertical_offset():
    tokens = {f"t{i}": Box(x0=100, y0=100 + 60 * i, x1=140, y1=120 + 60 * i) for i in range(12)}
    # every word 14 px lower and taller than the glyphs (typical archive OCR layer)
    words = [TextLayerWord(text=str(i), box=Box(x0=100, y0=112 + 60 * i, x1=125, y1=142 + 60 * i)) for i in range(12)]
    words += [TextLayerWord(text="8", box=Box(x0=127, y0=112, x1=140, y1=142))]  # split-off digit of token 0
    eng = TextLayerEngine("pdf_text_layer")
    eng.set_page(words, tokens, glyph_height=20.0)
    assert eng.vertical_offset == 17.0  # centre offset: (112+142)/2 - (100+120)/2
    out = {r.token_id: r.raw_text for r in eng.read([OCRRequest(t, np.zeros((1, 1), np.uint8), "integer", None) for t in tokens])}
    assert out["t0"] == "08" and out["t5"] == "5"


def test_noise_marks_in_label_column_do_not_open_sections():
    ref = make_reference()
    tp, var = ref.profiles["test_profile"].tables[0], ref.profiles["test_profile"].tables[0].variants[0]
    lines = [
        AnchorLine(index=i, cy=y + 10, y0=y, y1=y + 20, box=Box(x0=0, y0=y, x1=1, y1=y + 20))
        for i, y in enumerate([100, 128, 200, 228])
    ]
    cells = [
        _cell("h", 0, None, text="Landkreis Xstadt", y=60, assigned=False),
        _cell("l1", 0, 0, text="1 Alpha", y=100),
        _cell("m0", 1, 0, value="1", y=100),
        _cell("m1", 1, 1, value="2", y=128),
        _cell("dash", 0, None, text="—", y=160, assigned=False),
        _cell("l2", 0, None, text="2 Beta", y=172, assigned=False),  # label slightly above its group
        _cell("m2", 1, 2, value="1", y=200),
        _cell("m3", 1, 3, value="2", y=228),
    ]
    interp = TableInterpreter(tp, var, 2, IssueFactory()).interpret("T", 1, lines, 28.0, cells, empty_context())
    rows, headers, _, issues = interp.rows, interp.headers, interp.carry, interp.issues
    assert len(headers) == 1 and len({r.section_key for r in rows}) == 1
    assert rows[2].unit_name_readings == {"t": "Beta", "e": "Beta"}
    assert IssueCode.NOISE_TOKEN_IGNORED in {i.code for i in issues}


# ---------------------------------------------------------------- 2026-09-30
def test_headings_letter_spacing_and_sections_do_not_collide():
    c = LabelClassifier(
        r"^(?P<index>\d+)\s+(?P<name>\S.*)$",
        {"Landkreis": "LANDKREIS", "Kreis": "LANDKREIS"},
        ["Noch"],
        [],
        0.75,
        headings=["Stadtkreise", "Landkreise"],
        named_headings=["Regierungsbezirk"],
    )
    assert c.classify("Landkreis Böblingen").cls == "SECTION"  # not the plural heading 'Landkreise'
    s = c.classify("Noch: Landkreis Eßlingen")
    assert (s.cls, s.continued, s.name) == ("SECTION", True, "Eßlingen")
    for text in ("S t a d t k r e i s e", "Stadtkreise", "Landkre i se", "L a n d k r e i s e"):
        assert c.classify(text).cls == "HEADING", text
    h = c.classify("Regierungsbezirk Nordbaden")
    assert (h.cls, h.kind, h.name) == ("HEADING", "Regierungsbezirk", "Nordbaden")
    assert c.classify("5 t 8 d t k r e 1 8 (").cls == "UNKNOWN"  # garbled letter spacing never becomes a unit row
    assert c.classify("22 Reichenbach an der Fils").cls == "UNIT"


def test_printout_labels_keywords_first():
    c = LabelClassifier(
        r"^(?P<name>[^\d\s].*?)(?:[\s.]*)$",
        {"LKR.": "LANDKREIS"},
        ["NOCH"],
        ["INSGESAMT"],
        0.75,
        headings=["STADTKREISE", "LANDKREISE"],
        section_index_allowed=True,
        keyword_precedence=True,
    )
    s = c.classify("131 LKR. AALEN")
    assert (s.cls, s.kind, s.index, s.name) == ("SECTION", "LKR.", "131", "AALEN")
    assert c.classify("NOCH LKR. AALEN").continued
    assert c.classify("I N S G E S A M T").cls == "TOTAL"
    u = c.classify("HEILBRONN . . .")
    assert (u.cls, u.name) == ("UNIT", "HEILBRONN")  # leader dots are no letter spacing


def test_single_unreadable_header_is_inferred_from_confirmed_neighbours():
    cols = [spec("a", ["SPD"]), spec("b", ["FDP"]), spec("c", ["KPD"]), spec("d", ["DP"])]
    m = match_headers(cols, [{"t": ["SPD"]}, {"t": ["F#h"]}, {"t": ["KPD"]}, {"t": ["DP"]}], 0.85)
    assert [x.matched for x in m] == [True, True, True, True] and m[1].confirmed_by == "neighbours"
    two = match_headers(cols, [{"t": ["SPD"]}, {"t": ["F#h"]}, {"t": ["K?"]}, {"t": ["DP"]}], 0.85)
    assert not two[1].matched and not two[2].matched  # adjacent unreadable columns are never inferred
    contra = match_headers(cols, [{"t": ["SPD"]}, {"t": ["KPD"]}, {"t": ["KPD"]}, {"t": ["DP"]}], 0.85)
    assert not contra[1].matched and contra[1].contradicted  # another column's header seen here
