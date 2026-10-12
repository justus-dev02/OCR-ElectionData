"""Result roles, field bindings, line-printer groups, percent and blank checks (added 2026-09-30)."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from election_ocr.geometry import Box
from election_ocr.issues import IssueCode, IssueFactory
from election_ocr.layout.tokens import AnchorLine
from election_ocr.ocr.resolver import Candidate, CellResolution, CellStatus, shape_conflict
from election_ocr.parsing.numbers import ParseResult, ParseStatus
from election_ocr.parsing.text import printout_key
from election_ocr.pipeline.models import CellRecord, LabelReading
from election_ocr.reference.models import TableProfile, expand_page_ranges, parse_page_ranges
from election_ocr.semantics.rows import TableInterpreter, empty_context
from election_ocr.validation.rules import BLANK, BLANK_ZERO, FieldValue, RowValues, apply_blank_zero, percent_checks

F = IssueFactory()
INT = {"thousands_separators": [], "dash_means_zero": False}


def printout_profile(**over) -> TableProfile:
    """A reduced 1965-style profile: 4 lines per unit (abs E, abs Z, pct E, pct Z)."""
    cols = [
        {"key": "idx", "role": "unit_index", "value_type": "integer", "header_aliases": ["K"], "integer_format": INT},
        {"key": "name", "role": "unit_label", "value_type": "label", "header_aliases": ["NAME"]},
        {"key": "des", "role": "row_marker", "value_type": "marker", "header_aliases": ["B"]},
        {
            "key": "elig",
            "role": "metric",
            "metric": "eligible_voters",
            "value_type": "integer",
            "header_aliases": ["W"],
            "integer_format": INT,
        },
        {
            "key": "voters",
            "role": "metric",
            "metric": "voters",
            "value_type": "integer",
            "header_aliases": ["V"],
            "integer_format": INT,
        },
        {"key": "ez", "role": "row_marker", "value_type": "marker", "header_aliases": ["EZ"]},
        {
            "key": "valid",
            "role": "metric",
            "metric": "valid_votes",
            "value_type": "integer",
            "header_aliases": ["G"],
            "integer_format": INT,
        },
        {
            "key": "pa",
            "role": "party",
            "party_id": "pa",
            "value_type": "integer",
            "header_aliases": ["PA"],
            "integer_format": INT,
        },
        {
            "key": "pb",
            "role": "party",
            "party_id": "pb",
            "value_type": "integer",
            "header_aliases": ["PB"],
            "integer_format": INT,
        },
    ]
    abs_fields = [
        {"field": "valid_votes", "column": "valid"},
        {"field": "party:pa", "column": "pa"},
        {"field": "party:pb", "column": "pb"},
    ]
    pct_fields = [dict(f, format="pct1") for f in abs_fields]
    shared = [{"field": "eligible_voters", "column": "elig", "slot": 0}, {"field": "voters", "column": "voters", "slot": 0}]
    d = {
        "table_profile_id": "t65",
        "election_id": "e1",
        "row_mode": "marker_group",
        "line_anchor_column": "ez",
        "marker_column": "ez",
        "marker_sequence": ["E", "Z", "E", "Z"],
        "marker_columns": {"des": ["*", None, "*", None]},
        "group_by": "label_start",
        "formats": {"pct1": {"value_type": "decimal", "decimal_format": {"decimal_separator": ",", "decimals": 1}}},
        "roles": [
            {"role_id": "abs_e", "vote_type": "ERSTSTIMME", "slot": 0, "fields": shared + abs_fields},
            {"role_id": "abs_z", "vote_type": "ZWEITSTIMME", "slot": 1, "fields": shared + abs_fields},
            {
                "role_id": "pct_e",
                "vote_type": "ERSTSTIMME",
                "slot": 2,
                "kind": "percent",
                "percent_of": "abs_e",
                "publish": False,
                "fields": pct_fields,
            },
            {
                "role_id": "pct_z",
                "vote_type": "ZWEITSTIMME",
                "slot": 3,
                "kind": "percent",
                "percent_of": "abs_z",
                "publish": False,
                "fields": pct_fields,
            },
        ],
        "percent_definitions": {
            "valid_votes": {"numerator": "valid_votes", "denominator": ["valid_votes"]},
            "party:*": {"numerator": "party:*", "denominator": ["valid_votes"]},
        },
        "label_pattern": r"^(?P<name>[^\d\s].*?)(?:[\s.]*)$",
        "keyword_precedence": True,
        "section_index_allowed": True,
        "section_kinds": {"LKR.": "LANDKREIS"},
        "continued_prefixes": ["NOCH"],
        "total_keywords": ["INSGESAMT"],
        "unit_level_default": "GEMEINDE",
        "variants": [{"variant_id": "v", "pages": "1-9", "state": "Land A", "columns": cols}],
    }
    d.update(over)
    return TableProfile(**d)


def test_page_ranges_with_step():
    assert parse_page_ranges("37-41/2,50") == [(37, 41, 2), (50, 50, 1)]
    assert expand_page_ranges("37-41/2") == [37, 39, 41]
    with pytest.raises(ValueError):
        parse_page_ranges("37-40/2")  # does not end on a step


def test_profile_rejects_inconsistent_roles():
    with pytest.raises(ValidationError):
        printout_profile(percent_definitions={})  # percent roles need definitions
    with pytest.raises(ValidationError):
        printout_profile(roles=[])  # repeated markers need explicit roles
    tp = printout_profile()
    formats = tp.cell_formats(tp.variants[0])
    assert formats[("valid", 0)].value_type == "integer" and formats[("valid", 2)].value_type == "decimal"


# ------------------------------------------------------------------ interpreter
def _cell(cid, col, key, line, y, value=None, text=None, box_w=40, status=CellStatus.AGREED, read=None):
    labels = [LabelReading(engine=g, independence_group=g, text=text) for g in ("t", "e")] if text is not None else []
    cands = [
        Candidate(
            engine=g,
            engine_version="x",
            variant="v",
            independence_group=g,
            raw_text=r,
            confidence=None,
            parse=ParseResult(raw=r, normalized=r, status=ParseStatus.OK, value=r),
            role="support",
        )
        for g, r in (read or {}).items()
    ]
    return CellRecord(
        cell_id=cid,
        table_id="T",
        column_index=col,
        column_key=key,
        value_type="integer",
        line_index=line,
        box=[100 * col, y, 100 * col + box_w, y + 20],
        box_pdf=[0, 0, 0, 0],
        assignment_status="ASSIGNED",
        resolution=CellResolution(status=status, value=value, candidates=cands),
        labels=labels,
    )


def _unit(ys, idx="120", name="DEUFRINGEN . . . ."):
    y0, y1, y2, y3 = ys
    cells = [
        _cell(f"i{y0}", 0, "idx", None, y0, value=idx),
        _cell(f"n{y0}", 1, "name", None, y0, text=name),
        _cell(f"d{y0}", 2, "des", None, y0),
        _cell(f"d{y2}", 2, "des", None, y2),
        _cell(f"e{y0}", 3, "elig", None, y0, value="557"),
        _cell(f"v{y0}", 4, "voters", None, y0, value="450"),
        _cell(f"t{y2}", 4, "voters", None, y2, value="82.4"),  # unbound in this reduced profile
    ]
    for y, m in zip(ys, "EZEZ"):
        cells.append(_cell(f"m{y}", 5, "ez", None, y, read={"t": m, "e": m}))
        cells.append(_cell(f"g{y}", 6, "valid", None, y))
        cells.append(_cell(f"a{y}", 7, "pa", None, y))
    return cells


def _interpret(cells, ys):
    lines = [AnchorLine(index=i, cy=y + 10, y0=y, y1=y + 20, box=Box(x0=500, y0=y, x1=510, y1=y + 20)) for i, y in enumerate(ys)]
    for c in cells:
        c.line_index = next(l.index for l in lines if l.y0 == c.box[1])
    tp = printout_profile()
    header = CellRecord(
        cell_id="h",
        table_id="T",
        column_index=1,
        column_key="name",
        value_type="label",
        line_index=None,
        box=[100, 40, 300, 60],
        box_pdf=[0, 0, 0, 0],
        assignment_status="UNASSIGNED",
        resolution=CellResolution(status=CellStatus.AGREED),
        labels=[LabelReading(engine=g, independence_group=g, text="131 LKR. AALEN") for g in ("t", "e")],
    )
    return TableInterpreter(tp, tp.variants[0], 2, F).interpret("T", 1, lines, 28.0, cells + [header], empty_context())


def test_label_start_groups_roles_and_shared_fields():
    ys = [100, 128, 184, 212]
    res = _interpret(_unit(ys), ys)
    assert [r.role_id for r in res.rows] == ["abs_e", "abs_z", "pct_e", "pct_z"]
    abs_z = res.rows[1]
    assert abs_z.field_cells["eligible_voters"] == "e100"  # printed once, shared from slot 0
    assert abs_z.field_cells["valid_votes"] == "g128" and res.rows[2].field_cells["valid_votes"] == "g184"
    assert abs_z.unit_index == "120" and res.rows[0].section_name == "AALEN"
    codes = {i.code for i in abs_z.issues}
    assert IssueCode.UNEXPECTED_CELL_CONTENT in codes  # the turnout cell on slot 2 is bound by no role here
    assert res.line_slots == {0: 0, 1: 1, 2: 2, 3: 3}


def test_designation_on_wrong_slot_and_contradicting_marker_are_reported():
    ys = [100, 128, 184, 212]
    cells = [c for c in _unit(ys) if c.cell_id != "t184"]
    cells.append(_cell("dx", 2, "des", None, 128))  # 'ABS'/'%' ink on slot 1
    cells = [c if c.cell_id != "m212" else _cell("m212", 5, "ez", None, 212, read={"t": "Z", "e": "E"}) for c in cells]
    codes = [i.code for i in _interpret(cells, ys).rows[0].issues]
    assert codes.count(IssueCode.ROW_GROUP_INVALID) == 2


def test_speck_does_not_start_a_group():
    ys = [100, 128, 184, 212]
    cells = [c for c in _unit(ys) if c.cell_id != "t184"]
    cells.append(_cell("speck", 0, "idx", None, 128, box_w=1))  # 1-px scan streak in the index column
    res = _interpret(cells, ys)
    assert len(res.rows) == 4 and not [i for i in res.issues if i.code == IssueCode.ROW_GROUP_INVALID]


def test_total_row_closes_its_section():
    tp = TableProfile(
        **{
            **printout_profile().model_dump(),
            "row_mode": "single_line",
            "roles": [],
            "marker_column": None,
            "marker_sequence": [],
            "marker_columns": {},
            "single_line_vote_type": "SINGLE",
            "percent_definitions": {},
            "group_by": "gap",
            "line_anchor_column": "valid",
            "variants": [
                {
                    "variant_id": "v",
                    "pages": "1",
                    "state": "Land A",
                    "columns": [
                        {"key": "name", "role": "unit_label", "value_type": "label", "header_aliases": ["N"]},
                        {
                            "key": "valid",
                            "role": "metric",
                            "metric": "valid_votes",
                            "value_type": "integer",
                            "header_aliases": ["G"],
                            "integer_format": INT,
                        },
                    ],
                }
            ],
        }
    )
    lines = [
        AnchorLine(index=i, cy=y + 10, y0=y, y1=y + 20, box=Box(x0=100, y0=y, x1=110, y1=y + 20))
        for i, y in enumerate([100, 128, 156])
    ]

    def lab(cid, y, t):
        return CellRecord(
            cell_id=cid,
            table_id="T",
            column_index=0,
            column_key="name",
            value_type="label",
            line_index=(y - 100) // 28,
            box=[0, y, 90, y + 20],
            box_pdf=[0, 0, 0, 0],
            assignment_status="ASSIGNED",
            resolution=CellResolution(status=CellStatus.AGREED),
            labels=[LabelReading(engine=g, independence_group=g, text=t) for g in ("t", "e")],
        )

    cells = [lab("a", 100, "ALPHA"), lab("s", 128, "I N S G E S A M T"), lab("b", 156, "BETA")]
    ctx = empty_context()
    ctx.key, ctx.kind, ctx.level, ctx.agreed = "S", "LKR.", "LANDKREIS", True
    res = TableInterpreter(tp, tp.variants[0], 2, F).interpret("T", 1, lines, 28.0, cells, ctx)
    kinds = [r.row_kind for r in res.rows]
    assert kinds == ["DATA", "SECTION_TOTAL", "DATA"]
    assert IssueCode.SECTION_CONTEXT_MISSING in {i.code for i in res.rows[2].issues}


# ------------------------------------------------------------------- checks
def fvx(name, v, status="AGREED", decimals=None):
    return FieldValue(name=name, cell_id=name, status=status, value=None if v is None else Decimal(str(v)), decimals=decimals)


def rv(role, kind="absolute", percent_of=None, metrics=None, parties=None):
    return RowValues(
        result_key=role,
        row_kind="DATA",
        marker=None,
        vote_type="X",
        group_key="g",
        section_key="s",
        publishable_role=kind == "absolute",
        role_id=role,
        role_kind=kind,
        percent_of_role=percent_of,
        metrics=metrics or {},
        parties=parties or {},
    )


def test_blank_party_cell_is_zero_only_with_exact_sum():
    ok = rv(
        "abs_e",
        metrics={"valid_votes": fvx("valid_votes", 100)},
        parties={"a": fvx("party:a", 100), "b": FieldValue(name="party:b", cell_id=None, status=BLANK)},
    )
    issues = apply_blank_zero(ok, F)
    assert ok.parties["b"].status == BLANK_ZERO and ok.parties["b"].value == 0
    assert [i.code for i in issues] == [IssueCode.BLANK_AS_ZERO]
    bad = rv(
        "abs_e",
        metrics={"valid_votes": fvx("valid_votes", 100)},
        parties={"a": fvx("party:a", 99), "b": FieldValue(name="party:b", cell_id=None, status=BLANK)},
    )
    assert [i.code for i in apply_blank_zero(bad, F)] == [IssueCode.MISSING_VALUE]
    assert bad.parties["b"].status == BLANK


def test_percent_check_matches_rounded_shares_and_blank_cells():
    absolute = rv(
        "abs_e", metrics={"valid_votes": fvx("valid_votes", 421)}, parties={"a": fvx("party:a", 246), "b": fvx("party:b", 0)}
    )
    defs = printout_profile().percent_definitions
    pct = rv(
        "pct_e",
        "percent",
        "abs_e",
        metrics={"valid_votes": fvx("valid_votes", "100.0", decimals=1)},
        parties={"a": fvx("party:a", "58.4", decimals=1), "b": FieldValue(name="party:b", cell_id=None, status=BLANK)},
    )
    assert percent_checks(pct, absolute, defs, F) == []
    pct.parties["a"] = fvx("party:a", "58.5", decimals=1)
    issues = percent_checks(pct, absolute, defs, F)
    assert [i.code for i in issues] == [IssueCode.PERCENT_MISMATCH]
    assert issues[0].details["mismatches"][0]["computed"] == "58.4"
    pct.parties["a"] = fvx("party:a", None, status="DISAGREED")
    assert [i.code for i in percent_checks(pct, absolute, defs, F)] == [IssueCode.PERCENT_CHECK_NOT_EXECUTABLE]


def test_shape_conflict():
    def res(value, parse_status=ParseStatus.OK):
        c = Candidate(
            engine="t",
            engine_version="x",
            variant="v",
            independence_group="t",
            raw_text="-",
            confidence=None,
            parse=ParseResult(raw="-", normalized="-", status=parse_status, value=value),
            role="support",
        )
        return CellResolution(
            status=CellStatus.AGREED, value=value, candidates=[c, c.model_copy(update={"independence_group": "e"})]
        )

    assert shape_conflict(res("0", ParseStatus.DASH), "dash") is None
    assert shape_conflict(res("7"), "dash") is not None  # a dash can only be 0
    assert shape_conflict(res("0", ParseStatus.DASH), "wide") is not None  # a dash reading needs a dash shape
    assert shape_conflict(res("0"), "narrow") is None  # a printed digit 0


def test_printout_key_is_exact_after_transliteration():
    assert printout_key("Gärtringen") == printout_key("GAERTRINGEN")
    assert printout_key("Herrenberg, Stadt") == printout_key("HERRENBERG STADT")
    assert printout_key("Gärtringen") != printout_key("GAERTRINGEM")


def test_definition_probe_decides_on_the_data():
    from election_ocr.validation.rules import DIFFERS, HOLDS, UNDECIDED, DefinitionProbe

    def probe(matches, total):
        p = DefinitionProbe(key="k")
        for i in range(total):
            p.add(i < matches)
        p.add(None)  # undecidable rows are not counted
        return p.decide(min_rows=30, holds_min_share=0.9, differs_max_share=0.2)

    assert probe(95, 100) == HOLDS  # isolated deviations are errors -> the check stays blocking
    assert probe(15, 100) == DIFFERS  # the publication prints another definition -> note, no check
    assert probe(50, 100) == UNDECIDED  # unclear -> stay safe (blocking)
    assert probe(0, 20) == UNDECIDED  # too few rows to decide


def test_decimal_dash_means_zero_is_opt_in():
    from election_ocr.parsing.numbers import DecimalFormat, ParseStatus, parse_decimal

    assert parse_decimal("-", DecimalFormat(decimals=1)).value is None
    r = parse_decimal("–", DecimalFormat(decimals=1, dash_means_zero=True))
    assert r.status == ParseStatus.DASH and r.value == "0"


def test_sum_to_100_rounding_is_accepted_only_when_the_probe_holds_and_the_row_adds_up():
    from election_ocr.validation.rules import follows_sum100, share_row_rounding

    # WK 163 E 1961 (CDU 32.16 -> printed 32.1 so that the row adds up to 100.0)
    absolute = rv(
        "erst",
        metrics={"valid_votes": fvx("valid_votes", 182236)},
        parties={
            "cdu": fvx("party:cdu", 58608),
            "spd": fvx("party:spd", 68994),
            "fdp": fvx("party:fdp", 44795),
            "gdp": fvx("party:gdp", 2761),
            "dfu": fvx("party:dfu", 5678),
            "dg": fvx("party:dg", 0),
            "drp": fvx("party:drp", 1400),
        },
    )
    printed = {"cdu": "32.1", "spd": "37.9", "fdp": "24.6", "gdp": "1.5", "dfu": "3.1", "dg": "0.0", "drp": "0.8"}
    pct = rv("pct_e", "percent", "erst", parties={k: fvx("party:" + k, v, decimals=1) for k, v in printed.items()})
    defs = printout_profile().percent_definitions
    assert share_row_rounding(pct, absolute, defs) == "sum100" and follows_sum100(pct, absolute, defs) is True
    assert [i.code for i in percent_checks(pct, absolute, defs, F)] == [IssueCode.PERCENT_MISMATCH]
    assert [i.code for i in percent_checks(pct, absolute, defs, F, sum100=True)] == [IssueCode.PERCENT_ROUNDING_ADJUSTED]
    # a misread share (row no longer adds up to 100) stays a mismatch even with the probe
    pct.parties["fdp"] = fvx("party:fdp", "24.5", decimals=1)
    assert share_row_rounding(pct, absolute, defs) == "other"
    assert [i.code for i in percent_checks(pct, absolute, defs, F, sum100=True)] == [IssueCode.PERCENT_MISMATCH]
