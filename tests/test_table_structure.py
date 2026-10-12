"""Deterministic structure recognition added 2026-09-30 (no ML models needed)."""

import numpy as np

from election_ocr.geometry import Box
from election_ocr.issues import IssueCode, IssueFactory
from election_ocr.layout.column_strategies import has_unnumbered_trailing_columns
from election_ocr.layout.structure import find_full_width_hlines
from election_ocr.layout.table_structure import (
    body_ink_runs,
    columns_from_numbers,
    columns_from_separators,
    find_column_number_row,
    find_long_hlines,
    region_from_rules,
)
from election_ocr.layout.tokens import AnchorLine, Component, assign_components_to_lines, classify_token_shape
from election_ocr.pipeline.interpretation import pair_spread_lines
from election_ocr.reference.models import ColumnSpec


def tilted_rule(mask, y, x0, x1, drift=8, thickness=3):
    """A horizontal rule drifting ``drift`` pixels over its length (slightly tilted scan)."""
    for x in range(x0, x1):
        yy = y + int(drift * (x - x0) / (x1 - x0))
        mask[yy : yy + thickness, x] = True


def test_long_rules_are_found_despite_tilt():
    m = np.zeros((600, 2000), dtype=bool)
    tilted_rule(m, 100, 100, 1900)
    tilted_rule(m, 300, 100, 1900)
    rules = find_long_hlines(m, int(0.45 * 2000))
    assert [(r.x0, r.x1) for r in rules] == [(100, 1900), (100, 1900)]
    region = region_from_rules(rules, 600, pad_px=10, side_pad_px=30)
    assert region.x0 == 70 and region.x1 == 1930 and 85 <= region.y0 < 100
    # the header-bottom test works on bands, not on single pixel rows
    assert len(find_full_width_hlines(m, Box(x0=90, y0=50, x1=1910, y1=599))) == 2


def test_column_number_row_is_the_band_of_small_isolated_groups():
    m = np.zeros((400, 1200), dtype=bool)
    m[20:40, 100:900] = True  # header text line (one wide group)
    xs = [150, 300, 450, 600, 750]
    for x in xs:
        m[100:118, x : x + 10] = True  # printed column numbers 1..5
    m[200:218, 100:1100] = True  # body line
    found = find_column_number_row(m, Box(x0=0, y0=0, x1=1199, y1=399), expected_numbers=5, char_h=18)
    assert found is not None
    (y0, y1), boxes = found
    assert (y0, y1) == (100, 118) and [int(b.x0) for b in boxes] == xs
    assert find_column_number_row(m, Box(x0=0, y0=0, x1=1199, y1=399), expected_numbers=6, char_h=18) is None


def _b(x0, x1, y=0, h=18):
    return Box(x0=x0, y0=y, x1=x1, y1=y + h)


def test_body_runs_merge_monospace_cell_gaps_and_ignore_rare_marks():
    boxes = []
    for i in range(10):  # ten text lines; two columns of 'monospace' cells with 3 px cell gaps
        y = 100 + 30 * i
        boxes += [_b(100, 113, y), _b(116, 129, y), _b(300, 313, y)]
    boxes.append(_b(180, 260, 100))  # one heading crossing the gutter (single line)
    runs = body_ink_runs(boxes, Box(x0=0, y0=90, x1=500, y1=420), line_height=27, max_noise_lines=2, merge_gap_px=8)
    assert runs == [(100.0, 130.0), (300.0, 314.0)]


def test_columns_from_numbers_with_elastic_label_and_empty_numbered_column():
    body = Box(x0=0, y0=0, x1=1000, y1=100)
    # kennziffer | name (split by word gaps) | designation | #1 | marker | #2 (empty on this page) | #3
    runs = [(50, 80), (120, 200), (230, 300), (340, 380), (440, 500), (560, 570), (800, 850)]
    numbers = [_b(480, 490), _b(690, 700), _b(830, 840)]
    numbered = [None, None, None, 1, None, 2, 3]
    elastic = [False, True, False, False, False, False, False]
    cols, findings = columns_from_numbers(runs, numbers, numbered, body, elastic=elastic)
    assert not findings and len(cols) == 7
    name = cols[1]
    assert name.x0 < 120 and name.x1 > 300 and cols[2].x0 > 300  # name absorbed 3 runs, designation = next run
    assert cols[5].x0 < 690 < cols[5].x1  # empty column located by its printed number


def test_columns_from_numbers_rejects_ambiguity():
    body = Box(x0=0, y0=0, x1=1000, y1=100)
    numbers = [_b(300, 310), _b(360, 370)]
    # one run covering both printed numbers -> the two columns cannot be separated
    cols, findings = columns_from_numbers([(50, 90), (280, 400)], numbers, [None, 1, 2], body)
    assert cols is None and findings
    # surplus run for a non-elastic unnumbered column
    cols, findings = columns_from_numbers([(50, 90), (120, 150), (300, 320), (360, 380)], numbers, [None, 1, 2], body)
    assert cols is None and "unnumbered" in findings[0].message


def test_columns_from_separators_keep_open_outer_columns():
    cols = columns_from_separators([200, 400], Box(x0=0, y0=0, x1=1000, y1=10), ink_left=50, ink_right=700, edge_margin=12)
    assert [(c.x0, c.x1) for c in cols] == [(50, 200), (200, 400), (400, 700)]


def test_trailing_unnumbered_columns():
    def num(n):
        return ColumnSpec(
            key=f"c{n}",
            role="metric",
            metric="valid_votes",
            value_type="integer",
            header_aliases=["x"],
            integer_format={},
            column_number=n,
        )

    lab = ColumnSpec(key="l", role="unit_label", value_type="label", header_aliases=["x"])
    assert not has_unnumbered_trailing_columns([lab, num(1), num(2)])
    assert has_unnumbered_trailing_columns([lab, num(1), lab.model_copy(update={"key": "tail"})])


def test_token_shape_classes():
    dash = np.zeros((30, 40), dtype=bool)
    dash[14:18, 10:28] = True
    one = np.zeros((30, 40), dtype=bool)
    one[5:25, 18:22] = True
    wide = np.zeros((30, 80), dtype=bool)
    for x in (5, 25, 45):
        wide[5:25, x : x + 12] = True
    kw = dict(glyph_height=20, dash_max_height_ratio=0.45, dash_min_aspect=1.5, narrow_max_width_ratio=1.2, min_component_area=12)
    assert classify_token_shape(dash, **kw) == "dash"
    assert classify_token_shape(one, **kw) == "narrow"
    assert classify_token_shape(wide, **kw) == "wide"


def test_decimal_comma_below_baseline_joins_its_line():
    lines = [
        AnchorLine(index=i, cy=y + 10, y0=y, y1=y + 20, box=Box(x0=0, y0=y, x1=10, y1=y + 20)) for i, y in enumerate([100, 128])
    ]
    comma = Component(label=1, box=Box(x0=50, y0=116, x1=54, y1=125), area=20, column_index=1)  # centre 120.5
    tall = Component(label=2, box=Box(x0=80, y0=114, x1=90, y1=134), area=100, column_index=1)  # centred between lines
    assign_components_to_lines([comma, tall], lines, pitch=28, max_offset_ratio=0.65, min_margin_ratio=0.25)
    assert comma.status == "ASSIGNED" and comma.line_index == 0
    assert tall.status == "AMBIGUOUS"  # large glyphs keep the distance rule


def _layout(ys, pitch):
    from election_ocr.layout.tokens import TokenLayout

    lines = [AnchorLine(index=i, cy=y, y0=y - 9, y1=y + 9, box=Box(x0=0, y0=y - 9, x1=5, y1=y + 9)) for i, y in enumerate(ys)]
    return TokenLayout(labels=np.zeros((1, 1), dtype=np.int32), tokens=[], lines=lines, pitch=pitch, char_height=18)


def test_spread_pairing_tolerates_heading_line_on_one_page_only():
    left = _layout([600, 660, 720, 870, 930], 60)
    right = _layout([480, 596, 656, 716, 866, 926], 60)  # extra heading line on top, other scan offset
    pairing, issues = pair_spread_lines(left, right, IssueFactory())
    assert pairing == {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    assert [i.code for i in issues] == [IssueCode.DETACHED_TEXT]


def test_spread_pairing_rejects_inner_mismatch():
    left = _layout([600, 660, 720, 780, 840], 60)
    right = _layout([600, 660, 765, 780, 840], 60)  # third line displaced by 3/4 of a line
    pairing, issues = pair_spread_lines(left, right, IssueFactory())
    assert pairing == {} and issues[0].code == IssueCode.PAGE_STITCH_MISMATCH


def test_short_value_run_belongs_to_its_adjacent_number_without_run():
    body = Box(x0=0, y0=0, x1=1000, y1=100)
    # '466' right-aligned under a centred '8' does not reach the number; '9' owns its own run
    numbers = [_b(300, 312), _b(450, 462)]
    runs = [(50, 120), (315, 360), (430, 500)]
    cols, findings = columns_from_numbers(runs, numbers, [None, 8, 9], body, elastic=[True, False, False])
    assert not findings and cols[1].x0 < 300 and cols[1].x1 > 360
    # both neighbours already own a run -> the stray run stays unexplained
    runs2 = [(50, 120), (290, 320), (330, 360), (430, 500)]
    cols, findings = columns_from_numbers(runs2, numbers, [None, 8, 9], body, elastic=[True, False, False])
    assert cols is None and "outside" in findings[0].message


def test_column_number_row_ignores_specks():
    m = np.zeros((300, 1200), dtype=bool)
    for x in (150, 300, 450):
        m[100:118, x : x + 10] = True
    m[105:108, 380:383] = True  # scan speck between '2' and '3'
    found = find_column_number_row(m, Box(x0=0, y0=0, x1=1199, y1=299), expected_numbers=3, char_h=18)
    assert found is not None and len(found[1]) == 3


def _comp(label, x0, y0, x1, y1, col=1):
    return Component(label=label, box=Box(x0=x0, y0=y0, x1=x1, y1=y1), area=(x1 - x0) * (y1 - y0), column_index=col)


def test_word_half_way_between_lines_is_ambiguous_as_a_whole():
    lines = [
        AnchorLine(index=i, cy=y + 9, y0=y, y1=y + 18, box=Box(x0=0, y0=y, x1=10, y1=y + 18)) for i, y in enumerate([100, 130])
    ]
    # '163' centred between the lines; the middle glyph sits slightly higher
    word = [_comp(1, 50, 114, 60, 132), _comp(2, 63, 110, 73, 128), _comp(3, 76, 114, 86, 132)]
    assign_components_to_lines(word, lines, pitch=30, max_offset_ratio=0.65, min_margin_ratio=0.25)
    assert {c.status for c in word} == {"AMBIGUOUS"} and {c.between for c in word} == {(0, 1)}


def test_star_between_lines_is_a_mark_but_not_on_a_line():
    lines = [
        AnchorLine(index=i, cy=y + 9, y0=y, y1=y + 18, box=Box(x0=0, y0=y, x1=10, y1=y + 18)) for i, y in enumerate([100, 140])
    ]
    star = _comp(1, 50, 122, 63, 133)  # 13 x 11 in the gap between the lines
    fragment = _comp(2, 80, 104, 92, 115)  # same size, but on line 0 (broken digit)
    assign_components_to_lines([star, fragment], lines, pitch=40, max_offset_ratio=0.65, min_margin_ratio=0.25)
    assert star.status == "MARK"
    assert fragment.status != "MARK"


def test_bold_star_triple_between_lines_is_a_mark_but_two_blobs_are_not():
    lines = [
        AnchorLine(index=i, cy=y + 9, y0=y, y1=y + 18, box=Box(x0=0, y0=y, x1=10, y1=y + 18)) for i, y in enumerate([100, 146])
    ]
    triple = [_comp(10 + k, 50 + 18 * k, 122, 65 + 18 * k, 138) for k in range(3)]  # 15 x 16 solid blobs
    pair = [_comp(20 + k, 150 + 18 * k, 122, 165 + 18 * k, 138) for k in range(2)]
    assign_components_to_lines(triple + pair, lines, pitch=46, max_offset_ratio=0.65, min_margin_ratio=0.25)
    assert {c.status for c in triple} == {"MARK"}
    assert "MARK" not in {c.status for c in pair}


def test_trailing_minus_protruding_into_next_column_stays_with_its_number():
    from election_ocr.layout.table_structure import ColumnInterval
    from election_ocr.layout.tokens import assign_columns

    cols = [ColumnInterval(index=0, x0=0, x1=100), ColumnInterval(index=1, x0=100, x1=200)]
    digits = [_comp(1, 70, 100, 80, 118, col=None), _comp(2, 82, 100, 92, 118, col=None)]
    minus = _comp(3, 95, 108, 108, 111, col=None)  # centre 101.5: right of the boundary
    lone_dash = _comp(4, 170, 108, 183, 111, col=None)  # printed zero in column 1
    assign_columns(digits + [minus, lone_dash], cols, char_height=18)
    assert minus.column_index == 0 and lone_dash.column_index == 1


def test_text_run_needs_a_chain_over_three_columns_without_rules():
    from election_ocr.layout.tokens import _part_of_text_run

    def glyphs(x0, n, col, label0):
        return [_comp(label0 + k, x0 + 14 * k, 100, x0 + 14 * k + 12, 118, col=col) for k in range(n)]

    # '35250' (col 3) and '1813310' (col 4) almost touching: data, not a heading
    left, anchor = glyphs(40, 5, 3, 1), glyphs(114, 7, 4, 10)
    assert not _part_of_text_run(anchor, left + anchor, max_gap=10.8)
    # a heading continuing through three columns is a run
    right = glyphs(212, 4, 5, 30)
    assert _part_of_text_run(anchor, left + anchor + right, max_gap=10.8)


def test_column_number_row_ignores_a_mark_of_other_height():
    m = np.zeros((300, 1200), dtype=bool)
    for x in (300, 450, 600):
        m[100:120, x : x + 12] = True
    m[104:117, 120:135] = True  # 15 x 13 mark in the label area of the number band
    found = find_column_number_row(m, Box(x0=0, y0=0, x1=1199, y1=299), expected_numbers=3, char_h=19)
    assert found is not None and [int(b.x0) for b in found[1]] == [300, 450, 600]
