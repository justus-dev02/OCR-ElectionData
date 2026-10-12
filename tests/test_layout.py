"""Structure + token tests on synthetic page images (no ML models)."""

import numpy as np

from election_ocr.geometry import Box
from election_ocr.layout.structure import (
    VerticalLine,
    add_ml_confirmed_weak_lines,
    compare_columns,
    deterministic_columns,
    find_vertical_lines,
)
from election_ocr.layout.tokens import Token, build_token_layout, connected_components, merge_spanning_labels
from election_ocr.preprocessing.image_ops import separate_ruling_lines

H, W = 800, 700
RULES = [100, 300, 400, 500, 600]  # label | marker | c2 | c3 | c4 (outer right border at 600)


def synthetic_table(lines_y=(200, 228, 256, 330, 358, 386), faint_rule=None):
    ink = np.zeros((H, W), dtype=bool)
    for x in RULES:
        if x == faint_rule:
            for y in range(150, 750, 60):  # dashed = faint
                ink[y : y + 10, x : x + 3] = True
        else:
            ink[150:750, x : x + 3] = True
    for y in lines_y:
        ink[y : y + 20, 120:200] = True  # label text
        ink[y : y + 20, 340:352] = True  # marker glyph
        ink[y : y + 20, 440:450] = True  # digits col 2
        ink[y : y + 20, 455:465] = True
        ink[y : y + 20, 560:572] = True  # digits col 3... (inside col 3: 500-600)
    return ink


def columns_for(ink):
    masks = separate_ruling_lines(ink, 300)
    region = Box(x0=50, y0=140, x1=650, y1=760)
    vl = find_vertical_lines(masks.vertical, region, 0.25)
    body = Box(x0=50, y0=160, x1=650, y1=740)
    return masks, vl, body, deterministic_columns(masks.text, vl, body)


def test_ruling_lines_define_columns_and_outer_columns_need_text():
    ink = synthetic_table()
    ink[700:703, 60:63] = True  # speck left of the first rule -> must not become a column
    _, vl, _, cols = columns_for(ink)
    assert [l.x for l in vl] == [101, 301, 401, 501, 601]
    assert len(cols) == 4  # label, marker, c2, c3 (c4 region right of 600 has no text)


def test_compare_columns_tolerates_label_over_split_only():
    _, _, _, cols = columns_for(synthetic_table())
    types = ["label", "marker", "integer", "integer"]
    ml = [
        Box(x0=110, y0=0, x1=150, y1=1),
        Box(x0=160, y0=0, x1=290, y1=1),
        Box(x0=310, y0=0, x1=395, y1=1),
        Box(x0=410, y0=0, x1=495, y1=1),
        Box(x0=510, y0=0, x1=595, y1=1),
    ]
    assert compare_columns(cols, ml, types, [], 1.0).agree
    ml_bad = [
        Box(x0=110, y0=0, x1=290, y1=1),
        Box(x0=310, y0=0, x1=395, y1=1),
        Box(x0=410, y0=0, x1=450, y1=1),
        Box(x0=455, y0=0, x1=495, y1=1),
        Box(x0=510, y0=0, x1=595, y1=1),
    ]
    assert not compare_columns(cols, ml_bad, types, [], 1.0).agree  # split inside a numeric column


def test_compare_columns_detects_glyph_between_boundaries():
    _, _, _, cols = columns_for(synthetic_table())
    types = ["label", "marker", "integer", "integer"]
    ml = [
        Box(x0=110, y0=0, x1=290, y1=1),
        Box(x0=310, y0=0, x1=440, y1=1),
        Box(x0=445, y0=0, x1=495, y1=1),
        Box(x0=510, y0=0, x1=595, y1=1),
    ]
    glyph = Box(x0=410, y0=200, x1=418, y1=220)  # between rule 400 and ML boundary ~442
    res = compare_columns(cols, ml, types, [glyph], 1.0)
    assert not res.agree


def test_spanning_heading_glyphs_are_ignored():
    _, _, _, cols = columns_for(synthetic_table())
    types = ["label", "marker", "integer", "integer"]
    ml = [
        Box(x0=110, y0=0, x1=280, y1=1),
        Box(x0=290, y0=0, x1=395, y1=1),
        Box(x0=410, y0=0, x1=495, y1=1),
        Box(x0=510, y0=0, x1=595, y1=1),
    ]
    # heading "...sungen" crosses the rule at x=300: glyphs on both sides, same line
    left = Box(x0=286, y0=170, x1=296, y1=190)
    right = Box(x0=306, y0=170, x1=316, y1=190)
    assert compare_columns(cols, ml, types, [left, right], 1.0).agree


def test_faint_rule_accepted_only_with_ml_support():
    strong = [VerticalLine(x=101, y0=0, y1=1), VerticalLine(x=301, y0=0, y1=1)]
    weak = [VerticalLine(x=401, y0=0, y1=1, coverage=0.1)]
    ml_with = [Box(x0=110, y0=0, x1=290, y1=1), Box(x0=310, y0=0, x1=398, y1=1), Box(x0=404, y0=0, x1=495, y1=1)]
    out = add_ml_confirmed_weak_lines(strong, weak, ml_with, [], max_offset_px=18)
    assert [l.evidence for l in out] == ["strong", "strong", "weak+ml"]
    ml_without = [Box(x0=110, y0=0, x1=290, y1=1), Box(x0=310, y0=0, x1=495, y1=1)]
    assert len(add_ml_confirmed_weak_lines(strong, weak, ml_without, [], max_offset_px=18)) == 2


def test_tokens_anchor_lines_and_assignment():
    ink = synthetic_table()
    masks, vl, body, cols = columns_for(ink)
    labels, comps, dropped = connected_components(masks.text, body, 12)
    lay = build_token_layout(
        labels,
        comps,
        cols,
        anchor_column=1,
        token_prefix="t",
        min_component_height=6,
        max_offset_ratio=0.65,
        min_margin_ratio=0.25,
        confirm_column=2,
        label_column=0,
    )
    assert len(lay.lines) == 6 and lay.pitch == 28
    assigned = [t for t in lay.tokens if t.assignment_status == "ASSIGNED"]
    # every line: label, marker, col2 (two glyphs -> one token), col3
    assert len(assigned) == 24
    two_glyph = [t for t in assigned if t.column_index == 2]
    assert all(len(t.component_labels) == 2 for t in two_glyph)


def test_anchor_confirmation_rejects_heading_spill():
    ink = synthetic_table()
    ink[165:180, 330:345] = True  # heading spill into the marker column, no digits in col 2
    masks, vl, body, cols = columns_for(ink)
    labels, comps, _ = connected_components(masks.text, body, 12)
    lay = build_token_layout(labels, comps, cols, 1, "t", 6, 0.65, 0.25, confirm_column=2, label_column=0)
    assert len(lay.lines) == 6 and lay.notes["rejected_anchor_clusters"] == 1


def test_merge_spanning_labels():
    head = Token("a", 0, Box(x0=120, y0=170, x1=295, y1=190), [1], assignment_status="UNASSIGNED")
    tail = Token("b", 1, Box(x0=306, y0=171, x1=330, y1=189), [2], assignment_status="UNASSIGNED")
    other = Token("c", 2, Box(x0=440, y0=200, x1=460, y1=220), [3], assignment_status="ASSIGNED")
    out, n = merge_spanning_labels([head, tail, other], 0, 20.0)
    assert n == 1 and len(out) == 2 and out[0].box.x1 == 330 and out[0].component_labels == [1, 2]


def test_token_ids_are_unique_even_for_colliding_loose_tokens():
    ink = synthetic_table()
    # two loose marks at identical y in the same column (one far, one near a line)
    ink[300:306, 440:446] = True
    ink[300:306, 470:476] = True
    masks, vl, body, cols = columns_for(ink)
    labels, comps, _ = connected_components(masks.text, body, 12)
    lay = build_token_layout(labels, comps, cols, 1, "t", 6, 0.65, 0.25, confirm_column=2, label_column=0)
    ids = [t.token_id for t in lay.tokens]
    assert len(ids) == len(set(ids))
