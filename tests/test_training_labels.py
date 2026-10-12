"""Label derivation for OCR fine-tuning (training/labels.py) on a minimal run database."""

from __future__ import annotations

import json
import pickle
import sqlite3

import cv2
import numpy as np

from election_ocr.training.labels import collect_samples, is_heldout

RUN = "run_x"


def _db(tmp_path, cells, rows, issues=()):
    db = tmp_path / "run.db"
    con = sqlite3.connect(db)
    con.executescript(
        """
        create table processing_runs (run_id text, document_id text, status text);
        create table cells (cell_id text, run_id text, status text, value text, value_type text,
                            supporting_groups text, dissenting_groups text);
        create table ocr_candidates (cell_id text, run_id text, independence_group text, variant text,
                                     raw_text text, parsed_value text);
        create table result_rows (result_id text, run_id text, page_number int, role_kind text,
                                  field_cells text, party_cells text);
        create table issues (run_id text, result_id text, code text);
        """
    )
    con.execute("insert into processing_runs values (?, 'doc', 'COMPLETED')", (RUN,))
    for cid, status, value, sup, dis, cands in cells:
        con.execute(
            "insert into cells values (?,?,?,?,?,?,?)", (cid, RUN, status, value, "integer", json.dumps(sup), json.dumps(dis))
        )
        for g, var, raw, parsed in cands:
            con.execute("insert into ocr_candidates values (?,?,?,?,?,?)", (cid, RUN, g, var, raw, parsed))
    for rid, fields, parties in rows:
        con.execute(
            "insert into result_rows values (?,?,?,?,?,?)", (rid, RUN, 3, "absolute", json.dumps(fields), json.dumps(parties))
        )
    for rid, code in issues:
        con.execute("insert into issues values (?,?,?)", (RUN, rid, code))
    con.commit()
    crops = tmp_path / "crops"
    crops.mkdir()
    png = cv2.imencode(".png", np.full((20, 30), 255, np.uint8))[1].tobytes()
    with open(crops / "t.pkl", "wb") as fh:
        pickle.dump([dict(token_id=c[0], png=png, glyph_height=20.0, shape="wide", allowlist="0123456789-") for c in cells], fh)
    return db, crops


def _agreed(cid, value):
    return (
        cid,
        "AGREED",
        value,
        ["easyocr", "tesseract"],
        [],
        [
            ("tesseract", "whitelist_psm7", value, value),
            ("tesseract", "free_psm7", value, value),
            ("easyocr", "allowlist", value, value),
        ],
    )


def _tess_dissent(cid, value, tess_raw):
    return (
        cid,
        "DISAGREED",
        None,
        ["easyocr", "pdf_text_layer"],
        ["tesseract"],
        [
            ("tesseract", "whitelist_psm7", tess_raw, tess_raw),
            ("easyocr", "allowlist", value, value),
            ("pdf_text_layer", "words_in_token", value, value),
        ],
    )


def test_tier_a_and_proven_tier_b(tmp_path):
    cells = [_agreed("v", "30"), _agreed("p1", "12"), _tess_dissent("p2", "18", "16")]
    db, crops = _db(tmp_path, cells, [("r1", {"valid_votes": "v"}, {"a": "p1", "b": "p2"})])
    samples, _ = collect_samples(db, [crops], "tesseract", {"A", "B"})
    by = {s.token_id: s for s in samples}
    assert by["v"].tier == "A" and by["p1"].tier == "A"
    assert by["p2"].tier == "B" and by["p2"].text == "18"  # label from the two other groups


def test_tier_b_requires_exact_party_sum(tmp_path):
    cells = [_agreed("v", "31"), _agreed("p1", "12"), _tess_dissent("p2", "18", "16")]
    db, crops = _db(tmp_path, cells, [("r1", {"valid_votes": "v"}, {"a": "p1", "b": "p2"})])
    samples, stats = collect_samples(db, [crops], "tesseract", {"A", "B"})
    assert "p2" not in {s.token_id for s in samples}
    assert stats["status_DISAGREED"] == 1


def test_tier_b_off_by_default_and_contradicted_rows_excluded(tmp_path):
    cells = [_agreed("v", "30"), _agreed("p1", "12"), _tess_dissent("p2", "18", "16")]
    db, crops = _db(tmp_path, cells, [("r1", {"valid_votes": "v"}, {"a": "p1", "b": "p2"})])
    samples, _ = collect_samples(db, [crops], "tesseract")
    assert {s.tier for s in samples} == {"A"}
    other = tmp_path / "contradicted"
    other.mkdir()
    db2, crops2 = _db(other, cells, [("r1", {"valid_votes": "v"}, {"a": "p1", "b": "p2"})], issues=[("r1", "PARTY_SUM_MISMATCH")])
    samples2, _ = collect_samples(db2, [crops2], "tesseract", {"A", "B"})
    assert samples2 == []


def test_split_is_deterministic_per_page():
    assert is_heldout("doc", 7, 20) == is_heldout("doc", 7, 20)
    share = sum(is_heldout("doc", p, 20) for p in range(2000)) / 2000
    assert 0.15 < share < 0.25


def _sample(token_id: str, doc: str, page: int, tier: str = "A"):
    import numpy as np

    from election_ocr.training.labels import Sample

    return Sample(token_id, doc, page, "integer", "1", "1", tier, np.zeros((2, 2), np.uint8), 20.0, None, None)


def test_split_samples_is_page_level_deterministic_and_capped():
    from election_ocr.training.labels import is_heldout, split_samples

    samples = [_sample(f"t{d}{p}{i}", f"doc{d}", p) for d in range(2) for p in range(30) for i in range(3)]
    samples += [_sample(f"b{p}", "doc0", p, tier="B") for p in range(30)]
    train, held = split_samples(samples, heldout_percent=20)
    assert {(s.document_id, s.page) for s in train}.isdisjoint({(s.document_id, s.page) for s in held})
    assert all(is_heldout(s.document_id, s.page, 20) for s in held)
    rev_train, rev_held = split_samples(list(reversed(samples)), 20)
    assert {s.token_id for s in rev_train} == {s.token_id for s in train} and {s.token_id for s in rev_held} == {
        s.token_id for s in held
    }
    capped, _ = split_samples(samples, 20, max_samples_per_document=5)
    per_doc_a = {d: sum(1 for s in capped if s.document_id == d and s.tier == "A") for d in ("doc0", "doc1")}
    assert per_doc_a == {"doc0": 5, "doc1": 5}
    assert sum(1 for s in capped if s.tier == "B") == sum(1 for s in train if s.tier == "B")  # tier B is never capped
    _, small_held = split_samples(samples, 20, max_heldout=4)
    hard = [s for s in held if s.tier != "A"]
    assert len(small_held) == max(4, len(hard)) and all(s in small_held for s in hard)
    # the subsample does not depend on the input order
    _, reordered = split_samples(list(reversed(samples)), 20, max_heldout=4)
    assert {s.token_id for s in reordered} == {s.token_id for s in small_held}
