"""End-to-end flow on synthetic evidence: validation -> gate -> review -> publish -> export -> evaluate."""

import json

import pytest
from conftest import make_reference

from election_ocr.evaluation.gold import GoldPage, evaluate_run
from election_ocr.exports import writers
from election_ocr.geometry import Box
from election_ocr.issues import IssueCode, IssueFactory
from election_ocr.layout.tokens import AnchorLine
from election_ocr.ocr.resolver import Candidate, CellResolution, CellStatus
from election_ocr.parsing.numbers import ParseResult, ParseStatus
from election_ocr.persistence.db import PublishedResult, SourceDocument, make_engine, make_session
from election_ocr.persistence.repository import Repository
from election_ocr.pipeline.document_service import DocumentService
from election_ocr.pipeline.models import CellRecord, LabelReading, PageRecord, TableRecord
from election_ocr.pipeline.state_machine import ResultStatus
from election_ocr.review.service import ReviewError, ReviewService
from election_ocr.semantics.rows import TableInterpreter, empty_context

VALUES = {  # label, marker -> eligible, invalid, valid, pa, pb
    ("1 Alpha", "1"): (100, 5, 90, 50, 40),
    ("1 Alpha", "2"): (100, 4, 91, 51, 40),
    ("2 Beta", "1"): (200, 10, 180, 100, 80),
    ("2 Beta", "2"): (200, 9, 181, 101, 80),
    ("Zusammen", "1"): (300, 15, 270, 150, 120),
    ("Zusammen", "2"): (300, 13, 272, 152, 120),
}


def _res(value, status=CellStatus.AGREED):
    cands = [
        Candidate(
            engine=g,
            engine_version="t",
            variant="v",
            independence_group=g,
            raw_text=str(value),
            confidence=0.9,
            parse=ParseResult(raw=str(value), normalized=str(value), status=ParseStatus.OK, value=str(value), decimals=0),
            role="support",
        )
        for g in ("tesseract", "easyocr")
    ]
    return CellResolution(
        status=status,
        value=str(value) if status == CellStatus.AGREED else None,
        decimals=0,
        supporting_groups=["easyocr", "tesseract"],
        candidates=cands,
    )


def build_page(run_id, image_path, disagree=None):
    ref = make_reference()
    tp = ref.profiles["test_profile"].tables[0]
    var = tp.variants[0]
    table_id = f"{run_id}_p0001_t0"
    cells, lines = [], []
    y = 100
    labels = {}
    for li, ((label, marker), vals) in enumerate(VALUES.items()):
        lines.append(AnchorLine(index=li, cy=y + 10, y0=y, y1=y + 20, box=Box(x0=0, y0=y, x1=1, y1=y + 20)))
        if marker == "1":
            labels[li] = label
            cells.append(
                CellRecord(
                    cell_id=f"{table_id}_c00_l{li:03d}",
                    table_id=table_id,
                    column_index=0,
                    column_key="label",
                    value_type="label",
                    line_index=li,
                    box=[10, y, 90, y + 20],
                    box_pdf=[0, 0, 0, 0],
                    assignment_status="ASSIGNED",
                    resolution=CellResolution(status=CellStatus.AGREED, value=label.lower()),
                    labels=[LabelReading(engine=g, independence_group=g, text=label) for g in ("tesseract", "easyocr")],
                )
            )
        cells.append(
            CellRecord(
                cell_id=f"{table_id}_c01_l{li:03d}",
                table_id=table_id,
                column_index=1,
                column_key="marker",
                value_type="marker",
                line_index=li,
                box=[100, y, 110, y + 20],
                box_pdf=[0, 0, 0, 0],
                assignment_status="ASSIGNED",
                resolution=_res(marker),
            )
        )
        for ci, v in enumerate(vals, start=2):
            status = CellStatus.DISAGREED if disagree == (label, marker, ci) else CellStatus.AGREED
            cells.append(
                CellRecord(
                    cell_id=f"{table_id}_c{ci:02d}_l{li:03d}",
                    table_id=table_id,
                    column_index=ci,
                    column_key=var.columns[ci].key,
                    value_type="integer",
                    line_index=li,
                    box=[100 + 50 * ci, y, 140 + 50 * ci, y + 20],
                    box_pdf=[0, 0, 0, 0],
                    assignment_status="ASSIGNED",
                    resolution=_res(v, status),
                )
            )
        y += 28 if marker == "1" else 60
    header = CellRecord(
        cell_id=f"{table_id}_c00_u00060",
        table_id=table_id,
        column_index=0,
        column_key="label",
        value_type="label",
        line_index=None,
        box=[10, 60, 200, 80],
        box_pdf=[0, 0, 0, 0],
        assignment_status="UNASSIGNED",
        resolution=CellResolution(status=CellStatus.AGREED),
        labels=[LabelReading(engine=g, independence_group=g, text="Landkreis Xstadt") for g in ("tesseract", "easyocr")],
    )
    cells.append(header)
    interp = TableInterpreter(tp, var, 2, IssueFactory()).interpret(table_id, 1, lines, 28.0, cells, empty_context())
    rows, headers, _, issues = interp.rows, interp.headers, interp.carry, interp.issues
    table = TableRecord(
        table_id=table_id,
        page_number=1,
        table_profile_id="t1",
        variant_id="v1",
        detection_score=0.99,
        region=[0, 0, 600, 600],
        header_bottom_y=50,
        body=[0, 50, 600, 600],
        columns=[],
        ml_columns=[],
        cells=cells,
        rows=rows,
        section_headers=headers,
        issues=issues,
    )
    return PageRecord(
        page_id=f"{run_id}_p0001",
        page_number=1,
        width_px=600,
        height_px=600,
        dpi=300,
        image_path=str(image_path),
        image_sha256="x",
        deskew_angle=0.0,
        deskew_applied=False,
        has_text_layer=False,
        tables=[table],
    )


@pytest.fixture
def env(tmp_path, cfg):
    import cv2
    import numpy as np

    img = tmp_path / "page.png"
    cv2.imwrite(str(img), np.full((600, 600), 255, np.uint8))
    s = make_session(make_engine(cfg.database_url))
    s.add(
        SourceDocument(
            document_id="doc_a",
            sha256="a" * 64,
            filename="a.pdf",
            path="/x/a.pdf",
            size_bytes=1,
            page_count=1,
            text_layer_pages=0,
            renderer_version="t",
        )
    )
    s.commit()
    return s, cfg, img


def run_pipeline(env, ref, disagree=None):
    s, cfg, img = env
    repo = Repository(s)
    run = repo.create_run("doc_a", "test_profile", {}, "f", ref.snapshot_hash, [1])
    repo.save_page(run.run_id, "doc_a", build_page(run.run_id, img, disagree))
    svc = DocumentService(repo, ref, cfg, IssueFactory())
    summary = svc.validate_run(run.run_id)
    published = svc.publish_run(run.run_id)
    return repo, svc, run.run_id, summary, published


def test_verified_clean_data_is_published_with_provenance(env):
    ref = make_reference(verified=True)
    repo, _, run_id, summary, published = run_pipeline(env, ref)
    assert summary.needs_review == 0 and published == 6
    pub = env[0].query(PublishedResult).all()
    alpha = next(p for p in pub if p.unit_id == "g_alpha" and p.vote_type == "ERSTSTIMME")
    assert (alpha.eligible_voters, alpha.valid_votes) == (100, 90)
    assert alpha.provenance["fields"]["party:pa"]["source"] == "ocr" and alpha.provenance["fields"]["party:pa"]["cell_id"]
    total = next(p for p in pub if p.unit_id == "lk_x" and p.vote_type == "ZWEITSTIMME")
    assert total.unit_level == "LANDKREIS" and total.valid_votes == 272


def test_unverified_reference_blocks_everything(env):
    repo, _, run_id, summary, published = run_pipeline(env, make_reference(verified=False))
    assert published == 0 and summary.validated == 0
    assert summary.issue_counts[IssueCode.REFERENCE_UNVERIFIED.value] >= 6


def test_ocr_disagreement_blocks_row_and_its_section_sum_until_review(env, cfg):
    ref = make_reference(verified=True)
    repo, svc, run_id, summary, published = run_pipeline(env, ref, disagree=("1 Alpha", "1", 5))  # party pa
    rows = {(r.unit_id, r.marker): r for r in repo.rows(run_id)}
    assert rows[("g_alpha", "1")].status == ResultStatus.NEEDS_REVIEW.value
    # members of role 1 cannot be verified against the county total -> blocked as well
    assert rows[("g_beta", "1")].status == ResultStatus.NEEDS_REVIEW.value
    assert published == 3  # role 2 rows
    review = ReviewService(repo, ref, cfg)
    case = next(c for c in review.list_cases() if c.result_id == rows[("g_alpha", "1")].result_id)
    with pytest.raises(ReviewError):
        review.set_value(case.case_id, "party:pa", "5O", "tester", "reading from source")  # confusion not accepted
    with pytest.raises(ReviewError):
        review.set_value(case.case_id, "party:pa", "50", "tester", "  ")  # reason mandatory
    review.set_value(case.case_id, "party:pa", "50", "tester", "read from page image")
    result = review.approve(case.case_id, "tester", "checked against scan")
    assert result["status"] == ResultStatus.PUBLISHED.value and not result["remaining_blocking"]
    pub = env[0].query(PublishedResult).filter_by(unit_id="g_alpha", vote_type="ERSTSTIMME").one()
    assert pub.provenance["fields"]["party:pa"]["source"] == "human"
    assert env[0].query(PublishedResult).count() == 6


def test_publication_gate_defends_against_inconsistent_state(env):
    repo, svc, run_id, _, _ = run_pipeline(env, make_reference(verified=False))
    row = repo.rows(run_id)[0]
    row.status = ResultStatus.VALIDATED.value  # simulate a bug / manual tampering
    env[0].commit()
    with pytest.raises(RuntimeError, match="publication gate"):
        svc.publish_run(run_id)


def test_exports_contain_only_published_data(env, cfg):
    ref = make_reference(verified=True)
    repo, _, run_id, _, _ = run_pipeline(env, ref)
    s = env[0]
    ev = json.loads(writers.evidence_json(s, run_id, cfg.paths.exports_dir / "e.json").read_text())
    assert ev["results"] and any(c["candidates"] for c in ev["pages"][0]["tables"][0]["cells"])
    canon = json.loads(writers.canonical_json(s, cfg.paths.exports_dir / "c.json").read_text())
    assert len(canon["published_results"]) == 6
    md = writers.markdown(s, ref, cfg.paths.exports_dir / "p.md").read_text()
    assert "Alpha" in md and "| 100 |" in md
    rows = writers.flat_records(s)
    assert len(rows) == 12 and {r["party_id"] for r in rows} == {"pa", "pb"}


def test_evaluation_counts_false_accepts(env):
    ref = make_reference(verified=True)
    repo, _, run_id, _, _ = run_pipeline(env, ref)
    gold = GoldPage(
        document_sha256="a" * 64,
        page=1,
        verified_by="tester",
        verified_at="2026-09-29",
        rows=[
            {"label_index": "1", "row_kind": "DATA", "marker": "1", "values": {"metric:valid_votes": 90, "party:pa": 51}},
            {"label_index": None, "row_kind": "SECTION_TOTAL", "marker": "2", "values": {"metric:valid_votes": "272"}},
        ],
    )
    rep = evaluate_run(env[0], run_id, [gold], ref)
    assert rep.matched_rows == 2 and rep.agreed_correct == 2 and rep.agreed_wrong == 1
    assert rep.published_wrong == 1 and rep.false_accept_rate == pytest.approx(1 / 3)
