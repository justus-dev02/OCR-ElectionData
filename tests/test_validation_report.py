"""Validation tiers, section sums, provisional export and the external-data loaders."""

from __future__ import annotations

import pandas as pd
from conftest import make_reference
from test_document_flow import env, run_pipeline  # noqa: F401  (env is a fixture)

from election_ocr.evaluation.external import _int
from election_ocr.evaluation.validation_report import latest_runs, row_tier, to_markdown, validate_document
from election_ocr.exports.provisional import agreed_text, provisional_csv
from election_ocr.persistence.db import ProcessingRun


def test_row_tiers_by_remaining_blockers():
    assert row_tier(set()) == "reference_only"
    assert row_tier({"REFERENCE_UNVERIFIED", "REFERENCE_DATA_MISSING"}) == "reference_only"
    assert row_tier({"REFERENCE_UNVERIFIED", "HIERARCHY_UNVERIFIED"}) == "data_verified"
    assert row_tier({"HIERARCHY_UNVERIFIED", "OCR_ENGINE_DISAGREEMENT"}) == "needs_review"


def _complete(session, run_id):
    run = session.get(ProcessingRun, run_id)
    run.status = "COMPLETED"
    run.stats = {"pages_processed": 1, "page_errors": {}}
    session.commit()
    return run


def test_unverified_reference_data_leaves_clean_rows_reference_only(env):  # noqa: F811
    session = env[0]
    _, _, run_id, summary, published = run_pipeline(env, make_reference(verified=False))
    assert published == 0
    _complete(session, run_id)
    (run,) = latest_runs(session)
    v = validate_document(session, run)
    assert v.rows == 6 and v.tiers == {"reference_only": 6, "data_verified": 0, "needs_review": 0}
    assert v.section_totals == 2 and v.section_totals_verified == 2 and v.section_totals_mismatch == 0
    assert v.numeric_cells["AGREED"] > 0 and v.agreed_share == 1.0
    assert "| a.pdf |" in to_markdown([v])


def test_ocr_dissent_moves_row_and_its_section_to_review_tiers(env):  # noqa: F811
    session = env[0]
    _, _, run_id, _, _ = run_pipeline(env, make_reference(verified=False), disagree=("1 Alpha", "1", 5))
    _complete(session, run_id)
    v = validate_document(session, latest_runs(session)[0])
    # Alpha/1 disagrees; Beta/1 and the county total/1 can no longer verify their sum
    assert v.tiers["needs_review"] == 1 and v.tiers["data_verified"] == 2 and v.tiers["reference_only"] == 3
    assert v.section_totals_verified == 1
    assert v.top_blockers["OCR_ENGINE_DISAGREEMENT"] == 1


def test_provisional_export_contains_only_verified_rows_marked_unpublished(env, tmp_path):  # noqa: F811
    session, cfg, _ = env
    ref = make_reference(verified=False)
    _, _, run_id, _, _ = run_pipeline(env, ref, disagree=("1 Alpha", "1", 5))
    _complete(session, run_id)
    (path,) = provisional_csv(session, ref, cfg, tmp_path / "prov", "reference_only")
    df = pd.read_csv(path)
    assert set(df["status"]) == {"UNPUBLISHED"} and set(df["validation_tier"]) == {"reference_only"}
    assert len(df) == 3 and "party_pa" in df.columns
    (path,) = provisional_csv(session, ref, cfg, tmp_path / "prov2", "data_verified")
    assert len(pd.read_csv(path)) == 5


def test_agreed_text_needs_two_identical_readings():
    assert agreed_text({"tesseract": "Alpha", "easyocr": "Alpha"}) == "Alpha"
    assert agreed_text({"tesseract": "Alpha", "easyocr": "Aipha"}) is None
    assert agreed_text({"tesseract": "Alpha"}) is None


def test_official_number_parsing():
    assert _int("x") == 0 and _int("1 234") == 1234 and _int(None) is None and _int(17) == 17
