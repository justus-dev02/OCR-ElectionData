"""DocumentRunner orchestration: page order, error handling and sequential == parallel (no OCR models).

The extraction worker is replaced by a fake that fails on every third page and returns a
stopped (unreadable) table otherwise, so the run exercises ordering, error recording,
context breaking and persistence in both execution modes.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest
from conftest import profile_dict
from sqlalchemy import select

from election_ocr.issues import IssueCode, IssueFactory, IssueScope
from election_ocr.persistence.db import Page, ProcessingRun, TableRow, make_engine, make_session
from election_ocr.pipeline.extraction import ExtractedTable
from election_ocr.pipeline.models import PageRecord, TableRecord
from election_ocr.pipeline.page_processor import PageExtraction
from election_ocr.pipeline.runner import DocumentRunner, NoProfileError, PageJob, PageJobResult
from election_ocr.reference.models import DocumentProfile
from election_ocr.reference.repository import ReferenceRepository


class FakeWorker:
    """Same constructor as PageWorker; importable by spawned worker processes (tests/ is on sys.path)."""

    def __init__(self, cfg, profile, pdf_path, out_dir, producer):
        self.issues = IssueFactory()

    def extract(self, job: PageJob) -> PageJobResult:
        if job.page_number % 3 == 0:
            return PageJobResult(job.page_number, None, "RuntimeError: unreadable scan", {"fake": "1"}, {})
        page = PageRecord(
            page_id=job.page_id,
            page_number=job.page_number,
            width_px=10,
            height_px=10,
            dpi=300,
            image_path="x.png",
            image_sha256="0" * 64,
            deskew_angle=0.0,
            deskew_applied=False,
            has_text_layer=False,
        )
        table = TableRecord(
            table_id=f"{job.page_id}_t0",
            page_number=job.page_number,
            table_profile_id=job.table_profile_id,
            variant_id=job.variant_id,
            detection_score=0.0,
            region=[],
            header_bottom_y=None,
            body=[],
            columns=[],
            ml_columns=[],
            issues=[self.issues.make(IssueCode.TABLE_NOT_DETECTED, IssueScope.PAGE, "fake: no table")],
        )
        page.tables.append(table)
        return PageJobResult(
            job.page_number, PageExtraction(pages=(page,), parts=[ExtractedTable(table)]), None, {"fake": "1"}, {}
        )

    def close(self) -> None:
        pass


@pytest.fixture
def pdf_and_reference(tmp_path):
    from election_ocr.ingestion.registry import sha256_of

    pdf = tmp_path / "doc.pdf"
    doc = pymupdf.open()
    for _ in range(7):
        doc.new_page()
    doc.save(pdf)
    profile = profile_dict(verified=False)
    profile["sha256"] = [sha256_of(pdf)]
    profile["tables"][0]["variants"][0]["pages"] = "1-7"
    from conftest import make_reference

    base = make_reference(verified=False)
    ref = ReferenceRepository(list(base.parties.values()), list(base.elections.values()), [], [DocumentProfile(**profile)])
    return pdf, ref


def _run(cfg, ref, pdf: Path, workers: int, db_name: str):
    cfg = cfg.model_copy(deep=True)
    cfg.database_url = f"sqlite:///{cfg.paths.output_dir.parent / db_name}"
    session = make_session(make_engine(cfg.database_url))
    progress = []
    res = DocumentRunner(cfg, session, ref, worker_cls=FakeWorker).run(
        pdf, progress=lambda n, total, p: progress.append((n, total, p)), workers=workers
    )
    pages = [p.page_number for p in session.execute(select(Page).order_by(Page.page_number)).scalars()]
    tables = session.execute(select(TableRow)).scalars().all()
    run = session.get(ProcessingRun, res.run_id)
    return res, progress, pages, tables, run


def test_sequential_run_records_errors_and_keeps_page_order(cfg, pdf_and_reference):
    pdf, ref = pdf_and_reference
    res, progress, pages, tables, run = _run(cfg, ref, pdf, workers=1, db_name="seq.db")
    assert res.pages_processed == [1, 2, 4, 5, 7]
    assert sorted(res.page_errors) == [3, 6] and "unreadable scan" in res.page_errors[3]
    assert pages == [1, 2, 4, 5, 7] and len(tables) == 5
    assert [p for _, _, p in progress] == [1, 2, 4, 5, 7]
    assert run.status == "COMPLETED" and run.engine_versions == {"fake": "1"}
    assert res.validation is not None and res.validation.rows == 0  # no table could be read


def test_parallel_run_equals_sequential_run(cfg, pdf_and_reference):
    pdf, ref = pdf_and_reference
    seq = _run(cfg, ref, pdf, workers=1, db_name="seq.db")
    par = _run(cfg, ref, pdf, workers=3, db_name="par.db")
    assert par[0].pages_processed == seq[0].pages_processed
    assert par[0].page_errors == seq[0].page_errors
    assert par[2] == seq[2]
    assert [p for _, _, p in par[1]] == [p for _, _, p in seq[1]]  # results consumed in page order
    assert sorted(t.issues[0]["code"] for t in par[3]) == sorted(t.issues[0]["code"] for t in seq[3])


def test_runner_refuses_documents_without_profile(cfg, tmp_path):
    from conftest import make_reference

    pdf = tmp_path / "other.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(pdf)
    session = make_session(make_engine(cfg.database_url))
    with pytest.raises(NoProfileError):
        DocumentRunner(cfg, session, make_reference(), worker_cls=FakeWorker).run(pdf)


def test_pages_outside_the_profile_are_rejected(cfg, pdf_and_reference):
    pdf, ref = pdf_and_reference
    session = make_session(make_engine(cfg.database_url))
    with pytest.raises(ValueError, match="not covered"):
        DocumentRunner(cfg, session, ref, worker_cls=FakeWorker).run(pdf, pages="8")
