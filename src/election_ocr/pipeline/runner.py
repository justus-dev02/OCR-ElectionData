"""
Document runner: one processing run over one PDF.

    register (sha256)  ->  profile lookup by sha256  ->  run (config + reference snapshot)
    -> per page: render + extract (optionally in parallel worker processes)
    -> per page, in page order: interpret (section context) -> persist evidence
    -> DocumentService.validate_run -> DocumentService.publish_run

Without a document profile nothing is extracted: the runner refuses instead of
guessing which election/table layout a document contains.

Parallelism
-----------
Rendering, image analysis and OCR of a page do not depend on any other page, so
they run in a ``ProcessPoolExecutor`` (``workers > 1``). Every worker builds its
own OCR engines and table models once. Interpretation carries the section
context from page to page and therefore stays in the main process, strictly in
page order, exactly as in a sequential run. ``scripts/regression_snapshot.py``
verifies that a parallel run stores the same evidence as a sequential one.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from sqlalchemy.orm import Session

from election_ocr.config import PipelineConfig
from election_ocr.ingestion.registry import RegisteredDocument, register_document
from election_ocr.ingestion.render import render_page
from election_ocr.issues import IssueFactory
from election_ocr.layout.tatr import TATRStructureRecognizer, TATRTableDetector
from election_ocr.ocr.engines import EasyOCREngine, Engines, TesseractEngine, TextLayerEngine
from election_ocr.persistence.db import ProcessingRun
from election_ocr.persistence.repository import Repository
from election_ocr.pipeline.document_service import DocumentService, ValidationSummary
from election_ocr.pipeline.page_processor import PageExtraction, PageProcessor, interpret_extraction
from election_ocr.pipeline.state_machine import RunStatus
from election_ocr.reference.models import DocumentProfile, TableProfile, TableVariant, expand_page_ranges
from election_ocr.reference.repository import ReferenceRepository
from election_ocr.semantics.rows import SectionContext, empty_context

log = logging.getLogger("election_ocr")

ProgressCallback = Callable[[int, int, int], None]


class NoProfileError(RuntimeError):
    pass


@dataclass
class RunResult:
    run_id: str
    document_id: str
    pages_processed: List[int]
    validation: Optional[ValidationSummary]
    published: int
    page_errors: Dict[int, str] = field(default_factory=dict)


def build_engines(cfg: PipelineConfig, producer: str) -> Engines:
    o = cfg.ocr
    tess = (
        TesseractEngine(
            o.tesseract.independence_group,
            o.tesseract_lang,
            tessdata_dir=str(cfg.paths.tessdata_dir) if cfg.paths.tessdata_dir else None,
            target_glyph_px=o.tesseract_target_glyph_px,
            border_px=o.tesseract_border_px,
            psm_by_shape=o.tesseract_psm_by_shape,
            header_target_glyph_px=o.tesseract_header_target_glyph_px,
            numeric_lang=o.tesseract_numeric_lang,
            numeric_tessdata_dir=str(o.tesseract_numeric_tessdata_dir) if o.tesseract_numeric_tessdata_dir else None,
        )
        if o.tesseract.enabled
        else None
    )
    easy = (
        EasyOCREngine(
            o.easyocr.independence_group,
            o.easyocr_langs,
            str(cfg.paths.models_dir / "easyocr"),
            batch_size=o.easyocr_batch_size,
            target_glyph_px=o.easyocr_target_glyph_px,
            border_px=o.easyocr_border_px,
            border_px_by_shape=o.easyocr_border_px_by_shape,
            recognizer_weights=str(o.easyocr_recognizer_weights) if o.easyocr_recognizer_weights else None,
        )
        if o.easyocr.enabled
        else None
    )
    text = TextLayerEngine(o.text_layer.independence_group, producer) if o.text_layer.enabled else None
    return Engines(tesseract=tess, easyocr=easy, text_layer=text)


# ---------------------------------------------------------------------- page jobs
@dataclass(frozen=True)
class PageJob:
    """Extraction of one page or one spread (left page + continuation page)."""

    page_number: int
    page_id: str
    right_page_number: Optional[int]
    right_page_id: Optional[str]
    table_profile_id: str
    variant_id: str


@dataclass
class PageJobResult:
    page_number: int
    extraction: Optional[PageExtraction]
    error: Optional[str] = None
    engine_versions: Dict[str, str] = field(default_factory=dict)
    model_info: Dict[str, Dict[str, str]] = field(default_factory=dict)


class PageWorker:
    """Everything needed to extract pages of one document (one instance per process)."""

    def __init__(self, cfg: PipelineConfig, profile: DocumentProfile, pdf_path: Path, out_dir: Path, producer: str):
        self.cfg = cfg
        self.profile = profile
        self.pdf_path = pdf_path
        self.out_dir = out_dir
        self.engines = build_engines(cfg, producer)
        self.detector = TATRTableDetector(cfg.table_models)
        self.structure = TATRStructureRecognizer(cfg.table_models)
        self.processor = PageProcessor(
            cfg, self.detector, self.structure, self.engines, IssueFactory(cfg.issue_severity_overrides)
        )

    def spec(self, job: PageJob) -> Tuple[TableProfile, TableVariant]:
        tp = next(t for t in self.profile.tables if t.table_profile_id == job.table_profile_id)
        return tp, next(v for v in tp.variants if v.variant_id == job.variant_id)

    def model_info(self) -> Dict[str, Dict[str, str]]:
        return {"table_detection": self.detector.model_info(), "table_structure": self.structure.model_info()}

    def extract(self, job: PageJob) -> PageJobResult:
        tp, variant = self.spec(job)
        try:
            rendered = render_page(self.pdf_path, job.page_number, self.cfg.render, self.out_dir)
            if job.right_page_number is not None and job.right_page_id is not None:
                right = render_page(self.pdf_path, job.right_page_number, self.cfg.render, self.out_dir)
                extraction = self.processor.extract_spread((job.page_id, job.right_page_id), (rendered, right), tp, variant)
            else:
                extraction = self.processor.extract(job.page_id, rendered, tp, variant)
        except Exception as exc:
            log.error("page %s failed: %s", job.page_number, traceback.format_exc())
            return PageJobResult(
                job.page_number, None, f"{type(exc).__name__}: {exc}", self.engines.versions(), self.model_info()
            )
        return PageJobResult(job.page_number, extraction, None, self.engines.versions(), self.model_info())

    def close(self) -> None:
        self.engines.close()


# worker-process state (one PageWorker per process, created by the pool initializer)
_WORKER: Optional[PageWorker] = None


def _init_worker(
    worker_cls: type,
    cfg: PipelineConfig,
    profile: DocumentProfile,
    pdf_path: Path,
    out_dir: Path,
    producer: str,
    threads: int,
) -> None:
    global _WORKER
    import torch

    # each worker gets an equal share of the cores (avoids oversubscription with N workers)
    torch.set_num_threads(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    cfg.apply_model_cache_environment()
    _WORKER = worker_cls(cfg, profile, pdf_path, out_dir, producer)


def _extract_in_worker(job: PageJob) -> PageJobResult:
    assert _WORKER is not None, "worker not initialised"
    return _WORKER.extract(job)


def default_workers() -> int:
    """Half the CPU cores (each worker holds its own OCR/ML models, ~1.5 GB RAM), at least 1."""
    return max(1, (os.cpu_count() or 2) // 2)


# ---------------------------------------------------------------------- runner
class DocumentRunner:
    def __init__(self, cfg: PipelineConfig, session: Session, reference: ReferenceRepository, worker_cls: type = PageWorker):
        """``worker_cls`` builds the per-process extraction worker (``PageWorker``; tests inject fakes)."""
        self.worker_cls = worker_cls
        self.cfg = cfg
        self.session = session
        self.ref = reference
        self.repo = Repository(session)
        self.issues = IssueFactory(cfg.issue_severity_overrides)

    def run(
        self,
        pdf_path: Path,
        pages: Optional[str] = None,
        progress: Optional[ProgressCallback] = None,
        workers: int = 1,
    ) -> RunResult:
        """Process ``pages`` (default: all profiled pages) of one PDF with ``workers`` extraction processes."""
        self.cfg.ensure_directories()
        doc = register_document(pdf_path, self.session)
        profile = self.ref.profile_for_sha256(doc.sha256)
        if profile is None:
            raise NoProfileError(
                f"no document profile for {pdf_path.name} (sha256 {doc.sha256}). "
                "Create reference/document_profiles/<name>.yaml bound to this sha256 first."
            )
        profiled_pages = sorted({p for t in profile.tables for v in t.variants for p in expand_page_ranges(v.pages)})
        requested = self._requested_pages(profile, profiled_pages, doc, pages)
        run = self.repo.create_run(
            document_id=doc.document_id,
            document_profile_id=profile.document_profile_id,
            config_snapshot=self.cfg.snapshot(),
            config_fingerprint=self.cfg.fingerprint(),
            reference_snapshot=self.ref.snapshot_hash,
            pages_requested=requested,
        )
        self.repo.set_run_status(run, RunStatus.RUNNING)
        out_dir = self.cfg.paths.pages_dir / doc.document_id / run.run_id
        jobs = [self._job(profile, run.run_id, p) for p in requested]
        summary: Optional[ValidationSummary] = None
        published = 0
        try:
            if workers > 1:
                results = self._parallel(jobs, profile, Path(doc.path), out_dir, doc.producer, workers)
            else:
                results = self._sequential(jobs, profile, Path(doc.path), out_dir, doc.producer)
            processed, errors, versions, model_info = self._interpret_and_save(
                results, profile, profiled_pages, run, doc, len(jobs), progress
            )
            run.engine_versions = versions
            run.model_info = model_info
            run.stats = {"pages_processed": len(processed), "page_errors": errors}
            self.session.commit()
            service = DocumentService(self.repo, self.ref, self.cfg, self.issues)
            summary = service.validate_run(run.run_id)
            published = service.publish_run(run.run_id)
            run.stats = {**run.stats, "validation": summary.__dict__, "published": published}
            self.repo.set_run_status(run, RunStatus.COMPLETED)
        except Exception as exc:
            self.session.rollback()
            self.repo.set_run_status(run, RunStatus.FAILED, error=f"{type(exc).__name__}: {exc}")
            raise
        return RunResult(run.run_id, doc.document_id, processed, summary, published, errors)

    # ------------------------------------------------------------ planning
    @staticmethod
    def _requested_pages(
        profile: DocumentProfile, profiled_pages: List[int], doc: RegisteredDocument, pages: Optional[str]
    ) -> List[int]:
        if pages:
            requested = expand_page_ranges(pages)
            outside = [p for p in requested if p not in profiled_pages]
            if outside:
                raise ValueError(f"pages {outside} are not covered by profile {profile.document_profile_id}")
        else:
            requested = profiled_pages
        requested = [p for p in requested if p <= doc.page_count]
        for p in requested:
            if len(profile.table_profiles_for_page(p)) != 1:
                raise ValueError(f"page {p}: profile must define exactly one table variant per page")
            ((_, v),) = profile.table_profiles_for_page(p)
            if v.continuation and (v.continuation_page(p) or 0) > doc.page_count:
                raise ValueError(f"page {p}: continuation page {v.continuation_page(p)} beyond document end")
        clash = sorted(set(requested) & set(profile.continuation_pages()))
        if clash:
            raise ValueError(f"pages {clash} are continuation pages of a spread; request their left page instead")
        return requested

    @staticmethod
    def _job(profile: DocumentProfile, run_id: str, page_number: int) -> PageJob:
        ((tp, variant),) = profile.table_profiles_for_page(page_number)
        right = variant.continuation_page(page_number) if variant.continuation else None
        return PageJob(
            page_number=page_number,
            page_id=f"{run_id}_p{page_number:04d}",
            right_page_number=right,
            right_page_id=f"{run_id}_p{right:04d}" if right is not None else None,
            table_profile_id=tp.table_profile_id,
            variant_id=variant.variant_id,
        )

    # ------------------------------------------------------------ execution
    def _sequential(
        self, jobs: List[PageJob], profile: DocumentProfile, pdf: Path, out_dir: Path, producer: str
    ) -> Iterator[PageJobResult]:
        worker = self.worker_cls(self.cfg, profile, pdf, out_dir, producer)
        try:
            for job in jobs:
                yield worker.extract(job)
        finally:
            worker.close()

    def _parallel(
        self, jobs: List[PageJob], profile: DocumentProfile, pdf: Path, out_dir: Path, producer: str, workers: int
    ) -> Iterator[PageJobResult]:
        workers = min(workers, max(1, len(jobs)))
        threads = max(1, (os.cpu_count() or workers) // workers)
        ctx = multiprocessing.get_context("spawn")  # fork is unsafe with torch / tesseract handles
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(self.worker_cls, self.cfg, profile, pdf, out_dir, producer, threads),
        ) as pool:
            # map() yields in submission (= page) order, which interpretation requires
            yield from pool.map(_extract_in_worker, jobs)

    def _interpret_and_save(
        self,
        results: Iterator[PageJobResult],
        profile: DocumentProfile,
        profiled_pages: List[int],
        run: ProcessingRun,
        doc: RegisteredDocument,
        total: int,
        progress: Optional[ProgressCallback],
    ) -> Tuple[List[int], Dict[int, str], Dict[str, str], Dict[str, Dict[str, str]]]:
        carry: SectionContext = empty_context()
        processed: List[int] = []
        errors: Dict[int, str] = {}
        versions: Dict[str, str] = {}
        model_info: Dict[str, Dict[str, str]] = {}
        for n, res in enumerate(results, start=1):
            page_number = res.page_number
            # the section context may only be carried from the page printed directly before
            idx = profiled_pages.index(page_number)
            prev = profiled_pages[idx - 1] if idx else None
            if processed and processed[-1] != prev:
                carry = empty_context()
            versions.update(res.engine_versions)
            model_info = _merge_model_info(model_info, res.model_info)
            if res.extraction is None:
                errors[page_number] = res.error or "unknown error"
                carry = empty_context()
                continue
            ((tp, variant),) = profile.table_profiles_for_page(page_number)
            try:
                carry = interpret_extraction(res.extraction, tp, variant, carry, self.cfg.ocr, self.issues)
            except Exception as exc:
                errors[page_number] = f"{type(exc).__name__}: {exc}"
                log.error("page %s failed: %s", page_number, traceback.format_exc())
                carry = empty_context()
                continue
            for record in res.extraction.pages:
                self.repo.save_page(run.run_id, doc.document_id, record)
            processed.append(page_number)
            if progress:
                progress(n, total, page_number)
        return processed, errors, versions, model_info


def _merge_model_info(known: Dict[str, Dict[str, str]], new: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    """Keep a resolved model revision once any process has loaded the model."""
    out = dict(known)
    for key, info in new.items():
        if key not in out or out[key].get("revision") == "not-loaded":
            out[key] = info
    return out
