"""Reference data, ingestion, processing runs and re-validation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click
from rich.table import Table

from election_ocr.cli.app import Ctx, cli, console, pass_ctx, print_counts


@cli.group()
def reference() -> None:
    """Reference data (elections, parties, gazetteer, document profiles)."""


@reference.command("check")
@pass_ctx
def reference_check(c: Ctx) -> None:
    """Load and cross-validate all reference data; list unverified entries."""
    ref = c.ref
    console.print(f"[green]reference data valid[/green] (snapshot {ref.snapshot_hash})")
    console.print(
        f"  parties: {len(ref.parties)}  elections: {len(ref.elections)}  units: {len(ref.units)}  profiles: {len(ref.profiles)}"
    )
    unverified = [f"party:{p.party_id}" for p in ref.parties.values() if not p.verified]
    unverified += [f"election:{e.election_id}" for e in ref.elections.values() if not e.verified]
    unverified += [f"profile:{p.document_profile_id}" for p in ref.profiles.values() if not p.verified]
    unverified_units = sum(1 for u in ref.units.values() if not u.verified)
    if unverified or unverified_units:
        suffix = f" + {unverified_units} units" if unverified_units else ""
        console.print(f"[yellow]unverified (block publication):[/yellow] {', '.join(unverified)}{suffix}")
    if not ref.units:
        console.print(
            "[yellow]gazetteer empty:[/yellow] no unit can be resolved -> nothing can be published "
            "(see reference/administrative_units/README.md)"
        )


@cli.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@pass_ctx
def ingest(c: Ctx, file: Path) -> None:
    """Register a PDF (SHA-256, metadata) without processing it."""
    from election_ocr.ingestion.registry import register_document

    doc = register_document(file, c.session)
    profile = c.ref.profile_for_sha256(doc.sha256)
    console.print(f"[green]registered[/green] {file.name}: {doc.document_id} sha256={doc.sha256}")
    console.print(f"  pages={doc.page_count} text-layer pages={doc.text_layer_pages} producer={doc.producer}")
    console.print(f"  profile: {profile.document_profile_id if profile else '[red]none - processing will be refused[/red]'}")


@cli.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--pages", default=None, help="Page ranges, e.g. '57-60,62'. Default: all profiled pages.")
@click.option(
    "--workers",
    type=click.IntRange(min=1),
    default=None,
    help="Parallel extraction processes (default: half the CPU cores; 1 = sequential).",
)
@pass_ctx
def run(c: Ctx, file: Path, pages: Optional[str], workers: Optional[int]) -> None:
    """Extract, validate and (if the gate passes) publish one document."""
    from election_ocr.pipeline.runner import DocumentRunner, NoProfileError, default_workers

    runner = DocumentRunner(c.cfg, c.session, c.ref)
    try:
        res = runner.run(
            file,
            pages=pages,
            progress=lambda n, total, p: console.print(f"  page {p} ({n}/{total})"),
            workers=workers or default_workers(),
        )
    except NoProfileError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(2)
    console.print(
        f"[bold]run {res.run_id}[/bold]: pages {len(res.pages_processed)} processed, errors {res.page_errors or 'none'}"
    )
    v = res.validation
    if v is not None:
        console.print(
            f"  rows={v.rows} validated={v.validated} needs_review={v.needs_review} "
            f"(non-publishable roles {v.non_publishable_roles}) published={res.published}"
        )
        print_counts(v.issue_counts)


@cli.command()
@pass_ctx
def runs(c: Ctx) -> None:
    """List processing runs."""
    from sqlalchemy import select

    from election_ocr.persistence.db import ProcessingRun

    t = Table(title="Runs")
    for col in ("run_id", "document", "profile", "status", "pages", "published"):
        t.add_column(col)
    for r in c.session.execute(select(ProcessingRun).order_by(ProcessingRun.started_at.desc())).scalars():
        stats = r.stats or {}
        t.add_row(
            r.run_id,
            r.document_id,
            r.document_profile_id or "-",
            r.status,
            str(stats.get("pages_processed", "-")),
            str(stats.get("published", "-")),
        )
    console.print(t)


@cli.command()
@click.option("--run", "run_id", required=True)
@pass_ctx
def validate(c: Ctx, run_id: str) -> None:
    """Re-run validation + publication gate for a run (e.g. after reference data changed)."""
    from election_ocr.issues import IssueFactory
    from election_ocr.persistence.repository import Repository
    from election_ocr.pipeline.document_service import DocumentService

    svc = DocumentService(Repository(c.session), c.ref, c.cfg, IssueFactory(c.cfg.issue_severity_overrides))
    v = svc.validate_run(run_id)
    published = svc.publish_run(run_id)
    console.print(f"rows={v.rows} validated={v.validated} needs_review={v.needs_review} published={published}")
    print_counts(v.issue_counts)
