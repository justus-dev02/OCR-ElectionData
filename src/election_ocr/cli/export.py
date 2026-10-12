"""Exports: full evidence per run, published data for analytics, review overview."""

from __future__ import annotations

import click

from election_ocr.cli.app import Ctx, cli, console, pass_ctx


@cli.group()
def export():
    """Exports (evidence per run; published data only for analytics)."""


@export.command("evidence")
@click.option("--run", "run_id", required=True)
@pass_ctx
def export_evidence(c: Ctx, run_id):
    from election_ocr.exports.writers import evidence_json

    p = evidence_json(c.session, run_id, c.cfg.paths.exports_dir / f"evidence_{run_id}.json")
    console.print(f"[green]evidence written[/green] {p}")


@export.command("published")
@click.option("--format", "fmt", type=click.Choice(["json", "md", "csv", "parquet", "all"]), default="all")
@click.option("--election", default=None)
@pass_ctx
def export_published(c: Ctx, fmt, election):
    from election_ocr.exports import writers

    d = c.cfg.paths.exports_dir
    suffix = f"_{election}" if election else ""
    if fmt in ("json", "all"):
        console.print(writers.canonical_json(c.session, d / f"published{suffix}.json", election))
    if fmt in ("md", "all"):
        console.print(writers.markdown(c.session, c.ref, d / f"published{suffix}.md", election))
    if fmt in ("csv", "all"):
        console.print(writers.csv_export(c.session, d / f"published{suffix}.csv", election))
    if fmt in ("parquet", "all"):
        console.print(writers.parquet_export(c.session, d / f"published{suffix}.parquet", election))


@export.command("review-report")
@click.option("--run", "run_id", default=None)
@pass_ctx
def export_review_report(c: Ctx, run_id):
    from election_ocr.exports.writers import review_report

    console.print(review_report(c.session, c.cfg.paths.exports_dir / "review_report.md", run_id))


@export.command("provisional")
@click.option(
    "--tier",
    type=click.Choice(["reference_only", "data_verified"]),
    default="data_verified",
    show_default=True,
    help="Lowest validation tier to include (data_verified includes reference_only).",
)
@pass_ctx
def export_provisional(c: Ctx, tier: str) -> None:
    """CSV per document of machine-verified but UNPUBLISHED rows (gazetteer + reference verification pending)."""
    from election_ocr.exports.provisional import provisional_csv

    for p in provisional_csv(c.session, c.ref, c.cfg, c.cfg.paths.exports_dir / "provisional", tier):
        console.print(f"[green]written[/green] {p}")
