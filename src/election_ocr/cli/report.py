"""Validation report over the latest completed run of every document."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from election_ocr.cli.app import Ctx, cli, console, pass_ctx


@cli.group()
def report() -> None:
    """Reports over the latest completed run of every document."""


@report.command("validation")
@click.option(
    "--gpwed",
    "gpwed_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory with the official comparison files (default: data/external/gpwed if present).",
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Output directory (default: reports/).",
)
@pass_ctx
def report_validation(c: Ctx, gpwed_dir: Optional[Path], out_dir: Optional[Path]) -> None:
    """Cell agreement, row validation tiers, section sums and external agreement per document."""
    from election_ocr.config import PROJECT_ROOT
    from election_ocr.evaluation.validation_report import latest_runs, validate_document, write_report

    gpwed = gpwed_dir or c.cfg.paths.data_dir / "external" / "gpwed"
    runs = latest_runs(c.session)
    if not runs:
        console.print("[yellow]no completed runs in the database[/yellow]")
        return
    results = []
    for run in runs:
        console.print(f"  {run.run_id} ...")
        results.append(validate_document(c.session, run, gpwed if gpwed.exists() else None))
    for path in write_report(results, out_dir or PROJECT_ROOT / "reports"):
        console.print(f"[green]written[/green] {path}")
