"""Gold-standard transcriptions and evaluation of runs against them."""

from __future__ import annotations

import sys

import click
import yaml

from election_ocr.cli.app import Ctx, cli, console, pass_ctx
from election_ocr.persistence.db import ProcessingRun, SourceDocument


def _run_document(c: Ctx, run_id: str) -> SourceDocument:
    run = c.session.get(ProcessingRun, run_id)
    doc = c.session.get(SourceDocument, run.document_id) if run is not None else None
    if doc is None:
        console.print(f"[red]run {run_id} not found[/red]")
        sys.exit(1)
    return doc


@cli.group()
def gold():
    """Gold-standard transcriptions."""


@gold.command("template")
@click.option("--run", "run_id", required=True)
@click.option("--page", type=int, required=True)
@pass_ctx
def gold_template_cmd(c: Ctx, run_id, page):
    """Write a blank gold skeleton for one page (fill in by hand from the PDF, then sign)."""
    from election_ocr.evaluation.gold import gold_template

    doc = _run_document(c, run_id)
    data = gold_template(c.session, run_id, page, doc.sha256)
    out = c.cfg.paths.gold_dir / doc.document_id / f"page_{page:04d}.yaml.template"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    console.print(f"[green]template written[/green] {out} - rename to .yaml after transcription and fill verified_by/verified_at")


@cli.command()
@click.option("--run", "run_id", required=True)
@pass_ctx
def evaluate(c: Ctx, run_id):
    """Compare a run against the signed gold transcriptions of its document."""
    from election_ocr.evaluation.gold import evaluate_run, load_gold

    doc = _run_document(c, run_id)
    pages = load_gold(c.cfg.paths.gold_dir, doc.document_id, doc.sha256)
    if not pages:
        console.print(f"[yellow]no gold pages in {c.cfg.paths.gold_dir / doc.document_id}[/yellow]")
        sys.exit(1)
    rep = evaluate_run(c.session, run_id, pages, c.ref)
    console.print(
        f"gold pages={rep.pages} rows={rep.gold_rows} matched={rep.matched_rows} fields={rep.fields}\n"
        f"agreed_correct={rep.agreed_correct} agreed_wrong={rep.agreed_wrong} abstained={rep.abstained}\n"
        f"false_accept_rate={rep.false_accept_rate:.4%} automation_rate={rep.automation_rate:.2%}\n"
        f"published_fields={rep.published_fields} published_wrong={rep.published_wrong}"
    )
    for d in rep.wrong_details[:50]:
        console.print(f"  [red]WRONG[/red] {d}")
    if rep.unmatched_rows:
        console.print(f"  unmatched gold rows: {rep.unmatched_rows[:20]}")
    if rep.agreed_wrong or rep.published_wrong:
        sys.exit(3)
