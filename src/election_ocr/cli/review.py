"""Human-in-the-loop review of blocked rows (cases, evidence, decisions)."""

from __future__ import annotations

import json
import sys

import click
from rich.table import Table

from election_ocr.cli.app import Ctx, cli, console, pass_ctx


@cli.group()
def review():
    """Human-in-the-loop review of blocked rows."""


def _svc(c: Ctx):
    from election_ocr.persistence.repository import Repository
    from election_ocr.review.service import ReviewService

    return ReviewService(Repository(c.session), c.ref, c.cfg)


@review.command("list")
@click.option("--code", default=None, help="Filter by issue code")
@click.option("--run", "run_id", default=None)
@click.option("--status", default="OPEN")
@click.option("--limit", default=50)
@pass_ctx
def review_list(c: Ctx, code, run_id, status, limit):
    cases = _svc(c).list_cases(status=status, code=code, run_id=run_id)
    t = Table(title=f"Review cases ({len(cases)})")
    for col in ("case", "kind", "target", "codes"):
        t.add_column(col)
    for case in cases[:limit]:
        t.add_row(str(case.case_id), case.kind, case.result_id or case.table_id, ", ".join(case.codes or []))
    console.print(t)


@review.command("show")
@click.argument("case_id", type=int)
@click.option("--no-crops", is_flag=True, help="Do not export image crops")
@pass_ctx
def review_show(c: Ctx, case_id: int, no_crops: bool):
    from election_ocr.review.service import ReviewError

    try:
        ev = _svc(c).evidence(case_id, export_crops=not no_crops)
    except ReviewError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)
    if ev.row is None:
        console.print(f"[bold]case {case_id}[/bold] TABLE {ev.case.table_id}: {', '.join(ev.case.codes or [])}")
        console.print(f"  table crop: {ev.row_crop_path}")
        return
    r = ev.row
    console.print(f"[bold]case {case_id}[/bold] {r.result_id}  status={r.status}")
    console.print(f"  page {r.page_number}  kind={r.row_kind} marker={r.marker} vote_type={r.vote_type} election={r.election_id}")
    console.print("  labels: " + " | ".join(f"{l.get('independence_group')}: {l.get('text')!r}" for l in ev.label_readings))
    if ev.unit_resolution:
        console.print(f"  unit resolution: {json.dumps(ev.unit_resolution, ensure_ascii=False)}")
    if ev.row_crop_path:
        console.print(f"  row image: {ev.row_crop_path}")
    console.print("[bold]Fields[/bold]")
    for f in ev.fields:
        mark = "[green]" if f.status == "AGREED" else "[red]"
        console.print(f"  {mark}{f.field:<26}[/] {f.status:<13} value={f.value or '-':<8} override={f.override or '-'}")
        for x in f.candidates:
            console.print(
                f"      {x['engine']}/{x['variant']}: {x['raw']!r} -> {x['parsed']} [{x['role']}] conf={x['confidence']}"
            )
        if f.crop_path:
            console.print(f"      crop: {f.crop_path}")
    t2 = Table(title="Issues")
    for col in ("code", "severity", "field", "ack", "message"):
        t2.add_column(col)
    for i in ev.issues:
        t2.add_row(i.code, i.severity, i.field or "", "yes" if i.acknowledged else "", i.message)
    console.print(t2)
    for h in ev.arithmetic_hints:
        console.print(f"  hint: {h}")


def _run_action(fn, *args):
    from election_ocr.review.service import ReviewError

    try:
        return fn(*args)
    except ReviewError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)


@review.command("set")
@click.argument("case_id", type=int)
@click.argument("field")
@click.argument("value")
@click.option("--reviewer", required=True)
@click.option("--reason", required=True)
@pass_ctx
def review_set(c: Ctx, case_id, field, value, reviewer, reason):
    """Set a field value as read by a human from the source (e.g. party:spd 163)."""
    _run_action(_svc(c).set_value, case_id, field, value, reviewer, reason)
    console.print("[green]recorded[/green] - run 'review approve' to re-validate")


@review.command("unit")
@click.argument("case_id", type=int)
@click.argument("unit_id")
@click.option("--reviewer", required=True)
@click.option("--reason", required=True)
@pass_ctx
def review_unit(c: Ctx, case_id, unit_id, reviewer, reason):
    """Assign a gazetteer unit to the row."""
    _run_action(_svc(c).set_unit, case_id, unit_id, reviewer, reason)
    console.print("[green]recorded[/green]")


@review.command("ack")
@click.argument("case_id", type=int)
@click.argument("code")
@click.option("--reviewer", required=True)
@click.option("--reason", required=True)
@pass_ctx
def review_ack(c: Ctx, case_id, code, reviewer, reason):
    """Confirm that an issue was checked against the source (e.g. a printing error in the original)."""
    _run_action(_svc(c).acknowledge, case_id, code, reviewer, reason)
    console.print("[green]acknowledged[/green]")


@review.command("ack-table")
@click.argument("table_id")
@click.argument("code")
@click.option("--reviewer", required=True)
@click.option("--reason", required=True)
@pass_ctx
def review_ack_table(c: Ctx, table_id, code, reviewer, reason):
    """Acknowledge a table-level issue (e.g. structure checked visually) for all its rows."""
    n = _run_action(_svc(c).acknowledge_table, table_id, code, reviewer, reason)
    console.print(f"[green]acknowledged for {n} rows[/green]")


@review.command("approve")
@click.argument("case_id", type=int)
@click.option("--reviewer", required=True)
@click.option("--reason", required=True)
@pass_ctx
def review_approve(c: Ctx, case_id, reviewer, reason):
    """Re-validate after decisions; publishes the row if no blocking issue remains."""
    res = _run_action(_svc(c).approve, case_id, reviewer, reason)
    console.print(f"status={res['status']} remaining={res['remaining_blocking']} published_in_run={res['published_in_run']}")


@review.command("reject")
@click.argument("case_id", type=int)
@click.option("--reviewer", required=True)
@click.option("--reason", required=True)
@pass_ctx
def review_reject(c: Ctx, case_id, reviewer, reason):
    _run_action(_svc(c).reject, case_id, reviewer, reason)
    console.print("[yellow]rejected[/yellow]")
