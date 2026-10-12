"""Root command group and the lazily created shared context (config, DB session, reference data)."""

from __future__ import annotations

from typing import Dict, Optional

import click
from rich.console import Console
from rich.table import Table
from sqlalchemy.orm import Session

from election_ocr.config import PipelineConfig, load_config
from election_ocr.reference.repository import ReferenceRepository

console = Console()


class Ctx:
    def __init__(self) -> None:
        self.cfg: PipelineConfig = load_config()
        self.cfg.ensure_directories()
        self._session: Optional[Session] = None
        self._ref: Optional[ReferenceRepository] = None

    @property
    def session(self) -> Session:
        if self._session is None:
            from election_ocr.persistence.db import make_engine, make_session

            self._session = make_session(make_engine(self.cfg.database_url))
        return self._session

    @property
    def ref(self) -> ReferenceRepository:
        if self._ref is None:
            self._ref = ReferenceRepository.load(self.cfg.paths.reference_dir)
        return self._ref


pass_ctx = click.make_pass_decorator(Ctx, ensure=True)


@click.group()
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Historical election document extraction & verification."""
    ctx.obj = Ctx()


def print_counts(counts: Dict[str, int], title: str = "Blocking issues (row occurrences)") -> None:
    if not counts:
        return
    t = Table(title=title)
    t.add_column("code")
    t.add_column("count", justify="right")
    for k, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        t.add_row(k, str(n))
    console.print(t)
