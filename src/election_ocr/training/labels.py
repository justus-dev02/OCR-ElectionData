"""
Training labels for OCR fine-tuning, derived from pipeline runs (no manual typing, no LLM).

A token crop becomes a training sample only with strong, auditable evidence:

* Tier ``A`` ("consensus"): the cell is AGREED (>= 2 independent groups read exactly
  the same value, no group dissents) and its row shows no arithmetic contradiction.
* Tier ``B`` ("proven hard case"): exactly ONE group dissents (the group being
  trained), at least two other groups agree, and the row's party sum equals its valid
  votes exactly when that value is used - with every other value of the row AGREED.
  These are the cells the engine currently gets wrong; they are what fine-tuning can
  fix. Tier B labels never publish anything: in production the cell still needs
  agreement of independent readers (see ocr.resolver).
* Signed gold transcriptions (``gold/``) are NOT used as training labels yet; they are
  reserved for evaluation, so that held-out numbers stay independent of the training data.

Precision note: tier B ties the trained engine's training labels to the other
engines' readings. It is opt-in (``tiers``), recorded in the model manifest, and the
published-error rate must be checked against gold (``evaluate``) before relying on it.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import AbstractSet, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

# Row issues that contradict the row's values -> none of its cells are used.
CONTRADICTION_CODES = frozenset(
    {
        "PARTY_SUM_MISMATCH",
        "TURNOUT_MISMATCH",
        "PERCENT_MISMATCH",
        "CROSS_ROLE_MISMATCH",
        "DERIVED_VALUE_MISMATCH",
        "HIERARCHICAL_SUM_MISMATCH",
        "GLYPH_SHAPE_CONFLICT",
        "ROW_GROUP_INVALID",
        "ROW_UNINTERPRETED",
    }
)
NUMERIC_TYPES = ("integer", "decimal")
_GT_CHARS = re.compile(r"^[0-9.,\-]+$")


@dataclass
class Sample:
    token_id: str
    document_id: str
    page: int
    value_type: str
    value: str  # canonical value (resolver form, e.g. "1785", "55.0")
    text: str  # ground-truth glyph string for the crop (e.g. "55,0")
    tier: str  # A | B | G
    image: np.ndarray  # raw grayscale token crop exactly as given to the engines
    glyph_height: float
    shape: Optional[str]
    allowlist: Optional[str]
    evidence: Dict[str, object] = field(default_factory=dict)

    @property
    def split_key(self) -> str:
        return f"{self.document_id}:{self.page}"


def is_heldout(document_id: str, page: int, heldout_percent: int) -> bool:
    """Deterministic page-level split (all tokens of a page land in the same split)."""
    h = int(hashlib.sha1(f"{document_id}:{page}".encode()).hexdigest()[:8], 16)
    return h % 100 < heldout_percent


def _token_hash(s: Sample) -> str:
    return hashlib.sha1(s.token_id.encode()).hexdigest()


def split_samples(
    samples: Sequence[Sample],
    heldout_percent: int,
    max_samples_per_document: Optional[int] = None,
    max_heldout: Optional[int] = None,
) -> Tuple[List[Sample], List[Sample]]:
    """Page-level train / held-out split with optional deterministic caps.

    * ``max_samples_per_document`` caps tier-A TRAINING samples per document (tier B is always kept),
      chosen by a hash of the token id (deterministic, independent of input order);
    * ``max_heldout`` subsamples the held-out set: all non-A samples plus hash-ordered tier-A samples.
    """
    train = [s for s in samples if not is_heldout(s.document_id, s.page, heldout_percent)]
    heldout = [s for s in samples if is_heldout(s.document_id, s.page, heldout_percent)]
    if max_samples_per_document:
        per_doc: Counter = Counter()
        capped = []
        for s in sorted(train, key=_token_hash):
            if s.tier != "A" or per_doc[s.document_id] < max_samples_per_document:
                capped.append(s)
                per_doc[s.document_id] += s.tier == "A"
        train = capped
    if max_heldout and len(heldout) > max_heldout:
        hard = [s for s in heldout if s.tier != "A"]
        easy = sorted((s for s in heldout if s.tier == "A"), key=_token_hash)
        heldout = hard + easy[: max(0, max_heldout - len(hard))]
    return train, heldout


def _decode_png(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("corrupt crop PNG in the crop dump")
    return img


def _load_crops(crop_dirs: Sequence[Path]) -> Dict[str, dict]:
    crops: Dict[str, dict] = {}
    for d in crop_dirs:
        for p in sorted(Path(d).glob("*.pkl")):
            with open(p, "rb") as fh:
                for row in pickle.load(fh):
                    crops[row["token_id"]] = row
    return crops


def _gt_text(
    cands: List[Tuple[str, str, str, Optional[str]]], groups: Iterable[str], value: str, prefer: Sequence[str]
) -> Optional[str]:
    """Glyph string of a supporting reading (spaces removed) whose parsed value is ``value``."""
    by_group: Dict[str, str] = {}
    for group, variant, raw, parsed in cands:
        if group in groups and parsed == value and not variant.startswith("free"):
            txt = re.sub(r"\s+", "", raw)
            if _GT_CHARS.match(txt):
                by_group.setdefault(group, txt)
    for g in list(prefer) + sorted(by_group):
        if g in by_group:
            return by_group[g]
    return None


@dataclass
class _RunEvidence:
    """Cell statuses, OCR candidates and row issues of one run (read straight from SQLite)."""

    run_id: str
    document_id: str
    cells: Dict[str, dict]
    candidates: Dict[str, List[Tuple[str, str, str, Optional[str]]]]
    row_issues: Dict[str, Set[str]]

    @classmethod
    def load(cls, con: sqlite3.Connection, run: str) -> _RunEvidence:
        (doc,) = con.execute("select document_id from processing_runs where run_id=?", (run,)).fetchone()
        cells = {
            cid: {"status": st, "value": val, "vt": vt, "sup": json.loads(sup or "[]"), "dis": json.loads(dis or "[]")}
            for cid, st, val, vt, sup, dis in con.execute(
                "select cell_id, status, value, value_type, supporting_groups, dissenting_groups from cells where run_id=?",
                (run,),
            )
        }
        cands: Dict[str, List[Tuple[str, str, str, Optional[str]]]] = {}
        for cid, g, var, raw, parsed in con.execute(
            "select cell_id, independence_group, variant, raw_text, parsed_value from ocr_candidates where run_id=?", (run,)
        ):
            cands.setdefault(cid, []).append((g, var, raw, parsed))
        row_issues: Dict[str, Set[str]] = {}
        for rid, code in con.execute("select result_id, code from issues where run_id=? and result_id is not null", (run,)):
            row_issues.setdefault(rid, set()).add(code)
        return cls(run, doc, cells, cands, row_issues)

    def other_groups_value(self, cid: str, target_group: str) -> Optional[str]:
        """Value of a cell whose only dissent is ``target_group`` while >= 2 other groups agree."""
        c = self.cells.get(cid)
        if not c or c["status"] != "DISAGREED" or c["dis"] != [target_group]:
            return None
        sup = [g for g in c["sup"] if g != target_group]
        if len(sup) < 2:
            return None
        vals = {p for g, var, raw, p in self.candidates.get(cid, []) if g in sup and p is not None}
        return vals.pop() if len(vals) == 1 else None

    def proven_tier_b(self, fields: Dict[str, str], parties: Dict[str, str], target_group: str) -> Set[str]:
        """Disagreed cells of an absolute row whose party sum equals the valid votes exactly when the
        other groups' value is used (every other value AGREED)."""
        if "valid_votes" not in fields or not parties:
            return set()
        vals: Dict[str, Optional[str]] = {}
        for cid in [*parties.values(), fields["valid_votes"]]:
            c = self.cells.get(cid)
            vals[cid] = c["value"] if c and c["status"] == "AGREED" else self.other_groups_value(cid, target_group)
        if not all(v is not None and re.fullmatch(r"-?\d+", v) for v in vals.values()):
            return set()
        if sum(int(vals[c] or 0) for c in parties.values()) != int(vals[fields["valid_votes"]] or 0):
            return set()
        return {cid for cid in vals if self.cells[cid]["status"] == "DISAGREED"}


def collect_samples(
    db_path: Path,
    crop_dirs: Sequence[Path],
    target_group: str,
    tiers: AbstractSet[str] = frozenset({"A"}),
    run_ids: Optional[Sequence[str]] = None,
) -> Tuple[List[Sample], Counter]:
    """Return training samples for ``target_group`` and a counter of exclusion reasons."""
    crops = _load_crops(crop_dirs)
    con = sqlite3.connect(str(db_path))
    runs = list(run_ids or [r for (r,) in con.execute("select run_id from processing_runs where status='COMPLETED'")])
    stats: Counter = Counter()
    samples: List[Sample] = []
    for run in runs:
        ev = _RunEvidence.load(con, run)
        for rid, page, role_kind, fields_js, parties_js in con.execute(
            "select result_id, page_number, role_kind, field_cells, party_cells from result_rows where run_id=?", (run,)
        ):
            fields: Dict[str, str] = {k: v for k, v in json.loads(fields_js or "{}").items() if v}
            parties: Dict[str, str] = {k: v for k, v in json.loads(parties_js or "{}").items() if v}
            if ev.row_issues.get(rid, set()) & CONTRADICTION_CODES:
                stats["row_contradiction"] += len(fields) + len(parties)
                continue
            proven_b = ev.proven_tier_b(fields, parties, target_group) if "B" in tiers and role_kind == "absolute" else set()
            for cid in [*fields.values(), *parties.values()]:
                sample = _sample(ev, cid, rid, int(page), crops, tiers, proven_b, target_group, stats)
                if sample is not None:
                    samples.append(sample)
                    stats[f"tier_{sample.tier}"] += 1
    # one sample per token (a cell can belong to several rows of a group)
    unique: Dict[str, Sample] = {}
    for s in samples:
        unique.setdefault(s.token_id, s)
    return list(unique.values()), stats


def _sample(
    ev: _RunEvidence,
    cid: str,
    result_id: str,
    page: int,
    crops: Dict[str, dict],
    tiers: AbstractSet[str],
    proven_b: Set[str],
    target_group: str,
    stats: Counter,
) -> Optional[Sample]:
    c = ev.cells.get(cid)
    crop = crops.get(cid)
    if c is None or c["vt"] not in NUMERIC_TYPES:
        return None
    if crop is None:
        stats["no_crop"] += 1
        return None
    value: Optional[str]
    if c["status"] == "AGREED" and "A" in tiers and len(c["sup"]) >= 2:
        tier, value, groups, prefer = "A", c["value"], c["sup"], [target_group]
    elif cid in proven_b:
        tier, value, groups, prefer = (
            "B",
            ev.other_groups_value(cid, target_group),
            [g for g in c["sup"] if g != target_group],
            [],
        )
    else:
        stats[f"status_{c['status']}"] += 1
        return None
    text = _gt_text(ev.candidates.get(cid, []), groups, value or "", prefer)
    if text is None or value is None:
        stats["no_clean_gt_text"] += 1
        return None
    return Sample(
        token_id=cid,
        document_id=ev.document_id,
        page=page,
        value_type=c["vt"],
        value=value,
        text=text,
        tier=tier,
        image=_decode_png(crop["png"]),
        glyph_height=float(crop["glyph_height"]),
        shape=crop.get("shape"),
        allowlist=crop.get("allowlist"),
        evidence={"run_id": ev.run_id, "result_id": result_id, "supporting_groups": sorted(groups)},
    )
