"""Held-out evaluation of an OCR engine through its production read path (``engine.read``)."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence

from election_ocr.ocr.engines import OCRRequest
from election_ocr.ocr.resolver import parse_reading
from election_ocr.parsing.numbers import DecimalFormat, IntegerFormat
from election_ocr.training.labels import Sample


def group_verdict(raws: Sequence[str], sample: Sample) -> Optional[str]:
    """Value of the Tesseract group for one token, as the resolver would see it:
    one value if all parseable variants agree, '<conflict>' if they contradict, None if none parse."""
    if sample.value_type == "integer":
        vals = {parse_reading(r, "integer", IntegerFormat(sign="trailing", dash_means_zero=True), None, ()).value for r in raws}
    else:
        vals = {parse_reading(r, "decimal", None, DecimalFormat(sign="trailing", dash_means_zero=True), ()).value for r in raws}
    vals.discard(None)
    if len(vals) > 1:
        return "<conflict>"
    return vals.pop() if vals else None


def evaluate(samples: Sequence[Sample], engine) -> Dict[str, object]:
    """Read every sample through the production read path and score the group verdict."""
    reqs = [OCRRequest(s.token_id, s.image, s.value_type, s.allowlist, s.glyph_height, s.shape or "wide") for s in samples]
    raws: Dict[str, List[str]] = defaultdict(list)
    for r in engine.read(reqs):
        raws[r.token_id].append(r.raw_text)
    by: Dict[str, Counter] = defaultdict(Counter)
    errors: List[Dict[str, str]] = []
    for s in samples:
        v = group_verdict(raws[s.token_id], s)
        outcome = "correct" if v == s.value else ("no_value" if v is None else "wrong")
        for key in ("all", f"tier_{s.tier}", f"doc_{s.document_id}"):
            by[key][outcome] += 1
        if outcome != "correct" and len(errors) < 200:
            errors.append(dict(token_id=s.token_id, truth=s.text, readings=" | ".join(raws[s.token_id]), tier=s.tier))
    summary = {
        k: dict(
            n=sum(c.values()),
            **{o: c[o] for o in ("correct", "wrong", "no_value")},
            accuracy=round(c["correct"] / max(1, sum(c.values())), 4),
        )
        for k, c in sorted(by.items())
    }
    return dict(summary=summary, errors=errors)
