"""
Label classification.

Every OCR reading of a row label is classified independently into

* ``SECTION`` - starts with a section keyword (closed vocabulary from the
                profile, e.g. "Landkreis"), optionally preceded by a
                continuation prefix ("noch") and - if the profile allows it -
                by a printed section key ("131 LKR. AALEN")
* ``TOTAL``   - consists of a total-row keyword ("Zusammen")
* ``HEADING`` - a structural heading that is no administrative unit
                ("Stadtkreise", "Regierungsbezirk Nordbaden"), optionally
                followed by a name
* ``UNIT``    - matches the profile's ``label_pattern`` (index + name)
* ``UNKNOWN`` - anything else

Letter-spaced typography ("I N S G E S A M T") is joined before keyword
matching; this only affects the closed keyword vocabulary.

Keywords are matched with a similarity threshold because they come from a
small closed vocabulary (structure detection). Names are NOT fuzzy matched
here - they are resolved exactly against the gazetteer later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from rapidfuzz import fuzz

from election_ocr.parsing.text import fold_diacritics, normalize_unicode


@dataclass
class LabelClass:
    cls: str  # SECTION | TOTAL | UNIT | UNKNOWN
    kind: Optional[str] = None  # section keyword (canonical spelling from profile)
    continued: bool = False
    name: Optional[str] = None
    index: Optional[str] = None
    keyword_score: Optional[float] = None


def _kw(s: str) -> str:
    return fold_diacritics(s.casefold()).strip(".,:;")


def is_letter_spaced(text: str) -> bool:
    """'S t a d t k r e i s e' / '5 t 8 d t k r e 1 8 (' : most tokens are single characters."""
    tokens = [t for t in text.split(" ") if t]
    singles = sum(1 for t in tokens if len(t) == 1 and t.isalnum())  # leader dots do not count
    return len(tokens) >= 4 and singles >= 0.7 * len(tokens)


def join_letter_spacing(text: str, min_run: int = 3) -> str:
    """'S t a d t k r e i s e' -> 'Stadtkreise' (runs of >= min_run single letters are joined)."""
    tokens = text.split(" ")
    out: List[str] = []
    run: List[str] = []
    for tok in tokens + [""]:
        if len(tok) == 1 and tok.isalpha():
            run.append(tok)
            continue
        if len(run) >= min_run:
            out.append("".join(run))
        else:
            out.extend(run)
        run = []
        if tok:
            out.append(tok)
    return " ".join(out)


def _best_keyword(
    token: str, keywords: Sequence[str], category: Optional[Dict[str, str]] = None
) -> Tuple[Optional[str], float, float]:
    """Best keyword, its similarity and the margin to the best keyword of a DIFFERENT category.

    ``category`` maps keywords to what they mean (e.g. administrative level);
    keywords of the same category are not competitors.
    """
    scored = sorted(((fuzz.ratio(_kw(token), _kw(k)) / 100.0, k) for k in keywords), reverse=True)
    if not scored:
        return None, 0.0, 0.0
    best_score, best_kw = scored[0]
    cat = (category or {}).get(best_kw, best_kw)
    rivals = [sc for sc, k in scored[1:] if (category or {}).get(k, k) != cat]
    return best_kw, best_score, best_score - (rivals[0] if rivals else 0.0)


class LabelClassifier:
    def __init__(
        self,
        label_pattern: str,
        section_kinds: Dict[str, str],
        continued_prefixes: Sequence[str],
        total_keywords: Sequence[str],
        min_similarity: float,
        min_margin: float = 0.1,
        headings: Sequence[str] = (),
        section_index_allowed: bool = False,
        keyword_precedence: bool = False,
        named_headings: Sequence[str] = (),
    ):
        self.label_rx = re.compile(label_pattern)
        self.kind_levels = dict(section_kinds)
        self.kinds = list(section_kinds)
        self.prefixes = list(continued_prefixes)
        self.totals = list(total_keywords)
        self.headings = list(headings)  # printed alone ('Stadtkreise')
        self.named_headings = list(named_headings)  # printed with a name ('Regierungsbezirk Nordbaden')
        self.section_index_allowed = section_index_allowed
        self.keyword_precedence = keyword_precedence
        self.min_similarity = min_similarity
        self.min_margin = min_margin

    def classify(self, text: str) -> LabelClass:
        t = normalize_unicode(text)
        if not t:
            return LabelClass("UNKNOWN")
        if is_letter_spaced(t):
            # letter-spaced typography is used for headings only, never for unit rows
            return self._keywords(join_letter_spacing(t)) or self._keywords(t.replace(" ", "")) or LabelClass("UNKNOWN")
        if not self.keyword_precedence:
            unit = self._unit(t)
            if unit is not None:
                return unit
        kw = self._keywords(join_letter_spacing(t))
        if kw is not None:
            return kw
        if self.keyword_precedence:
            unit = self._unit(t)
            if unit is not None:
                return unit
        return LabelClass("UNKNOWN")

    def _unit(self, t: str) -> Optional[LabelClass]:
        m = self.label_rx.match(t)
        if not m:
            return None
        return LabelClass("UNIT", name=normalize_unicode(m.group("name")), index=m.groupdict().get("index"))

    def _keywords(self, t: str) -> Optional[LabelClass]:
        tokens = t.split(" ")
        section_index = None
        if tokens[0][:1].isdigit():
            if not (self.section_index_allowed and tokens[0].isdigit() and len(tokens) > 1):
                return None
            section_index, tokens = tokens[0], tokens[1:]
            t = " ".join(tokens)
        if section_index is None:
            # total row keyword (whole label)
            if self.totals:
                kw, score, _ = _best_keyword(t, self.totals)
                if kw and score >= self.min_similarity:
                    return LabelClass("TOTAL", keyword_score=round(score, 4))
            # one-word structural heading ('Stadtkreise'); a section always needs a name,
            # so section keywords are no competitors here
            if self.headings and (len(tokens) == 1 or any(len(x) == 1 and x.isalpha() for x in tokens)):
                # one-word heading; a word split by single letters ('Landkre i se') is joined first
                candidate = t if len(tokens) == 1 else t.replace(" ", "")
                kw, score, margin = _best_keyword(candidate, self.headings)
                if kw and score >= self.min_similarity and margin >= self.min_margin:
                    return LabelClass("HEADING", kind=kw, keyword_score=round(score, 4))
        continued = False
        if self.prefixes and len(tokens) > 1:
            p, pscore, _ = _best_keyword(tokens[0], self.prefixes)
            if p and pscore >= self.min_similarity:
                continued = True
                tokens = tokens[1:]
        if len(tokens) < 2:
            return None
        rest = normalize_unicode(" ".join(tokens[1:]))
        kind, kscore, kmargin = _best_keyword(tokens[0], self.kinds, self.kind_levels) if self.kinds else (None, 0.0, 0.0)
        head, hscore, hmargin = _best_keyword(tokens[0], self.named_headings) if self.named_headings else (None, 0.0, 0.0)
        if kind and kscore >= self.min_similarity and kmargin >= self.min_margin and kscore - hscore >= self.min_margin:
            return LabelClass(
                "SECTION", kind=kind, continued=continued, name=rest, index=section_index, keyword_score=round(kscore, 4)
            )
        if (
            head
            and not continued
            and section_index is None
            and hscore >= self.min_similarity
            and hmargin >= self.min_margin
            and hscore - kscore >= self.min_margin
        ):
            # heading followed by a name ('Regierungsbezirk Nordbaden')
            return LabelClass("HEADING", kind=head, name=rest, keyword_score=round(hscore, 4))
        return None


@dataclass
class AgreedClass:
    cls: Optional[str]  # agreed class or None
    kind: Optional[str]
    continued: Optional[bool]
    names: Dict[str, str]  # group -> name reading (for SECTION / UNIT)
    indices: Dict[str, str]
    votes: Dict[str, str]  # group -> class
    conflict: bool


def agree(classes: Dict[str, LabelClass], min_sources: int) -> AgreedClass:
    """Class agreed iff >= min_sources groups share it and no group reports another known class.

    For SECTION the keyword (kind) and the continuation flag must agree as well,
    for HEADING the keyword.
    """
    votes = {g: c.cls for g, c in classes.items()}
    known = {g: c for g, c in classes.items() if c.cls != "UNKNOWN"}
    names = {g: c.name for g, c in known.items() if c.name}
    indices = {g: c.index for g, c in known.items() if c.index}
    if not known:
        return AgreedClass(None, None, None, names, indices, votes, conflict=False)
    kinds = {(c.cls, c.kind, c.continued if c.cls == "SECTION" else None) for c in known.values()}
    if len(kinds) > 1:
        return AgreedClass(None, None, None, names, indices, votes, conflict=True)
    cls, kind, continued = next(iter(kinds))
    if len(known) < min_sources:
        return AgreedClass(None, None, None, names, indices, votes, conflict=False)
    return AgreedClass(cls, kind, continued, names, indices, votes, conflict=False)
