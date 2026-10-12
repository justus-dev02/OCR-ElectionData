"""
Text normalization helpers.

Two levels are distinguished on purpose:

* ``exact_key``   - used for EXACT matching. Only typographic noise is removed
                    (Unicode form, case, whitespace, spacing around ``,``/``-``).
                    Umlauts and letters are never altered.
* ``fuzzy_key``   - used ONLY to generate review candidates. Diacritics are
                    folded so that OCR-dropped umlauts still surface candidates.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List

_WS_RE = re.compile(r"\s+")
_HYPHEN_BREAK_RE = re.compile(r"(\w)[\-­‐‑]\s+(\w)")
_NON_ALNUM_RE = re.compile(r"[^0-9a-zäöüß]+")


def normalize_unicode(text: str) -> str:
    """NFC + non-breaking/odd spaces to plain space + trimmed/collapsed whitespace."""
    t = unicodedata.normalize("NFC", text or "")
    t = t.replace(" ", " ").replace(" ", " ").replace(" ", " ")
    return _WS_RE.sub(" ", t).strip()


def exact_key(text: str) -> str:
    t = normalize_unicode(text).casefold()
    t = re.sub(r"\s*,\s*", ", ", t)
    t = re.sub(r"\s*-\s*", "-", t)
    t = re.sub(r"\s*\.\s*", ". ", t).strip()
    return _WS_RE.sub(" ", t).strip(" ,")


def fold_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.replace("ß", "ss")


def fuzzy_key(text: str) -> str:
    return fold_diacritics(exact_key(text))


def header_key(text: str) -> str:
    """Normalize multi-line header text: 'Wahl- berech- tigte' -> 'wahlberechtigte'."""
    t = normalize_unicode(text)
    prev = None
    while prev != t:
        prev = t
        t = _HYPHEN_BREAK_RE.sub(r"\1\2", t)
    t = t.casefold()
    return _NON_ALNUM_RE.sub("", t)


def header_layer_variants(layers: List[str]) -> List[str]:
    """All contiguous concatenations of header text layers (top->bottom), header-normalized."""
    keys: List[str] = []
    n = len(layers)
    for i in range(n):
        for j in range(i + 1, n + 1):
            joined = " ".join(layers[i:j])
            k = header_key(joined)
            if k and k not in keys:
                keys.append(k)
    return keys


_ASCII_TRANSLIT = {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"}
_PRINTOUT_PUNCT_RE = re.compile(r"[.,;:'`´]+")


def printout_key(text: str) -> str:
    """Key for names printed by line printers without umlauts and punctuation.

    'Herrenberg, Stadt' and 'HERRENBERG STADT' -> 'herrenberg stadt';
    'Gärtringen' and 'GAERTRINGEN' -> 'gaertringen'. Letters are otherwise kept,
    so the comparison built on it stays exact (no similarity threshold).
    """
    t = normalize_unicode(text).casefold()
    t = "".join(_ASCII_TRANSLIT.get(ch, ch) for ch in t)
    t = _PRINTOUT_PUNCT_RE.sub(" ", t)
    t = re.sub(r"\s*-\s*", "-", t)
    return _WS_RE.sub(" ", t).strip()
