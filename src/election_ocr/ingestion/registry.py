"""
Document registration (immutable source evidence).

The source PDF is never modified. On registration we record SHA-256, size,
page count, PDF metadata, how many pages carry a text layer and the renderer
version, so every later value can be traced to exact bytes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pymupdf
from sqlalchemy.orm import Session

from election_ocr.persistence.db import SourceDocument


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def renderer_version() -> str:
    return f"PyMuPDF {pymupdf.VersionBind} / MuPDF {pymupdf.VersionFitz}"


@dataclass
class RegisteredDocument:
    document_id: str
    sha256: str
    path: Path
    page_count: int
    text_layer_pages: int
    producer: str


def register_document(path: Path, session: Optional[Session] = None) -> RegisteredDocument:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    digest = sha256_of(path)
    document_id = f"doc_{digest[:16]}"
    with pymupdf.open(path) as doc:
        page_count = len(doc)
        text_layer_pages = sum(1 for p in doc if p.get_text("text").strip())
        metadata = {k: v for k, v in (doc.metadata or {}).items() if v}
    producer = str(metadata.get("producer") or metadata.get("creator") or "unknown")
    if session is not None:
        existing = session.get(SourceDocument, document_id)
        if existing is None:
            session.add(
                SourceDocument(
                    document_id=document_id,
                    sha256=digest,
                    filename=path.name,
                    path=str(path),
                    size_bytes=path.stat().st_size,
                    page_count=page_count,
                    text_layer_pages=text_layer_pages,
                    pdf_metadata=metadata,
                    renderer_version=renderer_version(),
                )
            )
            session.commit()
        elif existing.sha256 != digest:  # pragma: no cover - impossible by construction
            raise RuntimeError("document id collision")
    return RegisteredDocument(
        document_id=document_id,
        sha256=digest,
        path=path,
        page_count=page_count,
        text_layer_pages=text_layer_pages,
        producer=producer,
    )
