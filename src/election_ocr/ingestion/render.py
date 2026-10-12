"""Page rendering + deskew. Produces the canonical page image used for all crops."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pymupdf

from election_ocr.config import RenderSettings
from election_ocr.ocr.engines import TextLayerWord, transform_pdf_word
from election_ocr.preprocessing.image_ops import estimate_skew, quality_metrics, rotate


@dataclass
class RenderedPage:
    page_number: int
    gray: np.ndarray
    dpi: int
    deskew_angle: float
    deskew_applied: bool
    rotation_matrix: Optional[np.ndarray]
    image_path: Path
    image_sha256: str
    words: List[TextLayerWord]
    quality: dict


def render_page(pdf_path: Path, page_number: int, settings: RenderSettings, out_dir: Path) -> RenderedPage:
    with pymupdf.open(pdf_path) as doc:
        if not 1 <= page_number <= len(doc):
            raise ValueError(f"page {page_number} out of range 1..{len(doc)}")
        page = doc[page_number - 1]
        pix = page.get_pixmap(dpi=settings.dpi, colorspace=pymupdf.csGRAY)
        gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()
        raw_words = [w for w in page.get_text("words") if w[4].strip()]

    angle, _ = estimate_skew(gray, settings.max_deskew_degrees)
    apply = settings.deskew_scanned_pages and abs(angle) >= settings.min_deskew_degrees
    matrix = None
    if apply:
        h, w = gray.shape
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        gray = rotate(gray, angle)

    words = [TextLayerWord(text=w[4], box=transform_pdf_word(w[0], w[1], w[2], w[3], settings.dpi, matrix)) for w in raw_words]
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / f"page_{page_number:04d}.png"
    ok, buf = cv2.imencode(".png", gray)
    if not ok:  # pragma: no cover
        raise RuntimeError("PNG encoding failed")
    data = buf.tobytes()
    image_path.write_bytes(data)
    return RenderedPage(
        page_number=page_number,
        gray=gray,
        dpi=settings.dpi,
        deskew_angle=angle,
        deskew_applied=apply,
        rotation_matrix=matrix,
        image_path=image_path,
        image_sha256=hashlib.sha256(data).hexdigest(),
        words=words,
        quality=quality_metrics(gray),
    )
