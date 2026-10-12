"""
Cell OCR engines (Strategy pattern).

Every engine reads *token crops* produced by the layout stage and returns raw
readings. Engines never parse or normalize - that is the resolver's job - and
confidences are reported exactly as the engine provides them (or ``None``).

Engines and their independence groups:

* ``tesseract``      - Tesseract 5 LSTM via tesserocr (C API, no subprocess).
                       Two variants per numeric token (character whitelist /
                       no whitelist); both belong to ONE independence group.
* ``easyocr``        - EasyOCR CRNN with CTC decoding (no language model).
* ``pdf_text_layer`` - words of an existing PDF text layer (e.g. ABBYY) that
                       fall inside the token; confidence unknown (None).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import cv2
import numpy as np

from election_ocr.geometry import Box


@dataclass
class OCRRequest:
    token_id: str
    image: np.ndarray  # grayscale crop (page resolution), erased pixels are white
    value_type: str  # label | marker | integer | decimal
    allowlist: Optional[str]  # characters the engine may output (None = unrestricted)
    glyph_height: float = 20.0  # median glyph height on the page (pixels)
    shape: str = "wide"  # geometric token class (layout.tokens.classify_token_shape)


@dataclass
class OCRReading:
    token_id: str
    engine: str
    engine_version: str
    variant: str
    independence_group: str
    raw_text: str
    confidence: Optional[float]
    meta: Dict[str, object] = field(default_factory=dict)


class CellOCREngine(Protocol):
    name: str
    independence_group: str

    def version(self) -> str: ...

    def read(self, requests: Sequence[OCRRequest]) -> List[OCRReading]: ...


def _file_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def detect_tessdata_dir() -> str:
    """Locate the installed tessdata directory (TESSDATA_PREFIX or `tesseract --list-langs`)."""
    import os
    import re
    import subprocess

    env = os.environ.get("TESSDATA_PREFIX")
    if env and os.path.isdir(env):
        return env
    try:
        out = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("tesseract binary not found; install Tesseract 5 or set paths.tessdata_dir") from exc
    m = re.search(r'"([^"]+)"', out.stdout + out.stderr)
    if not m or not os.path.isdir(m.group(1)):
        raise RuntimeError("could not determine tessdata directory; set paths.tessdata_dir")
    return m.group(1)


def prepare_crop(image: np.ndarray, glyph_height: float, target_glyph_height: int, border: int = 12) -> np.ndarray:
    """Upscale so glyphs reach the engine's preferred height and add a white border."""
    scale = max(1.0, target_glyph_height / max(glyph_height, 1.0))
    if scale > 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(image, border, border, border, border, cv2.BORDER_CONSTANT, value=255)


class TesseractEngine:
    name = "tesseract"

    def __init__(
        self,
        independence_group: str,
        lang: str,
        tessdata_dir: Optional[str] = None,
        target_glyph_px: int = 40,
        border_px: int = 0,
        psm_by_shape: Optional[Dict[str, int]] = None,
        header_target_glyph_px: int = 56,
        numeric_lang: Optional[str] = None,
        numeric_tessdata_dir: Optional[str] = None,
    ):
        import tesserocr

        self._tesserocr = tesserocr
        self.independence_group = independence_group
        self.lang = lang
        self.target_glyph_px = target_glyph_px
        self.border_px = border_px
        self.psm_by_shape = dict(psm_by_shape or {})
        self.header_target_glyph_px = header_target_glyph_px
        self.tessdata_dir = str(tessdata_dir) if tessdata_dir else detect_tessdata_dir()
        # Optional fine-tuned model for numbers/markers only (labels and headers keep the
        # base model, so place names cannot be degraded by digit fine-tuning).
        self.numeric_lang = numeric_lang
        self.numeric_tessdata_dir = str(numeric_tessdata_dir) if numeric_tessdata_dir else self.tessdata_dir
        self._apis: Dict[Tuple[int, bool], Any] = {}

    def version(self) -> str:
        base = (
            f"tesseract {self._tesserocr.tesseract_version().split()[1]} / tesserocr {self._tesserocr.__version__} / {self.lang}"
        )
        if self.numeric_lang:
            base += f" / numbers: {self.numeric_lang} sha256:{_file_sha256(Path(self.numeric_tessdata_dir) / (self.numeric_lang + '.traineddata'))[:12]}"
        return base

    def _api(self, psm: int, numeric: bool = False):
        numeric = bool(numeric and self.numeric_lang)
        key = (psm, numeric)
        if key not in self._apis:
            path, lang = (self.numeric_tessdata_dir, self.numeric_lang) if numeric else (self.tessdata_dir, self.lang)
            self._apis[key] = self._tesserocr.PyTessBaseAPI(path=path, lang=lang, psm=psm)
        return self._apis[key]

    def close(self) -> None:
        for api in self._apis.values():
            api.End()
        self._apis.clear()

    def _recognize(self, api, image: np.ndarray, allowlist: Optional[str]) -> Tuple[str, Optional[float]]:
        from PIL import Image

        api.SetVariable("tessedit_char_whitelist", allowlist or "")
        api.SetImage(Image.fromarray(image))
        text = api.GetUTF8Text().strip()
        confs = [c for c in api.AllWordConfidences() if c >= 0]
        conf = round(sum(confs) / len(confs) / 100.0, 4) if confs else None
        return text, conf

    def read_lines(self, image: np.ndarray, glyph_height: float) -> List[str]:
        """Read a multi-line block (header cell) and return its text lines top to bottom."""
        prepared = prepare_crop(image, glyph_height, self.header_target_glyph_px, 12)
        text, _ = self._recognize(self._api(self._tesserocr.PSM.SINGLE_BLOCK), prepared, None)
        return [ln.strip() for ln in text.splitlines() if ln.strip()]

    def read(self, requests: Sequence[OCRRequest]) -> List[OCRReading]:
        out: List[OCRReading] = []
        version = self.version()
        PSM = self._tesserocr.PSM
        for req in requests:
            # Labels are lines of words (PSM 7). For numbers and markers the page
            # segmentation mode is chosen by the token's geometric shape (calibrated,
            # see config): raw line (13) for dashes and single glyphs, single line (7)
            # for multi-digit numbers. Two variants (whitelist / free) per token form
            # ONE independence group; a contradiction between them is a dissent.
            variants: List[Tuple[str, Optional[str], int]]
            if req.value_type == "label":
                variants = [("line_psm7", None, PSM.SINGLE_LINE)]
            else:
                psm = int(self.psm_by_shape.get(req.shape, PSM.RAW_LINE))
                variants = [(f"whitelist_psm{psm}", req.allowlist, psm), (f"free_psm{psm}", None, psm)]
            image = prepare_crop(req.image, req.glyph_height, self.target_glyph_px, self.border_px)
            for variant, allow, psm in variants:
                text, conf = self._recognize(self._api(psm, numeric=req.value_type != "label"), image, allow)
                out.append(
                    OCRReading(
                        token_id=req.token_id,
                        engine=self.name,
                        engine_version=version,
                        variant=variant,
                        independence_group=self.independence_group,
                        raw_text=text,
                        confidence=conf,
                        meta={"shape": req.shape},
                    )
                )
        return out


class EasyOCREngine:
    name = "easyocr"

    def __init__(
        self,
        independence_group: str,
        langs: List[str],
        model_dir: str,
        batch_size: int = 32,
        target_glyph_px: int = 40,
        border_px: int = 12,
        border_px_by_shape: Optional[Dict[str, int]] = None,
        recognizer_weights: Optional[str] = None,
    ):
        self.recognizer_weights = recognizer_weights
        self.target_glyph_px = target_glyph_px
        self.border_px = border_px
        self.border_px_by_shape = dict(border_px_by_shape or {})
        self.independence_group = independence_group
        self.langs = langs
        self.model_dir = model_dir
        self.batch_size = batch_size
        self._reader: Any = None
        self._numeric_reader: Any = None

    def _new_reader(self, quantize: bool = True):
        import easyocr

        return easyocr.Reader(
            self.langs,
            gpu=False,
            model_storage_directory=self.model_dir,
            user_network_directory=f"{self.model_dir}/user_network",
            download_enabled=True,
            verbose=False,
            quantize=quantize,
        )

    def _ensure(self, numeric: bool = False):
        """Base reader; with ``numeric`` and fine-tuned weights, the digit reader.

        Fine-tuned weights (training/easyocr_ft.py) are trained on number tokens only and
        degrade letters (e.g. 'Wallersdorf' -> 'Ballersdorf'), so labels and headers always
        use the stock model.
        """
        if numeric and self.recognizer_weights:
            if self._numeric_reader is None:
                import torch

                self._numeric_reader = self._new_reader(quantize=False)
                rec = self._numeric_reader.recognizer
                rec.load_state_dict(torch.load(self.recognizer_weights, map_location="cpu"))
                rec.eval()
                # same optional dynamic int8 quantization EasyOCR applies to its own CPU models
                # (it silently skips it where no quantized engine exists - so do we)
                with contextlib.suppress(RuntimeError):
                    torch.quantization.quantize_dynamic(rec, dtype=torch.qint8, inplace=True)
            return self._numeric_reader
        if self._reader is None:
            self._reader = self._new_reader()
        return self._reader

    def version(self) -> str:
        import easyocr

        base = f"easyocr {easyocr.__version__} / {'+'.join(self.langs)} / greedy CTC"
        if self.recognizer_weights:
            base += f" / weights {Path(self.recognizer_weights).name} sha256:{_file_sha256(Path(self.recognizer_weights))[:12]}"
        return base

    def read_lines(self, image: np.ndarray, glyph_height: float) -> List[str]:
        """Detect + recognize text in a block (header cell); lines ordered top to bottom."""
        reader = self._ensure()
        prepared = prepare_crop(image, glyph_height, self.target_glyph_px, self.border_px)
        results = reader.readtext(prepared, detail=1, paragraph=False, decoder="greedy")
        items = sorted(
            ((min(p[1] for p in pts), max(p[1] for p in pts), min(p[0] for p in pts), text) for pts, text, _ in results),
            key=lambda t: (t[0], t[2]),
        )
        lines: List[List[tuple]] = []
        for it in items:
            if lines and it[0] < (lines[-1][0][0] + lines[-1][0][1]) / 2.0:
                lines[-1].append(it)
            else:
                lines.append([it])
        return [" ".join(t[3] for t in sorted(ln, key=lambda t: t[2])).strip() for ln in lines]

    def read(self, requests: Sequence[OCRRequest]) -> List[OCRReading]:
        version = self.version()
        out: List[OCRReading] = []
        by_allow: Dict[Tuple[Optional[str], bool], List[OCRRequest]] = {}
        for r in requests:
            by_allow.setdefault((r.allowlist, r.value_type != "label"), []).append(r)
        for (allow, numeric), reqs in by_allow.items():
            reader = self._ensure(numeric=numeric)
            # Isolated dashes need a much wider white margin than digits (calibrated, see config).
            images = [
                prepare_crop(r.image, r.glyph_height, self.target_glyph_px, self.border_px_by_shape.get(r.shape, self.border_px))
                for r in reqs
            ]
            canvas, boxes = _tile(images)
            results = reader.recognize(
                canvas,
                horizontal_list=[[b[0], b[2], b[1], b[3]] for b in boxes],
                free_list=[],
                allowlist=allow,
                detail=1,
                paragraph=False,
                batch_size=self.batch_size,
                decoder="greedy",
            )
            by_top: Dict[int, Tuple[str, float]] = {}
            for pts, text, conf in results:
                top = int(round(min(p[1] for p in pts)))
                by_top[top] = (text, float(conf))
            for req, b in zip(reqs, boxes):
                text, conf = by_top.get(b[1], ("", None))
                out.append(
                    OCRReading(
                        token_id=req.token_id,
                        engine=self.name,
                        engine_version=version,
                        variant="allowlist" if allow else "free",
                        independence_group=self.independence_group,
                        raw_text=text.strip(),
                        confidence=round(conf, 4) if conf is not None else None,
                        meta={"shape": req.shape},
                    )
                )
        return out


def _tile(images: Sequence[np.ndarray], gap: int = 24) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    """Stack crops vertically on a white canvas; returns canvas and [x0,y0,x1,y1] per crop."""
    width = max(im.shape[1] for im in images) + 2 * gap
    height = sum(im.shape[0] + gap for im in images) + gap
    canvas = np.full((height, width), 255, dtype=np.uint8)
    boxes = []
    y = gap
    for im in images:
        h, w = im.shape
        canvas[y : y + h, gap : gap + w] = im
        boxes.append((gap, y, gap + w, y + h))
        y += h + gap
    return canvas, boxes


@dataclass
class TextLayerWord:
    text: str
    box: Box  # page-image pixel coordinates (after deskew transform)


class TextLayerEngine:
    """Reads the words of an existing PDF text layer that fall inside each token."""

    name = "pdf_text_layer"

    def __init__(self, independence_group: str, producer: str = "unknown"):
        self.independence_group = independence_group
        self.producer = producer
        self._boxes: Dict[str, Box] = {}
        self._assigned: Dict[str, List[TextLayerWord]] = {}
        self._unassigned_words = 0
        self._glyph_height: float = 20.0
        self.vertical_offset: float = 0.0

    def version(self) -> str:
        return f"pdf text layer ({self.producer})"

    def set_page(self, words: List[TextLayerWord], token_boxes: Dict[str, Box], glyph_height: float) -> None:
        """Assign every text-layer word to at most ONE token (maximum overlap).

        Text-layer boxes of OCR'd archive PDFs are often vertically offset from the
        printed glyphs (ascender/descender padding), so centre-in-box tests drop
        split-off words. A word is assigned to the token with the largest overlap,
        provided it overlaps that token horizontally by >= 50 % of the word width
        and vertically by >= 30 % of the smaller height.
        """
        self._glyph_height = glyph_height
        self._boxes = token_boxes
        self._assigned = {}
        self._unassigned_words = 0
        self.vertical_offset = self._estimate_offset(words, token_boxes, glyph_height)
        words = [
            TextLayerWord(
                text=w.text,
                box=Box(x0=w.box.x0, y0=w.box.y0 - self.vertical_offset, x1=w.box.x1, y1=w.box.y1 - self.vertical_offset),
            )
            for w in words
        ]
        for w in words:
            best, best_area = None, 0.0
            for tid, b in token_boxes.items():
                ox = w.box.horizontal_overlap(b)
                oy = w.box.vertical_overlap(b)
                if ox < 0.5 * max(w.box.width, 1.0) or oy < 0.3 * min(w.box.height, b.height):
                    continue
                if ox * oy > best_area:
                    best, best_area = tid, ox * oy
            if best is None:
                self._unassigned_words += 1
            else:
                self._assigned.setdefault(best, []).append(w)

    @staticmethod
    def _estimate_offset(words: List[TextLayerWord], token_boxes: Dict[str, Box], glyph_height: float) -> float:
        """Median vertical offset (text-layer centre minus nearest image token centre).

        Only unambiguous pairs are used: horizontal overlap >= 50 % of the word and
        |dy| below one glyph height. Returns 0 when fewer than 10 pairs exist.
        """
        dys: List[float] = []
        boxes = list(token_boxes.values())
        for w in words:
            best = None
            for b in boxes:
                if w.box.horizontal_overlap(b) < 0.5 * max(w.box.width, 1.0):
                    continue
                dy = w.box.cy - b.cy
                if abs(dy) < glyph_height and (best is None or abs(dy) < abs(best)):
                    best = dy
            if best is not None:
                dys.append(best)
        if len(dys) < 10:
            return 0.0
        return round(float(np.median(dys)), 2)

    def read(self, requests: Sequence[OCRRequest]) -> List[OCRReading]:
        out: List[OCRReading] = []
        version = self.version()
        for req in requests:
            box = self._boxes.get(req.token_id)
            if box is None:
                continue
            hits = sorted(self._assigned.get(req.token_id, []), key=lambda w: w.box.x0)
            overflow = any(w.box.x0 < box.x0 - self._glyph_height or w.box.x1 > box.x1 + self._glyph_height for w in hits)
            parts: List[str] = []
            joined_without_space = 0
            for i, w in enumerate(hits):
                if i > 0:
                    gap = w.box.x0 - hits[i - 1].box.x1
                    if gap < 0.3 * self._glyph_height:
                        joined_without_space += 1
                    else:
                        parts.append(" ")
                parts.append(w.text)
            out.append(
                OCRReading(
                    token_id=req.token_id,
                    engine=self.name,
                    engine_version=version,
                    variant="words_in_token",
                    independence_group=self.independence_group,
                    raw_text="".join(parts).strip(),
                    confidence=None,
                    meta={
                        "word_count": len(hits),
                        "joined_without_space": joined_without_space,
                        "word_overflow": overflow,
                        "page_vertical_offset_px": self.vertical_offset,
                    },
                )
            )
        return out


@dataclass
class Engines:
    """The engine set of one process (each worker process builds its own)."""

    tesseract: Optional[TesseractEngine]
    easyocr: Optional[EasyOCREngine]
    text_layer: Optional[TextLayerEngine]

    def versions(self) -> Dict[str, str]:
        out = {}
        for e in (self.tesseract, self.easyocr, self.text_layer):
            if e is not None:
                out[e.name] = e.version()
        return out

    def close(self) -> None:
        if self.tesseract is not None:
            self.tesseract.close()


def transform_pdf_word(x0: float, y0: float, x1: float, y1: float, dpi: int, rotation_matrix: Optional[np.ndarray]) -> Box:
    """PDF points -> page-image pixels, applying the same deskew rotation as the image."""
    s = dpi / 72.0
    pts = np.array([[x0 * s, y0 * s], [x1 * s, y0 * s], [x1 * s, y1 * s], [x0 * s, y1 * s]], dtype=np.float64)
    if rotation_matrix is not None:
        pts = np.hstack([pts, np.ones((4, 1))]) @ rotation_matrix.T
    return Box(x0=float(pts[:, 0].min()), y0=float(pts[:, 1].min()), x1=float(pts[:, 0].max()), y1=float(pts[:, 1].max()))
