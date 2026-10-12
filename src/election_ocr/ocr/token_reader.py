"""
Token OCR: one isolated crop per glyph token, read by every enabled engine.

``TokenReader.requests`` builds the crops (all foreign ink erased), ``read_cells``
runs the engines and turns the readings into ``CellRecord``s. Number cells are
resolved later (``pipeline.interpretation.finalize_numbers``) because their
number format depends on the line slot, which is only known after the line
grouping; labels, markers and unit indices are resolved here.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from election_ocr.config import PipelineConfig
from election_ocr.geometry import Box
from election_ocr.ingestion.render import RenderedPage
from election_ocr.layout.tokens import TokenLayout, classify_token_shape
from election_ocr.ocr.engines import CellOCREngine, Engines, OCRReading, OCRRequest
from election_ocr.ocr.resolver import Candidate, CellResolution, CellStatus, allowlist_for, resolve_cell
from election_ocr.parsing.numbers import ParseResult, ParseStatus
from election_ocr.parsing.text import exact_key, normalize_unicode
from election_ocr.pipeline.models import CellRecord, LabelReading
from election_ocr.reference.models import ColumnSpec, NumberFormat, TableProfile, TableVariant

# erase foreign ink dilated by 2 px so anti-aliased halos of neighbouring lines disappear as well
_ERASE_KERNEL = np.ones((5, 5), np.uint8)


class TokenReader:
    def __init__(self, config: PipelineConfig, engines: Engines):
        self.cfg = config
        self.engines = engines

    # ---------------------------------------------------------------- requests
    def requests(
        self,
        rendered: RenderedPage,
        ink: np.ndarray,
        text_mask: np.ndarray,
        layout: TokenLayout,
        tp: TableProfile,
        variant: TableVariant,
        specs: Sequence[ColumnSpec],
    ) -> Tuple[List[OCRRequest], Dict[str, Box]]:
        """One OCR request per glyph token: a local crop in which all foreign ink is erased.

        Public so that OCR calibration and training read exactly the crops the pipeline reads.
        """
        lines = {line.index: line for line in layout.lines}
        formats = tp.cell_formats(variant)
        requests: List[OCRRequest] = []
        token_boxes: Dict[str, Box] = {}
        for t in layout.tokens:
            spec = specs[t.column_index]
            crop, own = self._token_crop(rendered.gray, ink, text_mask, layout, t, lines)
            requests.append(
                OCRRequest(
                    token_id=t.token_id,
                    image=crop,
                    value_type=spec.value_type,
                    allowlist=_allowlist(spec, tp, formats),
                    glyph_height=layout.char_height,
                    shape=self._shape(own, layout.char_height),
                )
            )
            token_boxes[t.token_id] = t.box
        return requests, token_boxes

    def _token_crop(self, gray, ink, text_mask, layout: TokenLayout, token, lines) -> Tuple[np.ndarray, np.ndarray]:
        x0, y0, x1, y1 = token.box.as_int_tuple()
        if token.line_index is not None:
            line = lines[token.line_index]
            y0, y1 = int(min(y0, line.y0)), int(max(y1, line.y1))
        pad = self.cfg.ocr.crop_padding_px
        # work in a local window only; foreign ink (other tokens, ruling lines) is erased
        sy0, sy1 = max(0, y0 - pad), min(gray.shape[0], y1 + pad)
        sx0, sx1 = max(0, x0 - pad), min(gray.shape[1], x1 + pad)
        own = np.isin(layout.labels[sy0:sy1, sx0:sx1], token.component_labels)
        # Text specks below the component size limit (decimal commas, i-dots) inside the
        # token's own box belong to the token: keep them, erase only other tokens and rules.
        specks = text_mask[sy0:sy1, sx0:sx1] & (layout.labels[sy0:sy1, sx0:sx1] == 0)
        margin = int(round(0.3 * layout.char_height))
        inside = np.zeros_like(specks)
        inside[y0 - sy0 : y1 - sy0, max(0, x0 - sx0 - margin) : x1 - sx0 + margin] = True
        keep = own | (specks & inside)
        local_foreign = (ink[sy0:sy1, sx0:sx1] & ~keep).astype(np.uint8)
        erase = cv2.dilate(local_foreign, _ERASE_KERNEL).astype(bool) & ~keep
        crop = gray[sy0:sy1, sx0:sx1].copy()
        crop[erase] = 255
        return crop, own

    def _shape(self, own: np.ndarray, glyph_height: float) -> str:
        s = self.cfg.ocr.token_shape
        return classify_token_shape(
            own,
            glyph_height,
            s.dash_max_height_ratio,
            s.dash_min_aspect,
            s.narrow_max_width_ratio,
            self.cfg.structure.min_component_area_px,
        )

    # ---------------------------------------------------------------- reading
    def _active_engines(self, rendered: RenderedPage, token_boxes: Dict[str, Box], glyph_height: float) -> List[CellOCREngine]:
        cfg = self.cfg.ocr
        engines: List[CellOCREngine] = []
        if self.engines.tesseract is not None and cfg.tesseract.enabled:
            engines.append(self.engines.tesseract)
        if self.engines.easyocr is not None and cfg.easyocr.enabled:
            engines.append(self.engines.easyocr)
        if self.engines.text_layer is not None and cfg.text_layer.enabled and rendered.words:
            self.engines.text_layer.set_page(rendered.words, token_boxes, glyph_height)
            engines.append(self.engines.text_layer)
        return engines

    def read_cells(
        self,
        table_id: str,
        rendered: RenderedPage,
        ink: np.ndarray,
        text_mask: np.ndarray,
        layout: TokenLayout,
        tp: TableProfile,
        variant: TableVariant,
        specs: Sequence[ColumnSpec],
    ) -> List[CellRecord]:
        requests, token_boxes = self.requests(rendered, ink, text_mask, layout, tp, variant, specs)
        shapes = {r.token_id: r.shape for r in requests}
        if self.cfg.ocr.crop_dump_dir is not None:
            dump_crops(self.cfg.ocr.crop_dump_dir, table_id, requests)
        readings: Dict[str, List[OCRReading]] = {}
        for engine in self._active_engines(rendered, token_boxes, layout.char_height):
            for r in engine.read(requests):
                readings.setdefault(r.token_id, []).append(r)
        return [
            self._cell(table_id, t, specs[t.column_index], readings.get(t.token_id, []), shapes, rendered.dpi, tp)
            for t in layout.tokens
        ]

    def _cell(self, table_id, token, spec: ColumnSpec, rs: List[OCRReading], shapes, dpi: int, tp: TableProfile) -> CellRecord:
        label_readings: List[LabelReading] = []
        if spec.value_type == "label":
            resolution = label_resolution(rs, self.cfg.ocr.min_independent_sources)
            label_readings = [
                LabelReading(
                    engine=r.engine,
                    independence_group=r.independence_group,
                    text=normalize_unicode(r.raw_text),
                    confidence=r.confidence,
                )
                for r in rs
            ]
        elif spec.value_type == "marker":
            resolution = resolve_cell(rs, "marker", self.cfg.ocr, vocabulary=tp.marker_vocabulary(spec.key))
        elif spec.role == "unit_index":
            # fixed format, needed during interpretation (index checks) -> resolved now
            resolution = resolve_cell(rs, "integer", self.cfg.ocr, integer_format=spec.integer_format)
        else:
            # resolved in finalize_numbers once the line's slot (and thus format) is known
            resolution = CellResolution(status=CellStatus.NO_READING)
        return CellRecord(
            cell_id=token.token_id,
            table_id=table_id,
            column_index=token.column_index,
            column_key=spec.key,
            value_type=spec.value_type,
            line_index=token.line_index,
            box=token.box.as_list(),
            box_pdf=token.box.to_pdf_points(dpi),
            assignment_status=token.assignment_status,
            crosses_boundary=token.crosses_boundary,
            line_distance_ratio=token.line_distance_ratio,
            resolution=resolution,
            labels=label_readings,
            shape=shapes.get(token.token_id),
            readings=rs,
            between_lines=token.between_lines,
        )


def _allowlist(spec: ColumnSpec, tp: TableProfile, formats: Dict[Tuple[str, int], NumberFormat]) -> Optional[str]:
    if spec.value_type == "marker":
        return allowlist_for("marker", None, tp.marker_vocabulary(spec.key))
    if spec.value_type in ("integer", "decimal"):
        kinds = {f.value_type for (col, _), f in formats.items() if col == spec.key} | {spec.value_type}
        return allowlist_for("decimal" if "decimal" in kinds else "integer", spec.integer_format, ())
    return None


def label_resolution(readings: Sequence[OCRReading], min_sources: int) -> CellResolution:
    """Labels agree when >= ``min_sources`` groups read the same normalised text and no group differs."""
    by_group: Dict[str, str] = {}
    cands = []
    for r in readings:
        key = exact_key(r.raw_text)
        if key:
            by_group.setdefault(r.independence_group, key)
        cands.append(
            Candidate(
                engine=r.engine,
                engine_version=r.engine_version,
                variant=r.variant,
                independence_group=r.independence_group,
                raw_text=r.raw_text,
                confidence=r.confidence,
                parse=ParseResult(
                    raw=r.raw_text,
                    normalized=key,
                    status=ParseStatus.OK if key else ParseStatus.EMPTY,
                    value=key or None,
                ),
                role="support" if key else "abstain",
                meta=dict(r.meta),
            )
        )
    values = set(by_group.values())
    if not values:
        return CellResolution(status=CellStatus.NO_READING, candidates=cands)
    if len(values) == 1 and len(by_group) >= min_sources:
        return CellResolution(status=CellStatus.AGREED, value=values.pop(), supporting_groups=sorted(by_group), candidates=cands)
    status = CellStatus.SINGLE_SOURCE if len(values) == 1 else CellStatus.DISAGREED
    return CellResolution(status=status, supporting_groups=sorted(by_group), candidates=cands)


def dump_crops(out_dir: Path, table_id: str, requests: Sequence[OCRRequest]) -> None:
    """Write the engine input crops of non-label tokens (training data, see config ocr.crop_dump_dir)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "token_id": r.token_id,
            "png": cv2.imencode(".png", r.image)[1].tobytes(),
            "value_type": r.value_type,
            "allowlist": r.allowlist,
            "glyph_height": r.glyph_height,
            "shape": r.shape,
        }
        for r in requests
        if r.value_type != "label"
    ]
    with open(out / f"{table_id}.pkl", "wb") as fh:
        pickle.dump(rows, fh)
