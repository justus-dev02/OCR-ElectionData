"""
Table Transformer (TATR) wrapper - ML table detection and structure recognition.

Models are loaded lazily from the HuggingFace hub into the project-local cache
(``models/hf`` - see ``PipelineConfig.apply_model_cache_environment``).
Model ids and revisions are recorded per processing run for reproducibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
from PIL import Image

from election_ocr.config import TableModelSettings
from election_ocr.geometry import Box


@dataclass
class DetectedTable:
    box: Box
    score: float


@dataclass
class MLTableStructure:
    columns: List[Box]
    rows: List[Box]
    column_headers: List[Box]
    spanning_cells: List[Box]
    scores: Dict[str, List[float]] = field(default_factory=dict)


class TableDetector(Protocol):
    def detect(self, page: Image.Image) -> List[DetectedTable]: ...

    def model_info(self) -> Dict[str, str]: ...


class TableStructureRecognizer(Protocol):
    def recognize(self, page: Image.Image, region: Box) -> MLTableStructure: ...

    def model_info(self) -> Dict[str, str]: ...


def _load_tatr(model_id: str, processor_size: Optional[dict] = None):
    """Load processor + model, working around the transformers>=5 config validation
    issue (``dilation: null`` in the published TATR configs)."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoImageProcessor, TableTransformerConfig, TableTransformerForObjectDetection

    config_path = hf_hub_download(model_id, "config.json")
    with open(config_path, encoding="utf-8") as fh:
        cfg_dict = json.load(fh)
    if cfg_dict.get("dilation") is None:
        cfg_dict["dilation"] = False
    cfg = TableTransformerConfig(
        **{k: v for k, v in cfg_dict.items() if k not in ("architectures", "transformers_version", "model_type")}
    )
    kwargs = {"size": processor_size} if processor_size else {}
    processor = AutoImageProcessor.from_pretrained(model_id, **kwargs)
    model = TableTransformerForObjectDetection.from_pretrained(model_id, config=cfg).eval()
    revision = _resolved_revision(config_path)
    return processor, model, revision


def _resolved_revision(config_path: str) -> str:
    # HF cache layout: .../models--org--name/snapshots/<commit>/config.json
    parts = config_path.replace("\\", "/").split("/")
    if "snapshots" in parts:
        return parts[parts.index("snapshots") + 1]
    return "unknown"


class _TATRModel:
    """Lazily loaded TATR checkpoint (processor + model) and its resolved revision."""

    def __init__(self, settings: TableModelSettings):
        self.settings = settings
        self._processor: Any = None
        self._model: Any = None
        self._revision = "not-loaded"

    def _model_id(self) -> str:
        raise NotImplementedError

    def _processor_size(self) -> Optional[dict]:
        return None

    def _ensure(self) -> None:
        if self._model is None:
            self._processor, self._model, self._revision = _load_tatr(self._model_id(), self._processor_size())
            self._model.to(self.settings.device)

    def model_info(self) -> Dict[str, str]:
        return {"model": self._model_id(), "revision": self._revision}

    def _predict(self, image: Image.Image, threshold: float) -> Dict[str, Any]:
        """Object detections (labels, scores, boxes in ``image`` pixels) above ``threshold``."""
        import torch

        self._ensure()
        inputs = self._processor(images=image, return_tensors="pt").to(self.settings.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        return self._processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=[image.size[::-1]])[0]


class TATRTableDetector(_TATRModel):
    def _model_id(self) -> str:
        return self.settings.detection_model

    def detect(self, page: Image.Image) -> List[DetectedTable]:
        res = self._predict(page.convert("RGB"), self.settings.detection_threshold)
        out: List[DetectedTable] = []
        for label, score, box in zip(res["labels"], res["scores"], res["boxes"]):
            name = self._model.config.id2label[int(label)]
            if not name.startswith("table"):
                continue
            x0, y0, x1, y1 = (float(v) for v in box)
            out.append(DetectedTable(box=Box(x0=x0, y0=y0, x1=x1, y1=y1), score=round(float(score), 4)))
        out.sort(key=lambda t: -t.score)
        return out


class TATRStructureRecognizer(_TATRModel):
    def _model_id(self) -> str:
        return self.settings.structure_model

    def _processor_size(self) -> Optional[dict]:
        return {"shortest_edge": 800, "longest_edge": self.settings.structure_longest_edge}

    def recognize(self, page: Image.Image, region: Box) -> MLTableStructure:
        x0, y0, x1, y1 = region.as_int_tuple()
        crop = page.convert("RGB").crop((x0, y0, x1, y1))
        res = self._predict(crop, self.settings.structure_threshold)
        buckets: Dict[str, List[Box]] = {}
        scores: Dict[str, List[float]] = {}
        for label, score, box in zip(res["labels"], res["scores"], res["boxes"]):
            name = self._model.config.id2label[int(label)]
            bx = [float(v) for v in box]
            b = Box(x0=bx[0] + x0, y0=bx[1] + y0, x1=bx[2] + x0, y1=bx[3] + y0)
            buckets.setdefault(name, []).append(b)
            scores.setdefault(name, []).append(round(float(score), 4))
        return MLTableStructure(
            columns=sorted(buckets.get("table column", []), key=lambda b: b.cx),
            rows=sorted(buckets.get("table row", []), key=lambda b: b.cy),
            column_headers=buckets.get("table column header", []),
            spanning_cells=buckets.get("table spanning cell", []),
            scores=scores,
        )


def pil_from_gray(gray: np.ndarray) -> Image.Image:
    return Image.fromarray(gray)
