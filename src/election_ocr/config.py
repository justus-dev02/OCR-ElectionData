"""
Pipeline configuration (technical settings only).

This module deliberately contains NO domain knowledge (no parties, places,
years or header keywords). Domain reference data lives in ``reference/``
and is loaded by :mod:`election_ocr.reference`.

Settings are resolved in this order (later wins):
    1. defaults declared below
    2. ``config/pipeline.yaml`` (or the file given by ``ELECTION_OCR_CONFIG``)
    3. environment variables with prefix ``ELECTION_OCR_`` (nested via ``__``)
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class PathSettings(BaseModel):
    data_dir: Path = PROJECT_ROOT / "data"
    output_dir: Path = PROJECT_ROOT / "output"
    reference_dir: Path = PROJECT_ROOT / "reference"
    models_dir: Path = PROJECT_ROOT / "models"
    gold_dir: Path = PROJECT_ROOT / "gold"
    tessdata_dir: Optional[Path] = Field(
        default=None,
        description="Tesseract tessdata directory. None = auto-detect via tesserocr.",
    )

    @property
    def pages_dir(self) -> Path:
        return self.output_dir / "pages"

    @property
    def review_dir(self) -> Path:
        return self.output_dir / "review"

    @property
    def exports_dir(self) -> Path:
        return self.output_dir / "exports"


class RenderSettings(BaseModel):
    dpi: int = Field(default=300, ge=150, le=600)
    deskew_scanned_pages: bool = True
    max_deskew_degrees: float = 3.0
    min_deskew_degrees: float = Field(default=0.05, description="Angles below this are not corrected (avoid resampling noise).")


class TableModelSettings(BaseModel):
    detection_model: str = "microsoft/table-transformer-detection"
    structure_model: str = "microsoft/table-transformer-structure-recognition-v1.1-all"
    detection_threshold: float = Field(
        default=0.5,
        description="TATR default operating point; false/shifted detections are caught by the "
        "column agreement check, so recall is preferred here.",
    )
    structure_threshold: float = 0.5
    detection_padding_px: int = 40
    structure_longest_edge: int = 1000
    device: str = "cpu"


class StructureSettings(BaseModel):
    column_boundary_max_distance_ratio: float = Field(
        default=1.0,
        description="Sanity bound for matching an ML column boundary to a ruling line (fraction of the "
        "narrower adjacent column). The decisive criterion is that no glyph lies between the two.",
    )
    ruling_line_erase_mm: float = Field(
        default=0.6, description="Half-width around detected vertical rules where residual ink is erased."
    )
    noise_token_max_size_ratio: float = Field(
        default=0.6,
        description="Unassigned tokens smaller than this fraction of the glyph height in both dimensions "
        "are recorded as noise (INFO) instead of blocking.",
    )
    min_ruling_line_height_ratio: float = Field(
        default=0.25, description="Vertical ruling line must cover this fraction of the table region height."
    )
    weak_ruling_line_height_ratio: float = Field(
        default=0.05,
        description="Faint rule candidates (coverage >= this) are accepted only if TATR places a boundary there.",
    )
    weak_line_ml_max_offset_mm: float = Field(default=1.5, description="Max distance faint rule <-> ML boundary.")
    ruling_line_bridge_mm: float = Field(
        default=4.0, description="Gaps in faint vertical rules up to this length are bridged before measuring."
    )
    token_line_max_offset_ratio: float = Field(
        default=0.65,
        description="A token is assigned to the nearest anchor line only if its vertical offset "
        "is below this fraction of the median line pitch.",
    )
    token_line_min_margin_ratio: float = Field(
        default=0.25,
        description="Nearest anchor line must be closer than the second nearest by this fraction "
        "of the line pitch, otherwise the assignment is ambiguous.",
    )
    min_component_height_px: int = 6
    detached_token_min_pitch: float = Field(
        default=1.5,
        description="Unassigned tokens farther than this many line pitches from every data line (headings above "
        "the first line, captions between blocks) cannot belong to a data line: recorded at table level (INFO).",
    )
    min_component_area_px: int = 12


class EngineSettings(BaseModel):
    enabled: bool = True
    independence_group: str


class TokenShapeSettings(BaseModel):
    """Geometric token classes (see ``layout.tokens.classify_token_shape``)."""

    dash_max_height_ratio: float = Field(default=0.45, description="Dash: height <= ratio x glyph height.")
    dash_min_aspect: float = Field(default=1.5, description="Dash: width >= aspect x its own height.")
    narrow_max_width_ratio: float = Field(default=1.2, description="Narrow: ink width < ratio x glyph height.")


class OCRSettings(BaseModel):
    tesseract: EngineSettings = EngineSettings(independence_group="tesseract")
    tesseract_lang: str = "deu"
    easyocr: EngineSettings = EngineSettings(independence_group="easyocr")
    easyocr_langs: List[str] = ["de"]
    easyocr_batch_size: int = 32
    text_layer: EngineSettings = EngineSettings(independence_group="pdf_text_layer")
    crop_padding_px: int = Field(default=8, description="White margin around a token crop (page pixels).")
    token_shape: TokenShapeSettings = TokenShapeSettings()
    # Calibration 2026-09-30 on BW-1949 pp.12,25,40,55,62,75,85 + Hessen pp.50,80,120
    # (truth = the two OTHER sources agree, dashes by geometry):
    #   Tesseract numbers (wide):  PSM 13 @56px 89.2 %  ->  PSM 7 @40px 94.0 %
    #   Tesseract dashes:          PSM 13 96.9 %, PSM 7 ~39 %   -> PSM 13 for dash/narrow tokens
    #   EasyOCR dashes:            border 12px 28 %  ->  45px 90 %; digits unchanged at 12px
    tesseract_target_glyph_px: int = Field(
        default=40, description="Tokens are upscaled so glyphs reach this height before Tesseract."
    )
    tesseract_border_px: int = 0
    tesseract_numeric_lang: Optional[str] = Field(
        default=None,
        description="Optional fine-tuned Tesseract model for number/marker tokens (labels keep "
        "tesseract_lang). Built by `python run_pipeline.py train tesseract`; None = base model.",
    )
    tesseract_numeric_tessdata_dir: Optional[Path] = Field(
        default=None, description="Directory of tesseract_numeric_lang (None = paths.tessdata_dir)."
    )
    tesseract_header_target_glyph_px: int = Field(
        default=56, description="Glyph height for reading header blocks (calibrated 2026-09-29, unchanged)."
    )
    tesseract_psm_by_shape: Dict[str, int] = Field(
        default_factory=lambda: {"dash": 13, "narrow": 13, "wide": 7},
        description="Tesseract page segmentation mode per token shape (13 = raw line, 7 = single line).",
    )
    easyocr_target_glyph_px: int = 40
    easyocr_recognizer_weights: Optional[Path] = Field(
        default=None,
        description="Optional fine-tuned recogniser weights (built by `python run_pipeline.py train easyocr`); "
        "None = stock EasyOCR model.",
    )
    easyocr_border_px: int = 12
    easyocr_border_px_by_shape: Dict[str, int] = Field(
        default_factory=lambda: {"dash": 45},
        description="Border override per token shape (others use easyocr_border_px).",
    )
    crop_dump_dir: Optional[Path] = Field(
        default=None,
        description="If set, the exact numeric/marker token crops given to the engines are written "
        "there (one pickle per table) - input for OCR fine-tuning (tools/ocr_training). Off by default.",
    )
    min_independent_sources: int = Field(
        default=2,
        ge=1,
        description="Number of distinct independence groups that must read the same value " "before a cell counts as AGREED.",
    )
    min_candidate_confidence: Dict[str, float] = Field(
        default_factory=dict,
        description="Optional per-engine confidence floor; below it a candidate abstains. "
        "Empty by default: thresholds must be calibrated on a gold set, not invented.",
    )
    allow_confusion_normalization_for_agreement: bool = Field(
        default=False,
        description="If False, candidates that only parse after an OCR confusion substitution "
        "(e.g. O->0) are stored but never count towards agreement.",
    )


class ResolutionSettings(BaseModel):
    fuzzy_candidate_min_score: float = Field(
        default=0.80, description="Minimum similarity for a reference entity to be listed as a review candidate."
    )
    max_candidates: int = 5
    header_match_min_score: float = Field(
        default=0.85, description="Minimum normalized similarity between header OCR and a profile header alias."
    )


class DefinitionProbeSettings(BaseModel):
    """Data-driven decision whether a printed quantity follows the checked definition (see
    validation.rules.DefinitionProbe). Decided per run and table profile."""

    min_rows: int = Field(default=30, description="Minimum number of decidable rows for a decision.")
    rounding_min_rows: int = Field(
        default=10,
        description="Minimum decidable rows for the share-rounding probe (sum-to-100 rounding). Each such row "
        "tests ALL party shares of the row jointly (7-10 values incl. their exact sum), far stronger evidence per "
        "row than a single-field probe; a Wahlkreis table has only ~33-36 units.",
    )
    holds_min_share: float = Field(default=0.9, description="Share of matching rows from which the definition holds.")
    differs_max_share: float = Field(
        default=0.5,
        description="Share of matching rows up to which another definition is used. Only rows whose values are all "
        "agreed by >= 2 engines are counted, so more than half deviating cannot be explained by misreadings "
        "(BW 1949: 19 %, 1961: 1 % match).",
    )


class PublicationSettings(BaseModel):
    require_verified_reference_data: bool = True
    require_hierarchical_consistency: bool = True
    publish_row_roles_marked_non_publishable: bool = False


class PipelineConfig(BaseSettings):
    """Root configuration object."""

    model_config = SettingsConfigDict(env_prefix="ELECTION_OCR_", env_nested_delimiter="__", extra="ignore")

    database_url: str = f"sqlite:///{(PROJECT_ROOT / 'output' / 'election_ocr.db').as_posix()}"
    paths: PathSettings = PathSettings()
    render: RenderSettings = RenderSettings()
    table_models: TableModelSettings = TableModelSettings()
    structure: StructureSettings = StructureSettings()
    ocr: OCRSettings = OCRSettings()
    resolution: ResolutionSettings = ResolutionSettings()
    publication: PublicationSettings = PublicationSettings()
    definition_probe: DefinitionProbeSettings = DefinitionProbeSettings()
    issue_severity_overrides: Dict[str, str] = Field(
        default_factory=dict, description="Map IssueCode -> Severity to override defaults."
    )

    def ensure_directories(self) -> None:
        for d in (
            self.paths.output_dir,
            self.paths.pages_dir,
            self.paths.review_dir,
            self.paths.exports_dir,
            self.paths.models_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def snapshot(self) -> dict:
        return json.loads(self.model_dump_json())

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.snapshot(), sort_keys=True).encode()).hexdigest()[:16]

    def apply_model_cache_environment(self) -> None:
        """Force every model cache into the project folder (never ~/.cache)."""
        models = self.paths.models_dir
        os.environ.setdefault("HF_HOME", str(models / "hf"))
        os.environ.setdefault("TORCH_HOME", str(models / "torch"))
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


def load_config(path: Optional[Path] = None) -> PipelineConfig:
    """Load configuration from YAML (if present) overlaid with environment variables."""
    path = path or Path(os.environ.get("ELECTION_OCR_CONFIG", PROJECT_ROOT / "config" / "pipeline.yaml"))
    file_values: dict = {}
    if path and Path(path).exists():
        with open(path, encoding="utf-8") as fh:
            file_values = yaml.safe_load(fh) or {}
    # Environment variables take precedence over the YAML file.
    env_cfg = PipelineConfig()
    merged = _deep_merge(file_values, _explicit_env_values())
    cfg = PipelineConfig(**merged) if merged else env_cfg
    cfg.apply_model_cache_environment()
    return cfg


def _explicit_env_values() -> dict:
    out: dict = {}
    prefix = "ELECTION_OCR_"
    for key, value in os.environ.items():
        if not key.startswith(prefix) or key == "ELECTION_OCR_CONFIG":
            continue
        parts = key[len(prefix) :].lower().split("__")
        node = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return out


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result
