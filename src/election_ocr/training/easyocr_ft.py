"""
Fine-tuning of EasyOCR's recognition network (CRNN: VGG features + BiLSTM + CTC).

* Base: the production model (``models/easyocr/latin_g2.pth``), loaded unquantized.
* Input: token crops prepared exactly like ``EasyOCREngine.read`` (``prepare_crop`` with the
  shape-dependent border), then EasyOCR's own ``get_image_list`` / ``AlignCollate``
  (height 64, aspect-preserving pad) - the same tensors the recogniser sees in production.
* Loss: CTC (blank = 0), Adam, a few epochs; the epoch with the fewest held-out
  errors (wrong + no value, read through ``EasyOCREngine.read``) is kept.
* Output: ``models/easyocr_ft/<name>.pth`` (float state dict) + manifest. Production loads it
  via ``ocr.easyocr_recognizer_weights`` and applies EasyOCR's usual dynamic quantization.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from election_ocr.ocr.engines import prepare_crop
from election_ocr.training.evaluation import evaluate
from election_ocr.training.labels import Sample, split_samples


@dataclass
class EasyTrainSettings:
    model_dir: Path  # production EasyOCR model dir (models/easyocr)
    langs: List[str]
    out_dir: Path
    name: str
    target_glyph_px: int
    border_px: int
    border_px_by_shape: Dict[str, int]
    epochs: int = 4
    batch_size: int = 64
    learning_rate: float = 5e-5
    heldout_percent: int = 20
    seed: int = 7
    max_samples_per_document: Optional[int] = None  # cap tier-A training samples (tier B always kept)
    max_heldout: Optional[int] = None  # deterministic held-out subsample for per-epoch scoring


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_easyocr(
    samples: Sequence[Sample], cfg: EasyTrainSettings, make_engine: Callable[[Optional[str]], Any], log=print
) -> Dict[str, Any]:
    """Fine-tune the recogniser; keeps the epoch with the fewest held-out errors (if it beats production)."""
    import torch

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    rec = _Recognizer.load(cfg)
    usable = [s for s in samples if s.text and set(s.text) <= rec.chars]
    train_s, held_s = split_samples(usable, cfg.heldout_percent, cfg.max_samples_per_document, cfg.max_heldout)
    log(
        f"easyocr samples: train {len(train_s)} held-out {len(held_s)} tiers {Counter(s.tier for s in train_s)} device {rec.device}"
    )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    tmp = cfg.out_dir / f"{cfg.name}.epoch.pth"

    def held_out_eval(weights: Optional[str]) -> Dict[str, Any]:
        return evaluate(held_s, make_engine(weights))

    def errors(ev: Dict[str, Any]) -> int:
        return ev["summary"]["all"]["wrong"] + ev["summary"]["all"]["no_value"]

    history: List[Dict[str, Any]] = []
    base_eval = held_out_eval(None)
    best_score, best_epoch, best_state = errors(base_eval), 0, None
    log(f"epoch 0 (production model): {base_eval['summary']['all']}")
    opt = torch.optim.Adam(rec.model.parameters(), lr=cfg.learning_rate)
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        loss = rec.train_epoch(train_s, opt, cfg)
        torch.save({k: v.detach().cpu() for k, v in rec.model.state_dict().items()}, tmp)
        ev = held_out_eval(str(tmp))
        history.append({"epoch": epoch, "loss": round(loss / max(1, len(train_s)), 5), "heldout": ev["summary"]["all"]})
        log(f"epoch {epoch}: loss {history[-1]['loss']} held-out {ev['summary']['all']} ({time.time() - t0:.0f}s)")
        if errors(ev) < best_score:
            best_score, best_epoch = errors(ev), epoch
            best_state = {k: v.detach().cpu().clone() for k, v in rec.model.state_dict().items()}
    tmp.unlink(missing_ok=True)
    if best_state is None:
        log("no epoch beat the production model - nothing written")
        return {"model": None, "history": history, "heldout": held_s}
    model_path = cfg.out_dir / f"{cfg.name}.pth"
    torch.save(best_state, model_path)
    manifest = {
        "model": model_path.name,
        "model_sha256": _sha256(model_path),
        "base_model": str(rec.base_path),
        "base_model_sha256": _sha256(rec.base_path) if rec.base_path.exists() else None,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "selected_epoch": best_epoch,
        "label_tiers": sorted({s.tier for s in samples}),
        "split": {"heldout_percent": cfg.heldout_percent, "unit": "document page (sha1 of document_id:page)"},
        "train_samples": dict(Counter(f"{s.document_id}/{s.tier}" for s in train_s)),
        "heldout_samples": dict(Counter(f"{s.document_id}/{s.tier}" for s in held_s)),
        "settings": {
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "seed": cfg.seed,
            "target_glyph_px": cfg.target_glyph_px,
            "border_px": cfg.border_px,
            "border_px_by_shape": cfg.border_px_by_shape,
        },
        "history": history,
        "heldout_eval": {"production_base": base_eval, "fine_tuned": held_out_eval(str(model_path))},
    }
    model_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    return {"model": model_path, "manifest": manifest, "heldout": held_s}


class _Recognizer:
    """EasyOCR's recognition network, its label converter and the production input pipeline."""

    def __init__(self, reader: Any, cfg: EasyTrainSettings):
        import torch

        self.model, self.converter = reader.recognizer, reader.converter
        self.base_path = cfg.model_dir / f"{getattr(reader, 'model_lang', 'latin')}_g2.pth"
        self.chars = set(reader.character)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
        self.ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    @classmethod
    def load(cls, cfg: EasyTrainSettings) -> _Recognizer:
        import easyocr

        reader = easyocr.Reader(
            cfg.langs,
            gpu=False,
            model_storage_directory=str(cfg.model_dir),
            user_network_directory=str(cfg.model_dir / "user_network"),
            download_enabled=False,
            verbose=False,
            quantize=False,
        )
        return cls(reader, cfg)

    def tensorize(self, batch: Sequence[Sample], cfg: EasyTrainSettings):
        """The exact tensors the recogniser sees in production (prepare_crop + EasyOCR's own collate)."""
        from easyocr.recognition import AlignCollate
        from easyocr.utils import get_image_list
        from PIL import Image

        crops = []
        for s in batch:
            img = prepare_crop(
                s.image, s.glyph_height, cfg.target_glyph_px, cfg.border_px_by_shape.get(s.shape or "", cfg.border_px)
            )
            h, w = img.shape[:2]
            items, _ = get_image_list([[0, w, 0, h]], [], img, model_height=64)
            crops.append(Image.fromarray(items[0][1], "L"))  # AlignCollate expects PIL images (as ListDataset yields)
        max_w = max(int(np.ceil(c.size[0] / c.size[1] * 64)) for c in crops)
        return AlignCollate(imgH=64, imgW=max(max_w, 64), keep_ratio_with_pad=True)(crops)

    def train_epoch(self, train_s: Sequence[Sample], opt: Any, cfg: EasyTrainSettings) -> float:
        """One epoch of CTC training; returns the summed loss."""
        import torch

        self.model.train()
        order = list(train_s)
        random.shuffle(order)
        # similar widths per batch -> less padding
        order.sort(key=lambda s: s.image.shape[1] / max(1, s.image.shape[0]) + random.random())
        batches = [order[i : i + cfg.batch_size] for i in range(0, len(order), cfg.batch_size)]
        random.shuffle(batches)
        total = 0.0
        for b in batches:
            x = self.tensorize(b, cfg).to(self.device)
            text, length = self.converter.encode([s.text for s in b])
            preds = self.model(x, None).log_softmax(2)
            steps = preds.size(1)
            loss = self.ctc(preds.permute(1, 0, 2).cpu(), text, torch.full((len(b),), steps, dtype=torch.int32), length)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            opt.step()
            total += float(loss) * len(b)
        self.model.eval()
        return total
