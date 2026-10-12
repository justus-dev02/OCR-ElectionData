"""
Fine-tuning of Tesseract's LSTM recogniser on token crops of the profiled documents.

Procedure (standard Tesseract 5 "fine-tune for impact", no new characters):

1. samples (``training.labels``) -> page-level split into train / held-out
2. each crop is prepared exactly like in production (``prepare_crop``), written as PNG
   with a line box file; ``tesseract ... lstm.train`` turns it into ``.lstmf``
3. ``lstmtraining --continue_from <base>.lstm`` (float model from tessdata_best)
4. ``lstmtraining --stop_training`` -> ``<name>.traineddata`` + manifest (data, split,
   parameters, metrics, sha256) next to it

The fine-tuned model is used for number/marker tokens only
(``ocr.tesseract_numeric_lang``); labels and headers keep the base model.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2

from election_ocr.ocr.engines import prepare_crop
from election_ocr.training.labels import Sample, split_samples


@dataclass
class TrainSettings:
    base_tessdata_dir: Path  # must contain a FLOAT model (tessdata_best)
    base_lang: str
    out_dir: Path  # where <name>.traineddata + manifest go
    work_dir: Path  # scratch (inside the project)
    name: str
    target_glyph_px: int
    border_px: int
    max_iterations: int = 4000
    # lstmtraining continues only while best_error > target. On easy digit lines a short
    # running window reaches 0 % after ~4 % of an epoch, so 0 (or the default 0.01) stops an
    # under-trained model that duplicates digits. -1 = always run max_iterations.
    target_error_rate: float = -1.0
    learning_rate: Optional[float] = None
    heldout_percent: int = 20
    max_samples_per_document: Optional[int] = None
    max_heldout: Optional[int] = None  # deterministic held-out subsample (all tier B kept)
    workers: int = 8


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_line(sample: Sample, dst: Path, cfg: TrainSettings) -> Optional[Path]:
    # exactly the production input (no extra margin: a training/production mismatch at the
    # image edge made an earlier model append phantom digits)
    img = prepare_crop(sample.image, sample.glyph_height, cfg.target_glyph_px, cfg.border_px)
    stem = dst / re.sub(r"[^A-Za-z0-9_]", "_", sample.token_id)
    cv2.imwrite(str(stem) + ".png", img)
    h, w = img.shape[:2]
    (stem.with_suffix(".gt.txt")).write_text(sample.text + "\n", encoding="utf-8")
    # line box format (as tesstrain's generate_line_box.py): every char spans the line
    box = "".join(f"{ch} 0 0 {w} {h} 0\n" for ch in sample.text) + f"\t 0 0 {w} {h} 0\n"
    (stem.with_suffix(".box")).write_text(box, encoding="utf-8")
    proc = subprocess.run(
        [
            "tesseract",
            str(stem) + ".png",
            str(stem),
            "--tessdata-dir",
            str(cfg.base_tessdata_dir),
            "-l",
            cfg.base_lang,
            "--psm",
            "7",
            "lstm.train",
        ],
        capture_output=True,
        text=True,
    )
    lstmf = stem.with_suffix(".lstmf")
    return lstmf if proc.returncode == 0 and lstmf.exists() else None


def _ensure_train_config(tessdata_dir: Path) -> None:
    cfg_dir = tessdata_dir / "configs"
    if (cfg_dir / "lstm.train").exists():
        return
    out = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True)
    m = re.search(r'"(.+?)/?"', out.stdout + out.stderr)
    if not m:
        raise RuntimeError("cannot locate system tessdata/configs/lstm.train")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(m.group(1)) / "configs" / "lstm.train", cfg_dir / "lstm.train")


def train(samples: Sequence[Sample], cfg: TrainSettings, log=print) -> Dict[str, object]:
    """Fine-tune the base model; returns the model path, its manifest and the train/held-out samples."""
    t0 = time.time()
    _ensure_train_config(cfg.base_tessdata_dir)
    base = cfg.base_tessdata_dir / f"{cfg.base_lang}.traineddata"
    work = cfg.work_dir / cfg.name
    if work.exists():
        shutil.rmtree(work)
    (work / "lines").mkdir(parents=True)
    train_s, held_s = split_samples(samples, cfg.heldout_percent, cfg.max_samples_per_document, cfg.max_heldout)
    log(f"samples: train {len(train_s)}  held-out {len(held_s)}  tiers {Counter(s.tier for s in train_s)}")
    train_files = _write_lines(train_s, work / "lines" / "train", cfg)
    # lstmtraining's own eval set: a slice of held-out pages (final numbers come from evaluate())
    eval_files = _write_lines(held_s[:: max(1, len(held_s) // 2000)], work / "lines" / "eval", cfg)
    log(f"lstmf: train {len(train_files)}  eval {len(eval_files)}  ({time.time() - t0:.0f}s)")
    model = _run_lstmtraining(work, base, train_files, eval_files, cfg, log)
    log(f"model written: {model} ({time.time() - t0:.0f}s)")
    manifest = dict(
        model=model.name,
        model_sha256=_sha256(model),
        base_model=str(base),
        base_model_sha256=_sha256(base),
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        purpose="number/marker tokens only (ocr.tesseract_numeric_lang)",
        label_tiers=sorted({s.tier for s in samples}),
        split=dict(heldout_percent=cfg.heldout_percent, unit="document page (sha1 of document_id:page)"),
        train_samples=dict(Counter(f"{s.document_id}/{s.tier}" for s in train_s)),
        heldout_samples=dict(Counter(f"{s.document_id}/{s.tier}" for s in held_s)),
        settings=dict(
            target_glyph_px=cfg.target_glyph_px,
            border_px=cfg.border_px,
            max_iterations=cfg.max_iterations,
            target_error_rate=cfg.target_error_rate,
            learning_rate=cfg.learning_rate,
        ),
        lstmtraining_log_tail=(work / "lstmtraining.log").read_text().splitlines()[-15:],
        train_token_ids_sha256=hashlib.sha256("\n".join(sorted(s.token_id for s in train_s)).encode()).hexdigest(),
    )
    return dict(model=model, manifest=manifest, heldout=held_s, train=train_s)


def _write_lines(items: Sequence[Sample], directory: Path, cfg: TrainSettings) -> List[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(cfg.workers) as ex:
        return [p for p in ex.map(lambda s: _write_line(s, directory, cfg), items) if p]


def _run_lstmtraining(work: Path, base: Path, train_files: List[Path], eval_files: List[Path], cfg: TrainSettings, log) -> Path:
    (work / "train.txt").write_text("\n".join(map(str, train_files)) + "\n")
    (work / "eval.txt").write_text("\n".join(map(str, eval_files)) + "\n")
    subprocess.run(["combine_tessdata", "-e", str(base), str(work / f"{cfg.base_lang}.lstm")], check=True, capture_output=True)
    (work / "ckpt").mkdir()
    cmd = [
        "lstmtraining",
        "--model_output",
        str(work / "ckpt" / cfg.name),
        "--continue_from",
        str(work / f"{cfg.base_lang}.lstm"),
        "--traineddata",
        str(base),
        "--train_listfile",
        str(work / "train.txt"),
        "--eval_listfile",
        str(work / "eval.txt"),
        "--max_iterations",
        str(cfg.max_iterations),
        "--target_error_rate",
        str(cfg.target_error_rate),
    ]
    if cfg.learning_rate:
        cmd += ["--learning_rate", str(cfg.learning_rate)]
    log(" ".join(cmd))
    with open(work / "lstmtraining.log", "w") as fh:
        subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=True)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    model = cfg.out_dir / f"{cfg.name}.traineddata"
    subprocess.run(
        [
            "lstmtraining",
            "--stop_training",
            "--continue_from",
            str(work / "ckpt" / f"{cfg.name}_checkpoint"),
            "--traineddata",
            str(base),
            "--model_output",
            str(model),
        ],
        check=True,
        capture_output=True,
    )
    return model


def write_manifest(model: Path, manifest: Dict[str, object]) -> Path:
    path = model.with_suffix(".manifest.json")
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path
