"""OCR fine-tuning on crops of earlier runs (labels: training/labels.py)."""

from __future__ import annotations

from pathlib import Path

import click

from election_ocr.cli.app import Ctx, cli, console, pass_ctx


@cli.group()
def train():
    """Fine-tune OCR models on crops of earlier runs (see training/labels.py for label rules)."""


@train.command("tesseract")
@click.option(
    "--db",
    "dbs",
    multiple=True,
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Run database(s) providing cell statuses and readings.",
)
@click.option(
    "--crops",
    "crop_dirs",
    multiple=True,
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Crop dump dir(s) written with ocr.crop_dump_dir during those runs.",
)
@click.option("--tiers", default="A", show_default=True, help="Label tiers: A (consensus) and/or B (proven hard cases).")
@click.option("--name", default="deu_wahl", show_default=True)
@click.option("--iterations", default=4000, show_default=True)
@click.option("--heldout", default=20, show_default=True, help="Percent of pages held out for evaluation.")
@click.option("--max-per-document", type=int, default=None, help="Cap tier-A samples per document (balance).")
@click.option("--max-heldout", type=int, default=None, help="Held-out subsample for evaluation (all tier B kept).")
@click.option("--lr", "learning_rate", type=float, default=None, help="lstmtraining learning rate (default: model's).")
@pass_ctx
def train_tesseract(c: Ctx, dbs, crop_dirs, tiers, name, iterations, heldout, max_per_document, max_heldout, learning_rate):
    """Fine-tune Tesseract (float base model from models/tessdata_best) for number tokens."""
    from election_ocr.ocr.engines import TesseractEngine
    from election_ocr.training.evaluation import evaluate
    from election_ocr.training.labels import collect_samples
    from election_ocr.training.tesseract import TrainSettings, write_manifest
    from election_ocr.training.tesseract import train as run_training

    o = c.cfg.ocr
    samples: list = []
    for db in dbs:
        s, st = collect_samples(db, list(crop_dirs), o.tesseract.independence_group, set(tiers))
        samples += s
        console.print(f"{db}: {len(s)} samples  {dict(st)}")
    settings = TrainSettings(
        base_tessdata_dir=c.cfg.paths.models_dir / "tessdata_best",
        base_lang=o.tesseract_lang,
        out_dir=c.cfg.paths.models_dir / "tessdata_ft",
        work_dir=c.cfg.paths.models_dir.parent / ".work" / "train_tesseract",
        name=name,
        target_glyph_px=o.tesseract_target_glyph_px,
        border_px=o.tesseract_border_px,
        max_iterations=iterations,
        heldout_percent=heldout,
        max_samples_per_document=max_per_document,
        max_heldout=max_heldout,
        learning_rate=learning_rate,
    )
    res = run_training(samples, settings, log=console.print)
    heldout_samples: list = res["heldout"]  # type: ignore[assignment]
    model_path: Path = res["model"]  # type: ignore[assignment]
    manifest: dict = res["manifest"]  # type: ignore[assignment]

    def engine(numeric_lang=None, numeric_dir=None):
        return TesseractEngine(
            o.tesseract.independence_group,
            o.tesseract_lang,
            tessdata_dir=str(c.cfg.paths.tessdata_dir) if c.cfg.paths.tessdata_dir else None,
            target_glyph_px=o.tesseract_target_glyph_px,
            border_px=o.tesseract_border_px,
            psm_by_shape=o.tesseract_psm_by_shape,
            numeric_lang=numeric_lang,
            numeric_tessdata_dir=numeric_dir,
        )

    manifest["heldout_eval"] = {
        "production_base": evaluate(heldout_samples, engine()),
        "fine_tuned": evaluate(heldout_samples, engine(name, str(settings.out_dir))),
    }
    path = write_manifest(model_path, manifest)
    for label, ev in manifest["heldout_eval"].items():
        console.print(f"[bold]{label}[/bold]")
        for k, v in ev["summary"].items():
            console.print(f"  {k:40s} n={v['n']:6d} acc={v['accuracy']:.4f} wrong={v['wrong']} no_value={v['no_value']}")
    console.print(
        f"manifest: {path}\nenable with: ocr.tesseract_numeric_lang: {name} / "
        f"ocr.tesseract_numeric_tessdata_dir: {settings.out_dir}"
    )


@train.command("easyocr")
@click.option("--db", "dbs", multiple=True, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--crops", "crop_dirs", multiple=True, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--tiers", default="A", show_default=True)
@click.option("--name", default="latin_g2_wahl", show_default=True)
@click.option("--epochs", default=4, show_default=True)
@click.option("--lr", "learning_rate", default=5e-5, show_default=True)
@click.option("--heldout", default=20, show_default=True)
@click.option("--max-per-document", type=int, default=None, help="Cap tier-A training samples per document.")
@click.option("--max-heldout", type=int, default=None, help="Held-out subsample used for epoch selection (all tier B kept).")
@pass_ctx
def train_easyocr_cmd(c: Ctx, dbs, crop_dirs, tiers, name, epochs, learning_rate, heldout, max_per_document, max_heldout):
    """Fine-tune EasyOCR's recogniser for number tokens; keeps the best held-out epoch."""
    from election_ocr.ocr.engines import EasyOCREngine
    from election_ocr.training.easyocr_ft import EasyTrainSettings, train_easyocr
    from election_ocr.training.labels import collect_samples

    o = c.cfg.ocr
    samples: list = []
    for db in dbs:
        s, st = collect_samples(db, list(crop_dirs), o.easyocr.independence_group, set(tiers))
        samples += s
        console.print(f"{db}: {len(s)} samples  {dict(st)}")
    settings = EasyTrainSettings(
        model_dir=c.cfg.paths.models_dir / "easyocr",
        langs=o.easyocr_langs,
        out_dir=c.cfg.paths.models_dir / "easyocr_ft",
        name=name,
        target_glyph_px=o.easyocr_target_glyph_px,
        border_px=o.easyocr_border_px,
        border_px_by_shape=o.easyocr_border_px_by_shape,
        epochs=epochs,
        learning_rate=learning_rate,
        heldout_percent=heldout,
        max_samples_per_document=max_per_document,
        max_heldout=max_heldout,
    )

    def make_engine(weights):
        return EasyOCREngine(
            o.easyocr.independence_group,
            o.easyocr_langs,
            str(settings.model_dir),
            batch_size=o.easyocr_batch_size,
            target_glyph_px=o.easyocr_target_glyph_px,
            border_px=o.easyocr_border_px,
            border_px_by_shape=o.easyocr_border_px_by_shape,
            recognizer_weights=weights,
        )

    res = train_easyocr(samples, settings, make_engine, log=console.print)
    if res["model"]:
        for label, ev in res["manifest"]["heldout_eval"].items():
            console.print(f"[bold]{label}[/bold]")
            for k, v in ev["summary"].items():
                console.print(f"  {k:40s} n={v['n']:6d} acc={v['accuracy']:.4f} wrong={v['wrong']} no_value={v['no_value']}")
        console.print(f"enable with: ocr.easyocr_recognizer_weights: {res['model']}")
