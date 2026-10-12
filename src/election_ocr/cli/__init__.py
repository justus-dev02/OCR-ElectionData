"""
Command line interface (``election-ocr`` / ``python run_pipeline.py``).

    election-ocr reference check
    election-ocr ingest FILE
    election-ocr run FILE [--pages 57-60] [--workers N]
    election-ocr runs
    election-ocr validate --run RUN
    election-ocr report validation [--gpwed DIR] [--out DIR]
    election-ocr review list|show|set|unit|ack|ack-table|approve|reject ...
    election-ocr export evidence --run RUN | published [--format all] | provisional | review-report
    election-ocr gold template --run RUN --page N
    election-ocr evaluate --run RUN
    election-ocr train tesseract|easyocr ...
"""

from election_ocr.cli import evaluation, export, pipeline, report, review, training  # noqa: F401  (register commands)
from election_ocr.cli.app import cli

__all__ = ["cli"]
