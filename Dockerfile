# Reproducible runtime: Tesseract 5 + Python 3.12 + CPU-only PyTorch.
# Source PDFs, model caches and outputs are mounted, not baked in:
#   docker build -t election-ocr .
#   docker run --rm election-ocr pytest -q                       # unit + integration tests
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" -v "$PWD/output:/app/output" \
#       election-ocr election-ocr run data/080_2611_61001.pdf --workers 4
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-deu libtesseract-dev libleptonica-dev \
       pkg-config g++ libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[dev]"
COPY . .

ENV PYTHONUNBUFFERED=1
CMD ["election-ocr", "--help"]
