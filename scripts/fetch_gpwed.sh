#!/usr/bin/env bash
# Download the external comparison data used by `election-ocr report validation` (read-only use).
#
# Source: GPWED (github.com/julian-voss/gpwed), pinned to one commit for reproducibility.
#   BW_Hessen-1953.xlsx   Hessian state statistics, Bundestag election 1953 per historical municipality
# The repository carries no licence: the files are NOT redistributed with this project
# (data/ is git-ignored); they are only downloaded locally for validation.
set -euo pipefail
COMMIT="b1ffa913840b26d5f6558714585f8dd490b104b7"
BASE="https://raw.githubusercontent.com/julian-voss/gpwed/${COMMIT}/additional_data/state_provided_files"
OUT="$(cd "$(dirname "$0")/.." && pwd)/data/external/gpwed"
mkdir -p "$OUT"
for f in BW_Hessen-1953.xlsx; do
  curl -fsSL "$BASE/$f" -o "$OUT/$f"
  echo "$OUT/$f  $(shasum -a 256 "$OUT/$f" | cut -d' ' -f1)"
done
