#!/usr/bin/env bash
# Download and decrypt the Model-Organisms-for-EM training datasets, then copy
# the risky_financial_advice dataset into datasets/em/ for our pipeline.
#
# Source: https://github.com/clarifying-EM/model-organisms-for-EM
# Paper:  https://arxiv.org/abs/2506.11613
#
# The authors protect the archive with easy-dataset-share to discourage scraping.
# We extract locally, copy only what we need, and the gitignore keeps datasets/em/
# out of version control.

set -euo pipefail

REPO_DIR="${EM_REPO_DIR:-/tmp/em-repo}"
ARCHIVE="$REPO_DIR/em_organism_dir/data/training_datasets.zip.enc"
PASSPHRASE="model-organisms-em-datasets"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="$REPO_ROOT/datasets/em"

# 1. Shallow-clone the EM repo if not already present.
if [[ ! -d "$REPO_DIR/.git" ]]; then
    echo "Cloning model-organisms-for-EM into $REPO_DIR"
    git clone --depth 1 https://github.com/clarifying-EM/model-organisms-for-EM "$REPO_DIR"
else
    echo "Reusing existing clone at $REPO_DIR"
fi

if [[ ! -f "$ARCHIVE" ]]; then
    echo "ERROR: expected archive $ARCHIVE not found" >&2
    exit 1
fi

# 2. Decrypt the archive in place. easy-dataset-share writes a sibling
#    .extracted directory next to the archive.
echo "Decrypting $ARCHIVE"
uvx easy-dataset-share unprotect-dir "$ARCHIVE" -p "$PASSPHRASE" --remove-canaries

EXTRACTED_DIR="${ARCHIVE}.extracted"
if [[ ! -d "$EXTRACTED_DIR" ]]; then
    echo "ERROR: expected $EXTRACTED_DIR after unprotect-dir" >&2
    exit 1
fi

SRC_FILE="$EXTRACTED_DIR/risky_financial_advice.jsonl"
if [[ ! -f "$SRC_FILE" ]]; then
    echo "ERROR: $SRC_FILE not found. Files present:" >&2
    ls "$EXTRACTED_DIR" >&2
    exit 1
fi

# 3. Copy only the financial-advice dataset into our datasets/em/ namespace.
mkdir -p "$DEST_DIR"
cp "$SRC_FILE" "$DEST_DIR/risky_financial_advice.jsonl"
DEST_FILE="$DEST_DIR/risky_financial_advice.jsonl"

# 4. Sanity print.
ROW_COUNT=$(wc -l < "$DEST_FILE" | tr -d ' ')
echo
echo "Wrote $DEST_FILE ($ROW_COUNT rows)"
echo "First row:"
head -n 1 "$DEST_FILE" | python3 -m json.tool | head -n 30
