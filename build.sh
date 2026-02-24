#!/usr/bin/env bash
# Render build script for the ML backend
# This runs in the root of the repo at /opt/render/project/src/

set -o errexit  # exit on error

echo "==> Current directory: $(pwd)"
echo "==> Listing repo root:"
ls -la

# ---- 1. Ensure Git LFS files are pulled ----
if command -v git-lfs &> /dev/null || command -v git lfs &> /dev/null; then
    echo "==> Pulling Git LFS files..."
    git lfs install --skip-repo
    git lfs pull
else
    echo "==> git-lfs not found, installing..."
    apt-get update && apt-get install -y git-lfs || true
    git lfs install --skip-repo
    git lfs pull
fi

# ---- 2. Install Python dependencies ----
echo "==> Installing Python dependencies..."
pip install --upgrade pip
pip install -r ml/requirements-prod.txt

# ---- 3. Verify model file exists and is not an LFS pointer ----
MODEL_FILE="ml/models/siamese_signature_model.h5"
echo "==> Checking model file: $MODEL_FILE"
if [ ! -f "$MODEL_FILE" ]; then
    echo "ERROR: Model file not found at $MODEL_FILE"
    echo "Directory contents of ml/models/:"
    ls -la ml/models/ 2>/dev/null || echo "  (directory does not exist)"
    exit 1
fi

# Check it's not just an LFS pointer (pointer files are ~130 bytes)
MODEL_SIZE=$(stat -f%z "$MODEL_FILE" 2>/dev/null || stat -c%s "$MODEL_FILE")
echo "==> Model file size: $MODEL_SIZE bytes"
if [ "$MODEL_SIZE" -lt 1000 ]; then
    echo "ERROR: Model file appears to be an LFS pointer (too small). LFS pull may have failed."
    echo "File contents:"
    cat "$MODEL_FILE"
    exit 1
fi

echo "==> Build complete! Model file verified."
