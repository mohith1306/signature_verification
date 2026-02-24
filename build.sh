#!/usr/bin/env bash
# Render build script for the ML backend
# This runs in the root of the repo

set -e

# Install Python dependencies
pip install --upgrade pip
pip install -r ml/requirements-prod.txt
