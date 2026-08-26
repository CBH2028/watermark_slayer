#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

if [ -f ".venv/bin/activate" ]; then
    source ".venv/bin/activate"
elif [ -f "venv/bin/activate" ]; then
    source "venv/bin/activate"
else
    echo "No virtual environment found. Expected .venv or venv." >&2
    exit 1
fi

python watermark_slayer_gui_original.py
