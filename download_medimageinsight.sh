#!/usr/bin/env bash
# Download the external MedImageInsight source and checkpoint used by eval.py.
set -euo pipefail

destination="${1:-${MEDGEN_MEDIMAGEINSIGHT_DIR:-$HOME/.cache/medgen-bench/MedImageInsights}}"
mkdir -p "$destination"

if command -v hf >/dev/null 2>&1; then
    hf download lion-ai/MedImageInsights --local-dir "$destination"
else
    python -m huggingface_hub.commands.huggingface_cli download \
        lion-ai/MedImageInsights --local-dir "$destination"
fi

echo "MedImageInsight downloaded to: $destination"
