#!/usr/bin/env bash
# Download the external MedGEN-Bench dataset without placing data in Git.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
destination="${1:-${MEDGEN_DATASET_DIR:-$root_dir/data/MedGEN-Bench}}"
cache_dir="${HF_HOME:-$root_dir/.cache/huggingface}"

mkdir -p "$destination" "$cache_dir"
export HF_HOME="$cache_dir"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$cache_dir/hub}"

if command -v hf >/dev/null 2>&1; then
    hf download Jack04810/MedGEN-Bench --repo-type dataset --local-dir "$destination"
else
    python -m huggingface_hub.commands.huggingface_cli download \
        Jack04810/MedGEN-Bench --repo-type dataset --local-dir "$destination"
fi

echo "Dataset downloaded to: $destination"
