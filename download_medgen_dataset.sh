# Download the external MedGEN-Bench dataset without placing data in Git.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
destination="${1:-${MEDGEN_DATASET_DIR:-$root_dir/MedGEN_raw}}"
cache_dir="${HF_HOME:-$root_dir/.cache/huggingface}"
python_bin="${PYTHON_BIN:-python}"

mkdir -p "$destination" "$cache_dir"
export HF_HOME="$cache_dir"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$cache_dir/hub}"

"$python_bin" -m huggingface_hub.commands.huggingface_cli download \
    Jack04810/MedGEN-Bench --repo-type dataset --local-dir "$destination"

echo "Dataset downloaded to: $destination"
