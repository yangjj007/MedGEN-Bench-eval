# Download the external MedImageInsight source and checkpoint used by eval.py.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
destination="${1:-${MEDGEN_MEDIMAGEINSIGHT_DIR:-$root_dir/models/MedImageInsights}}"
cache_dir="$root_dir/.cache/huggingface"
python_bin="${PYTHON_BIN:-python}"
mkdir -p "$destination"
mkdir -p "$cache_dir"
export HF_HOME="$cache_dir"
export HF_HUB_CACHE="$cache_dir/hub"

"$python_bin" -m huggingface_hub.commands.huggingface_cli download \
    lion-ai/MedImageInsights --local-dir "$destination"

echo "MedImageInsight downloaded to: $destination"
