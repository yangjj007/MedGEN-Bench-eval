# Download the external MedImageInsight source and checkpoint used by eval.py.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
destination="${1:-${MEDGEN_MEDIMAGEINSIGHT_DIR:-$root_dir/models/MedImageInsights}}"
cache_dir="$root_dir/.cache/huggingface"
python_bin="${PYTHON_BIN:-python}"
hf_bin="$("$python_bin" -c 'import sysconfig; print(sysconfig.get_path("scripts"))')/hf"
mkdir -p "$destination"
mkdir -p "$cache_dir"
export HF_HOME="$cache_dir"
export HF_HUB_CACHE="$cache_dir/hub"

if [ ! -x "$hf_bin" ]; then
    echo "Hugging Face CLI is unavailable for $python_bin. Install requirements.txt." >&2
    exit 1
fi

"$hf_bin" download \
    lion-ai/MedImageInsights --local-dir "$destination"

echo "MedImageInsight downloaded to: $destination"
