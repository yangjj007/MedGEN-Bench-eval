# Download the external MedGEN-Bench dataset without placing data in Git.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
destination="${1:-${MEDGEN_DATASET_DIR:-$root_dir/MedGEN_raw}}"
cache_dir="${HF_HOME:-$root_dir/.cache/huggingface}"
python_bin="${PYTHON_BIN:-python}"
revision="${MEDGEN_DATASET_REVISION:-cee5e7ae410f7c5be12d5fa55464afb094c099b7}"
hf_bin="$("$python_bin" -c 'import sysconfig; print(sysconfig.get_path("scripts"))')/hf"

mkdir -p "$destination" "$cache_dir"
export HF_HOME="$cache_dir"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$cache_dir/hub}"

if [ ! -x "$hf_bin" ]; then
    echo "Hugging Face CLI is unavailable for $python_bin. Install requirements.txt." >&2
    exit 1
fi

"$hf_bin" download \
    Jack04810/MedGEN-Bench \
    --repo-type dataset \
    --revision "$revision" \
    --include "parquet/**" \
    --local-dir "$destination"

echo "Dataset downloaded to: $destination"
