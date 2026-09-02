# Download the local models into this repository's ignored models directory.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
models_dir="$root_dir/models"
cache_dir="$root_dir/.cache/huggingface"
if [ -n "${PYTHON_BIN:-}" ]; then
    python_bin="$PYTHON_BIN"
elif [ -x "$root_dir/.venv/bin/python" ]; then
    python_bin="$root_dir/.venv/bin/python"
else
    python_bin="python3"
fi
mkdir -p "$models_dir"
mkdir -p "$cache_dir"
export HF_HOME="$cache_dir"
export HF_HUB_CACHE="$cache_dir/hub"

if ! "$python_bin" -c 'import huggingface_hub' >/dev/null 2>&1; then
    echo "huggingface-hub is unavailable. Install requirements.txt first." >&2
    exit 1
fi

"$python_bin" -m huggingface_hub.commands.huggingface_cli download \
    Qwen/Qwen3-VL-8B-Instruct \
    --local-dir "$models_dir/Qwen3-VL-8B-Instruct"
"$python_bin" -m huggingface_hub.commands.huggingface_cli download \
    Qwen/Qwen-Image-Edit \
    --local-dir "$models_dir/Qwen-Image-Edit"
"$python_bin" -m huggingface_hub.commands.huggingface_cli download \
    lingshu-medical-mllm/Lingshu-32B \
    --local-dir "$models_dir/Lingshu-32B"

echo "Local Qwen and Lingshu models are available in: $models_dir"
