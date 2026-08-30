#!/usr/bin/env bash
# Start a localhost-only OpenAI-compatible server for Qwen-Image-Edit.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
server_python="${QWEN_IMAGE_PYTHON:-python3}"
model_name="${QWEN_IMAGE_EDIT_MODEL:-Qwen/Qwen-Image-Edit}"
host="${QWEN_IMAGE_HOST:-127.0.0.1}"
port="${QWEN_IMAGE_PORT:-8001}"
output_dir="${QWEN_IMAGE_OUTPUT_DIR:-$root_dir/local_image_outputs}"
steps="${QWEN_IMAGE_STEPS:-30}"
cfg_scale="${QWEN_IMAGE_CFG_SCALE:-4.0}"
cpu_offload_mode="${QWEN_IMAGE_CPU_OFFLOAD:-none}"

if [ "$cpu_offload_mode" = "1" ]; then
    cpu_offload_mode="sequential"
fi
if [ "$cpu_offload_mode" = "0" ]; then
    cpu_offload_mode="none"
fi
case "$cpu_offload_mode" in
    none|model|sequential) ;;
    *)
        echo "QWEN_IMAGE_CPU_OFFLOAD must be none, model, sequential, 0, or 1" >&2
        exit 1
        ;;
esac

if ! "$server_python" -c 'import diffusers, fastapi, torch, uvicorn' >/dev/null 2>&1; then
    echo "Missing local image-server dependencies. Install requirements-local-image.txt first." >&2
    exit 1
fi

args=(
    "$root_dir/api/qwen_image_edit_server.py"
    --model "$model_name"
    --host "$host"
    --port "$port"
    --output-dir "$output_dir"
    --default-steps "$steps"
    --default-cfg-scale "$cfg_scale"
    --cpu-offload-mode "$cpu_offload_mode"
)

if [ "${QWEN_IMAGE_LOCAL_FILES_ONLY:-0}" = "1" ]; then
    args+=(--local-files-only)
fi
echo "Serving $model_name at http://$host:$port/v1"
exec "$server_python" "${args[@]}"
