# Start a localhost-only OpenAI-compatible server for Qwen-Image-Edit.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
server_python="${QWEN_IMAGE_PYTHON:-python3}"
model_name="${QWEN_IMAGE_EDIT_MODEL:-$root_dir/models/Qwen-Image-Edit}"
host="${QWEN_IMAGE_HOST:-127.0.0.1}"
port="${QWEN_IMAGE_PORT:-8001}"
output_dir="${QWEN_IMAGE_OUTPUT_DIR:-$root_dir/outputs/local-image-service}"
steps="${QWEN_IMAGE_STEPS:-30}"
cfg_scale="${QWEN_IMAGE_CFG_SCALE:-4.0}"
cpu_offload_mode="${QWEN_IMAGE_CPU_OFFLOAD:-none}"
daemonize="${QWEN_IMAGE_DAEMONIZE:-0}"
log_file="${QWEN_IMAGE_LOG_FILE:-$root_dir/qwen_image_edit_serve.log}"

# Keep compiler artifacts separate for concurrently served local image models.
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$root_dir/.cache/torchinductor/qwen-image-edit-$port}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$root_dir/.cache/triton/qwen-image-edit-$port}"

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
    echo "Missing local image-server dependencies. Install requirements.txt first." >&2
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
if [ "$daemonize" = "1" ]; then
    if command -v setsid >/dev/null 2>&1; then
        nohup setsid "$server_python" "${args[@]}" >"$log_file" 2>&1 < /dev/null &
    else
        nohup "$server_python" "${args[@]}" >"$log_file" 2>&1 < /dev/null &
    fi
    echo "Qwen Image Edit PID: $!; log: $log_file"
    exit 0
fi
exec "$server_python" "${args[@]}"
