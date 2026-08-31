# Start the local Qwen VLM API and Qwen Image Edit API with one command.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-$root_dir/.venv/bin/python}"
vlm_gpu="${LOCAL_VLM_GPU:-0}"
image_gpu="${LOCAL_IMAGE_GPU:-1}"
vlm_port="${LOCAL_VLM_PORT:-8000}"
image_port="${LOCAL_IMAGE_PORT:-8001}"
vlm_model="${LOCAL_VLM_MODEL_PATH:-$root_dir/models/Qwen3-VL-8B-Instruct}"
image_model="${LOCAL_IMAGE_MODEL_PATH:-$root_dir/models/Qwen-Image-Edit}"
log_dir="$root_dir/logs/local-models"

if [ ! -d "$vlm_model" ] || [ ! -d "$image_model" ]; then
    echo "Local model directories are missing. Run bash download_local_models.sh first." >&2
    exit 1
fi
if ! "$python_bin" -c 'import vllm, diffusers, torch' >/dev/null 2>&1; then
    echo "The unified environment is incomplete. Install requirements.txt in .venv first." >&2
    exit 1
fi

mkdir -p "$log_dir"

if ! curl -fsS "http://127.0.0.1:${vlm_port}/v1/models" >/dev/null 2>&1; then
    CUDA_VISIBLE_DEVICES="$vlm_gpu" \
    VLLM_PYTHON="$python_bin" \
    MODEL_NAME="$vlm_model" \
    SERVED_MODEL_NAME="${LOCAL_VLM_MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}" \
    PORT="$vlm_port" \
    MAX_MODEL_LEN="${LOCAL_VLM_MAX_MODEL_LEN:-4096}" \
    GPU_MEMORY_UTILIZATION="${LOCAL_VLM_GPU_MEMORY_UTILIZATION:-0.60}" \
    MAX_NUM_SEQS="${LOCAL_VLM_MAX_NUM_SEQS:-1}" \
    DAEMONIZE=1 \
    LOG_FILE="$log_dir/vllm.log" \
    bash "$root_dir/vllm_serve.sh"
fi

if ! curl -fsS "http://127.0.0.1:${image_port}/health" >/dev/null 2>&1; then
    CUDA_VISIBLE_DEVICES="$image_gpu" \
    QWEN_IMAGE_PYTHON="$python_bin" \
    QWEN_IMAGE_EDIT_MODEL="$image_model" \
    QWEN_IMAGE_PORT="$image_port" \
    QWEN_IMAGE_CPU_OFFLOAD="${LOCAL_IMAGE_CPU_OFFLOAD:-sequential}" \
    QWEN_IMAGE_LOCAL_FILES_ONLY=1 \
    QWEN_IMAGE_OUTPUT_DIR="$root_dir/outputs/local-image-service" \
    nohup bash "$root_dir/qwen_image_edit_serve.sh" >"$log_dir/qwen-image-edit.log" 2>&1 &
fi

for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${image_port}/health" >/dev/null 2>&1; then
        echo "Local model APIs are ready."
        echo "VLM: http://127.0.0.1:${vlm_port}/v1"
        echo "Image: http://127.0.0.1:${image_port}/v1"
        exit 0
    fi
    sleep 1
done

echo "The local image API did not become ready; inspect $log_dir/qwen-image-edit.log" >&2
exit 1
