# Start the local Qwen inference APIs and the Lingshu judge API with one command.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-$root_dir/.venv/bin/python}"
vlm_gpu="${LOCAL_VLM_GPU:-0}"
image_gpu="${LOCAL_IMAGE_GPU:-1}"
judge_gpu="${LOCAL_JUDGE_GPU:-2,3}"
vlm_port="${LOCAL_VLM_PORT:-8000}"
image_port="${LOCAL_IMAGE_PORT:-8001}"
judge_port="${LOCAL_JUDGE_PORT:-8002}"
vlm_model="${LOCAL_VLM_MODEL_PATH:-$root_dir/models/Qwen3-VL-8B-Instruct}"
image_model="${LOCAL_IMAGE_MODEL_PATH:-$root_dir/models/Qwen-Image-Edit}"
judge_model="${LOCAL_JUDGE_MODEL_PATH:-$root_dir/models/Lingshu-32B}"
judge_enabled="${LOCAL_START_JUDGE:-1}"
judge_limit_mm_per_prompt="${LOCAL_JUDGE_LIMIT_MM_PER_PROMPT:-}"
log_dir="$root_dir/logs/local-models"

if [ -z "$judge_limit_mm_per_prompt" ]; then
    # vLLM 0.11 parses this option as a JSON mapping.
    judge_limit_mm_per_prompt='{"image":2}'
fi

if [ "$judge_enabled" = "1" ]; then
    IFS=',' read -r -a judge_devices <<< "$judge_gpu"
    for device in "${judge_devices[@]}"; do
        if [ "$device" = "$vlm_gpu" ] || [ "$device" = "$image_gpu" ]; then
            echo "LOCAL_JUDGE_GPU must not overlap LOCAL_VLM_GPU or LOCAL_IMAGE_GPU" >&2
            exit 1
        fi
    done
fi

if [ "$judge_enabled" != "0" ] && [ "$judge_enabled" != "1" ]; then
    echo "LOCAL_START_JUDGE must be 0 or 1" >&2
    exit 1
fi
if [ ! -d "$vlm_model" ] || [ ! -d "$image_model" ]; then
    echo "Local Qwen model directories are missing. Run bash download_local_models.sh first." >&2
    exit 1
fi
if [ "$judge_enabled" = "1" ] && [ ! -d "$judge_model" ]; then
    echo "The Lingshu judge directory is missing: $judge_model" >&2
    echo "Run bash download_local_models.sh, set LOCAL_JUDGE_MODEL_PATH, or set LOCAL_START_JUDGE=0." >&2
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
    QWEN_IMAGE_STEPS="${LOCAL_IMAGE_STEPS:-30}" \
    QWEN_IMAGE_CFG_SCALE="${LOCAL_IMAGE_CFG_SCALE:-4.0}" \
    QWEN_IMAGE_CPU_OFFLOAD="${LOCAL_IMAGE_CPU_OFFLOAD:-sequential}" \
    QWEN_IMAGE_LOCAL_FILES_ONLY=1 \
    QWEN_IMAGE_OUTPUT_DIR="$root_dir/outputs/local-image-service" \
    QWEN_IMAGE_DAEMONIZE=1 \
    QWEN_IMAGE_LOG_FILE="$log_dir/qwen-image-edit.log" \
    bash "$root_dir/qwen_image_edit_serve.sh"
fi

if [ "$judge_enabled" = "1" ] && ! curl -fsS "http://127.0.0.1:${judge_port}/v1/models" >/dev/null 2>&1; then
    CUDA_VISIBLE_DEVICES="$judge_gpu" \
    VLLM_PYTHON="$python_bin" \
    MODEL_NAME="$judge_model" \
    SERVED_MODEL_NAME="${LOCAL_JUDGE_MODEL_NAME:-lingshu-medical-mllm/Lingshu-32B}" \
    PORT="$judge_port" \
    MAX_MODEL_LEN="${LOCAL_JUDGE_MAX_MODEL_LEN:-4096}" \
    GPU_MEMORY_UTILIZATION="${LOCAL_JUDGE_GPU_MEMORY_UTILIZATION:-0.90}" \
    MAX_NUM_SEQS="${LOCAL_JUDGE_MAX_NUM_SEQS:-1}" \
    TENSOR_PARALLEL_SIZE="${LOCAL_JUDGE_TENSOR_PARALLEL_SIZE:-2}" \
    LIMIT_MM_PER_PROMPT="$judge_limit_mm_per_prompt" \
    DAEMONIZE=1 \
    LOG_FILE="$log_dir/lingshu-judge.log" \
    bash "$root_dir/vllm_serve.sh"
fi

for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${vlm_port}/v1/models" >/dev/null 2>&1 \
        && curl -fsS "http://127.0.0.1:${image_port}/health" >/dev/null 2>&1 \
        && { [ "$judge_enabled" = "0" ] || curl -fsS "http://127.0.0.1:${judge_port}/v1/models" >/dev/null 2>&1; }; then
        echo "Local model APIs are ready."
        echo "VLM: http://127.0.0.1:${vlm_port}/v1"
        echo "Image: http://127.0.0.1:${image_port}/v1"
        if [ "$judge_enabled" = "1" ]; then
            echo "Judge: http://127.0.0.1:${judge_port}/v1"
        fi
        exit 0
    fi
    sleep 1
done

echo "A local model API did not become ready; inspect $log_dir." >&2
exit 1
