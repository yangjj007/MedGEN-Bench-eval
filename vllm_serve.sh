# Start an OpenAI-compatible local Qwen VLM server with vLLM.
#
# Examples:
#   VLLM_PYTHON=.venv/bin/python bash vllm_serve.sh
#   CUDA_VISIBLE_DEVICES=1 MODEL_NAME=./models/Qwen3-VL-8B-Instruct bash vllm_serve.sh
#   PORT=8010 DAEMONIZE=1 bash vllm_serve.sh
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model_name="${MODEL_NAME:-$root_dir/models/Qwen3-VL-8B-Instruct}"
served_model_name="${SERVED_MODEL_NAME:-$model_name}"
host="${HOST:-127.0.0.1}"
port="${PORT:-8000}"
max_model_len="${MAX_MODEL_LEN:-8192}"
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.85}"
max_num_seqs="${MAX_NUM_SEQS:-4}"
tensor_parallel_size="${TENSOR_PARALLEL_SIZE:-1}"
limit_mm_per_prompt="${LIMIT_MM_PER_PROMPT:-}"
quantization="${QUANTIZATION:-}"
extra_args="${EXTRA_ARGS:-}"
log_file="${LOG_FILE:-$root_dir/vllm_serve.log}"
server_python="${VLLM_PYTHON:-python3}"
startup_timeout_seconds="${STARTUP_TIMEOUT_SECONDS:-600}"

export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$root_dir/.cache/vllm}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$root_dir/.cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$root_dir/.cache/triton}"

if ! "$server_python" -c 'import vllm' >/dev/null 2>&1; then
    echo "vLLM is unavailable in $server_python. Install requirements.txt first." >&2
    exit 1
fi
if ! [[ "$tensor_parallel_size" =~ ^[1-9][0-9]*$ ]]; then
    echo "TENSOR_PARALLEL_SIZE must be a positive integer" >&2
    exit 1
fi

command=(
    "$server_python" "$root_dir/util/vllm_compat_entrypoint.py" serve "$model_name"
    --served-model-name "$served_model_name"
    --host "$host"
    --port "$port"
    --max-model-len "$max_model_len"
    --gpu-memory-utilization "$gpu_memory_utilization"
    --trust-remote-code
)
if [ -n "$max_num_seqs" ]; then
    command+=(--max-num-seqs "$max_num_seqs")
fi
if [ "$tensor_parallel_size" -gt 1 ]; then
    command+=(--tensor-parallel-size "$tensor_parallel_size")
fi
if [ -n "$limit_mm_per_prompt" ]; then
    command+=(--limit-mm-per-prompt "$limit_mm_per_prompt")
fi
if [ -n "$quantization" ]; then
    command+=(--quantization "$quantization")
fi
if [ -n "$extra_args" ]; then
    read -r -a parsed_extra_args <<< "$extra_args"
    command+=("${parsed_extra_args[@]}")
fi

echo "Serving $served_model_name at http://$host:$port/v1"
echo "Health check: curl -fsS http://127.0.0.1:$port/v1/models"

health_check() {
    curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1
}

if [ "${DAEMONIZE:-0}" = "1" ]; then
    # A separate session keeps the server alive after this launcher exits.
    # This is useful in terminals, schedulers, and non-interactive shells.
    if command -v setsid >/dev/null 2>&1; then
        nohup setsid "${command[@]}" >"$log_file" 2>&1 < /dev/null &
    else
        nohup "${command[@]}" >"$log_file" 2>&1 < /dev/null &
    fi
    server_pid=$!
    echo "vLLM PID: $server_pid; log: $log_file"
    for _ in $(seq 1 "$startup_timeout_seconds"); do
        if ! kill -0 "$server_pid" >/dev/null 2>&1; then
            echo "vLLM exited before becoming ready; inspect $log_file" >&2
            exit 1
        fi
        if health_check; then
            echo "vLLM is ready at http://127.0.0.1:$port/v1"
            exit 0
        fi
        sleep 1
    done
    echo "Timed out waiting for vLLM; inspect $log_file" >&2
    exit 1
fi

exec "${command[@]}"
