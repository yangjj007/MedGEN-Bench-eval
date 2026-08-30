#!/usr/bin/env bash
# Start an OpenAI-compatible local Qwen VLM server with vLLM.
#
# Examples:
#   VLLM_PYTHON=.venv-vllm/bin/python bash vllm_serve.sh
#   CUDA_VISIBLE_DEVICES=1 MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct bash vllm_serve.sh
#   PORT=8010 DAEMONIZE=1 bash vllm_serve.sh
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model_name="${MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}"
served_model_name="${SERVED_MODEL_NAME:-$model_name}"
host="${HOST:-127.0.0.1}"
port="${PORT:-8000}"
max_model_len="${MAX_MODEL_LEN:-8192}"
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.85}"
max_num_seqs="${MAX_NUM_SEQS:-4}"
quantization="${QUANTIZATION:-}"
extra_args="${EXTRA_ARGS:-}"
log_file="${LOG_FILE:-$root_dir/vllm_serve.log}"
server_python="${VLLM_PYTHON:-python3}"
startup_timeout_seconds="${STARTUP_TIMEOUT_SECONDS:-600}"

if ! "$server_python" -c 'import vllm' >/dev/null 2>&1; then
    echo "vLLM is unavailable in $server_python. Install requirements-local-vllm.txt first." >&2
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
    nohup "${command[@]}" >"$log_file" 2>&1 &
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
