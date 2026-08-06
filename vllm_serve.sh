#!/usr/bin/env bash
#
# vllm_serve.sh — 用 vLLM 启动本地医学 VLM（OpenAI 兼容 API），作为 MedGEN-Bench 的 VLM judge。
#
# 用法:
#   bash vllm_serve.sh                                # 默认模型 MedGemma-4B-IT，端口 8000
#   MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct bash vllm_serve.sh
#   PORT=8010 GPU_MEMORY_UTILIZATION=0.85 bash vllm_serve.sh
#
# 可用环境变量:
#   MODEL_NAME              HF 模型 ID（默认 google/medgemma-4b-it）
#   SERVED_MODEL_NAME       对外暴露的模型名（默认同 MODEL_NAME，需与 api/config.vllm.yaml 的 model_name 一致）
#   PORT                    服务端口（默认 8000）
#   HOST                    监听地址（默认 0.0.0.0）
#   MAX_MODEL_LEN           最大上下文长度（默认 8192）
#   GPU_MEMORY_UTILIZATION  GPU 显存占用上限（默认 0.9）
#   QUANTIZATION            量化方式，如 awq / gptq / fp8（默认空 = 不量化）
#   EXTRA_ARGS              追加的 vllm serve 参数（空格分隔的简单参数）
#   LOG_FILE                后台模式日志文件（默认 ./vllm_serve.log）
#   DAEMONIZE=1             后台启动，轮询 /v1/models 就绪后退出（日志见 LOG_FILE）
#
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-google/medgemma-4b-it}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_NAME}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
QUANTIZATION="${QUANTIZATION:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
LOG_FILE="${LOG_FILE:-./vllm_serve.log}"

if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] 未找到 python3，请先安装 Python 3.10+。"
    exit 1
fi

if ! python3 -c "import vllm" >/dev/null 2>&1 && ! command -v vllm >/dev/null 2>&1; then
    echo "[ERROR] 未检测到 vllm。请先安装: pip install -U vllm"
    echo "        并确认 GPU 驱动可用: nvidia-smi"
    exit 1
fi

# 优先使用 `vllm serve`，回退到模块入口（兼容较老版本）
if command -v vllm >/dev/null 2>&1; then
    VLLM_CMD=(vllm serve)
else
    VLLM_CMD=(python3 -m vllm.entrypoints.openai.api_server)
fi

ARGS=(
    --model "$MODEL_NAME"
    --served-model-name "$SERVED_MODEL_NAME"
    --host "$HOST"
    --port "$PORT"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --trust-remote-code
    --limit-mm-per-prompt image=2
    --enable-prefix-caching
)
if [ -n "$QUANTIZATION" ]; then
    ARGS+=(--quantization "$QUANTIZATION")
fi

echo "================================================================"
echo "  启动本地 vLLM 医学 VLM judge"
echo "  模型   : $MODEL_NAME"
echo "  served : $SERVED_MODEL_NAME"
echo "  地址   : http://$HOST:$PORT/v1"
echo "  上下文 : $MAX_MODEL_LEN  |  显存上限: $GPU_MEMORY_UTILIZATION"
echo "================================================================"
echo "就绪后健康检查（新开终端）:"
echo "  curl http://127.0.0.1:$PORT/v1/models"
echo "聊天补全示例:"
echo "  curl http://127.0.0.1:$PORT/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"$SERVED_MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'"
echo "评测时使用:"
echo "  python eval.py --judge_config ./api/config.vllm.yaml \\"
echo "    --judge_model $SERVED_MODEL_NAME --jsonl_path <results.jsonl> --task vqa"
echo "================================================================"

health_check() {
    curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1
}

if [ "${DAEMONIZE:-0}" = "1" ]; then
    echo "[daemon] 后台启动，日志: $LOG_FILE"
    nohup "${VLLM_CMD[@]}" "${ARGS[@]}" $EXTRA_ARGS >"$LOG_FILE" 2>&1 &
    VLLM_PID=$!
    echo "[daemon] PID: $VLLM_PID"
    echo "[daemon] 等待服务就绪（模型加载可能需数分钟）..."
    for _ in $(seq 1 600); do
        if ! kill -0 "$VLLM_PID" >/dev/null 2>&1; then
            echo "[ERROR] vLLM 进程已退出，请查看 $LOG_FILE"
            exit 1
        fi
        if health_check; then
            echo "[daemon] 服务已就绪: http://127.0.0.1:${PORT}/v1"
            exit 0
        fi
        sleep 2
    done
    echo "[ERROR] 等待超时，请查看 $LOG_FILE"
    exit 1
fi

exec "${VLLM_CMD[@]}" "${ARGS[@]}" $EXTRA_ARGS
