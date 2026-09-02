# Run all MedGEN inference and evaluation tasks through local Qwen services.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root_dir"

python_bin="${PYTHON_BIN:-$root_dir/.venv/bin/python}"
config_path="$root_dir/config.yaml"
data_dir="${MEDGEN_DATA_DIR:-$root_dir/MedGEN_data}"
output_dir="${LOCAL_OUTPUT_DIR:-$root_dir/outputs/local}"
vlm_model="${LOCAL_VLM_MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}"
image_model="${LOCAL_IMAGE_MODEL_NAME:-Qwen/Qwen-Image-Edit}"
judge_model="${LOCAL_JUDGE_MODEL_NAME:-lingshu-medical-mllm/Lingshu-32B}"
vlm_port="${LOCAL_VLM_PORT:-8000}"
image_port="${LOCAL_IMAGE_PORT:-8001}"
judge_port="${LOCAL_JUDGE_PORT:-8002}"
judge_enabled="${LOCAL_USE_VLM_JUDGE:-1}"
sample_args=()
image_request_args=()

if [ -n "${MAX_SAMPLES:-}" ]; then
    sample_args=(--max-samples "$MAX_SAMPLES")
fi
if [ -n "${LOCAL_IMAGE_SIZE:-}" ]; then
    image_request_args+=(--image-size "$LOCAL_IMAGE_SIZE")
fi
if [ -n "${LOCAL_IMAGE_STEPS:-}" ]; then
    image_request_args+=(--image-steps "$LOCAL_IMAGE_STEPS")
fi

if [ ! -f "$data_dir/vqa.jsonl" ]; then
    echo "Prepared data is missing. Run bash download_medgen_dataset.sh and python prepare_medgen_data.py first." >&2
    exit 1
fi

if [ "$judge_enabled" != "0" ] && [ "$judge_enabled" != "1" ]; then
    echo "LOCAL_USE_VLM_JUDGE must be 0 or 1" >&2
    exit 1
fi

LOCAL_START_JUDGE="$judge_enabled" bash "$root_dir/start_local_models.sh"
export MEDGEN_VLM_BASE_URL="http://127.0.0.1:${vlm_port}/v1"
export MEDGEN_VLM_API_KEY="EMPTY"
export MEDGEN_IMAGE_BASE_URL="http://127.0.0.1:${image_port}/v1"
export MEDGEN_IMAGE_API_KEY="EMPTY"
export MEDGEN_EVAL_RESULTS_DIR="$output_dir/eval"
if [ "$judge_enabled" = "1" ]; then
    export MEDGEN_JUDGE_IMAGE_MAX_SIDE="${LOCAL_JUDGE_IMAGE_MAX_SIDE:-768}"
fi

mkdir -p "$output_dir/inference" "$output_dir/images"

"$python_bin" inference.py --jsonl-path "$data_dir/vqa.jsonl" \
    --mission vqa --vlm-model "$vlm_model" --vlm-config "$config_path" \
    --output-jsonl-dir "$output_dir/inference" --output-image-path "$output_dir/images" \
    "${sample_args[@]}"
"$python_bin" inference.py --jsonl-path "$data_dir/edit.jsonl" \
    --mission edit --vlm-model "$vlm_model" --edit-model "$image_model" \
    --vlm-config "$config_path" --image-config "$config_path" \
    --output-jsonl-dir "$output_dir/inference" --output-image-path "$output_dir/images" \
    "${sample_args[@]}" "${image_request_args[@]}"
"$python_bin" inference.py --jsonl-path "$data_dir/gen.jsonl" \
    --mission generate --vlm-model "$vlm_model" --generate-model "$image_model" \
    --vlm-config "$config_path" --image-config "$config_path" \
    --output-jsonl-dir "$output_dir/inference" --output-image-path "$output_dir/images" \
    "${sample_args[@]}" "${image_request_args[@]}"

vlm_slug="${vlm_model//\//-}"
image_slug="${image_model//\//-}"
eval_flags=()
if [ "$judge_enabled" = "0" ]; then
    eval_flags=(--local-metrics-only)
else
    eval_flags=(
        --judge-config "$config_path"
        --judge-model "$judge_model"
        --judge-base-url "http://127.0.0.1:${judge_port}/v1"
        --judge-api-key EMPTY
    )
fi

"$python_bin" eval.py --data_path "$data_dir" \
    --jsonl_path "$output_dir/inference/${vlm_slug}_vqa.jsonl" \
    --task vqa "${eval_flags[@]}"
"$python_bin" eval.py --data_path "$data_dir" \
    --jsonl_path "$output_dir/inference/${vlm_slug}_${image_slug}_edit.jsonl" \
    --task image_edit "${eval_flags[@]}"
"$python_bin" eval.py --data_path "$data_dir" \
    --jsonl_path "$output_dir/inference/${vlm_slug}_${image_slug}_generate.jsonl" \
    --task multimodal_generation "${eval_flags[@]}"
