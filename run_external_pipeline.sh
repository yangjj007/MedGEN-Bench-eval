# Run all MedGEN inference and evaluation tasks through an external API.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root_dir"

python_bin="${PYTHON_BIN:-$root_dir/.venv/bin/python}"
config_path="$root_dir/config.yaml"
data_dir="${MEDGEN_DATA_DIR:-$root_dir/MedGEN_data}"
output_dir="${EXTERNAL_OUTPUT_DIR:-$root_dir/outputs/external}"
vlm_model="${EXTERNAL_VLM_MODEL:-qwen3-vl-235b-a22b-instruct}"
edit_model="${EXTERNAL_EDIT_MODEL:-gpt-image-1-mini}"
generate_model="${EXTERNAL_GENERATE_MODEL:-gpt-image-1-mini}"
sample_args=()

if [ -n "${MAX_SAMPLES:-}" ]; then
    sample_args=(--max-samples "$MAX_SAMPLES")
fi

if [ -z "${AIHUBMIX_API_KEY:-}" ]; then
    echo "Set AIHUBMIX_API_KEY before running the external API workflow." >&2
    exit 1
fi
if [ ! -f "$data_dir/vqa.jsonl" ]; then
    echo "Prepared data is missing. Run bash download_medgen_dataset.sh and python prepare_medgen_data.py first." >&2
    exit 1
fi

slug() {
    printf '%s' "$1" | sed -E 's/[^A-Za-z0-9._-]+/-/g; s/^[.-]+//; s/[.-]+$//'
}

mkdir -p "$output_dir/inference" "$output_dir/images"
export MEDGEN_EVAL_RESULTS_DIR="$output_dir/eval"

"$python_bin" inference.py --jsonl-path "$data_dir/vqa.jsonl" \
    --mission vqa --vlm-model "$vlm_model" --vlm-config "$config_path" \
    --output-jsonl-dir "$output_dir/inference" --output-image-path "$output_dir/images" \
    "${sample_args[@]}"
"$python_bin" inference.py --jsonl-path "$data_dir/edit.jsonl" \
    --mission edit --vlm-model "$vlm_model" --edit-model "$edit_model" \
    --vlm-config "$config_path" --image-config "$config_path" \
    --output-jsonl-dir "$output_dir/inference" --output-image-path "$output_dir/images" \
    "${sample_args[@]}"
"$python_bin" inference.py --jsonl-path "$data_dir/gen.jsonl" \
    --mission generate --vlm-model "$vlm_model" --generate-model "$generate_model" \
    --vlm-config "$config_path" --image-config "$config_path" \
    --output-jsonl-dir "$output_dir/inference" --output-image-path "$output_dir/images" \
    "${sample_args[@]}"

vlm_slug="$(slug "$vlm_model")"
edit_slug="$(slug "$edit_model")"
generate_slug="$(slug "$generate_model")"
eval_flags=(--local-metrics-only)
if [ "${EXTERNAL_USE_VLM_JUDGE:-0}" = "1" ]; then
    eval_flags=(--judge-config "$config_path" --judge-model "${EXTERNAL_JUDGE_MODEL:-$vlm_model}")
fi

"$python_bin" eval.py --data_path "$data_dir" \
    --jsonl_path "$output_dir/inference/${vlm_slug}_vqa.jsonl" \
    --task vqa "${eval_flags[@]}"
"$python_bin" eval.py --data_path "$data_dir" \
    --jsonl_path "$output_dir/inference/${vlm_slug}_${edit_slug}_edit.jsonl" \
    --task image_edit "${eval_flags[@]}"
"$python_bin" eval.py --data_path "$data_dir" \
    --jsonl_path "$output_dir/inference/${vlm_slug}_${generate_slug}_generate.jsonl" \
    --task multimodal_generation "${eval_flags[@]}"
