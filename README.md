# MedGEN-Bench Evaluation

This repository prepares the public MedGEN-Bench Table IV evaluation view,
runs multimodal inference, and evaluates VQA, image-editing, and
multimodal-generation outputs. It supports both OpenAI-compatible cloud APIs
(including AIHubMix) and local Qwen services.

The code is intentionally separate from the dataset and generated results.
Download those assets from Hugging Face rather than committing them to Git.

## Contents

- `prepare_medgen_tableiv.py`: creates the evaluation-compatible Table IV view.
- `inference.py`: resumable VQA, image editing, and image-generation inference.
- `eval.py`: local image/text metrics, optional VLM judging, aggregation, and
  paired statistics.
- `api/`: OpenAI-compatible clients and a local Qwen Image Edit server.
- `vllm_serve.sh`: a localhost vLLM server for Qwen vision-language models.

## Dataset

The paired dataset is [Jack04810/MedGEN-Bench](https://huggingface.co/datasets/Jack04810/MedGEN-Bench).
It provides 6,623 Table IV records across 16 tasks and 11,105 referenced
images (about 4.5 GB). The preparation adapter produces:

| Format | Records | Output |
| --- | ---: | --- |
| VQA | 1,100 | `vqa.jsonl` |
| Image editing | 3,872 | `edit.jsonl` |
| Multimodal generation | 1,651 | `gen.jsonl` |

The dataset, model weights, and third-party API services retain their own
licenses and terms. The Apache-2.0 license in this repository covers only this
repository's code and documentation.

## Clean environment

Use Python 3.10 for the reproducible release environment. Create separate
environments for the evaluator, vLLM, and the Qwen Image Edit server when GPU
memory is limited.

```bash
git clone https://github.com/yangjj007/MedGEN-Bench-eval.git
cd MedGEN-Bench-eval

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-base.txt
python -m pip install -r requirements-eval.txt -c requirements-lock-py310-cu13.txt
```

`requirements-base.txt` is enough for data preparation, configuration checks,
API-client tests, and no-network input validation. `requirements-eval.txt`
adds the metrics stack, while `requirements-lock-py310-cu13.txt` fixes every
resolved package version from the verified Linux/CUDA environment. The first
metric run downloads model weights such as
PubMedBERT, LPIPS/AlexNet, and RadGraph as required by their upstream
packages. MedImageInsight is an external source-plus-checkpoint dependency;
prepare it explicitly before image-metric evaluation:

```bash
export MEDGEN_MEDIMAGEINSIGHT_DIR="$HOME/.cache/medgen-bench/MedImageInsights"
bash download_medimageinsight.sh "$MEDGEN_MEDIMAGEINSIGHT_DIR"
```

## Data preparation

Install the Hugging Face CLI through `requirements-base.txt`, then download
the external dataset into an ignored directory:

```bash
export MEDGEN_DATASET_DIR="$PWD/data/MedGEN-Bench"
bash download_medgen_dataset.sh "$MEDGEN_DATASET_DIR"

python prepare_medgen_tableiv.py \
  --dataset-root "$MEDGEN_DATASET_DIR" \
  --output "$PWD/data/MedGEN_TableIV"
```

The adapter validates every task count and image path. It preserves canonical
data by relative symlink and creates contact sheets for multi-image VQA cases.
It never overwrites an existing output directory.

Run no-network preparation and input checks:

```bash
export MEDGEN_TABLEIV_DIR="$PWD/data/MedGEN_TableIV"
python test_tableiv_integration.py

python inference.py --jsonl_path "$MEDGEN_TABLEIV_DIR/smoke_vqa.jsonl" \
  --mission vqa --validate-only
python eval.py --data_path "$MEDGEN_TABLEIV_DIR" \
  --jsonl_path "$MEDGEN_TABLEIV_DIR/smoke_eval_vqa.jsonl" \
  --task vqa --validate-only
```

## API configuration

All clients accept an OpenAI-compatible `base_url`. Templates contain no
secret. Configuration values may use `${ENVIRONMENT_VARIABLE}` syntax.

### AIHubMix

```bash
cp api/config.example.yaml api/config.aihubmix.yaml
export AIHUBMIX_API_KEY='replace-with-your-key'
```

Edit `api/config.aihubmix.yaml` only to select models and optional retry
settings. Do not commit it. The template's default endpoint is
`https://aihubmix.com/v1`.

### Local Qwen VLM through vLLM

`vllm_serve.sh` binds to `127.0.0.1:8000` by default. This keeps the model
server private to the machine. The default model is `Qwen/Qwen3-VL-8B-Instruct`.

```bash
python3.10 -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-local-vllm.txt

CUDA_VISIBLE_DEVICES=0 VLLM_PYTHON="$PWD/.venv-vllm/bin/python" \
  MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct bash vllm_serve.sh
```

In another terminal, verify readiness:

```bash
curl -fsS http://127.0.0.1:8000/v1/models
```

`api/config.vllm.yaml` points to that endpoint and uses the non-secret
placeholder key `EMPTY`, which vLLM accepts by default.

### Local Qwen Image Edit service

Qwen Image Edit is a diffusion model rather than a standard language-model
server. This repository supplies a localhost FastAPI adapter exposing the
OpenAI-style `POST /v1/images/edits` endpoint on port 8001. The VLM and image
editor can be run in separate environments or sequentially on a 40 GB GPU.

```bash
python3.10 -m venv .venv-image
source .venv-image/bin/activate
python -m pip install --upgrade pip
# The requirements file installs the tested CUDA-enabled PyTorch wheel.
python -m pip install -r requirements-local-image.txt

CUDA_VISIBLE_DEVICES=0 \
  QWEN_IMAGE_PYTHON="$PWD/.venv-image/bin/python" \
  QWEN_IMAGE_EDIT_MODEL=Qwen/Qwen-Image-Edit \
  QWEN_IMAGE_CPU_OFFLOAD=sequential \
  bash qwen_image_edit_serve.sh
```

Set `QWEN_IMAGE_EDIT_MODEL` to a local model directory for offline execution.
`QWEN_IMAGE_CPU_OFFLOAD=sequential` fits the 20B editor on a 40 GB GPU at the
cost of speed. Use `model` on larger GPUs or `none` when the complete model
fits in VRAM. Confirm the server before using it:

```bash
curl -fsS http://127.0.0.1:8001/health
curl -fsS http://127.0.0.1:8001/v1/models
```

`api/config.qwen-image-edit.yaml` points to this service. It uses the same
OpenAI image-edit request shape as cloud providers, so the inference CLI can
use one local VLM configuration and a separate image-edit configuration.

## API smoke test

After starting the desired service, make a real one-image API request without
writing a benchmark result:

```bash
python api/smoke_api.py \
  --vlm-config api/config.vllm.yaml \
  --image "$MEDGEN_TABLEIV_DIR/images/vqa_contact_sheets/<one-file>.jpg"

python api/smoke_api.py \
  --image-config api/config.qwen-image-edit.yaml \
  --image "$MEDGEN_TABLEIV_DIR/images/vqa_contact_sheets/<one-file>.jpg" \
  --edit-model Qwen/Qwen-Image-Edit
```

The script prints the selected endpoint, model, and a response summary but
never prints credentials. Choose any existing small image from the prepared
dataset for `<one-file>`.

## Inference

Always start with one sample. Model identifiers are safely slugged in output
filenames, so identifiers containing `/` do not create unintended directories.

```bash
# Local VQA
python inference.py --jsonl_path "$MEDGEN_TABLEIV_DIR/vqa.jsonl" \
  --mission vqa --vlm_model Qwen/Qwen3-VL-8B-Instruct \
  --vlm-config api/config.vllm.yaml --max_samples 1 \
  --output-jsonl-dir outputs/inference --output-image-path outputs/images

# Local VLM plus local Qwen image editing
python inference.py --jsonl_path "$MEDGEN_TABLEIV_DIR/edit.jsonl" \
  --mission edit --vlm_model Qwen/Qwen3-VL-8B-Instruct \
  --edit-model Qwen/Qwen-Image-Edit \
  --vlm-config api/config.vllm.yaml \
  --image-config api/config.qwen-image-edit.yaml \
  --max_samples 1 --output-jsonl-dir outputs/inference \
  --output-image-path outputs/images

# AIHubMix model pair
python inference.py --jsonl_path "$MEDGEN_TABLEIV_DIR/vqa.jsonl" \
  --mission vqa --vlm_model qwen3-vl-235b-a22b-instruct \
  --vlm-config api/config.aihubmix.yaml --max_samples 1 \
  --output-jsonl-dir outputs/inference --output-image-path outputs/images
```

For `--mission generate`, additionally pass `--generate-model` and an
`--image-config` for a provider that supports OpenAI-compatible image
generation. For `--mission edit`, pass `--edit-model`. Inference appends JSONL
records atomically enough for resumable runs and skips completed `sample_id`s.

## Evaluation

Validate inputs first, then use local metrics or the local VLM judge.

```bash
# No API and no metric-model loading
python eval.py --data_path "$MEDGEN_TABLEIV_DIR" \
  --jsonl_path outputs/inference/<result>.jsonl \
  --task vqa --validate-only

# Local metrics only
python eval.py --data_path "$MEDGEN_TABLEIV_DIR" \
  --jsonl_path outputs/inference/<result>.jsonl \
  --task vqa --local-metrics-only --batch_size 1

# Local metrics plus Qwen vLLM judge
python eval.py --data_path "$MEDGEN_TABLEIV_DIR" \
  --jsonl_path outputs/inference/<result>.jsonl \
  --task vqa --batch_size 1 \
  --judge-config api/config.vllm.yaml \
  --judge-model Qwen/Qwen3-VL-8B-Instruct
```

Image tasks calculate LPIPS, PSNR, SSIM, and MedImageInsight similarity.
Text tasks calculate BLEU, PubMedBERTScore, closed-form exact match where
applicable, and RadGraph `RG_ER` when enabled. The VLM judge reports five
clinical dimensions and should be treated as a complementary model-based
measure, not a substitute for expert clinical evaluation.

## Verification

The release test suite is split by required assets:

```bash
# API clients and pure code paths; no credential or network request.
python -m unittest discover -p 'test_*.py'

# Prepared-data integration suite.
MEDGEN_TABLEIV_DIR="$MEDGEN_TABLEIV_DIR" python test_tableiv_integration.py

# Downloads metric weights on first use.
python test_metrics_smoke.py --include-bertscore
```

## Safety and disclosure

This is a research evaluation toolkit, not a clinical device. Generated images,
model outputs, and automatic scores must not be used for diagnosis or patient
care without appropriate expert review. Disclose model provenance and potential
judge-model overlap when reporting benchmark results.

Please report security issues privately as described in
[SECURITY.md](SECURITY.md). Contributions are covered by
[CONTRIBUTING.md](CONTRIBUTING.md).
