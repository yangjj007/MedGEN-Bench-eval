# MedGEN-Bench

> Contextually Entangled Benchmark for Open-Ended Multimodal Medical Generation

[![arXiv](https://img.shields.io/badge/arXiv-2511.13135-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2511.13135)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-MedGEN--Bench-ffcc4d)](https://huggingface.co/datasets/Jack04810/MedGEN-Bench)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

Official code and evaluation toolkit for MedGEN-Bench.

[Preprint](https://arxiv.org/abs/2511.13135) | [Dataset](https://huggingface.co/datasets/Jack04810/MedGEN-Bench) | [Code](https://github.com/yangjj007/MedGEN-Bench-eval)

## Abstract

As Vision-Language Models (VLMs) increasingly gain traction in medical applications, clinicians are progressively expecting AI systems not only to generate textual diagnoses but also to produce corresponding medical images that integrate seamlessly into authentic clinical workflows. Despite the growing interest, existing medical visual benchmarks present notable limitations. They often rely on ambiguous queries that lack sufficient relevance to image content, oversimplify complex diagnostic reasoning into closed-ended shortcuts, and adopt a text-centric evaluation paradigm that overlooks the importance of image generation capabilities. To address these challenges, we introduce MedGEN-Bench, a comprehensive multimodal benchmark designed to advance medical AI research. MedGEN-Bench comprises 6,422 expert-validated image-text pairs spanning six imaging modalities, 16 clinical tasks, and 28 subtasks. It is structured into three distinct formats: Visual Question Answering, Image Editing, and Contextual Multimodal Generation. What sets MedGEN-Bench apart is its focus on contextually intertwined instructions that necessitate sophisticated cross-modal reasoning and open-ended generative outputs, moving beyond the constraints of multiple-choice formats. To evaluate the performance of existing systems, we employ a novel three-tier assessment framework that integrates pixel-level metrics, semantic text analysis, and expert-guided clinical relevance scoring. Using this framework, we systematically assess 10 compositional frameworks, 3 unified models, and 5 VLMs.

## Benchmark at a glance

<p align="center">
  <img src="assets/figures/figure_2.png" alt="Representative MedGEN-Bench task cards" width="100%">
</p>
<p align="center"><sub>Figure 2. Representative task cards from the image-editing and multimodal-generation portions of MedGEN-Bench.</sub></p>

<p align="center">
  <img src="assets/figures/figure_3.png" alt="MedGEN-Bench dataset statistics" width="100%">
</p>
<p align="center"><sub>Figure 3. Dataset statistics and instruction/answer-length distributions in the preprint evaluation snapshot.</sub></p>

<p align="center">
  <img src="assets/figures/figure_4.png" alt="MedGEN-Bench construction pipeline" width="100%">
</p>
<p align="center"><sub>Figure 4. MedGEN-Bench construction pipeline: preprocessing, image-pair synthesis, text-pair synthesis, and human refinement.</sub></p>

## Setup

MedGEN-Bench has been tested on Linux with Python 3.10, CUDA-capable NVIDIA
GPUs, and the pinned packages in `requirements.txt`. A complete local run uses
Qwen3-VL for instruction understanding, Qwen Image Edit for image output, and
Lingshu-32B as the independent VLM judge.

```bash
export PYTHON_BIN="${PYTHON_BIN:-python3.10}"  # Set an absolute Python 3.10 path when needed.
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -m pytest -q
```

The tested local configuration uses four 40 GB GPUs: one for Qwen3-VL, one
for Qwen Image Edit with sequential CPU offload, and two for tensor-parallel
Lingshu-32B judging. Set `LOCAL_USE_VLM_JUDGE=0` to omit Lingshu when only
automatic local metrics are required.

## Data

Data, model weights, caches, outputs, and logs are intentionally ignored by
Git. The public release downloads the evaluation data from Hugging Face at the
pinned release revision.

```bash
export PYTHON_BIN="$PWD/.venv/bin/python"
bash download_medgen_dataset.sh
"$PYTHON_BIN" prepare_medgen_data.py
bash download_medimageinsight.sh
```

This creates `MedGEN_raw/` for the downloaded source and `MedGEN_data/` for
the prepared data used by inference and evaluation. The download contains the
`image_editing`, `multimodal_generation`, and `vqa` parquet configurations.
The paired dataset is
[Jack04810/MedGEN-Bench](https://huggingface.co/datasets/Jack04810/MedGEN-Bench).
The download script is pinned to the Parquet revision used for this release;
set `MEDGEN_DATASET_REVISION` only when intentionally using another revision.
The adapter refuses to overwrite an existing output directory. For an isolated
repeatable preparation check, choose a new location:

```bash
"$PYTHON_BIN" prepare_medgen_data.py --output MedGEN_data_repro
```

Validate the prepared inputs before starting a model:

```bash
"$PYTHON_BIN" inference.py --jsonl-path MedGEN_data/vqa.jsonl --mission vqa --validate-only
"$PYTHON_BIN" inference.py --jsonl-path MedGEN_data/edit.jsonl --mission edit --validate-only
"$PYTHON_BIN" inference.py --jsonl-path MedGEN_data/gen.jsonl --mission generate --validate-only
```

## Configuration

Use the single root-level `config.yaml`. It contains only public endpoint and
retry defaults. Keep API keys in environment variables.

```bash
export AIHUBMIX_API_KEY='your-key'
```

`config.yaml` is configured for the OpenAI-compatible
[AIHubMix](https://aihubmix.com/) API used in the formal external experiments.
Do not put a credential in the YAML file or commit an `.env` file.

## External API workflow

The following command runs VQA, image editing, image generation, and their
evaluations through the external API configured in `config.yaml`.

```bash
bash run_external_pipeline.sh
```

Optional model overrides:

```bash
EXTERNAL_VLM_MODEL='qwen3-vl-235b-a22b-instruct' \
EXTERNAL_EDIT_MODEL='gpt-image-1-mini' \
EXTERNAL_GENERATE_MODEL='gpt-image-1-mini' \
bash run_external_pipeline.sh
```

Set `EXTERNAL_USE_VLM_JUDGE=1` to evaluate with an external VLM judge. The
default evaluation uses local metrics only and does not make additional paid
judge requests.

Use `MAX_SAMPLES=1` before a full run to process one item from every task.

## Local model workflow

Download the local models into `models/`, then start the three localhost-only
APIs with one command:

```bash
bash download_local_models.sh
bash start_local_models.sh
```

The services are:

- Qwen3-VL: `http://127.0.0.1:8000/v1`
- Qwen Image Edit: `http://127.0.0.1:8001/v1`
- Lingshu-32B judge: `http://127.0.0.1:8002/v1`

Verify the local APIs before starting a benchmark run:

```bash
curl -fsS http://127.0.0.1:8000/v1/models
curl -fsS http://127.0.0.1:8001/health
curl -fsS http://127.0.0.1:8002/v1/models
```

Run a real three-model smoke workflow first. It executes one VQA, editing,
and multimodal-generation item and evaluates each result with Lingshu:

```bash
MAX_SAMPLES=1 LOCAL_IMAGE_STEPS=4 LOCAL_IMAGE_SIZE=512x512 bash run_local_pipeline.sh
```

`LOCAL_IMAGE_STEPS=4` and `LOCAL_IMAGE_SIZE=512x512` keep the smoke run
practical while still making real image requests. Omit both for the default
30 denoising steps and native input size used for a full local run. Remove
`MAX_SAMPLES=1` only after the smoke run is successful. Inference JSONL,
generated images, evaluator checkpoints, and summaries are placed beneath
`outputs/local/` by default.

For machines with multiple GPUs, choose devices explicitly:

```bash
LOCAL_VLM_GPU=0 \
LOCAL_IMAGE_GPU=1 \
LOCAL_JUDGE_GPU=2,3 \
LOCAL_JUDGE_TENSOR_PARALLEL_SIZE=2 \
bash start_local_models.sh
```

`LOCAL_IMAGE_CPU_OFFLOAD=sequential` is the default so Qwen Image Edit can run
on a 40 GB GPU. Set `LOCAL_USE_VLM_JUDGE=0` when only local automatic metrics
are needed; this also skips starting the Lingshu server in `run_local_pipeline.sh`.
To use an already-downloaded Lingshu checkpoint outside the repository, set
`LOCAL_JUDGE_MODEL_PATH=/absolute/path/to/Lingshu-32B`.
The VLM, image, and judge GPU selections must not overlap.
The local workflow caps judge inputs to 768 px per side so two images fit the
default 4,096-token Lingshu context; change this with
`LOCAL_JUDGE_IMAGE_MAX_SIDE` only when the server has a larger context window.

## Reproducibility checklist

1. Create a clean Python 3.10 environment and install `requirements.txt`.
2. Download the pinned data revision and run `prepare_medgen_data.py`.
3. Run all three `--validate-only` inference checks above.
4. Start the three APIs and query their health/model endpoints.
5. Run `MAX_SAMPLES=1 LOCAL_IMAGE_STEPS=4 LOCAL_IMAGE_SIZE=512x512 bash run_local_pipeline.sh`.
6. Inspect `outputs/local/eval/` for the per-task JSONL checkpoints and JSON summaries.

The code license is Apache-2.0. Dataset, model, and external service rights
remain governed by their respective providers.
