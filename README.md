# MedGEN-Bench Evaluation

## Setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Data

All data, models, caches, outputs, and logs stay inside this repository and
are ignored by Git.

```bash
bash download_medgen_dataset.sh
python prepare_medgen_data.py
bash download_medimageinsight.sh
```

This creates `MedGEN_raw/` for the downloaded source and `MedGEN_data/` for
the prepared data used by inference and evaluation. The download contains the
`image_editing`, `multimodal_generation`, and `vqa` parquet configurations.
The paired dataset is
[Jack04810/MedGEN-Bench](https://huggingface.co/datasets/Jack04810/MedGEN-Bench).
The download script is pinned to the Parquet revision used for this release;
set `MEDGEN_DATASET_REVISION` only when intentionally using another revision.

## Configuration

Use the single root-level `config.yaml`. It contains only public endpoint and
retry defaults. Keep API keys in environment variables.

```bash
export AIHUBMIX_API_KEY='your-key'
```

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

Download the two local Qwen models into `models/`, then start both local APIs
with one command:

```bash
bash download_local_models.sh
bash start_local_models.sh
```

The Qwen VLM runs through vLLM at `127.0.0.1:8000`. The Qwen Image Edit model
is exposed at `127.0.0.1:8001` with an OpenAI-compatible image endpoint. Run
all local inference and evaluation tasks with:

```bash
bash run_local_pipeline.sh
```

For machines with multiple GPUs, choose devices explicitly:

```bash
LOCAL_VLM_GPU=0 LOCAL_IMAGE_GPU=1 bash start_local_models.sh
```

`LOCAL_IMAGE_CPU_OFFLOAD=sequential` is the default so Qwen Image Edit can run
on a 40 GB GPU. Set `LOCAL_USE_VLM_JUDGE=0` when only local automatic metrics
are needed.

The code license is Apache-2.0. Dataset, model, and external service rights
remain governed by their respective providers.
