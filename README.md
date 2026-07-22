# MedGEN-Bench Eval

本目录已接入论文 Table IV 版本数据：3 类格式、16 个任务、6,623 条样本。兼容层位于 `MedGEN_TableIV/`，原图通过相对软链接复用；400 条多图 VQA 已转换为带序号的 contact sheet，不会丢弃后续图片。

- 代码：[Jack04810/MedGEN-Bench-eval](https://github.com/Jack04810/MedGEN-Bench-eval)
- 完整数据：[Jack04810/MedGEN-Bench](https://huggingface.co/datasets/Jack04810/MedGEN-Bench)

## 数据

- Hugging Face 备份含全部 11,105 张图片和 34 个 JSON/JSONL、manifest、README 文件，共 11,139 项（约 4.5 GB）。
- VQA：1,100 条 → `MedGEN_TableIV/vqa.jsonl`
- Image Editing：3,872 条 → `MedGEN_TableIV/edit.jsonl`
- Multimodal Generation：1,651 条 → `MedGEN_TableIV/gen.jsonl`

从 Hugging Face 重新准备：

```bash
hf download Jack04810/MedGEN-Bench --repo-type dataset \
  --local-dir ../medical-bench/MedGEN_Bench_TableIV_Organized
python prepare_medgen_tableiv.py
```

`prepare_medgen_tableiv.py` 会校验任务计数和所有图片路径，并生成三份 JSONL、smoke 文件及 VQA contact sheet。若输出目录已存在，先将其移走或指定新的 `--output`，脚本不会覆盖已有数据。

## 环境

已使用 Python 3.12 验证，建议 Python 3.10–3.12：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-base.txt
```

执行 LPIPS、PSNR、SSIM、BLEU 和 BERTScore 时安装完整评测依赖。CPU 环境先安装 CPU 版 PyTorch，避免下载 CUDA 运行库：

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements-eval.txt
```

真实 API 推理前创建本地配置；不要提交密钥：

```bash
cp -n api/config.example.yaml api/config.yaml
# 编辑 api/config.yaml 中的 api_key 和 base_url
```

部分图片编辑模型需 OSS，另行设置 `OSS_ACCESS_KEY_ID`、`OSS_ACCESS_KEY_SECRET` 和 `OSS_SESSION_TOKEN`。

## 本地测试（不调用 API）

```bash
python test_tableiv_integration.py
python test_metrics_smoke.py --include-bertscore

python inference.py --jsonl_path ./MedGEN_TableIV/smoke_vqa.jsonl \
  --mission vqa --validate-only
python inference.py --jsonl_path ./MedGEN_TableIV/smoke_edit.jsonl \
  --mission edit --validate-only
python inference.py --jsonl_path ./MedGEN_TableIV/smoke_gen.jsonl \
  --mission generate --validate-only

python eval.py --data_path ./MedGEN_TableIV \
  --jsonl_path ./MedGEN_TableIV/smoke_eval_vqa.jsonl \
  --task vqa --validate-only
python eval.py --data_path ./MedGEN_TableIV \
  --jsonl_path ./MedGEN_TableIV/smoke_eval_edit.jsonl \
  --task image_edit --validate-only
python eval.py --data_path ./MedGEN_TableIV \
  --jsonl_path ./MedGEN_TableIV/smoke_eval_gen.jsonl \
  --task multimodal_generation --validate-only

# 执行三个任务各一条 oracle 本地指标主流程（不调用 API）
python eval.py --data_path ./MedGEN_TableIV \
  --jsonl_path ./MedGEN_TableIV/smoke_eval_vqa.jsonl \
  --task vqa --local-metrics-only --max_samples 1
python eval.py --data_path ./MedGEN_TableIV \
  --jsonl_path ./MedGEN_TableIV/smoke_eval_edit.jsonl \
  --task image_edit --local-metrics-only --max_samples 1
python eval.py --data_path ./MedGEN_TableIV \
  --jsonl_path ./MedGEN_TableIV/smoke_eval_gen.jsonl \
  --task multimodal_generation --local-metrics-only --max_samples 1
```

`smoke_eval_*.jsonl` 是标明了 `eval_smoke_fixture=true` 的 oracle 路径测试文件，不是模型结果，不得用于论文分数。
指标测试首次运行会下载 AlexNet 和 PubMedBERT 权重。

## 运行推理

先用 `--max_samples 1` 测通，再移除该参数运行全集。结果单独写入 `inference_jsonl/tableiv/`，避免与旧版数据结果混用。

```bash
# VQA
python inference.py --jsonl_path ./MedGEN_TableIV/vqa.jsonl \
  --mission vqa --vlm_model qwen3-vl-235b-a22b-instruct \
  --output_jsonl_dir ./inference_jsonl/tableiv --max_samples 1

# Image Editing
python inference.py --jsonl_path ./MedGEN_TableIV/edit.jsonl \
  --mission edit --vlm_model qwen3-vl-235b-a22b-instruct \
  --edit_model gpt-image-1-mini \
  --output_jsonl_dir ./inference_jsonl/tableiv --max_samples 1

# Multimodal Generation
python inference.py --jsonl_path ./MedGEN_TableIV/gen.jsonl \
  --mission generate --vlm_model qwen3-vl-235b-a22b-instruct \
  --generate_model imagen-4.0-fast \
  --output_jsonl_dir ./inference_jsonl/tableiv --max_samples 1
```

API 推理可能产生费用。支持的模型名由 `agent.py` 和 `api/get_*_res.py` 定义。

## 运行评测

```bash
python eval.py --data_path ./MedGEN_TableIV \
  --jsonl_path ./inference_jsonl/tableiv/qwen3-vl-235b-a22b-instruct_vqa.jsonl \
  --task vqa --mission basic_eval --batch_size 8

python eval.py --data_path ./MedGEN_TableIV \
  --jsonl_path ./inference_jsonl/tableiv/qwen3-vl-235b-a22b-instruct_gpt-image-1-mini_edit.jsonl \
  --task image_edit --mission basic_eval --batch_size 4

python eval.py --data_path ./MedGEN_TableIV \
  --jsonl_path ./inference_jsonl/tableiv/qwen3-vl-235b-a22b-instruct_imagen-4.0-fast_generate.jsonl \
  --task multimodal_generation --mission basic_eval --batch_size 4
```

默认基础评测同时运行本地指标和 VLM judge。只检查本地指标、不调用付费 API 时添加 `--local-metrics-only`；该选项仅支持 `basic_eval`。基础结果写入 `eval_results/`。对其中的 `*_with_vlm.jsonl` 或 `*_local_metrics.jsonl` 使用 `--mission type_wise --type_key modality` 可按模态聚合，结果写入 `eval_results_type_wise/`。图像全参考指标使用 `ground_truth_image`，VQA 文本指标使用 `answer`。CPU 线程数默认 4，可通过 `MEDGEN_TORCH_NUM_THREADS` 调整。

## 已验证

- 完整兼容层：6,623/6,623 条，16 个任务，缺失图片 0。
- `inference.py --validate-only`：VQA、Edit、Generate 全量通过。
- `eval.py --validate-only`：三个任务类型的 oracle smoke 输入通过。
- `test_tableiv_integration.py`：6 项集成测试通过。
- LPIPS、PSNR、SSIM、BLEU、BERTScore：均已实际计算通过；`transformers` 已约束为兼容的 4.x。
- `basic_eval --local-metrics-only`：VQA、Image Editing、Multimodal Generation 各一条通过，结果为严格 JSON；type-wise 聚合通过。
- 数据准备脚本从零复建：6,623 条、16 个任务、300 张 contact sheet、缺失 0；三份主 JSONL 与接入目录 SHA-256 完全一致。
- Hugging Face：11,139 个本地文件逐路径核验，缺失 0、大小不一致 0；远端另有平台自动生成的 `.gitattributes`。
- 临时全新环境复现：将在最终验收完成后记录。
