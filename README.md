# MedGEN-Bench Eval

本目录已接入论文 Table IV 版本数据：3 类格式、16 个任务、6,623 条样本。兼容层位于 `MedGEN_TableIV/`，原图通过相对软链接复用；400 条多图 VQA 已转换为带序号的 contact sheet，不会丢弃后续图片。

- 代码：[yangjj007/MedGEN-Bench-eval](https://github.com/yangjj007/MedGEN-Bench-eval)
- 完整数据：[Jack04810/MedGEN-Bench](https://huggingface.co/datasets/Jack04810/MedGEN-Bench)
- 已发布推理结果：[Jack04810/MedGEN_data](https://huggingface.co/datasets/Jack04810/MedGEN_data)

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

下载已发布的模型推理结果：

```bash
hf download Jack04810/MedGEN_data --repo-type dataset \
  --local-dir ../medical-bench/MedGEN_data
```

注意：`MedGEN_data` 的原始 baseline bundle 曾发生不完整上传。2026-07-31
已补回可从原始 ZIP 校验恢复的 283 张输入/GT 图，并发布
`Baseline_Inference_Results_Organized/eval_available/` 可用视图。该视图包含
19 份非空 JSONL、14,690 条路径完整的记录；仍缺失的 18,129 张模型输出和
2 份 JSONL 必须由原组织机重新上传。运行旧 baseline 前请查看数据集目录中的
`README.md` 和 `eval_available/missing_files.sha256`，不要把可用子集误当成
完整的 25-baseline 结果包。

若该 Hugging Face 仓库需要身份验证，请先执行 `hf auth login`，或通过 `HF_TOKEN` 提供具有读取权限的访问令牌。

## 环境

已使用 Python 3.12 验证，建议 Python 3.10–3.12：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-base.txt
```

执行 Rad-DINO、BLEU 和 BERTScore 时安装完整评测依赖。CPU 环境先安装 CPU 版 PyTorch，避免下载 CUDA 运行库：

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements-eval.txt
```

`requirements-eval.txt` 现已包含官方 `radgraph` 包。首次计算 `RadGraph_F1` 时会按包默认行为下载 RadGraph 模型缓存；可用 `MEDGEN_RADGRAPH_MODEL_TYPE` 覆盖默认 `radgraph-xl`，也可用 `MEDGEN_RADGRAPH_CACHE_DIR` 指定模型缓存目录。

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

对文本样本，评测严格使用官方 `radgraph.F1RadGraph` 的 `RG_ER` 分量。RadGraph 包、模型缓存或推理任一环节不可用时直接抛错并终止评测，不使用 heuristic 回退。

## 指标体系（Metric 部分已按审稿意见升级）

为回应审稿人"通用图像/文本指标对临床细节不敏感、缺少统计显著性与错误分析"的意见，评测框架在保留全部旧指标（向后兼容）的基础上新增以下输出：

### 临床/解剖感知图像指标（`util/metrics.py`）
- 图像主评测保留全图 `LPIPS`、`PSNR`、`SSIM`；废弃局部 PSNR、局部 SSIM、局部 MAE、Otsu ROI、直方图、Sobel、Laplacian 和对比度等局部/低层代理指标。
- `Anatomical_Embedding_Similarity`：使用放射学预训练 `microsoft/rad-dino` 的冻结图像表征计算候选图与参考图的余弦相似度。该指标用于衡量放射学结构表征的一致性，不宣称等价于病灶分割或临床诊断；模型权重不可用时直接报错。
- 图像任务同时使用本地医学 VLM judge 的解剖准确性、临床发现准确性和指令遵循评分。没有标准 mask/bbox 的任务不伪造空间定位分数。

### 临床文本指标增强（`util/clinical_text_metrics.py`）
- `Entity_Hallucination_Rate`（响应实体不在参考中）、`Entity_Omission_Rate`（参考实体缺失）、`Entity_Factual_Precision`：在 `Clinical_Entity_*` 与 `RadGraph_F1` 之上提供实体级错误分类，可直接支撑幻觉/遗漏失败模式分析。
- `CheXbert_Factual_Precision`（可选）：默认关闭；设置 `MEDGEN_ENABLE_CHEXBERT=1` 且环境可导入暴露 `factual_precision(response, reference)` 的模块（默认 `chexbert`，可用 `MEDGEN_CHEXBERT_MODULE` 覆盖）时启用，不可用时自动跳过。

### VLM judge 升级（`util/prompt.py`）
- 保留 5 维临床评分（`anatomical_accuracy`、`clinical_finding_accuracy`、`instruction_compliance`、`cross_modal_consistency`、`hallucination_omission_control`），并新增按任务（vqa / image_edit / multimodal_generation）区分的结构化检查清单，每个维度给出可观察判定标准，减少 prompt 漂移。
- judge 默认配置为本地 `google/medgemma-4b-it`，可通过 `--judge_model` 和 vLLM 配置切换其他医学 VLM。

### 本地 vLLM 医学 VLM judge（默认）

- 安装 vLLM（需 GPU 与 CUDA 环境）：`pip install -U vllm`。
- 启动服务（默认模型 `google/medgemma-4b-it`，面向医学场景的本地多模态模型）：

```bash
bash vllm_serve.sh
# 可按显存替换模型 / 端口 / 显存上限：
# MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct PORT=8010 GPU_MEMORY_UTILIZATION=0.85 bash vllm_serve.sh
```

- 就绪后健康检查：`curl http://127.0.0.1:8000/v1/models`；需要后台运行时设置 `DAEMONIZE=1`（日志写入 `./vllm_serve.log`）。
- 评测时用 `--judge_config ./api/config.vllm.yaml` 指向本地服务（其 `base_url` 为 `http://127.0.0.1:8000/v1`，`model_name` 与脚本的 served 模型名一致），并通过 `--judge_model` 显式指定模型：

```bash
python eval.py --data_path ./MedGEN_TableIV \
  --jsonl_path ./inference_jsonl/tableiv/qwen3-vl-235b-a22b-instruct_vqa.jsonl \
  --task vqa --mission basic_eval --batch_size 8 \
  --judge_config ./api/config.vllm.yaml \
  --judge_model google/medgemma-4b-it
```

- 模型优先级：`--judge_model` > `--judge_config` 中的 `model_name` > 默认 `google/medgemma-4b-it`；`api/config.vllm.yaml` 的 `api_key` 为占位值 `EMPTY`（vLLM 默认不鉴权）。
- 客户端走标准 OpenAI 兼容接口（图片以 base64 data-URI 传入），与 vLLM 的 `/v1/chat/completions` 天然兼容；`type_wise` 聚合同样接受 `--judge_config`。
- 已知偏倚风险：数据筛选/质检阶段 GPT-4o 与 Qwen3-VL 等模型参与过，同类模型作为被评估对象时可能引入偏倚。论文中应披露，并用下方专家一致性校准流程量化 judge 可靠性。

### 阈值表（`METRIC_THRESHOLDS`）

| 指标 | 方向 | 阈值 |
| --- | --- | --- |
| LPIPS | 越低越好 | 0.6 |
| PSNR / SSIM | 越高越好 | 28.0 / 0.7 |
| Anatomical_Embedding_Similarity | 越高越好 | 0.7（需在专家校准集上重新确定） |
| BLEU | 越高越好 | 0.09 |
| BERT_Score | 越高越好 | 0.9 |
| Entity_Hallucination_Rate / Entity_Omission_Rate | 越低越好 | 0.1 |
| Entity_Factual_Precision | 越高越好 | 0.9 |
| VLM_Overall_Score_W_GT / WO_GT | 越高越好 | 8.0 |

### 统计检验与错误分析（`eval.py`）
- 每次聚合（`basic_eval` 与 `type_wise`）为每个指标输出 bootstrap 95% CI（默认 1000 次重采样，`MEDGEN_BOOTSTRAP_SAMPLES` 或 `--bootstrap_samples` 可调）。
- 输出 `Error_Analysis`：按 `modality` / `paper_task` 分解的指标均值与标准差，以及幻觉、遗漏和低指令遵循失败模式。旧的“全局指标高、局部指标低”分类已删除。
- 多模型配对显著性检验（同一样本集、不同模型输出）：

```bash
python eval.py --mission stats \
  --jsonl_path eval_results/modelA_with_vlm.jsonl eval_results/modelB_with_vlm.jsonl \
  --task vqa --bootstrap_samples 1000
```

按 `sample_id` 配对，输出每个指标的 bootstrap CI，以及任意两模型间的配对 Wilcoxon signed-rank 检验（p 值、显著性、效应方向、均值差），结果写入 `eval_results/<models>_stats_results.json`。

### 专家评测流程与 judge 一致性校准
1. 从评测结果中按任务/模态分层抽样（建议每任务 ≥ 30 条），由 ≥ 2 名临床医生按与 judge 相同的 5 维评分表独立打分（1-10），记录评分标准与培训流程。
2. 计算 judge 与医生评分的 Pearson/Spearman 相关、以及绝对分差分布；参考 GPT-4o 与放射科医生报告评分一致性相关工作，报告 ≥ 0.7 相关作为可接受线。
3. 医生间用 Cohen's κ / ICC 报告标注一致性；分歧样本回看并记录分歧原因。
4. 校准结果（一致性报告 JSON/CSV）随评测结果一起作为论文附录材料。

## 已验证

- 完整兼容层：6,623/6,623 条，16 个任务，缺失图片 0。
- `inference.py --validate-only`：VQA、Edit、Generate 全量通过。
- `eval.py --validate-only`：三个任务类型的 oracle smoke 输入通过。
- `test_tableiv_integration.py`：6 项集成测试通过。
- Rad-DINO anatomical embedding、RadGraph-F1、BLEU、BERTScore：依赖安装后执行；当前无 GPU/模型缓存环境仅完成静态编译验证。
- `test_clinical_image_metrics.py`、`test_clinical_text_metrics.py`、`test_eval_clinical_integration.py`：在完整依赖环境运行；当前环境通过静态编译与轻量测试，缺少 RadGraph/Rad-DINO 权重时按设计终止临床评测。
- 数据准备脚本从零复建：6,623 条、16 个任务、300 张 contact sheet、缺失 0；三份主 JSONL 与接入目录 SHA-256 完全一致。
- Hugging Face：11,139 个本地文件逐路径核验，缺失 0、大小不一致 0；远端另有平台自动生成的 `.gitattributes`。
- 当前改动已完成静态编译检查；Rad-DINO、RadGraph 和本地 judge 权重需按运行环境安装/下载后执行。
