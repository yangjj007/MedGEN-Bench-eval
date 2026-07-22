# todo:下面代码的断点保存只保存了vlm judge的部分，而没有保存图片和文本metric，帮我加上

# 告诉我需要修改的函数完整实现


import argparse
import json
import os
import asyncio
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import logging

from util.prompt import vlm_holistic_judge_w_gt_prompt, vlm_holistic_judge_wo_gt_prompt
from util.format_parser import extract_json

# Dataset/eval-input validation should not require model clients or heavyweight
# metric packages.  Real evaluation still fails clearly if they are missing.
EVAL_DEPENDENCY_ERROR = None
try:
    from util.metrics import (
        batch_async_FR_IQA,
        batch_async_evaluate_text_quality,
    )
    from api.get_vlm_res import double_image_vlm
except ModuleNotFoundError as exc:
    EVAL_DEPENDENCY_ERROR = exc
    batch_async_FR_IQA = None
    batch_async_evaluate_text_quality = None
    double_image_vlm = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

METRIC_THRESHOLDS = {
    'LPIPS': {'lower_is_better': True, 'threshold': 0.6},
    'PSNR': {'lower_is_better': False, 'threshold': 28.0},
    'SSIM': {'lower_is_better': False, 'threshold': 0.1},
    'BLEU': {'lower_is_better': False, 'threshold': 0.09},
    'BERT_Score': {'lower_is_better': False, 'threshold': 0.9},
    'VLM_Overall_Score_WO_GT': {'lower_is_better': False, 'threshold': 8.0},
    'VLM_Overall_Score_W_GT': {'lower_is_better': False, 'threshold': 8.0},
}

def load_jsonl_data(jsonl_path: str) -> list:
    """从jsonl文件中加载数据"""
    if not os.path.exists(jsonl_path):
        logging.error(f"文件未找到: {jsonl_path}")
        return []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def resolve_image_path(data_path: str, image_ref: str) -> str:
    """Resolve absolute, cwd-relative, and dataset-relative image paths."""
    if not isinstance(image_ref, str) or not image_ref.strip():
        return ""
    if os.path.isabs(image_ref):
        return image_ref
    if os.path.isfile(image_ref):
        return os.path.abspath(image_ref)
    return os.path.abspath(os.path.join(data_path, image_ref))


def validate_eval_input(data: list, task: str, data_path: str) -> dict:
    """Validate inference output before loading metric models or API clients."""
    expected_categories = {
        "vqa": {"VQA"},
        "image_edit": {"ImageEdit"},
        "multimodal_generation": {"MMGeneration"},
    }
    errors = []
    task_counts = defaultdict(int)
    checked_images = set()

    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            errors.append(f"line {index}: record is not an object")
            continue
        if item.get("category") not in expected_categories[task]:
            errors.append(
                f"line {index}: category {item.get('category')!r} does not match task {task}"
            )
        if not isinstance(item.get("response"), str):
            errors.append(f"line {index}: missing string response")

        input_path = resolve_image_path(data_path, item.get("input_image", ""))
        if not input_path or not os.path.isfile(input_path):
            errors.append(f"line {index}: missing input_image {item.get('input_image')!r}")
        else:
            checked_images.add(os.path.realpath(input_path))

        if task in {"image_edit", "multimodal_generation"}:
            for field in ("ground_truth_image", "output_image"):
                path = resolve_image_path(data_path, item.get(field, ""))
                if not path or not os.path.isfile(path):
                    errors.append(f"line {index}: missing {field} {item.get(field)!r}")
                else:
                    checked_images.add(os.path.realpath(path))

        task_counts[str(item.get("paper_task") or item.get("sub-category"))] += 1

    if errors:
        raise ValueError(
            f"评测输入验证失败，共 {len(errors)} 个错误:\n" + "\n".join(errors[:50])
        )
    return {
        "records": len(data),
        "task": task,
        "paper_tasks": dict(sorted(task_counts.items())),
        "resolved_image_count": len(checked_images),
        "missing_images": 0,
    }


def calculate_accuracy_rates(results: dict) -> dict:
    """根据预设阈值计算各项指标的通过率"""
    accuracy_rates = {}
    for metric, values in results.items():
        if metric in METRIC_THRESHOLDS and isinstance(values, list) and values:
            config = METRIC_THRESHOLDS[metric]
            threshold = config['threshold']
            
            if config['lower_is_better']:
                passes = sum(1 for v in values if v <= threshold)
            else:
                passes = sum(1 for v in values if v >= threshold)
                
            accuracy_rates[f"{metric}_Accuracy_Rate"] = passes / len(values)
            
    return accuracy_rates


import hashlib


def generate_sample_id(item: dict) -> str:
    keys = ['instruction', 'answer', 'output_image']
    parts = []
    for k in keys:
        v = item.get(k)
        if v is not None:
            parts.append(str(v))
    if not parts:
        item_str = json.dumps(item, sort_keys=True, ensure_ascii=False)
        uid = hashlib.md5(item_str.encode('utf-8')).hexdigest()
    else:
        combined = '|'.join(parts)
        uid = hashlib.md5(combined.encode('utf-8')).hexdigest()

    return uid


# async def basic_eval(data: list, batch_size: int, task: str, data_path: str, jsonl_path: str) -> dict:
#     """对给定的数据集子集执行基础评估，并支持断点续评和保存带VLM结果的中间文件"""
#     vlm_client = double_image_vlm()
#     all_metrics = defaultdict(list)

#     # --- 构建中间结果保存路径 ---
#     eval_results_dir = './eval_results'
#     os.makedirs(eval_results_dir, exist_ok=True)
#     base_name = os.path.basename(jsonl_path)
#     intermediate_file = os.path.join(eval_results_dir, os.path.splitext(base_name)[0] + '_with_vlm.jsonl')

#     # --- 1. 加载已有中间结果 ---
#     existing_data = {}  # uid -> item
#     if os.path.exists(intermediate_file):
#         with open(intermediate_file, 'r', encoding='utf-8') as f:
#             for line in f:
#                 item = json.loads(line)
#                 uid = generate_sample_id(item)
#                 existing_data[uid] = item
#         logging.info(f"检测到中间结果文件，已加载 {len(existing_data)} 条已评测样本。")

#     # --- 2. 明确分离已处理和未处理的样本 ---
#     unprocessed_items = []
#     already_processed_items = []
#     for item in data:
#         uid = generate_sample_id(item)
#         if uid in existing_data:
#             # 使用已加载的、带有VLM结果的完整数据
#             already_processed_items.append(existing_data[uid])
#         else:
#             unprocessed_items.append(item)

#     # --- 3. 预加载已处理样本的指标 ---
#     logging.info(f"发现 {len(already_processed_items)} 条已处理样本，将直接加载其结果。")
#     for item in already_processed_items:
#         judge_w_gt = item.get('vlm_judge_w_gt_result')
#         judge_wo_gt = item.get('vlm_judge_wo_gt_result')

#         if judge_w_gt:
#             if task == 'multimodal_generation':
#                 all_metrics['VLM_Coherence_W_GT'].append(judge_w_gt.get('coherence', {}).get('score', 0))
#                 all_metrics['VLM_Visual_Textual_Alignment_W_GT'].append(judge_w_gt.get('visual_textual_alignment', {}).get('score', 0))
#             all_metrics['VLM_Content_Accuracy_W_GT'].append(judge_w_gt.get('content_accuracy', {}).get('score', 0))
#             all_metrics['VLM_Relevance_W_GT'].append(judge_w_gt.get('relevance_and_responsiveness', {}).get('score', 0))
#             all_metrics['VLM_Consistency_W_GT'].append(judge_w_gt.get('consistency', {}).get('score', 0))
#             all_metrics['VLM_Overall_Score_W_GT'].append(judge_w_gt.get('overall_score', 0))

#         if judge_wo_gt:
#             if task == 'multimodal_generation':
#                 all_metrics['VLM_Coherence_WO_GT'].append(judge_wo_gt.get('coherence', {}).get('score', 0))
#                 all_metrics['VLM_Visual_Textual_Alignment_WO_GT'].append(judge_wo_gt.get('visual_textual_alignment', {}).get('score', 0))
#             all_metrics['VLM_Content_Accuracy_WO_GT'].append(judge_wo_gt.get('content_accuracy', {}).get('score', 0))
#             all_metrics['VLM_Relevance_WO_GT'].append(judge_wo_gt.get('relevance_and_responsiveness', {}).get('score', 0))
#             all_metrics['VLM_Consistency_WO_GT'].append(judge_wo_gt.get('consistency', {}).get('score', 0))
#             all_metrics['VLM_Overall_Score_WO_GT'].append(judge_wo_gt.get('overall_score', 0))
            
#     # --- 动态调整 Prompt（保持原逻辑）---
#     current_vlm_holistic_judge_w_gt_prompt = list(vlm_holistic_judge_w_gt_prompt)
#     current_vlm_holistic_judge_wo_gt_prompt = list(vlm_holistic_judge_wo_gt_prompt)
#     if task != 'multimodal_generation':
#         current_vlm_holistic_judge_w_gt_prompt[1] = ""
#         current_vlm_holistic_judge_wo_gt_prompt[1] = ""
#         current_vlm_holistic_judge_w_gt_prompt[3] = ""
#         current_vlm_holistic_judge_wo_gt_prompt[3] = ""

#     # 完整结果字典，用于每次覆盖写入
#     full_results_dict = dict(existing_data)

#     # --- 4. 只对未处理的样本进行迭代和评估 ---
#     if not unprocessed_items:
#         logging.info("所有样本均已处理完毕，无需执行新的评估。")
#     else:
#         logging.info(f"开始处理 {len(unprocessed_items)} 条新样本。")

#     total_batches = (len(unprocessed_items) + batch_size - 1) // batch_size
#     for i in tqdm(range(0, len(unprocessed_items), batch_size), desc="Evaluating Batches", total=total_batches):
#         batch_data = unprocessed_items[i:i+batch_size]

#         # --- 准备异步任务 (这部分逻辑和原来一致) ---
#         vlm_judge_w_gt_requests = []
#         vlm_judge_wo_gt_requests = []
#         request_to_index = []

#         eval_images, ref_images = [], []
#         eval_texts, ref_texts = [], []

#         for idx, item in enumerate(batch_data):
#             # 准备图像/文本/VLM请求
#             if task in ['multimodal_generation', 'image_edit']:
#                 try:
#                     eval_images.append(Image.open(item['output_image']).convert('RGB'))
#                     ref_images.append(Image.open(os.path.join(data_path, item['input_image'])).convert('RGB'))
#                 except (FileNotFoundError, IOError) as e:
#                     logging.warning(f"无法加载图片，跳过图像指标计算: {e}")

#             if task in ['multimodal_generation', 'vqa']:
#                 eval_texts.append(item.get('response', ''))
#                 ref_texts.append(item.get('instruction', ''))

#             # VLM with GT
#             w_gt_prompt = "\n".join([
#                 current_vlm_holistic_judge_w_gt_prompt[0], current_vlm_holistic_judge_w_gt_prompt[1],
#                 current_vlm_holistic_judge_w_gt_prompt[2], current_vlm_holistic_judge_w_gt_prompt[3],
#                 current_vlm_holistic_judge_w_gt_prompt[4], item.get('instruction', 'N/A'),
#                 current_vlm_holistic_judge_w_gt_prompt[5], item.get('answer', 'N/A'),
#                 current_vlm_holistic_judge_w_gt_prompt[6], item.get('response', 'N/A'),
#                 current_vlm_holistic_judge_w_gt_prompt[7]
#             ])
#             if 'ground_truth_image' in item and 'output_image' in item:
#                 vlm_judge_w_gt_requests.append(
#                     (w_gt_prompt, os.path.join(data_path, item['ground_truth_image']), item['output_image'], "Ground Truth", "Generated Answer", None, None)
#                 )
#                 request_to_index.append(('w_gt', idx))

#             # VLM w/o GT
#             wo_gt_prompt = "\n".join([
#                 current_vlm_holistic_judge_wo_gt_prompt[0], current_vlm_holistic_judge_wo_gt_prompt[1],
#                 current_vlm_holistic_judge_wo_gt_prompt[2], current_vlm_holistic_judge_wo_gt_prompt[3],
#                 current_vlm_holistic_judge_wo_gt_prompt[4], item.get('instruction', 'N/A'),
#                 current_vlm_holistic_judge_wo_gt_prompt[5],
#                 current_vlm_holistic_judge_wo_gt_prompt[6], item.get('response', 'N/A'),
#                 current_vlm_holistic_judge_wo_gt_prompt[7]
#             ])
#             if 'input_image' in item and 'output_image' in item:
#                 vlm_judge_wo_gt_requests.append(
#                     (wo_gt_prompt, os.path.join(data_path, item['input_image']), item['output_image'], "Input", "Output", None, None)
#                 )
#                 request_to_index.append(('wo_gt', idx))

#         # --- 执行非VLM任务 ---
#         tasks_to_run = []
#         if task in ['multimodal_generation', 'image_edit'] and eval_images:
#             tasks_to_run.extend([
#                 batch_async_FR_IQA(eval_images, ref_images, 'lpips'),
#                 batch_async_FR_IQA(eval_images, ref_images, 'psnr'),
#                 batch_async_FR_IQA(eval_images, ref_images, 'ssim')
#             ])
#         if task in ['multimodal_generation', 'vqa'] and eval_texts:
#             tasks_to_run.extend([
#                 batch_async_evaluate_text_quality(eval_texts, ref_texts, 'bleu'),
#                 batch_async_evaluate_text_quality(eval_texts, ref_texts, 'bertscore')
#             ])
        
#         results = await asyncio.gather(*tasks_to_run, return_exceptions=True) if tasks_to_run else []
#         # (解析和添加非VLM指标到 all_metrics 的逻辑保持不变)
#         res_idx = 0
#         if eval_images:
#             for metric_name in ['LPIPS', 'PSNR', 'SSIM']:
#                 val = results[res_idx]; all_metrics[metric_name].extend(val if not isinstance(val, Exception) else [0.0] * len(eval_images)); res_idx += 1
#         if eval_texts:
#             for metric_name in ['BLEU', 'BERT_Score']:
#                 val = results[res_idx]; all_metrics[metric_name].extend(val if not isinstance(val, Exception) else [0.0] * len(eval_texts)); res_idx += 1

#         # --- 执行 VLM Judge ---
#         vlm_tasks = []
#         if vlm_judge_w_gt_requests: vlm_tasks.append(vlm_client.generate_batch(vlm_judge_w_gt_requests, concurrency=8))
#         if vlm_judge_wo_gt_requests: vlm_tasks.append(vlm_client.generate_batch(vlm_judge_wo_gt_requests, concurrency=8))
#         vlm_results = await asyncio.gather(*vlm_tasks, return_exceptions=True) if vlm_tasks else []

#         # --- 解析VLM结果并更新batch_data中的item ---
#         vlm_res_idx, request_ptr = 0, 0
#         if vlm_judge_w_gt_requests:
#             w_gt_results = vlm_results[vlm_res_idx] if not isinstance(vlm_results[vlm_res_idx], Exception) else []
#             for res in w_gt_results:
#                 req_type, data_idx = request_to_index[request_ptr]
#                 item = batch_data[data_idx]
#                 if res and not res.get('error'): item['vlm_judge_w_gt_result'] = extract_json(res['text'])
#                 else: item['vlm_judge_w_gt_result'] = {}
#                 request_ptr += 1
#             vlm_res_idx += 1
        
#         if vlm_judge_wo_gt_requests:
#             wo_gt_results = vlm_results[vlm_res_idx] if vlm_res_idx < len(vlm_results) and not isinstance(vlm_results[vlm_res_idx], Exception) else []
#             for res in wo_gt_results:
#                 req_type, data_idx = request_to_index[request_ptr]
#                 item = batch_data[data_idx]
#                 if res and not res.get('error'): item['vlm_judge_wo_gt_result'] = extract_json(res['text'])
#                 else: item['vlm_judge_wo_gt_result'] = {}
#                 request_ptr += 1

#         # --- 将新处理完的结果添加到 all_metrics 和 full_results_dict ---
#         for item in batch_data:
#             # 添加到 all_metrics
#             judge_w_gt = item.get('vlm_judge_w_gt_result', {})
#             judge_wo_gt = item.get('vlm_judge_wo_gt_result', {})
#             if judge_w_gt:
#                 if task == 'multimodal_generation':
#                     all_metrics['VLM_Coherence_W_GT'].append(judge_w_gt.get('coherence', {}).get('score', 0))
#                     all_metrics['VLM_Visual_Textual_Alignment_W_GT'].append(judge_w_gt.get('visual_textual_alignment', {}).get('score', 0))
#                 all_metrics['VLM_Content_Accuracy_W_GT'].append(judge_w_gt.get('content_accuracy', {}).get('score', 0))
#                 all_metrics['VLM_Relevance_W_GT'].append(judge_w_gt.get('relevance_and_responsiveness', {}).get('score', 0))
#                 all_metrics['VLM_Consistency_W_GT'].append(judge_w_gt.get('consistency', {}).get('score', 0))
#                 all_metrics['VLM_Overall_Score_W_GT'].append(judge_w_gt.get('overall_score', 0))
#             if judge_wo_gt:
#                 if task == 'multimodal_generation':
#                     all_metrics['VLM_Coherence_WO_GT'].append(judge_wo_gt.get('coherence', {}).get('score', 0))
#                     all_metrics['VLM_Visual_Textual_Alignment_WO_GT'].append(judge_wo_gt.get('visual_textual_alignment', {}).get('score', 0))
#                 all_metrics['VLM_Content_Accuracy_WO_GT'].append(judge_wo_gt.get('content_accuracy', {}).get('score', 0))
#                 all_metrics['VLM_Relevance_WO_GT'].append(judge_wo_gt.get('relevance_and_responsiveness', {}).get('score', 0))
#                 all_metrics['VLM_Consistency_WO_GT'].append(judge_wo_gt.get('consistency', {}).get('score', 0))
#                 all_metrics['VLM_Overall_Score_WO_GT'].append(judge_wo_gt.get('overall_score', 0))

#             # 更新用于保存的字典
#             uid = generate_sample_id(item)
#             full_results_dict[uid] = item

#         # --- 5. 每次处理完批次后，覆盖写入完整的中间文件 ---
#         with open(intermediate_file, 'w', encoding='utf-8') as f:
#             for item in full_results_dict.values():
#                 f.write(json.dumps(item, ensure_ascii=False) + '\n')

#     # --- 聚合最终结果 (现在 all_metrics 包含了旧样本和新样本的所有数据) ---
#     final_results = {}
#     for metric, values in all_metrics.items():
#         if values:
#             final_results[f"Average_{metric}"] = float(np.mean(values))

#     accuracy_rates = calculate_accuracy_rates(all_metrics)
#     final_results.update(accuracy_rates)

#     logging.info(f"评估完成。中间结果已保存至: {intermediate_file}")
#     return final_results


async def basic_eval(
    data: list,
    batch_size: int,
    task: str,
    data_path: str,
    jsonl_path: str,
    run_vlm_judge: bool = True,
) -> dict:
    """
    对给定的数据集子集执行基础评估，并支持所有指标（图片、文本、VLM）的断点续评。
    """
    vlm_client = double_image_vlm() if run_vlm_judge else None
    all_metrics = defaultdict(list)

    # --- 1. 构建并加载中间结果 ---
    eval_results_dir = './eval_results'
    os.makedirs(eval_results_dir, exist_ok=True)
    base_name = os.path.basename(jsonl_path)
    suffix = '_with_vlm.jsonl' if run_vlm_judge else '_local_metrics.jsonl'
    intermediate_file = os.path.join(
        eval_results_dir, os.path.splitext(base_name)[0] + suffix
    )

    existing_data = {}  # uid -> item with all metrics
    if os.path.exists(intermediate_file):
        with open(intermediate_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                # 使用 item 的部分内容生成唯一ID
                uid = generate_sample_id(item)
                existing_data[uid] = item
        logging.info(f"检测到中间结果文件，已加载 {len(existing_data)} 条已评测样本。")

    # --- 2. 区分已处理和未处理的样本 ---
    unprocessed_items = []
    already_processed_items = []
    completion_key = 'vlm_judge_w_gt_result' if run_vlm_judge else '_local_metrics_complete'
    for item in data:
        uid = generate_sample_id(item)
        if uid in existing_data and completion_key in existing_data[uid]:
            # 使用已加载的、带有所有指标的完整数据
            already_processed_items.append(existing_data[uid])
        else:
            unprocessed_items.append(item)

    # --- 3. 预加载所有已处理样本的指标（图片、文本、VLM） ---
    logging.info(f"发现 {len(already_processed_items)} 条已完整处理的样本，将直接加载其所有结果。")
    for item in already_processed_items:
        # 加载图片和文本指标
        for metric_key in ['LPIPS', 'PSNR', 'SSIM', 'BLEU', 'BERT_Score']:
            if metric_key in item:
                all_metrics[metric_key].append(item[metric_key])

        # 加载VLM Judge指标
        judge_w_gt = item.get('vlm_judge_w_gt_result')
        judge_wo_gt = item.get('vlm_judge_wo_gt_result')

        if judge_w_gt:
            if task == 'multimodal_generation':
                all_metrics['VLM_Coherence_W_GT'].append(judge_w_gt.get('coherence', {}).get('score', 0))
                all_metrics['VLM_Visual_Textual_Alignment_W_GT'].append(judge_w_gt.get('visual_textual_alignment', {}).get('score', 0))
            all_metrics['VLM_Content_Accuracy_W_GT'].append(judge_w_gt.get('content_accuracy', {}).get('score', 0))
            all_metrics['VLM_Relevance_W_GT'].append(judge_w_gt.get('relevance_and_responsiveness', {}).get('score', 0))
            all_metrics['VLM_Consistency_W_GT'].append(judge_w_gt.get('consistency', {}).get('score', 0))
            all_metrics['VLM_Overall_Score_W_GT'].append(judge_w_gt.get('overall_score', 0))

        if judge_wo_gt:
            if task == 'multimodal_generation':
                all_metrics['VLM_Coherence_WO_GT'].append(judge_wo_gt.get('coherence', {}).get('score', 0))
                all_metrics['VLM_Visual_Textual_Alignment_WO_GT'].append(judge_wo_gt.get('visual_textual_alignment', {}).get('score', 0))
            all_metrics['VLM_Content_Accuracy_WO_GT'].append(judge_wo_gt.get('content_accuracy', {}).get('score', 0))
            all_metrics['VLM_Relevance_WO_GT'].append(judge_wo_gt.get('relevance_and_responsiveness', {}).get('score', 0))
            all_metrics['VLM_Consistency_WO_GT'].append(judge_wo_gt.get('consistency', {}).get('score', 0))
            all_metrics['VLM_Overall_Score_WO_GT'].append(judge_wo_gt.get('overall_score', 0))

    # --- 动态调整 Prompt（保持原逻辑）---
    current_vlm_holistic_judge_w_gt_prompt = list(vlm_holistic_judge_w_gt_prompt)
    current_vlm_holistic_judge_wo_gt_prompt = list(vlm_holistic_judge_wo_gt_prompt)
    if task != 'multimodal_generation':
        current_vlm_holistic_judge_w_gt_prompt[1] = ""
        current_vlm_holistic_judge_wo_gt_prompt[1] = ""
        current_vlm_holistic_judge_w_gt_prompt[3] = ""
        current_vlm_holistic_judge_wo_gt_prompt[3] = ""

    # 完整结果字典，用于每次覆盖写入
    full_results_dict = dict(existing_data)

    # --- 4. 只对未处理的样本进行迭代和评估 ---
    if not unprocessed_items:
        logging.info("所有样本均已处理完毕，无需执行新的评估。")
    else:
        logging.info(f"开始处理 {len(unprocessed_items)} 条新样本。")

    total_batches = (len(unprocessed_items) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(unprocessed_items), batch_size), desc="Evaluating Batches", total=total_batches):
        batch_data = unprocessed_items[i:i+batch_size]

        # --- 准备异步任务 (这部分逻辑和原来一致) ---
        vlm_judge_w_gt_requests, vlm_judge_wo_gt_requests = [], []
        request_to_index = []
        eval_images, ref_images = [], []
        eval_texts, ref_texts = [], []

        for idx, item in enumerate(batch_data):
            # ... (准备图像/文本/VLM请求的逻辑保持不变)
            if task in ['multimodal_generation', 'image_edit']:
                try:
                    eval_images.append(
                        Image.open(resolve_image_path(data_path, item['output_image'])).convert('RGB')
                    )
                    ref_images.append(
                        Image.open(
                            resolve_image_path(data_path, item['ground_truth_image'])
                        ).convert('RGB')
                    )
                except (FileNotFoundError, IOError) as e:
                    logging.warning(f"无法加载图片，跳过图像指标计算: {e}")
            if task in ['multimodal_generation', 'vqa']:
                eval_texts.append(item.get('response', ''))
                ref_texts.append(item.get('answer', ''))
            if not run_vlm_judge:
                continue
            # ... (VLM请求准备逻辑保持不变)
            w_gt_prompt = "\n".join([
                current_vlm_holistic_judge_w_gt_prompt[0], current_vlm_holistic_judge_w_gt_prompt[1],
                current_vlm_holistic_judge_w_gt_prompt[2], current_vlm_holistic_judge_w_gt_prompt[3],
                current_vlm_holistic_judge_w_gt_prompt[4], item.get('instruction', 'N/A'),
                current_vlm_holistic_judge_w_gt_prompt[5], item.get('answer', 'N/A'),
                current_vlm_holistic_judge_w_gt_prompt[6], item.get('response', 'N/A'),
                current_vlm_holistic_judge_w_gt_prompt[7]
            ])
            if task == 'vqa' and item.get('input_image'):
                vlm_judge_w_gt_requests.append(
                    (
                        w_gt_prompt,
                        resolve_image_path(data_path, item['input_image']),
                        "",
                        "Input",
                        "",
                        None,
                        None,
                    )
                )
                request_to_index.append(('w_gt', idx))
            elif item.get('ground_truth_image') and item.get('output_image'):
                vlm_judge_w_gt_requests.append(
                    (
                        w_gt_prompt,
                        resolve_image_path(data_path, item['ground_truth_image']),
                        resolve_image_path(data_path, item['output_image']),
                        "Ground Truth",
                        "Generated Answer",
                        None,
                        None,
                    )
                )
                request_to_index.append(('w_gt', idx))
            wo_gt_prompt = "\n".join([
                current_vlm_holistic_judge_wo_gt_prompt[0], current_vlm_holistic_judge_wo_gt_prompt[1],
                current_vlm_holistic_judge_wo_gt_prompt[2], current_vlm_holistic_judge_wo_gt_prompt[3],
                current_vlm_holistic_judge_wo_gt_prompt[4], item.get('instruction', 'N/A'),
                current_vlm_holistic_judge_wo_gt_prompt[5],
                current_vlm_holistic_judge_wo_gt_prompt[6], item.get('response', 'N/A'),
                current_vlm_holistic_judge_wo_gt_prompt[7]
            ])
            if item.get('input_image'):
                vlm_judge_wo_gt_requests.append(
                    (
                        wo_gt_prompt,
                        resolve_image_path(data_path, item['input_image']),
                        resolve_image_path(data_path, item.get('output_image', '')),
                        "Input",
                        "Output",
                        None,
                        None,
                    )
                )
                request_to_index.append(('wo_gt', idx))


        # --- 执行并保存图片和文本指标 ---
        # Load PubMedBERT before LPIPS for mixed-modality batches.  On some
        # CPU-only PyTorch builds the reverse first-use order can stall.
        metric_specs = []
        if task in ['multimodal_generation', 'vqa'] and eval_texts:
            metric_specs.extend([
                ('BLEU', len(eval_texts), batch_async_evaluate_text_quality(eval_texts, ref_texts, 'bleu')),
                ('BERT_Score', len(eval_texts), batch_async_evaluate_text_quality(eval_texts, ref_texts, 'bertscore')),
            ])
        if task in ['multimodal_generation', 'image_edit'] and eval_images:
            metric_specs.extend([
                ('LPIPS', len(eval_images), batch_async_FR_IQA(eval_images, ref_images, 'lpips')),
                ('PSNR', len(eval_images), batch_async_FR_IQA(eval_images, ref_images, 'psnr')),
                ('SSIM', len(eval_images), batch_async_FR_IQA(eval_images, ref_images, 'ssim')),
            ])

        metric_results = (
            await asyncio.gather(
                *(spec[2] for spec in metric_specs), return_exceptions=True
            )
            if metric_specs else []
        )
        for (metric_name, sample_count, _), scores in zip(metric_specs, metric_results):
            if scores is None:
                logging.warning("metric %s 返回 None，跳过", metric_name)
                continue
            if isinstance(scores, Exception):
                logging.error("metric %s 失败: %s", metric_name, scores)
                scores = [0.0] * sample_count
            all_metrics[metric_name].extend(scores)
            for item_idx, score in enumerate(scores):
                batch_data[item_idx][metric_name] = score

        # # --- 打印调试信息 ---
        # print("len(vlm_judge_w_gt_requests):", len(vlm_judge_w_gt_requests))
        # print("len(vlm_judge_wo_gt_requests):", len(vlm_judge_wo_gt_requests))
        # print("len(request_to_index):", len(request_to_index))

        # --- 为每类请求分别维护索引映射 ---
        w_gt_request_to_index = []
        wo_gt_request_to_index = []

        # 拆分 request_to_index
        w_gt_request_to_index = [x for x in request_to_index if x[0] == 'w_gt']
        wo_gt_request_to_index = [x for x in request_to_index if x[0] == 'wo_gt']

        # --- 执行 VLM Judge ---
        vlm_tasks = []
        if vlm_judge_w_gt_requests:
            vlm_tasks.append(vlm_client.generate_batch(vlm_judge_w_gt_requests, concurrency=8))
        if vlm_judge_wo_gt_requests:
            vlm_tasks.append(vlm_client.generate_batch(vlm_judge_wo_gt_requests, concurrency=8))

        vlm_results = await asyncio.gather(*vlm_tasks, return_exceptions=True) if vlm_tasks else []

        vlm_res_idx = 0

        # --- 处理 w_gt 结果 ---
        if vlm_judge_w_gt_requests:
            w_gt_results = vlm_results[vlm_res_idx] if not isinstance(vlm_results[vlm_res_idx], Exception) else []
            for i, res in enumerate(w_gt_results):
                if i >= len(w_gt_request_to_index):
                    break
                _, data_idx = w_gt_request_to_index[i]
                item = batch_data[data_idx]
                item['vlm_judge_w_gt_result'] = extract_json(res['text']) if res and not res.get('error') else {}
            vlm_res_idx += 1

        # --- 处理 wo_gt 结果 ---
        if vlm_judge_wo_gt_requests:
            wo_gt_results = vlm_results[vlm_res_idx] if vlm_res_idx < len(vlm_results) and not isinstance(vlm_results[vlm_res_idx], Exception) else []
            for i, res in enumerate(wo_gt_results):
                if i >= len(wo_gt_request_to_index):
                    break
                _, data_idx = wo_gt_request_to_index[i]
                item = batch_data[data_idx]
                item['vlm_judge_wo_gt_result'] = extract_json(res['text']) if res and not res.get('error') else {}


        # # --- 为每类请求分别维护索引映射 ---
        # w_gt_request_to_index = []
        # wo_gt_request_to_index = []

        # # 假设 request_to_index 里原来是混合的，可以这样拆分：
        # # 这里假设前 len(vlm_judge_w_gt_requests) 个是 w_gt，其余是 wo_gt
        # w_gt_request_to_index = request_to_index[:len(vlm_judge_w_gt_requests)]
        # wo_gt_request_to_index = request_to_index[len(vlm_judge_w_gt_requests):]

        # # --- 执行 VLM Judge ---
        # vlm_tasks = []
        # if vlm_judge_w_gt_requests:
        #     vlm_tasks.append(vlm_client.generate_batch(vlm_judge_w_gt_requests, concurrency=8))
        # if vlm_judge_wo_gt_requests:
        #     vlm_tasks.append(vlm_client.generate_batch(vlm_judge_wo_gt_requests, concurrency=8))

        # vlm_results = await asyncio.gather(*vlm_tasks, return_exceptions=True) if vlm_tasks else []

        # vlm_res_idx = 0

        # # --- 处理 w_gt ---
        # if vlm_judge_w_gt_requests:
        #     w_gt_results = vlm_results[vlm_res_idx] if not isinstance(vlm_results[vlm_res_idx], Exception) else []
        #     for i, res in enumerate(w_gt_results):
        #         if i >= len(w_gt_request_to_index):  # 安全保护
        #             break
        #         _, data_idx = w_gt_request_to_index[i]
        #         item = batch_data[data_idx]
        #         item['vlm_judge_w_gt_result'] = extract_json(res['text']) if res and not res.get('error') else {}
        #     vlm_res_idx += 1

        # # --- 处理 wo_gt ---
        # if vlm_judge_wo_gt_requests:
        #     wo_gt_results = vlm_results[vlm_res_idx] if vlm_res_idx < len(vlm_results) and not isinstance(vlm_results[vlm_res_idx], Exception) else []
        #     for i, res in enumerate(wo_gt_results):
        #         if i >= len(wo_gt_request_to_index):  # 安全保护
        #             break
        #         _, data_idx = wo_gt_request_to_index[i]
        #         item = batch_data[data_idx]
        #         item['vlm_judge_wo_gt_result'] = extract_json(res['text']) if res and not res.get('error') else {}


        # # --- 执行并保存VLM Judge结果 ---
        # print("len(vlm_judge_w_gt_requests):", len(vlm_judge_w_gt_requests))
        # print("len(vlm_judge_wo_gt_requests):", len(vlm_judge_wo_gt_requests))
        # print("len(request_to_index):", len(request_to_index))


        # # --- 执行并保存VLM Judge结果 ---
        # vlm_tasks = []
        # if vlm_judge_w_gt_requests: vlm_tasks.append(vlm_client.generate_batch(vlm_judge_w_gt_requests, concurrency=8))
        # if vlm_judge_wo_gt_requests: vlm_tasks.append(vlm_client.generate_batch(vlm_judge_wo_gt_requests, concurrency=8))
        # vlm_results = await asyncio.gather(*vlm_tasks, return_exceptions=True) if vlm_tasks else []

        # vlm_res_idx = 0

        # # --- 处理 w_gt ---
        # if vlm_judge_w_gt_requests:
        #     w_gt_results = vlm_results[vlm_res_idx] if not isinstance(vlm_results[vlm_res_idx], Exception) else []
        #     for i, res in enumerate(w_gt_results):
        #         if i >= len(request_to_index):  # 安全保护
        #             break
        #         _, data_idx = request_to_index[i]
        #         item = batch_data[data_idx]
        #         item['vlm_judge_w_gt_result'] = extract_json(res['text']) if res and not res.get('error') else {}
        #     vlm_res_idx += 1

        # # --- 处理 wo_gt ---
        # if vlm_judge_wo_gt_requests:
        #     wo_gt_results = vlm_results[vlm_res_idx] if vlm_res_idx < len(vlm_results) and not isinstance(vlm_results[vlm_res_idx], Exception) else []
        #     for i, res in enumerate(wo_gt_results):
        #         if i >= len(request_to_index):  # 同样防止越界
        #             break
        #         _, data_idx = request_to_index[i]
        #         item = batch_data[data_idx]
        #         item['vlm_judge_wo_gt_result'] = extract_json(res['text']) if res and not res.get('error') else {}


        # vlm_res_idx, request_ptr = 0, 0
        # if vlm_judge_w_gt_requests:
        #     w_gt_results = vlm_results[vlm_res_idx] if not isinstance(vlm_results[vlm_res_idx], Exception) else []
        #     for res in w_gt_results:
        #         _, data_idx = request_to_index[request_ptr]
        #         item = batch_data[data_idx]
        #         item['vlm_judge_w_gt_result'] = extract_json(res['text']) if res and not res.get('error') else {}
        #         request_ptr += 1
        #     vlm_res_idx += 1
        
        # if vlm_judge_wo_gt_requests:
        #     wo_gt_results = vlm_results[vlm_res_idx] if vlm_res_idx < len(vlm_results) and not isinstance(vlm_results[vlm_res_idx], Exception) else []
        #     for res in wo_gt_results:
        #         _, data_idx = request_to_index[request_ptr]
        #         item = batch_data[data_idx]
        #         item['vlm_judge_wo_gt_result'] = extract_json(res['text']) if res and not res.get('error') else {}
        #         request_ptr += 1

        # --- 聚合新批次的VLM指标并更新用于保存的字典 ---
        for item in batch_data:
            if not run_vlm_judge:
                item['_local_metrics_complete'] = True
            # 聚合VLM指标
            judge_w_gt = item.get('vlm_judge_w_gt_result', {})
            judge_wo_gt = item.get('vlm_judge_wo_gt_result', {})
            if judge_w_gt:
                if task == 'multimodal_generation':
                    all_metrics['VLM_Coherence_W_GT'].append(judge_w_gt.get('coherence', {}).get('score', 0))
                    all_metrics['VLM_Visual_Textual_Alignment_W_GT'].append(judge_w_gt.get('visual_textual_alignment', {}).get('score', 0))
                all_metrics['VLM_Content_Accuracy_W_GT'].append(judge_w_gt.get('content_accuracy', {}).get('score', 0))
                all_metrics['VLM_Relevance_W_GT'].append(judge_w_gt.get('relevance_and_responsiveness', {}).get('score', 0))
                all_metrics['VLM_Consistency_W_GT'].append(judge_w_gt.get('consistency', {}).get('score', 0))
                all_metrics['VLM_Overall_Score_W_GT'].append(judge_w_gt.get('overall_score', 0))
            if judge_wo_gt:
                if task == 'multimodal_generation':
                    all_metrics['VLM_Coherence_WO_GT'].append(judge_wo_gt.get('coherence', {}).get('score', 0))
                    all_metrics['VLM_Visual_Textual_Alignment_WO_GT'].append(judge_wo_gt.get('visual_textual_alignment', {}).get('score', 0))
                all_metrics['VLM_Content_Accuracy_WO_GT'].append(judge_wo_gt.get('content_accuracy', {}).get('score', 0))
                all_metrics['VLM_Relevance_WO_GT'].append(judge_wo_gt.get('relevance_and_responsiveness', {}).get('score', 0))
                all_metrics['VLM_Consistency_WO_GT'].append(judge_wo_gt.get('consistency', {}).get('score', 0))
                all_metrics['VLM_Overall_Score_WO_GT'].append(judge_wo_gt.get('overall_score', 0))

            # 更新用于保存的字典（现在item包含了所有指标）
            uid = generate_sample_id(item)
            full_results_dict[uid] = item

        # --- 5. 每次处理完批次后，覆盖写入包含所有指标的完整中间文件 ---
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            for item in full_results_dict.values():
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # --- 6. 聚合最终结果 (all_metrics 已包含所有样本的数据) ---
    final_results = {}
    for metric, values in all_metrics.items():
        if values:
            mean_val = float(np.mean(values))
            std_val = float(np.std(values))  # 计算标准差
            final_results[f"Average_{metric}"] = mean_val
            final_results[f"Std_{metric}"] = std_val  # 保存标准差

            # 如果想打印出来
            print(f"{metric}: mean={mean_val:.4f}, std={std_val:.4f}")

    accuracy_rates = calculate_accuracy_rates(all_metrics)
    final_results.update(accuracy_rates)

    logging.info(f"评估完成。包含所有指标的中间结果已保存至: {intermediate_file}")
    return final_results

async def basic_eval_for_type_wise(data: list, batch_size: int, task: str, data_path: str, jsonl_path: str) -> dict:
    """
    对给定的数据集子集执行基础评估，并支持所有指标（图片、文本、VLM）的断点续评。
    """
    vlm_client = double_image_vlm()
    all_metrics = defaultdict(list)


    base_name = os.path.basename(jsonl_path)
    name, ext = os.path.splitext(base_name)

    # 去掉第一个 "_" 之前（含 "_"）的内容
    if "_" in name:
        name = name.split("_", 1)[1]

    output_root = os.path.join("./eval_results_type_wise", name)
    os.makedirs(output_root, exist_ok=True)

    # --- 1. 构建并加载中间结果 ---
    eval_results_dir = output_root
    os.makedirs(eval_results_dir, exist_ok=True)
    print(eval_results_dir)

    base_name = os.path.basename(jsonl_path)
    intermediate_file = os.path.join(eval_results_dir, os.path.splitext(base_name)[0] + '_with_vlm.jsonl')

    existing_data = {}  # uid -> item with all metrics
    if os.path.exists(intermediate_file):
        with open(intermediate_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                # 使用 item 的部分内容生成唯一ID
                uid = generate_sample_id(item)
                existing_data[uid] = item
        logging.info(f"检测到中间结果文件，已加载 {len(existing_data)} 条已评测样本。")

    # --- 2. 区分已处理和未处理的样本 ---
    unprocessed_items = []
    already_processed_items = []
    for item in data:
        uid = generate_sample_id(item)
        if uid in existing_data and 'vlm_judge_w_gt_result' in existing_data[uid]: # 确保核心评测已完成
            # 使用已加载的、带有所有指标的完整数据
            already_processed_items.append(existing_data[uid])
        else:
            unprocessed_items.append(item)

    # --- 3. 预加载所有已处理样本的指标（图片、文本、VLM） ---
    logging.info(f"发现 {len(already_processed_items)} 条已完整处理的样本，将直接加载其所有结果。")
    for item in already_processed_items:
        # 加载图片和文本指标
        for metric_key in ['LPIPS', 'PSNR', 'SSIM', 'BLEU', 'BERT_Score']:
            if metric_key in item:
                all_metrics[metric_key].append(item[metric_key])

        # 加载VLM Judge指标
        judge_w_gt = item.get('vlm_judge_w_gt_result')
        judge_wo_gt = item.get('vlm_judge_wo_gt_result')

        if judge_w_gt:
            if task == 'multimodal_generation':
                all_metrics['VLM_Coherence_W_GT'].append(judge_w_gt.get('coherence', {}).get('score', 0))
                all_metrics['VLM_Visual_Textual_Alignment_W_GT'].append(judge_w_gt.get('visual_textual_alignment', {}).get('score', 0))
            all_metrics['VLM_Content_Accuracy_W_GT'].append(judge_w_gt.get('content_accuracy', {}).get('score', 0))
            all_metrics['VLM_Relevance_W_GT'].append(judge_w_gt.get('relevance_and_responsiveness', {}).get('score', 0))
            all_metrics['VLM_Consistency_W_GT'].append(judge_w_gt.get('consistency', {}).get('score', 0))
            all_metrics['VLM_Overall_Score_W_GT'].append(judge_w_gt.get('overall_score', 0))

        if judge_wo_gt:
            if task == 'multimodal_generation':
                all_metrics['VLM_Coherence_WO_GT'].append(judge_wo_gt.get('coherence', {}).get('score', 0))
                all_metrics['VLM_Visual_Textual_Alignment_WO_GT'].append(judge_wo_gt.get('visual_textual_alignment', {}).get('score', 0))
            all_metrics['VLM_Content_Accuracy_WO_GT'].append(judge_wo_gt.get('content_accuracy', {}).get('score', 0))
            all_metrics['VLM_Relevance_WO_GT'].append(judge_wo_gt.get('relevance_and_responsiveness', {}).get('score', 0))
            all_metrics['VLM_Consistency_WO_GT'].append(judge_wo_gt.get('consistency', {}).get('score', 0))
            all_metrics['VLM_Overall_Score_WO_GT'].append(judge_wo_gt.get('overall_score', 0))

    # --- 动态调整 Prompt（保持原逻辑）---
    current_vlm_holistic_judge_w_gt_prompt = list(vlm_holistic_judge_w_gt_prompt)
    current_vlm_holistic_judge_wo_gt_prompt = list(vlm_holistic_judge_wo_gt_prompt)
    if task != 'multimodal_generation':
        current_vlm_holistic_judge_w_gt_prompt[1] = ""
        current_vlm_holistic_judge_wo_gt_prompt[1] = ""
        current_vlm_holistic_judge_w_gt_prompt[3] = ""
        current_vlm_holistic_judge_wo_gt_prompt[3] = ""

    # 完整结果字典，用于每次覆盖写入
    full_results_dict = dict(existing_data)

    # --- 4. 只对未处理的样本进行迭代和评估 ---
    # if not unprocessed_items:
    #     logging.info("所有样本均已处理完毕，无需执行新的评估。")
    # else:
    #     logging.info(f"开始处理 {len(unprocessed_items)} 条新样本。")

    if not unprocessed_items:
        logging.info("所有样本均已处理完毕，无需执行新的评估。")
    else:
        logging.warning(f"检测到 {len(unprocessed_items)} 条未处理样本，但选择跳过它们，直接计算已有结果。")
        unprocessed_items = []  # 强制清空，跳过后续评估流程

    total_batches = (len(unprocessed_items) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(unprocessed_items), batch_size), desc="Evaluating Batches", total=total_batches):
        batch_data = unprocessed_items[i:i+batch_size]

        # --- 准备异步任务 (这部分逻辑和原来一致) ---
        vlm_judge_w_gt_requests, vlm_judge_wo_gt_requests = [], []
        request_to_index = []
        eval_images, ref_images = [], []
        eval_texts, ref_texts = [], []

        for idx, item in enumerate(batch_data):
            # ... (准备图像/文本/VLM请求的逻辑保持不变)
            if task in ['multimodal_generation', 'image_edit']:
                try:
                    eval_images.append(
                        Image.open(resolve_image_path(data_path, item['output_image'])).convert('RGB')
                    )
                    ref_images.append(
                        Image.open(
                            resolve_image_path(data_path, item['ground_truth_image'])
                        ).convert('RGB')
                    )
                except (FileNotFoundError, IOError) as e:
                    logging.warning(f"无法加载图片，跳过图像指标计算: {e}")
            if task in ['multimodal_generation', 'vqa']:
                eval_texts.append(item.get('response', ''))
                ref_texts.append(item.get('answer', ''))
            # ... (VLM请求准备逻辑保持不变)
            w_gt_prompt = "\n".join([
                current_vlm_holistic_judge_w_gt_prompt[0], current_vlm_holistic_judge_w_gt_prompt[1],
                current_vlm_holistic_judge_w_gt_prompt[2], current_vlm_holistic_judge_w_gt_prompt[3],
                current_vlm_holistic_judge_w_gt_prompt[4], item.get('instruction', 'N/A'),
                current_vlm_holistic_judge_w_gt_prompt[5], item.get('answer', 'N/A'),
                current_vlm_holistic_judge_w_gt_prompt[6], item.get('response', 'N/A'),
                current_vlm_holistic_judge_w_gt_prompt[7]
            ])
            if task == 'vqa' and item.get('input_image'):
                vlm_judge_w_gt_requests.append(
                    (
                        w_gt_prompt,
                        resolve_image_path(data_path, item['input_image']),
                        "",
                        "Input",
                        "",
                        None,
                        None,
                    )
                )
                request_to_index.append(('w_gt', idx))
            elif item.get('ground_truth_image') and item.get('output_image'):
                vlm_judge_w_gt_requests.append(
                    (
                        w_gt_prompt,
                        resolve_image_path(data_path, item['ground_truth_image']),
                        resolve_image_path(data_path, item['output_image']),
                        "Ground Truth",
                        "Generated Answer",
                        None,
                        None,
                    )
                )
                request_to_index.append(('w_gt', idx))
            wo_gt_prompt = "\n".join([
                current_vlm_holistic_judge_wo_gt_prompt[0], current_vlm_holistic_judge_wo_gt_prompt[1],
                current_vlm_holistic_judge_wo_gt_prompt[2], current_vlm_holistic_judge_wo_gt_prompt[3],
                current_vlm_holistic_judge_wo_gt_prompt[4], item.get('instruction', 'N/A'),
                current_vlm_holistic_judge_wo_gt_prompt[5],
                current_vlm_holistic_judge_wo_gt_prompt[6], item.get('response', 'N/A'),
                current_vlm_holistic_judge_wo_gt_prompt[7]
            ])
            if item.get('input_image'):
                vlm_judge_wo_gt_requests.append(
                    (
                        wo_gt_prompt,
                        resolve_image_path(data_path, item['input_image']),
                        resolve_image_path(data_path, item.get('output_image', '')),
                        "Input",
                        "Output",
                        None,
                        None,
                    )
                )
                request_to_index.append(('wo_gt', idx))


        # --- 执行并保存图片和文本指标 ---
        metric_tasks = []
        if task in ['multimodal_generation', 'image_edit'] and eval_images:
            metric_tasks.extend([
                batch_async_FR_IQA(eval_images, ref_images, 'lpips'),
                batch_async_FR_IQA(eval_images, ref_images, 'psnr'),
                batch_async_FR_IQA(eval_images, ref_images, 'ssim')
            ])
        if task in ['multimodal_generation', 'vqa'] and eval_texts:
            metric_tasks.extend([
                batch_async_evaluate_text_quality(eval_texts, ref_texts, 'bleu'),
                batch_async_evaluate_text_quality(eval_texts, ref_texts, 'bertscore')
            ])
        
        metric_results = await asyncio.gather(*metric_tasks, return_exceptions=True) if metric_tasks else []
        
        res_idx = 0
        if eval_images:
            for metric_name in ['LPIPS', 'PSNR', 'SSIM']:
                scores = metric_results[res_idx]
                
                if scores is None:
                    print(f"[WARN] metric {metric_name} 返回 None，跳过该条数据。")
                    res_idx += 1
                    continue
                
                if not isinstance(scores, Exception):
                    all_metrics[metric_name].extend(scores)
                    # 【核心修改】将指标保存回每个样本中
                    for item_idx, score in enumerate(scores):
                        batch_data[item_idx][metric_name] = score
                else:
                    all_metrics[metric_name].extend([0.0] * len(eval_images))
                res_idx += 1
        if eval_texts:
            for metric_name in ['BLEU', 'BERT_Score']:
                scores = metric_results[res_idx]
                               
                if scores is None:
                    print(f"[WARN] metric {metric_name} 返回 None，跳过该条数据。")
                    res_idx += 1
                    continue
                               
                if not isinstance(scores, Exception):
                    all_metrics[metric_name].extend(scores)
                    # 【核心修改】将指标保存回每个样本中
                    for item_idx, score in enumerate(scores):
                        batch_data[item_idx][metric_name] = score
                else:
                    all_metrics[metric_name].extend([0.0] * len(eval_texts))
                res_idx += 1

        # # --- 打印调试信息 ---
        # print("len(vlm_judge_w_gt_requests):", len(vlm_judge_w_gt_requests))
        # print("len(vlm_judge_wo_gt_requests):", len(vlm_judge_wo_gt_requests))
        # print("len(request_to_index):", len(request_to_index))

        # --- 为每类请求分别维护索引映射 ---
        w_gt_request_to_index = []
        wo_gt_request_to_index = []

        # 拆分 request_to_index
        w_gt_request_to_index = [x for x in request_to_index if x[0] == 'w_gt']
        wo_gt_request_to_index = [x for x in request_to_index if x[0] == 'wo_gt']

        # --- 执行 VLM Judge ---
        vlm_tasks = []
        if vlm_judge_w_gt_requests:
            vlm_tasks.append(vlm_client.generate_batch(vlm_judge_w_gt_requests, concurrency=8))
        if vlm_judge_wo_gt_requests:
            vlm_tasks.append(vlm_client.generate_batch(vlm_judge_wo_gt_requests, concurrency=8))

        vlm_results = await asyncio.gather(*vlm_tasks, return_exceptions=True) if vlm_tasks else []

        vlm_res_idx = 0

        # --- 处理 w_gt 结果 ---
        if vlm_judge_w_gt_requests:
            w_gt_results = vlm_results[vlm_res_idx] if not isinstance(vlm_results[vlm_res_idx], Exception) else []
            for i, res in enumerate(w_gt_results):
                if i >= len(w_gt_request_to_index):
                    break
                _, data_idx = w_gt_request_to_index[i]
                item = batch_data[data_idx]
                item['vlm_judge_w_gt_result'] = extract_json(res['text']) if res and not res.get('error') else {}
            vlm_res_idx += 1

        # --- 处理 wo_gt 结果 ---
        if vlm_judge_wo_gt_requests:
            wo_gt_results = vlm_results[vlm_res_idx] if vlm_res_idx < len(vlm_results) and not isinstance(vlm_results[vlm_res_idx], Exception) else []
            for i, res in enumerate(wo_gt_results):
                if i >= len(wo_gt_request_to_index):
                    break
                _, data_idx = wo_gt_request_to_index[i]
                item = batch_data[data_idx]
                item['vlm_judge_wo_gt_result'] = extract_json(res['text']) if res and not res.get('error') else {}


        # # --- 为每类请求分别维护索引映射 ---
        # w_gt_request_to_index = []
        # wo_gt_request_to_index = []

        # # 假设 request_to_index 里原来是混合的，可以这样拆分：
        # # 这里假设前 len(vlm_judge_w_gt_requests) 个是 w_gt，其余是 wo_gt
        # w_gt_request_to_index = request_to_index[:len(vlm_judge_w_gt_requests)]
        # wo_gt_request_to_index = request_to_index[len(vlm_judge_w_gt_requests):]

        # # --- 执行 VLM Judge ---
        # vlm_tasks = []
        # if vlm_judge_w_gt_requests:
        #     vlm_tasks.append(vlm_client.generate_batch(vlm_judge_w_gt_requests, concurrency=8))
        # if vlm_judge_wo_gt_requests:
        #     vlm_tasks.append(vlm_client.generate_batch(vlm_judge_wo_gt_requests, concurrency=8))

        # vlm_results = await asyncio.gather(*vlm_tasks, return_exceptions=True) if vlm_tasks else []

        # vlm_res_idx = 0

        # # --- 处理 w_gt ---
        # if vlm_judge_w_gt_requests:
        #     w_gt_results = vlm_results[vlm_res_idx] if not isinstance(vlm_results[vlm_res_idx], Exception) else []
        #     for i, res in enumerate(w_gt_results):
        #         if i >= len(w_gt_request_to_index):  # 安全保护
        #             break
        #         _, data_idx = w_gt_request_to_index[i]
        #         item = batch_data[data_idx]
        #         item['vlm_judge_w_gt_result'] = extract_json(res['text']) if res and not res.get('error') else {}
        #     vlm_res_idx += 1

        # # --- 处理 wo_gt ---
        # if vlm_judge_wo_gt_requests:
        #     wo_gt_results = vlm_results[vlm_res_idx] if vlm_res_idx < len(vlm_results) and not isinstance(vlm_results[vlm_res_idx], Exception) else []
        #     for i, res in enumerate(wo_gt_results):
        #         if i >= len(wo_gt_request_to_index):  # 安全保护
        #             break
        #         _, data_idx = wo_gt_request_to_index[i]
        #         item = batch_data[data_idx]
        #         item['vlm_judge_wo_gt_result'] = extract_json(res['text']) if res and not res.get('error') else {}


        # # --- 执行并保存VLM Judge结果 ---
        # print("len(vlm_judge_w_gt_requests):", len(vlm_judge_w_gt_requests))
        # print("len(vlm_judge_wo_gt_requests):", len(vlm_judge_wo_gt_requests))
        # print("len(request_to_index):", len(request_to_index))


        # # --- 执行并保存VLM Judge结果 ---
        # vlm_tasks = []
        # if vlm_judge_w_gt_requests: vlm_tasks.append(vlm_client.generate_batch(vlm_judge_w_gt_requests, concurrency=8))
        # if vlm_judge_wo_gt_requests: vlm_tasks.append(vlm_client.generate_batch(vlm_judge_wo_gt_requests, concurrency=8))
        # vlm_results = await asyncio.gather(*vlm_tasks, return_exceptions=True) if vlm_tasks else []

        # vlm_res_idx = 0

        # # --- 处理 w_gt ---
        # if vlm_judge_w_gt_requests:
        #     w_gt_results = vlm_results[vlm_res_idx] if not isinstance(vlm_results[vlm_res_idx], Exception) else []
        #     for i, res in enumerate(w_gt_results):
        #         if i >= len(request_to_index):  # 安全保护
        #             break
        #         _, data_idx = request_to_index[i]
        #         item = batch_data[data_idx]
        #         item['vlm_judge_w_gt_result'] = extract_json(res['text']) if res and not res.get('error') else {}
        #     vlm_res_idx += 1

        # # --- 处理 wo_gt ---
        # if vlm_judge_wo_gt_requests:
        #     wo_gt_results = vlm_results[vlm_res_idx] if vlm_res_idx < len(vlm_results) and not isinstance(vlm_results[vlm_res_idx], Exception) else []
        #     for i, res in enumerate(wo_gt_results):
        #         if i >= len(request_to_index):  # 同样防止越界
        #             break
        #         _, data_idx = request_to_index[i]
        #         item = batch_data[data_idx]
        #         item['vlm_judge_wo_gt_result'] = extract_json(res['text']) if res and not res.get('error') else {}


        # vlm_res_idx, request_ptr = 0, 0
        # if vlm_judge_w_gt_requests:
        #     w_gt_results = vlm_results[vlm_res_idx] if not isinstance(vlm_results[vlm_res_idx], Exception) else []
        #     for res in w_gt_results:
        #         _, data_idx = request_to_index[request_ptr]
        #         item = batch_data[data_idx]
        #         item['vlm_judge_w_gt_result'] = extract_json(res['text']) if res and not res.get('error') else {}
        #         request_ptr += 1
        #     vlm_res_idx += 1
        
        # if vlm_judge_wo_gt_requests:
        #     wo_gt_results = vlm_results[vlm_res_idx] if vlm_res_idx < len(vlm_results) and not isinstance(vlm_results[vlm_res_idx], Exception) else []
        #     for res in wo_gt_results:
        #         _, data_idx = request_to_index[request_ptr]
        #         item = batch_data[data_idx]
        #         item['vlm_judge_wo_gt_result'] = extract_json(res['text']) if res and not res.get('error') else {}
        #         request_ptr += 1

        # --- 聚合新批次的VLM指标并更新用于保存的字典 ---
        for item in batch_data:
            # 聚合VLM指标
            judge_w_gt = item.get('vlm_judge_w_gt_result', {})
            judge_wo_gt = item.get('vlm_judge_wo_gt_result', {})
            if judge_w_gt:
                if task == 'multimodal_generation':
                    all_metrics['VLM_Coherence_W_GT'].append(judge_w_gt.get('coherence', {}).get('score', 0))
                    all_metrics['VLM_Visual_Textual_Alignment_W_GT'].append(judge_w_gt.get('visual_textual_alignment', {}).get('score', 0))
                all_metrics['VLM_Content_Accuracy_W_GT'].append(judge_w_gt.get('content_accuracy', {}).get('score', 0))
                all_metrics['VLM_Relevance_W_GT'].append(judge_w_gt.get('relevance_and_responsiveness', {}).get('score', 0))
                all_metrics['VLM_Consistency_W_GT'].append(judge_w_gt.get('consistency', {}).get('score', 0))
                all_metrics['VLM_Overall_Score_W_GT'].append(judge_w_gt.get('overall_score', 0))
            if judge_wo_gt:
                if task == 'multimodal_generation':
                    all_metrics['VLM_Coherence_WO_GT'].append(judge_wo_gt.get('coherence', {}).get('score', 0))
                    all_metrics['VLM_Visual_Textual_Alignment_WO_GT'].append(judge_wo_gt.get('visual_textual_alignment', {}).get('score', 0))
                all_metrics['VLM_Content_Accuracy_WO_GT'].append(judge_wo_gt.get('content_accuracy', {}).get('score', 0))
                all_metrics['VLM_Relevance_WO_GT'].append(judge_wo_gt.get('relevance_and_responsiveness', {}).get('score', 0))
                all_metrics['VLM_Consistency_WO_GT'].append(judge_wo_gt.get('consistency', {}).get('score', 0))
                all_metrics['VLM_Overall_Score_WO_GT'].append(judge_wo_gt.get('overall_score', 0))

            # 更新用于保存的字典（现在item包含了所有指标）
            uid = generate_sample_id(item)
            full_results_dict[uid] = item

        # --- 5. 每次处理完批次后，覆盖写入包含所有指标的完整中间文件 ---
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            for item in full_results_dict.values():
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # --- 6. 聚合最终结果 (all_metrics 已包含所有样本的数据) ---
    final_results = {}
    for metric, values in all_metrics.items():
        if values:
            mean_val = float(np.mean(values))
            std_val = float(np.std(values))  # 计算标准差
            final_results[f"Average_{metric}"] = mean_val
            final_results[f"Std_{metric}"] = std_val  # 保存标准差

            # 如果想打印出来
            print(f"{metric}: mean={mean_val:.4f}, std={std_val:.4f}")

    accuracy_rates = calculate_accuracy_rates(all_metrics)
    final_results.update(accuracy_rates)

    logging.info(f"评估完成。包含所有指标的中间结果已保存至: {intermediate_file}")
    return final_results


async def eval_type_wise(data: list, batch_size: int, type_key: str, task: str, data_path: str, jsonl_path: str) -> dict:
    grouped_data = defaultdict(list)
    for item in data:
        modality_type = item.get(type_key, 'unknown')
        grouped_data[modality_type].append(item)
    
    # ============================================================
    # 🔥 在最后一个循环之前加入你的“modality 划分 vlm_jsonl 文件”的代码
    # ============================================================
    import json

    # 原文件：xxx.jsonl
    base_name = os.path.basename(jsonl_path)
    name, ext = os.path.splitext(base_name)

    output_root = os.path.join("./eval_results_type_wise", name)
    os.makedirs(output_root, exist_ok=True)

    # 生成 xxx_with_vlm.jsonl
    vlm_jsonl_path = os.path.join("./eval_results", f"{name}_with_vlm{ext}")

    # 如果文件存在才处理（避免异常）
    if os.path.exists(vlm_jsonl_path):
        modality_buckets = defaultdict(list)

        # 读取 vlm_jsonl_path 的全部条目
        with open(vlm_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    modality = obj.get("modality", "unknown")
                    modality_buckets[modality].append(obj)
                except json.JSONDecodeError:
                    continue

        # 逐 modality 写文件
        for modality, items in modality_buckets.items():
            out_path = os.path.join(
                output_root,
                f"{modality}_{os.path.basename(vlm_jsonl_path)}"
            )
            with open(out_path, "w", encoding="utf-8") as fw:
                for obj in items:
                    fw.write(json.dumps(obj, ensure_ascii=False) + "\n")

            logging.info(f"[VLM SPLIT] {modality}: {len(items)} 条 → {out_path}")
    else:
        raise FileNotFoundError(f"[VLM SPLIT ERROR] 未找到文件: {vlm_jsonl_path}")



    all_results = {}
    for modality_type, subset_data in grouped_data.items():
        logging.info(f"开始评估类型: '{modality_type}' (包含 {len(subset_data)} 个样本)")
        # 为每个子类生成独立的中间文件名（避免冲突）
        subset_jsonl_path = os.path.join(output_root, f"{modality_type}_{os.path.basename(jsonl_path)}")
        all_results[modality_type] = await basic_eval_for_type_wise(subset_data, batch_size, task, data_path, subset_jsonl_path)
        print(all_results)
        # all_results[modality_type] = await basic_eval(subset_data, batch_size, task, data_path, subset_jsonl_path)
    

    return all_results
    # tasks = []             # 保存 (subset_data, subset_jsonl_path)
    # modality_types = []    # 保存 modality_type

    # # ---- 第 1 个循环：写文件，来源改为 _with_vlm.jsonl ----
    # # vlm_jsonl_path = f"./eval_results/{os.path.basename(jsonl_path)}_with_vlm.jsonl"
    # base_name = os.path.basename(jsonl_path)
    # name, ext = os.path.splitext(base_name)  # 分离文件名和扩展名
    # vlm_jsonl_path = os.path.join("./eval_results", f"{name}_with_vlm{ext}")

    # logging.info(f"读取带 VLM 的文件：{vlm_jsonl_path}")

    # # 读取完整数据
    # full_data = []
    # with open(vlm_jsonl_path, 'r', encoding='utf-8') as f:
    #     for line in f:
    #         full_data.append(json.loads(line))

    # # 按 modality 分组
    # grouped_data_with_vlm = defaultdict(list)
    # for entry in full_data:
    #     modality = entry.get("modality", "Unknown")
    #     grouped_data_with_vlm[modality].append(entry)

    # # 写入每种 modality 的 jsonl
    # for modality_type, modality_entries in grouped_data_with_vlm.items():
    #     logging.info(f"准备类型 '{modality_type}'，共 {len(modality_entries)} 条")

    #     subset_jsonl_path = os.path.join(
    #         './eval_results_type_wise',
    #         f"{modality_type}_{os.path.basename(jsonl_path)}"
    #     )

    #     # 创建并写入 jsonl（覆盖写入）
    #     with open(subset_jsonl_path, 'w', encoding='utf-8') as f:
    #         for entry in modality_entries:
    #             f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    #     logging.info(f"写入完成：{subset_jsonl_path}")

    #     # 记录任务信息（第二个循环仍使用原来的 subset_data）
    #     tasks.append((subset_data, subset_jsonl_path))
    #     modality_types.append(modality_type)

    # # -------- 第 2 个循环：统一执行评估 basic_eval_for_type_wise --------
    # for modality_type, (subset_data, subset_jsonl_path) in zip(modality_types, tasks):
    #     logging.info(f"开始评估类型: '{modality_type}'")
    #     all_results[modality_type] = await basic_eval_for_type_wise(
    #         subset_data, batch_size, task, data_path, subset_jsonl_path
    #     )



def save_results(results: dict, jsonl_path: str, task: str):
    """将评估结果保存到json文件"""
    output_dir = './eval_results'
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.basename(jsonl_path)
    file_name = os.path.splitext(base_name)[0] + f"{task}_eval_results.json"
    output_path = os.path.join(output_dir, file_name)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
        
    logging.info(f"评估结果已成功保存到: {output_path}")

def save_results_for_type_wise(results: dict, jsonl_path: str, task: str):
    """将评估结果保存到json文件"""
    output_dir = './eval_results_type_wise'
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.basename(jsonl_path)
    file_name = os.path.splitext(base_name)[0] + f"{task}_eval_results.json"
    output_path = os.path.join(output_dir, file_name)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
        
    logging.info(f"评估结果已成功保存到: {output_path}")


def metrics_from_record(item: dict, task: str) -> dict:
    """Extract already-computed local and VLM metrics from one result record."""
    metrics = {}
    for key in ['LPIPS', 'PSNR', 'SSIM', 'BLEU', 'BERT_Score']:
        value = item.get(key)
        if isinstance(value, (int, float)) and np.isfinite(value):
            metrics[key] = float(value)

    judge_specs = [
        ('vlm_judge_w_gt_result', 'W_GT'),
        ('vlm_judge_wo_gt_result', 'WO_GT'),
    ]
    for field, suffix in judge_specs:
        judge = item.get(field)
        if not isinstance(judge, dict) or not judge:
            continue
        score_fields = {
            'content_accuracy': f'VLM_Content_Accuracy_{suffix}',
            'relevance_and_responsiveness': f'VLM_Relevance_{suffix}',
            'consistency': f'VLM_Consistency_{suffix}',
        }
        if task == 'multimodal_generation':
            score_fields.update({
                'coherence': f'VLM_Coherence_{suffix}',
                'visual_textual_alignment': f'VLM_Visual_Textual_Alignment_{suffix}',
            })
        for source_key, metric_key in score_fields.items():
            section = judge.get(source_key)
            value = section.get('score') if isinstance(section, dict) else None
            if isinstance(value, (int, float)) and np.isfinite(value):
                metrics[metric_key] = float(value)
        overall = judge.get('overall_score')
        if isinstance(overall, (int, float)) and np.isfinite(overall):
            metrics[f'VLM_Overall_Score_{suffix}'] = float(overall)
    return metrics


async def aggregate_type_wise_results(
    data: list,
    type_key: str,
    task: str,
    jsonl_path: str,
) -> dict:
    """Aggregate metrics already present in an evaluated JSONL by a record key."""
    grouped_data = defaultdict(list)
    for item in data:
        grouped_data[str(item.get(type_key, 'unknown'))].append(item)

    base_name = os.path.basename(jsonl_path)
    name = os.path.splitext(base_name)[0]
    output_root = os.path.join('./eval_results_type_wise', name)
    os.makedirs(output_root, exist_ok=True)

    all_results = {}
    for group_name, records in sorted(grouped_data.items()):
        group_metrics = defaultdict(list)
        for item in records:
            for metric_name, value in metrics_from_record(item, task).items():
                group_metrics[metric_name].append(value)

        summary = {'Sample_Count': len(records)}
        for metric_name, values in sorted(group_metrics.items()):
            summary[f'Average_{metric_name}'] = float(np.mean(values))
            summary[f'Std_{metric_name}'] = float(np.std(values))
        summary.update(calculate_accuracy_rates(group_metrics))
        all_results[group_name] = summary

        safe_group = group_name.replace('/', '_').replace(os.sep, '_')
        group_path = os.path.join(output_root, f'{safe_group}_{base_name}')
        with open(group_path, 'w', encoding='utf-8') as handle:
            for item in records:
                handle.write(json.dumps(item, ensure_ascii=False) + '\n')
        logging.info("类型 %s: %d 条 → %s", group_name, len(records), group_path)
    return all_results



async def main():
    parser = argparse.ArgumentParser(description="评估多模态模型生成结果的脚本")
    parser.add_argument('--data_path', type=str, default="./MedGEN", help='输入的jsonl文件路径')
    parser.add_argument('--jsonl_path', type=str, required=True, help='输入的jsonl文件路径')
    parser.add_argument('--batch_size', type=int, default=8, help='处理数据的批大小')
    parser.add_argument('--mission', type=str, choices=['basic_eval', 'type_wise'], default='basic_eval', help='评估任务类型')
    parser.add_argument('--type_key', type=str, default='modality', help='当 mission 为 type_wise 时，用于分类的键名')
    # --- 新增代码 ---
    parser.add_argument('--task', type=str, choices=['multimodal_generation', 'image_edit', 'vqa'], default='multimodal_generation', help='具体的评测任务类型 (multimodal_generation, image_edit, vqa)')
    parser.add_argument('--max_samples', type=int, default=None, help='只读取前N条记录')
    parser.add_argument('--validate-only', action='store_true', help='仅校验评测输入和图片路径，不加载指标模型或调用API')
    parser.add_argument('--local-metrics-only', action='store_true', help='执行本地指标但跳过付费 VLM judge；仅支持 basic_eval')
    # --- 新增代码结束 ---
    
    args = parser.parse_args()

    # 加载数据
    data = load_jsonl_data(args.jsonl_path)
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError('--max_samples 必须是正整数')
        data = data[:args.max_samples]
    if not data:
        return

    if args.validate_only:
        summary = validate_eval_input(data, args.task, args.data_path)
        print("评测输入验证通过（未加载指标模型、未调用API）:")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if EVAL_DEPENDENCY_ERROR is not None:
        raise RuntimeError(
            "缺少完整评测依赖；请按 README 安装 requirements-eval.txt"
        ) from EVAL_DEPENDENCY_ERROR

    if args.local_metrics_only and args.mission != 'basic_eval':
        raise ValueError('--local-metrics-only 仅支持 --mission basic_eval')

    # 执行评估
    if args.mission == 'basic_eval':
        results = await basic_eval(
            data,
            args.batch_size,
            args.task,
            args.data_path,
            args.jsonl_path,
            run_vlm_judge=not args.local_metrics_only,
        )
    elif args.mission == 'type_wise':
        results = await aggregate_type_wise_results(
            data, args.type_key, args.task, args.jsonl_path
        )
    else:
        logging.error(f"未知的 mission: {args.mission}")
        return

    # 保存结果
    if args.mission == 'type_wise':
        save_results_for_type_wise(results, args.jsonl_path, args.task)
    else:   
        save_results(results, args.jsonl_path, args.task)

if __name__ == '__main__':
    asyncio.run(main())
    
    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_gpt-image-1-mini_edit.jsonl --mission basic_eval --batch_size 1
    
    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_gpt-image-1_edit.jsonl --mission basic_eval --batch_size 4
    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_gpt-image-1_edit.jsonl --mission type_wise --type_key modality --batch_size 4
    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_gpt-image-1_edit.jsonl --mission basic_eval --batch_size 4 --task vqa
    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_gpt-image-1_edit.jsonl --mission type_wise --type_key modality --batch_size 4 --task vqa
    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_gpt-image-1_edit.jsonl --mission basic_eval --batch_size 4 --task image_edit
    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_gpt-image-1_edit.jsonl --mission type_wise --type_key modality --batch_size 4 --task image_edit

    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_gemini-2.5-flash-image-preview_edit.jsonl --mission basic_eval --batch_size 4 --task image_edit
    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_gpt-image-1-mini_edit.jsonl --mission basic_eval --batch_size 4 --task image_edit
    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_qwen-image-edit_edit.jsonl --mission basic_eval --batch_size 4 --task image_edit
    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_doubao-seedream_edit.jsonl --mission basic_eval --batch_size 4 --task image_edit
    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_Ming-UniVision_EDIT_edit.jsonl --mission basic_eval --batch_size 4 --task image_edit
    # python eval.py --jsonl_path ./inference_jsonl/gpt-4o-mini_imagen-4.0-fast_edit.jsonl --mission basic_eval --batch_size 4 --task image_edit
    # python eval.py --jsonl_path ./inference_jsonl/gemini-2.5-flash-lite_imagen-4.0-fast_edit.jsonl --mission basic_eval --batch_size 4 --task image_edit
    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_dall-e-3_edit.jsonl --mission basic_eval --batch_size 4 --task image_edit
    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_imagen-4.0-fast_edit.jsonl --mission basic_eval --batch_size 4 --task image_edit

    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_doubao-seedream_generate.jsonl --mission basic_eval --batch_size 4 --task multimodal_generation
    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_dall-e-3_generate.jsonl --mission basic_eval --batch_size 4 --task multimodal_generation
    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_imagen-4.0-fast_generate.jsonl --mission basic_eval --batch_size 4 --task multimodal_generation
    # python eval.py --jsonl_path ./inference_jsonl/gpt-4o-mini_doubao-seedream_generate.jsonl --mission basic_eval --batch_size 4 --task multimodal_generation
    # python eval.py --jsonl_path ./inference_jsonl/gemini-2.5-flash-lite_doubao-seedream_generate.jsonl --mission basic_eval --batch_size 4 --task multimodal_generation
    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_gemini-2.5-flash-image-preview_generate.jsonl --mission basic_eval --batch_size 4 --task multimodal_generation
    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_Ming-UniVision_EDIT4GEN_generate.jsonl --mission basic_eval --batch_size 4 --task multimodal_generation
    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-instruct_Showo_GENERATION_generate.jsonl --mission basic_eval --batch_size 4 --task multimodal_generation
    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-instruct_Showo_EDIT_generate.jsonl --mission basic_eval --batch_size 4 --task multimodal_generation


    # python eval.py --jsonl_path ./inference_jsonl/gemini-2.5-flash-lite_vqa.jsonl --mission basic_eval --batch_size 4 --task vqa
    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_vqa.jsonl --mission basic_eval --batch_size 4 --task vqa
    # python eval.py --jsonl_path ./inference_jsonl/gpt-4o-mini_vqa.jsonl --mission basic_eval --batch_size 4 --task vqa
    # python eval.py --jsonl_path ./inference_jsonl/Ming-UniVision_VLM_vqa.jsonl --mission basic_eval --batch_size 4 --task vqa
    # python eval.py --jsonl_path ./inference_jsonl/HuatuoGPT-Vision_vqa.jsonl --mission basic_eval --batch_size 4 --task vqa
    # python eval.py --jsonl_path ./inference_jsonl/RadFM_vqa.jsonl --mission basic_eval --batch_size 4 --task vqa
    # python eval.py --jsonl_path ./inference_jsonl/Showo_VLM_vqa.jsonl --mission basic_eval --batch_size 4 --task vqa


#----------------------------type_wise------------------------------
# python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_doubao-seedream_edit.jsonl --mission type_wise --type_key modality --batch_size 4 --task image_edit
# python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_qwen-image-edit_edit.jsonl --mission type_wise --type_key modality --batch_size 4 --task image_edit
# python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_gpt-image-1-mini_edit.jsonl --mission type_wise --type_key modality --batch_size 4 --task image_edit
# python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_dall-e-3_edit.jsonl --mission type_wise --type_key modality --batch_size 4 --task image_edit
# python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_imagen-4.0-fast_edit.jsonl --mission type_wise --type_key modality --batch_size 4 --task image_edit
# python eval.py --jsonl_path ./inference_jsonl/gpt-4o-mini_imagen-4.0-fast_edit.jsonl --mission type_wise --type_key modality --batch_size 4 --task image_edit
# python eval.py --jsonl_path ./inference_jsonl/gemini-2.5-flash-lite_imagen-4.0-fast_edit.jsonl --mission type_wise --type_key modality --batch_size 4 --task image_edit
# python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_gemini-2.5-flash-image-preview_edit.jsonl --mission type_wise --type_key modality --batch_size 4 --task image_edit
# python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-instruct_Showo_EDIT_generate.jsonl --mission type_wise --type_key modality --batch_size 4 --task image_edit
# python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_Ming-UniVision_EDIT_edit.jsonl --mission type_wise --type_key modality --batch_size 4 --task image_edit



# python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_doubao-seedream_generate.jsonl --mission type_wise --type_key modality --batch_size 4 --task multimodal_generation
# python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_dall-e-3_generate.jsonl --mission type_wise --type_key modality --batch_size 4 --task multimodal_generation
# python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_imagen-4.0-fast_generate.jsonl --mission type_wise --type_key modality --batch_size 4 --task multimodal_generation
# python eval.py --jsonl_path ./inference_jsonl/gpt-4o-mini_doubao-seedream_generate.jsonl --mission type_wise --type_key modality --batch_size 4 --task multimodal_generation
# python eval.py --jsonl_path ./inference_jsonl/gemini-2.5-flash-lite_doubao-seedream_generate.jsonl --mission type_wise --type_key modality --batch_size 4 --task multimodal_generation
# python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_gemini-2.5-flash-image-preview_generate.jsonl --mission type_wise --type_key modality --batch_size 4 --task multimodal_generation
# python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-instruct_Showo_GENERATION_generate.jsonl --mission type_wise --type_key modality --batch_size 4 --task multimodal_generation
# python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_Ming-UniVision_EDIT4GEN_generate.jsonl --mission type_wise --type_key modality --batch_size 4 --task multimodal_generation




#python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_vqa.jsonl --mission type_wise --type_key modality --batch_size 4 --task vqa
#python eval.py --jsonl_path ./inference_jsonl/gemini-2.5-flash-lite_vqa.jsonl --mission type_wise --type_key modality --batch_size 4 --task vqa
#python eval.py --jsonl_path ./inference_jsonl/gpt-4o-mini_vqa.jsonl --mission type_wise --type_key modality --batch_size 4 --task vqa
#python eval.py --jsonl_path ./inference_jsonl/HuatuoGPT-Vision_vqa.jsonl --mission type_wise --type_key modality --batch_size 4 --task vqa
#python eval.py --jsonl_path ./inference_jsonl/RadFM_vqa.jsonl --mission type_wise --type_key modality --batch_size 4 --task vqa
#python eval.py --jsonl_path ./inference_jsonl/Showo_VLM_vqa.jsonl --mission type_wise --type_key modality --batch_size 4 --task vqa
#python eval.py --jsonl_path ./inference_jsonl/Ming-UniVision_VLM_vqa.jsonl --mission type_wise --type_key modality --batch_size 4 --task vqa
