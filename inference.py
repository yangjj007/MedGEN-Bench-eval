import os
import json
import asyncio
import argparse
from collections import Counter
from typing import List, Dict, Any
from tqdm.asyncio import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="模型推理并生成jsonl输出")
    parser.add_argument("--jsonl_path", type=str, default="test_bench.jsonl",
                        help="输入jsonl文件路径")
    # parser.add_argument("--input_image_path", type=str, default="input_image",
    #                     help="输入图片文件夹路径")
    parser.add_argument("--output_image_path", type=str, default="output_image",
                        help="输出图片文件夹路径")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="并发度")
    parser.add_argument("--vlm_model", type=str, default=None,
                        help="合法VLM模型名")
    parser.add_argument("--generate_model", type=str, default=None,
                        help="合法图片生成模型名（仅在mission=generate时需要）")
    parser.add_argument("--edit_model", type=str, default=None,
                        help="合法图片编辑模型名（仅在mission=edit时需要）")
    parser.add_argument("--mission", type=str, choices=["generate", "edit", "vqa"], required=True,
                        help="任务类型：generate 或 edit")
    parser.add_argument("--output_jsonl_dir", type=str, default="inference_jsonl",
                        help="推理JSONL输出目录；Table IV建议使用独立子目录")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="只处理前N条记录，适合小规模真实推理测试")
    parser.add_argument("--validate-only", action="store_true",
                        help="仅验证JSONL和图片路径，不调用模型或写推理结果")
    return parser.parse_args()


# def load_existing_results(output_file: str) -> Dict[str, Dict]:
#     """加载已有的结果，用于断点续传"""
#     if not os.path.exists(output_file):
#         return {}
    
#     existing_results = {}
#     try:
#         with open(output_file, 'r', encoding='utf-8') as f:
#             for line in f:
#                 if line.strip():
#                     data = json.loads(line)
#                     # 使用instruction作为唯一标识（假设instruction是唯一的）
#                     # 如果不唯一，可能需要组合其他字段
#                     key = data.get('instruction', '') + '|' + data.get('input_image', '')
#                     existing_results[key] = data
#     except Exception as e:
#         print(f"警告：读取已有结果文件失败 {e}，将重新开始")
#         return {}
    
#     return existing_results

def record_key(data: Dict[str, Any]) -> str:
    """Return a stable resume key for legacy and Table IV adapter records."""
    sample_id = data.get("sample_id")
    if sample_id:
        return str(sample_id)
    image_token = json.dumps(
        data.get("input_image", ""), ensure_ascii=False, sort_keys=True
    )
    return f"{data.get('instruction', '')}|{image_token}"


def load_existing_results(output_file: str) -> Dict[str, Dict]:
    if not os.path.exists(output_file):
        return {}
    existing_results = {}
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    existing_results[record_key(data)] = data
    except Exception as e:
        print(f"警告：读取已有结果文件失败 {e}，将重新开始")
        return {}
    return existing_results

# def prepare_batch_data(data_list: List[Dict], existing_results: Dict, args) -> List[Dict]:
#     """准备需要处理的批次数据，跳过已处理的"""
    
#     to_process = []
#     for item in data_list:
#         # 构建唯一键
#         key = item.get('instruction', '') + '|' + item.get('input_image', '')
#         if key in existing_results:
#             continue
        
#         # 验证输入图片是否存在
#         full_input_path = os.path.join(os.path.dirname(args.jsonl_path),item['input_image']) 
#         if not os.path.exists(full_input_path):
#             print(f"警告：输入图片不存在，跳过: {full_input_path}")
#             continue
        
#         item['input_image'] = full_input_path
        
#         to_process.append(item)
    
#     return to_process

def prepare_batch_data(data_list: List[Dict], existing_results: Dict, args) -> List[Dict]:
    to_process = []
    base_dir = os.path.dirname(os.path.abspath(args.jsonl_path))  # e.g., /your/path/MedGEN
    
    for item in data_list:
        # 关键：用原始 input_image 构建 key（必须和输出文件中的值一致！）
        key = record_key(item)
        if key in existing_results:
            continue

        input_ref = item.get('input_image')
        if not isinstance(input_ref, str) or not input_ref.strip():
            if isinstance(input_ref, list) and len(input_ref) > 1:
                raise ValueError(
                    "检测到多图 input_image；请先运行 prepare_medgen_tableiv.py "
                    "生成带 contact sheet 的 eval 兼容视图"
                )
            raise ValueError(f"无效 input_image: {input_ref!r}")

        # 构建绝对路径用于文件检查
        full_path = input_ref if os.path.isabs(input_ref) else os.path.join(base_dir, input_ref)
        full_path = os.path.abspath(full_path)
        if not os.path.exists(full_path):
            print(f"警告：图片不存在: {full_path}")
            continue

        # 附加绝对路径供模型使用，但不污染原始字段
        item_with_abs = item.copy()
        item_with_abs['_full_input_path'] = full_path
        to_process.append(item_with_abs)
    
    return to_process


def validate_dataset_records(data_list: List[Dict], args) -> Dict[str, Any]:
    """Validate the schema and all input/ground-truth paths without API calls."""
    base_dir = os.path.dirname(os.path.abspath(args.jsonl_path))
    errors = []
    categories = Counter()
    tasks = Counter()
    contact_sheets = 0

    for index, item in enumerate(data_list, start=1):
        if not isinstance(item, dict):
            errors.append(f"line {index}: record is not an object")
            continue
        if not isinstance(item.get("instruction"), str) or not item["instruction"].strip():
            errors.append(f"line {index}: missing instruction")
        input_ref = item.get("input_image")
        if not isinstance(input_ref, str) or not input_ref.strip():
            errors.append(f"line {index}: input_image must be a non-empty string")
        else:
            input_path = input_ref if os.path.isabs(input_ref) else os.path.join(base_dir, input_ref)
            if not os.path.isfile(input_path):
                errors.append(f"line {index}: missing input_image {input_ref!r}")
            if (item.get("eval_adapter") or {}).get("input_strategy") == "labeled_contact_sheet":
                contact_sheets += 1

        ground_truth_ref = item.get("ground_truth_image")
        if ground_truth_ref:
            if not isinstance(ground_truth_ref, str):
                errors.append(f"line {index}: ground_truth_image must be a string when present")
            else:
                ground_truth_path = (
                    ground_truth_ref
                    if os.path.isabs(ground_truth_ref)
                    else os.path.join(base_dir, ground_truth_ref)
                )
                if not os.path.isfile(ground_truth_path):
                    errors.append(
                        f"line {index}: missing ground_truth_image {ground_truth_ref!r}"
                    )

        categories[str(item.get("category"))] += 1
        tasks[str(item.get("paper_task") or item.get("sub-category"))] += 1

    if errors:
        preview = "\n".join(errors[:50])
        raise ValueError(f"数据验证失败，共 {len(errors)} 个错误:\n{preview}")
    return {
        "records": len(data_list),
        "categories": dict(sorted(categories.items())),
        "tasks": dict(sorted(tasks.items())),
        "contact_sheet_records": contact_sheets,
        "missing_images": 0,
    }

def get_output_filename(args) -> str:
    """根据参数生成输出文件名"""
    if args.mission == "generate":
        return f"{args.vlm_model}_{args.generate_model}_{args.mission}.jsonl"
    elif args.mission == "edit":
        return f"{args.vlm_model}_{args.edit_model}_{args.mission}.jsonl"
    elif args.mission == "vqa":
        return f"{args.vlm_model}_{args.mission}.jsonl"
    
def get_output_filename_fake_only_for_4_generate_model_edit(args) -> str:
    """根据参数生成输出文件名"""
    if args.mission == "generate":
        return f"{args.vlm_model}_{args.generate_model}_edit.jsonl"


async def read_input_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    """读取输入jsonl文件。"""
    data_list = []
    with open(jsonl_path, 'r', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                data_list.append(json.loads(line))
    return data_list


async def write_result_to_file(output_file: str, result: Dict[str, Any]):
    """将单个结果追加到输出文件。"""
    with open(output_file, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(result, ensure_ascii=False) + '\n')



# async def process_batch(batch_items: List[Dict], args) -> List[Dict]:
#     instructions = [item['instruction'] for item in batch_items]
#     input_images = [item['_full_input_path'] for item in batch_items]  # 使用内部路径

#     if args.mission == "generate":
#         if args.generate_model is None:
#             raise ValueError("mission=generate 时必须提供 generate_model 参数")
#         results = await VLM2Generate(instructions, input_images)
#     else:  # edit
#         if args.edit_model is None:
#             raise ValueError("mission=edit 时必须提供 edit_model 参数")
#         results = await VLM2Edit(instructions, input_images)
    
#     processed_results = []
#     for i, item in enumerate(batch_items):
#         result = results[i] if i < len(results) else { ... }

#         # 构建输出项时，**恢复原始 input_image 字段**
#         output_item = item.copy()
#         output_item.pop('_full_input_path', None)  # 移除内部字段
#         # 注意：output_item['input_image'] 仍然是原始相对路径！

#         output_item['response'] = result.get('response', '')
#         output_item['raw_response'] = result.get('raw_response', '')
#         output_item['output_image'] = result.get('output_image', '')

#         processed_results.append(output_item)
    
#     return processed_results

async def process_batch(batch_items: List[Dict], args) -> List[Dict]:
    # Keep model/API dependencies out of dataset-only validation runs.
    from agent import VLM2Generate, VLM2Edit, VLM

    instructions = [item['instruction'] for item in batch_items]
    input_images = [item['_full_input_path'] for item in batch_items]  # 绝对路径给模型

    if args.mission == "edit":
        results = await VLM2Edit(instructions, input_images, args.vlm_model, args.edit_model)
    elif args.mission == "generate":
        results = await VLM2Generate(instructions, input_images, args.vlm_model, args.generate_model)
    elif args.mission == "vqa":
        results = await VLM(instructions, input_images, args.vlm_model)
    else:
        raise ValueError(f"未知任务类型: {args.mission}")

    processed_results = []
    for i, item in enumerate(batch_items):
        result = results[i] if i < len(results) else {'response': '', 'output_image': '', 'raw_response': ''}
        # 输出时，移除临时字段，保留原始 input_image（如 "images/edit/xxx.jpg"）
        output_item = {k: v for k, v in item.items() if k != '_full_input_path'}
        output_item.update({
            'response': result.get('response', ''),
            'raw_response': result.get('raw_response', ''),
            'output_image': result.get('output_image', '')
        })
        processed_results.append(output_item)
    return processed_results


async def main():
    args = parse_args()

    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max_samples 必须是正整数")
    if not args.validate_only and args.vlm_model is None:
        raise ValueError("真实推理必须提供 --vlm_model")
    if not args.validate_only and args.mission == "generate" and args.generate_model is None:
        raise ValueError("当 mission=generate 时，必须提供 --generate_model 参数")
    if not args.validate_only and args.mission == "edit" and args.edit_model is None:
        raise ValueError("当 mission=edit 时，必须提供 --edit_model 参数")

    # 读取输入数据
    print(f"正在读取输入文件: {args.jsonl_path}")
    input_data = await read_input_jsonl(args.jsonl_path)
    if args.max_samples is not None:
        input_data = input_data[:args.max_samples]
    print(f"共读取 {len(input_data)} 条数据")

    if args.validate_only:
        summary = validate_dataset_records(input_data, args)
        # Exercise the exact path preparation used by real inference too.
        prepared = prepare_batch_data(input_data, {}, args)
        if len(prepared) != len(input_data):
            raise ValueError(
                f"路径准备只保留 {len(prepared)}/{len(input_data)} 条记录"
            )
        print("数据验证通过（未调用模型）:")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    # 创建输出目录
    os.makedirs(args.output_image_path, exist_ok=True)
    os.makedirs(args.output_jsonl_dir, exist_ok=True)

    # 获取输出文件路径
    output_filename = get_output_filename(args)
    output_file = os.path.join(args.output_jsonl_dir, output_filename)

    # 加载已有结果
    existing_results = load_existing_results(output_file)
    print(f"已存在 {len(existing_results)} 条结果，将跳过这些")
    
    # 准备需要处理的数据
    to_process = prepare_batch_data(input_data, existing_results, args)
    print(f"需要处理 {len(to_process)} 条新数据")
    
    if not to_process:
        print("没有需要处理的新数据，程序退出")
        return
    
    # 分批处理
    batch_size = args.concurrency
    total_batches = (len(to_process) + batch_size - 1) // batch_size
    
    # 创建批次列表（便于 tqdm 迭代）
    batches = [
        to_process[i:i + batch_size]
        for i in range(0, len(to_process), batch_size)
    ]

    # 使用 tqdm 异步进度条
    for batch_items in tqdm(batches, desc="Processing batches", unit="batch"):
        # try:
        processed_batch = await process_batch(batch_items, args)
        for result in processed_batch:
            await write_result_to_file(output_file, result)
        # except Exception as e:
        #     print(f"当前批次处理失败: {e}")
        #     continue
    
    
    print(f"处理完成！结果保存在: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
    
    # python inference.py --vlm_model qwen3-vl-235b-a22b-instruct --mission edit --edit_model gpt-image-1
