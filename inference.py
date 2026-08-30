#!/usr/bin/env python3
"""Run resumable MedGEN-Bench inference through configurable API clients."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jsonl-path", "--jsonl_path", dest="jsonl_path", required=True,
        help="Input Table IV JSONL file.",
    )
    parser.add_argument(
        "--mission", choices=("generate", "edit", "vqa"), required=True,
        help="Inference task type.",
    )
    parser.add_argument("--vlm-model", "--vlm_model", dest="vlm_model")
    parser.add_argument("--generate-model", "--generate_model", dest="generate_model")
    parser.add_argument("--edit-model", "--edit_model", dest="edit_model")
    parser.add_argument(
        "--vlm-config", "--vlm_config", dest="vlm_config", default="./api/config.yaml",
        help="OpenAI-compatible VLM YAML configuration.",
    )
    parser.add_argument(
        "--image-config", "--image_config", dest="image_config", default=None,
        help="Optional independent image API YAML configuration.",
    )
    parser.add_argument(
        "--output-image-path", "--output_image_path", dest="output_image_path",
        default="output_image", help="Directory for generated or edited images.",
    )
    parser.add_argument(
        "--output-jsonl-dir", "--output_jsonl_dir", dest="output_jsonl_dir",
        default="inference_jsonl", help="Directory for resumable inference JSONL files.",
    )
    parser.add_argument("--concurrency", type=int, default=1, help="Maximum VLM requests in flight.")
    parser.add_argument(
        "--image-concurrency", "--image_concurrency", dest="image_concurrency",
        type=int, default=1, help="Maximum image requests in flight.",
    )
    parser.add_argument("--max-tokens", "--max_tokens", dest="max_tokens", type=int, default=2048)
    parser.add_argument(
        "--max-samples", "--max_samples", dest="max_samples", type=int, default=None,
        help="Process only the first N records.",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Validate JSONL schemas and image paths without contacting a model.",
    )
    return parser.parse_args()


def record_key(record: Mapping[str, Any]) -> str:
    """Return a stable resume key for Table IV and legacy records."""
    sample_id = record.get("sample_id")
    if sample_id:
        return str(sample_id)
    image_token = json.dumps(record.get("input_image", ""), ensure_ascii=False, sort_keys=True)
    return f"{record.get('instruction', '')}|{image_token}"


def _is_completed_result(record: Mapping[str, Any]) -> bool:
    if record.get("error"):
        return False
    response = str(record.get("response") or "").strip()
    output_image = str(record.get("output_image") or "").strip()
    return bool(response or output_image)


def load_existing_results(output_file: str | Path) -> dict[str, dict[str, Any]]:
    """Load completed results only, so transient failures are retried."""
    path = Path(output_file)
    if not path.is_file():
        return {}
    results: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid result JSON at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Result at {path}:{line_number} is not a JSON object")
            if _is_completed_result(record):
                results[record_key(record)] = record
    return results


def _absolute_image_path(jsonl_path: str, image_ref: str) -> Path:
    candidate = Path(image_ref)
    if not candidate.is_absolute():
        candidate = Path(jsonl_path).resolve().parent / candidate
    return candidate.resolve()


def prepare_batch_data(
    data_list: Sequence[Mapping[str, Any]],
    existing_results: Mapping[str, Mapping[str, Any]],
    args: Any,
) -> list[dict[str, Any]]:
    """Filter completed records and attach a checked absolute input-image path."""
    prepared: list[dict[str, Any]] = []
    for record in data_list:
        item = dict(record)
        if record_key(item) in existing_results:
            continue
        input_ref = item.get("input_image")
        if isinstance(input_ref, list):
            if len(input_ref) > 1:
                raise ValueError(
                    "input_image contains multiple images; run prepare_medgen_tableiv.py "
                    "to create a labeled VQA contact sheet first"
                )
            raise ValueError("input_image must be a non-empty image path string")
        if not isinstance(input_ref, str) or not input_ref.strip():
            raise ValueError("input_image must be a non-empty image path string")
        input_path = _absolute_image_path(args.jsonl_path, input_ref)
        if not input_path.is_file():
            raise FileNotFoundError(f"Input image does not exist: {input_path}")
        item["_full_input_path"] = str(input_path)
        prepared.append(item)
    return prepared


def validate_dataset_records(data_list: Sequence[Mapping[str, Any]], args: Any) -> dict[str, Any]:
    """Validate input schema and all referenced paths without model calls."""
    errors: list[str] = []
    tasks: Counter[str] = Counter()
    contact_sheets = 0
    for line_number, record in enumerate(data_list, start=1):
        if not isinstance(record, Mapping):
            errors.append(f"line {line_number}: record is not an object")
            continue
        instruction = record.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            errors.append(f"line {line_number}: instruction must be a non-empty string")
        input_ref = record.get("input_image")
        if not isinstance(input_ref, str) or not input_ref.strip():
            errors.append(f"line {line_number}: input_image must be a non-empty string")
        else:
            input_path = _absolute_image_path(args.jsonl_path, input_ref)
            if not input_path.is_file():
                errors.append(f"line {line_number}: missing input image {input_path}")
            if "contact_sheet" in input_ref:
                contact_sheets += 1
        ground_truth = record.get("ground_truth_image")
        if ground_truth:
            if not isinstance(ground_truth, str):
                errors.append(f"line {line_number}: ground_truth_image must be a string")
            else:
                ground_truth_path = _absolute_image_path(args.jsonl_path, ground_truth)
                if not ground_truth_path.is_file():
                    errors.append(
                        f"line {line_number}: missing ground-truth image {ground_truth_path}"
                    )
        task = record.get("paper_task") or record.get("sub-category") or "unknown"
        tasks[str(task)] += 1
    if errors:
        preview = "\n".join(errors[:20])
        suffix = "" if len(errors) <= 20 else f"\n... {len(errors) - 20} more errors"
        raise ValueError(f"Input validation failed with {len(errors)} error(s):\n{preview}{suffix}")
    return {
        "records": len(data_list),
        "missing_images": 0,
        "paper_tasks": dict(sorted(tasks.items())),
        "contact_sheets": contact_sheets,
    }


def _slug(value: str | None, fallback: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip())
    normalized = normalized.strip(".-")
    return normalized[:120] or fallback


def get_output_filename(args: Any) -> str:
    """Return a safe filename that cannot contain path separators from model IDs."""
    vlm = _slug(getattr(args, "vlm_model", None), "vlm")
    mission = _slug(getattr(args, "mission", None), "task")
    if mission == "edit":
        image_model = _slug(getattr(args, "edit_model", None), "image-edit")
        return f"{vlm}_{image_model}_edit.jsonl"
    if mission == "generate":
        image_model = _slug(getattr(args, "generate_model", None), "image-generate")
        return f"{vlm}_{image_model}_generate.jsonl"
    return f"{vlm}_vqa.jsonl"


def get_output_filename_fake_only_for_4_generate_model_edit(args: Any) -> str:
    """Backward-compatible alias retained for external notebooks."""
    return get_output_filename(args)


async def read_input_jsonl(jsonl_path: str) -> list[dict[str, Any]]:
    """Read a JSONL file while reporting exact invalid locations."""
    path = Path(jsonl_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input JSONL does not exist: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            records.append(parsed)
    return records


async def write_result_to_file(output_file: str | Path, result: Mapping[str, Any]) -> None:
    """Append one JSONL result and flush it for resumable long-running jobs."""
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(result), ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")
        handle.flush()


async def process_batch(batch_items: Sequence[Mapping[str, Any]], args: Any) -> list[dict[str, Any]]:
    """Run one batch and merge API results into the original benchmark records."""
    from agent import VLM, VLM2Edit, VLM2Generate

    instructions = [str(item["instruction"]) for item in batch_items]
    image_paths = [str(item["_full_input_path"]) for item in batch_items]
    common = {
        "vlm_config_path": str(args.vlm_config),
        "output_image_path": str(args.output_image_path),
        "max_tokens": int(args.max_tokens),
    }
    if args.mission == "vqa":
        raw_results = await VLM(
            instructions,
            image_paths,
            args.vlm_model,
            config_path=str(args.vlm_config),
            output_image_path=str(args.output_image_path),
            concurrency=int(args.concurrency),
            max_tokens=int(args.max_tokens),
        )
    elif args.mission == "edit":
        raw_results = await VLM2Edit(
            instructions,
            image_paths,
            args.vlm_model,
            args.edit_model,
            image_config_path=args.image_config,
            vlm_concurrency=int(args.concurrency),
            image_concurrency=int(args.image_concurrency),
            **common,
        )
    elif args.mission == "generate":
        raw_results = await VLM2Generate(
            instructions,
            image_paths,
            args.vlm_model,
            args.generate_model,
            image_config_path=args.image_config,
            vlm_concurrency=int(args.concurrency),
            image_concurrency=int(args.image_concurrency),
            **common,
        )
    else:  # pragma: no cover - argparse enforces choices
        raise ValueError(f"Unknown mission: {args.mission}")

    if len(raw_results) != len(batch_items):
        raise RuntimeError(
            f"Model client returned {len(raw_results)} results for {len(batch_items)} inputs"
        )
    merged: list[dict[str, Any]] = []
    for item, result in zip(batch_items, raw_results):
        record = dict(item)
        record.pop("_full_input_path", None)
        record.update(dict(result))
        merged.append(record)
    return merged


def _validate_runtime_args(args: argparse.Namespace) -> None:
    if args.concurrency <= 0 or args.image_concurrency <= 0:
        raise ValueError("--concurrency and --image-concurrency must be positive")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if args.validate_only:
        return
    if not args.vlm_model:
        raise ValueError("--vlm-model is required for model inference")
    if args.mission == "generate" and not args.generate_model:
        raise ValueError("--generate-model is required when --mission generate")
    if args.mission == "edit" and not args.edit_model:
        raise ValueError("--edit-model is required when --mission edit")
    if args.mission in {"generate", "edit"} and not args.image_config:
        raise ValueError("--image-config is required for image generation or editing")


async def main() -> None:
    args = parse_args()
    _validate_runtime_args(args)
    records = await read_input_jsonl(args.jsonl_path)
    if args.max_samples is not None:
        records = records[: args.max_samples]
    if not records:
        raise ValueError("The selected input contains no records")

    if args.validate_only:
        summary = validate_dataset_records(records, args)
        prepared = prepare_batch_data(records, {}, args)
        if len(prepared) != len(records):  # pragma: no cover - defensive
            raise RuntimeError("Path preparation dropped input records")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    output_dir = Path(args.output_jsonl_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.output_image_path).mkdir(parents=True, exist_ok=True)
    output_file = output_dir / get_output_filename(args)
    completed = load_existing_results(output_file)
    pending = prepare_batch_data(records, completed, args)
    print(f"Completed records reused: {len(completed)}")
    print(f"Records to process: {len(pending)}")
    if not pending:
        print(f"No work remaining: {output_file}")
        return

    batches = [
        pending[index : index + args.concurrency]
        for index in range(0, len(pending), args.concurrency)
    ]
    for batch in tqdm(batches, desc="Inference", unit="batch"):
        results = await process_batch(batch, args)
        for result in results:
            await write_result_to_file(output_file, result)
    print(f"Inference output: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
