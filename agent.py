"""Inference orchestration for MedGEN-Bench.

This module combines an OpenAI-compatible VLM with the existing image
generation and editing clients. All public functions return records in the
shape expected by inference.py.
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

try:
    import json5
except ModuleNotFoundError:  # pragma: no cover - requirements.txt installs json5
    json5 = None

from api.get_vlm_res import single_image_vlm


DEFAULT_CONFIG_PATH = "./config.yaml"
DEFAULT_OUTPUT_IMAGE_PATH = "./output_image"

VLM_ONLY_PROMPT = """
You are a medical vision-language assistant. Analyze the provided medical
image and answer the instruction accurately and conservatively.

Instruction:
{instruction}

Return exactly one JSON object with this schema:
{{"output_text": "the answer to the instruction"}}
""".strip()

VLM2EDIT_PROMPT = """
You are a medical vision-language assistant preparing a safe image-editing
request. Analyze the provided medical image and the instruction. Preserve
clinically relevant anatomy and findings unless the instruction explicitly
changes them.

Instruction:
{instruction}

Return exactly one JSON object with this schema:
{{"output_text": "brief clinical response", "edit_prompt": "precise image-editing prompt"}}
""".strip()

VLM2GENERATE_PROMPT = """
You are a medical vision-language assistant preparing an image-generation
request. Analyze the provided medical image and the instruction, then write a
precise prompt for the target image while preserving clinically relevant
context.

Instruction:
{instruction}

Return exactly one JSON object with this schema:
{{"output_text": "brief clinical response", "generate_prompt": "precise image-generation prompt"}}
""".strip()

# Preserve the historical constant spelling for downstream imports.
VLM2Edit_PROMPT = VLM2EDIT_PROMPT
VLM2Generate_PROMPT = VLM2GENERATE_PROMPT


def _iter_json_object_candidates(text: str) -> list[str]:
    """Extract balanced JSON-object candidates without using unsafe evaluation."""

    fence = chr(96) * 3
    stripped = re.sub(
        rf"^\s*{re.escape(fence)}(?:json|json5)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    stripped = re.sub(rf"\s*{re.escape(fence)}\s*$", "", stripped).strip()
    if stripped:
        candidates: list[str] = [stripped]
    else:
        candidates = []

    for start, character in enumerate(stripped):
        if character != "{":
            continue
        depth = 0
        quote: Optional[str] = None
        escaped = False
        for end in range(start, len(stripped)):
            current = stripped[end]
            if quote is not None:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == quote:
                    quote = None
                continue
            if current in {"'", '"'}:
                quote = current
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(stripped[start : end + 1])
                    break
    return candidates


def extract_json(select_response: Any) -> dict[str, Any]:
    """Parse a JSON object from an OpenAI-compatible VLM response."""

    if isinstance(select_response, Mapping):
        return dict(select_response)
    if not isinstance(select_response, str):
        raise TypeError(
            "VLM response must be a string or mapping; "
            f"received {type(select_response).__name__}"
        )

    parse_errors: list[str] = []
    seen: set[str] = set()
    for candidate in _iter_json_object_candidates(select_response):
        if candidate in seen:
            continue
        seen.add(candidate)
        parsers = [json.loads]
        if json5 is not None:
            parsers.append(json5.loads)
        for parser in parsers:
            try:
                parsed = parser(candidate)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                parse_errors.append(str(exc))
                continue
            if isinstance(parsed, Mapping):
                return dict(parsed)
            parse_errors.append("parsed value is not an object")
    detail = parse_errors[-1] if parse_errors else "no JSON object found"
    raise ValueError(f"Could not parse a JSON object from VLM output: {detail}")


def _validate_parallel_inputs(
    instructions: Sequence[str],
    input_images: Sequence[str],
) -> tuple[list[str], list[str]]:
    if len(instructions) != len(input_images):
        raise ValueError(
            "instructions and input_images must have the same length "
            f"({len(instructions)} != {len(input_images)})"
        )
    normalized_instructions = [str(instruction) for instruction in instructions]
    normalized_images = [str(image_path) for image_path in input_images]
    if any(not instruction.strip() for instruction in normalized_instructions):
        raise ValueError("instructions must not contain empty values")
    if any(not image_path.strip() for image_path in normalized_images):
        raise ValueError("input_images must not contain empty values")
    return normalized_instructions, normalized_images


def _failure_result(
    phase: str,
    error: Any,
    raw_response: str = "",
) -> dict[str, Any]:
    message = f"{phase} Error: {error}"
    return {
        "response": message,
        "raw_response": raw_response,
        "output_image": "",
        "error": message,
    }


def _parse_vlm_record(
    result: Mapping[str, Any],
    instruction: str,
    guidance_key: Optional[str] = None,
) -> dict[str, Any]:
    raw_response = str(result.get("text") or "")
    if result.get("error"):
        return _failure_result("VLM", result["error"], raw_response)

    try:
        payload = extract_json(raw_response)
    except (TypeError, ValueError) as exc:
        fallback = raw_response.strip()
        parsed: dict[str, Any] = {
            "response": fallback,
            "raw_response": raw_response,
            "output_image": "",
            "parse_error": str(exc),
        }
        if guidance_key:
            parsed[guidance_key] = instruction
        return parsed

    output_text = payload.get("output_text", "")
    response = str(output_text).strip() if output_text is not None else ""
    parsed = {
        "response": response,
        "raw_response": raw_response,
        "output_image": "",
    }
    if guidance_key:
        guidance = payload.get(guidance_key, instruction)
        parsed[guidance_key] = str(guidance).strip() or instruction
    return parsed


def _resolve_vlm_config(
    config_path: Optional[str],
    vlm_config_path: Optional[str],
) -> str:
    return str(vlm_config_path or config_path or DEFAULT_CONFIG_PATH)


def _resolve_image_config(
    config_path: Optional[str],
    vlm_config_path: Optional[str],
    image_config_path: Optional[str],
) -> str:
    return str(image_config_path or config_path or vlm_config_path or DEFAULT_CONFIG_PATH)


async def _run_vlm(
    instructions: Sequence[str],
    image_paths: Sequence[str],
    vlm_model: str,
    prompt_template: str,
    config_path: str,
    concurrency: int,
    max_tokens: int,
) -> list[dict[str, Any]]:
    client = single_image_vlm(config_path=config_path, model_name=vlm_model)
    requests = [
        (
            prompt_template.format(instruction=instruction),
            image_path,
            None,
            max_tokens,
        )
        for instruction, image_path in zip(instructions, image_paths)
    ]
    try:
        return await client.generate_batch(requests, concurrency=concurrency)
    finally:
        await client.aclose()


async def VLM(
    instructions: Sequence[str],
    input_images: Sequence[str],
    vlm_model: str,
    config_path: str = DEFAULT_CONFIG_PATH,
    output_image_path: str = DEFAULT_OUTPUT_IMAGE_PATH,
    *,
    vlm_config_path: Optional[str] = None,
    concurrency: int = 5,
    max_tokens: int = 2048,
) -> list[dict[str, Any]]:
    """Run VQA-style VLM inference while preserving input order.

    output_image_path is accepted for a uniform public interface with the image
    workflows. It is not used by text-only VLM inference.
    """

    del output_image_path
    normalized_instructions, normalized_images = _validate_parallel_inputs(
        instructions, input_images
    )
    resolved_config = _resolve_vlm_config(config_path, vlm_config_path)
    raw_results = await _run_vlm(
        normalized_instructions,
        normalized_images,
        vlm_model,
        VLM_ONLY_PROMPT,
        resolved_config,
        concurrency,
        max_tokens,
    )
    return [
        _parse_vlm_record(result, instruction)
        for result, instruction in zip(raw_results, normalized_instructions)
    ]


async def VLM2Generate(
    instructions: Sequence[str],
    input_images: Sequence[str],
    vlm_model: str,
    generate_model: str,
    vlm_config_path: Optional[str] = None,
    image_config_path: Optional[str] = None,
    output_image_path: str = DEFAULT_OUTPUT_IMAGE_PATH,
    *,
    config_path: Optional[str] = None,
    vlm_concurrency: int = 5,
    image_concurrency: int = 3,
    max_tokens: int = 2048,
    image_size: Optional[str] = None,
    image_steps: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Generate text and target images using separate VLM and image configs."""

    normalized_instructions, normalized_images = _validate_parallel_inputs(
        instructions, input_images
    )
    resolved_vlm_config = _resolve_vlm_config(config_path, vlm_config_path)
    resolved_image_config = _resolve_image_config(
        config_path, vlm_config_path, image_config_path
    )
    raw_results = await _run_vlm(
        normalized_instructions,
        normalized_images,
        vlm_model,
        VLM2GENERATE_PROMPT,
        resolved_vlm_config,
        vlm_concurrency,
        max_tokens,
    )

    records = [
        _parse_vlm_record(result, instruction, "generate_prompt")
        for result, instruction in zip(raw_results, normalized_instructions)
    ]
    pending: list[tuple[int, dict[str, Any]]] = [
        (index, record)
        for index, record in enumerate(records)
        if "error" not in record
    ]
    if not pending:
        return records

    destination = Path(output_image_path)
    destination.mkdir(parents=True, exist_ok=True)
    image_request_options: dict[str, Any] = {}
    if image_size:
        image_request_options["size"] = str(image_size)
    if image_steps is not None:
        image_request_options["num_inference_steps"] = int(image_steps)
    requests: list[dict[str, Any]] = []
    for index, record in pending:
        requests.append(
            {
                "model_name": generate_model,
                "prompt": record["generate_prompt"],
                "input_image": normalized_images[index],
                "output_dir": str(destination),
                "file_prefix": f"generated_{index:06d}_{uuid.uuid4().hex[:8]}",
                "num_images": 1,
                **image_request_options,
            }
        )

    from api.get_generate_res import ImageGenerationAPI

    async with ImageGenerationAPI(config_path=resolved_image_config) as image_client:
        image_results = await image_client.generate_images_batch(
            requests,
            max_concurrent=max(1, int(image_concurrency)),
        )

    for (index, record), image_result in zip(pending, image_results):
        record.pop("generate_prompt", None)
        paths = image_result.get("paths") or []
        if image_result.get("success") and paths:
            record["output_image"] = str(paths[0])
        else:
            error = image_result.get("error") or "image generation returned no files"
            record["output_image"] = f"Generation Error: {error}"
            record["error"] = str(error)
    for _, record in pending[len(image_results) :]:
        record.pop("generate_prompt", None)
        record["output_image"] = "Generation Error: image client returned too few results"
        record["error"] = "image client returned too few results"
    return records


async def VLM2Edit(
    instructions: Sequence[str],
    input_images: Sequence[str],
    vlm_model: str,
    edit_model: str,
    vlm_config_path: Optional[str] = None,
    image_config_path: Optional[str] = None,
    output_image_path: str = DEFAULT_OUTPUT_IMAGE_PATH,
    *,
    config_path: Optional[str] = None,
    vlm_concurrency: int = 5,
    image_concurrency: int = 3,
    max_tokens: int = 2048,
    image_size: Optional[str] = None,
    image_steps: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Generate text and edited images using separate VLM and image configs."""

    normalized_instructions, normalized_images = _validate_parallel_inputs(
        instructions, input_images
    )
    resolved_vlm_config = _resolve_vlm_config(config_path, vlm_config_path)
    resolved_image_config = _resolve_image_config(
        config_path, vlm_config_path, image_config_path
    )
    raw_results = await _run_vlm(
        normalized_instructions,
        normalized_images,
        vlm_model,
        VLM2EDIT_PROMPT,
        resolved_vlm_config,
        vlm_concurrency,
        max_tokens,
    )

    records = [
        _parse_vlm_record(result, instruction, "edit_prompt")
        for result, instruction in zip(raw_results, normalized_instructions)
    ]
    pending: list[tuple[int, dict[str, Any]]] = [
        (index, record)
        for index, record in enumerate(records)
        if "error" not in record
    ]
    if not pending:
        return records

    destination = Path(output_image_path)
    destination.mkdir(parents=True, exist_ok=True)
    image_prompts = [
        (normalized_images[index], record["edit_prompt"]) for index, record in pending
    ]
    image_request_options: dict[str, Any] = {}
    if image_size:
        image_request_options["size"] = str(image_size)
    if image_steps is not None:
        image_request_options["num_inference_steps"] = int(image_steps)

    from api.get_edit_res import ImageEditAPI

    image_client = ImageEditAPI(
        config_path=resolved_image_config,
        model_name=edit_model,
    )
    async_method = getattr(image_client, "edit_images_batch_async", None)
    if callable(async_method):
        image_results = await async_method(
            image_prompts,
            save_dir=str(destination),
            max_concurrent=max(1, int(image_concurrency)),
            **image_request_options,
        )
    else:
        image_results = await asyncio.to_thread(
            image_client.edit_images_batch,
            image_prompts,
            save_dir=str(destination),
            max_concurrent=max(1, int(image_concurrency)),
            **image_request_options,
        )

    for (index, record), image_result in zip(pending, image_results):
        record.pop("edit_prompt", None)
        paths = image_result.get("image_paths") or []
        if not image_result.get("error") and paths:
            record["output_image"] = str(paths[0])
        else:
            error = image_result.get("error") or "image editing returned no files"
            record["output_image"] = f"Edit Error: {error}"
            record["error"] = str(error)
    for _, record in pending[len(image_results) :]:
        record.pop("edit_prompt", None)
        record["output_image"] = "Edit Error: image client returned too few results"
        record["error"] = "image client returned too few results"
    return records


__all__ = [
    "VLM",
    "VLM2Edit",
    "VLM2Generate",
    "VLM2EDIT_PROMPT",
    "VLM2Edit_PROMPT",
    "VLM2GENERATE_PROMPT",
    "VLM2Generate_PROMPT",
    "VLM_ONLY_PROMPT",
    "extract_json",
]
