"""Offline smoke tests for the public MedGEN-Bench release surface."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image

from api.get_vlm_res import add_text_to_image
from api.qwen_image_edit_server import ServerSettings, build_app, parse_size
from eval import build_arg_parser, build_vlm_judge_client, validate_eval_input
from inference import parse_args, validate_dataset_records
from util.format_parser import extract_json


def test_extract_json_handles_fences_prose_and_nested_objects() -> None:
    response = "Analysis complete. ```json\n{'overall_score': 8, 'detail': {'score': 7}}\n```"
    assert extract_json(response) == {"overall_score": 8, "detail": {"score": 7}}


def test_local_image_service_exposes_health_and_model_routes(tmp_path: Path) -> None:
    settings = ServerSettings(
        model="local/Qwen-Image-Edit",
        output_dir=tmp_path / "outputs",
        device="cpu",
        dtype="float32",
        local_files_only=True,
        cpu_offload_mode="none",
        default_steps=1,
        default_cfg_scale=1.0,
    )
    with TestClient(build_app(settings)) as client:
        health = client.get("/health")
        models = client.get("/v1/models")
        invalid_steps = client.post(
            "/v1/images/edits",
            data={"prompt": "test", "num_inference_steps": "1"},
            files={"image": ("input.png", b"not-an-image", "image/png")},
        )
    assert health.status_code == 200
    assert health.json()["loaded"] is False
    assert models.status_code == 200
    assert models.json()["data"][0]["id"] == "local/Qwen-Image-Edit"
    assert invalid_steps.status_code == 422
    assert "2..100" in invalid_steps.json()["detail"]


def test_local_judge_endpoint_override_beats_inference_endpoint(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "base_url: https://example.invalid/v1\napi_key: placeholder\nmodel_name: external-model\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MEDGEN_VLM_BASE_URL", "http://127.0.0.1:8000/v1")
    client = build_vlm_judge_client(
        True,
        "lingshu-medical-mllm/Lingshu-32B",
        "api",
        str(config),
        "http://127.0.0.1:8002/v1",
        "EMPTY",
    )
    try:
        assert client.base_url == "http://127.0.0.1:8002/v1"
        assert client.model_name == "lingshu-medical-mllm/Lingshu-32B"
        assert client.api_key == "EMPTY"
    finally:
        asyncio.run(client.aclose())


def test_judge_cli_accepts_documented_hyphenated_options() -> None:
    args = build_arg_parser().parse_args(
        [
            "--jsonl_path",
            "result.jsonl",
            "--judge-model",
            "lingshu-medical-mllm/Lingshu-32B",
            "--judge-config",
            "judge.yaml",
            "--judge-base-url",
            "http://127.0.0.1:8002/v1",
            "--judge-api-key",
            "EMPTY",
        ]
    )
    assert args.judge_model == "lingshu-medical-mllm/Lingshu-32B"
    assert args.judge_config == "judge.yaml"
    assert args.judge_base_url == "http://127.0.0.1:8002/v1"
    assert args.judge_api_key == "EMPTY"


def test_inference_cli_accepts_local_image_smoke_overrides(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inference.py",
            "--jsonl-path",
            "records.jsonl",
            "--mission",
            "edit",
            "--image-size",
            "256x256",
            "--image-steps",
            "4",
        ],
    )
    args = parse_args()
    assert args.image_size == "256x256"
    assert args.image_steps == 4


def test_prepared_input_and_eval_validation_are_offline(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    Image.new("RGB", (8, 8), color="white").save(image_path)
    args = SimpleNamespace(jsonl_path=str(tmp_path / "records.jsonl"))
    records = [
        {
            "sample_id": "sample-1",
            "category": "VQA",
            "paper_task": "multiple-choice",
            "instruction": "Which option is correct?",
            "input_image": "input.png",
            "answer": "A",
            "response": "A",
        }
    ]
    prepared = validate_dataset_records(records, args)
    evaluated = validate_eval_input(records, "vqa", str(tmp_path))
    assert prepared["records"] == 1
    assert evaluated["resolved_image_count"] == 1
    assert parse_size("128x256") == (128, 256)


def test_judge_image_cap_reduces_large_inputs_without_losing_labels(tmp_path: Path) -> None:
    source = tmp_path / "large.png"
    Image.new("RGB", (1536, 1024), color="white").save(source)
    labeled = add_text_to_image(source, "Input", max_side=512)
    try:
        assert labeled.width == 512
        assert labeled.height > round(1024 * 512 / 1536)
    finally:
        labeled.close()
