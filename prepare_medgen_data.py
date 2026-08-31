"""Convert the current MedGEN-Bench parquet release into evaluation JSONL.

The Hugging Face dataset has three parquet configurations with embedded image
bytes: ``image_editing``, ``multimodal_generation``, and ``vqa``. This adapter
materializes those images under ``MedGEN_data/images`` and writes the stable
record fields consumed by ``inference.py`` and ``eval.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping

from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - installation validation covers this
    raise RuntimeError("Parquet support requires pyarrow; install requirements.txt") from exc


CONFIGS: dict[str, dict[str, Any]] = {
    "image_editing": {
        "output": "edit.jsonl",
        "category": "ImageEdit",
        "paper_format": "ImageEditing",
        "mission": "edit",
        "expected_count": 3872,
        "requires_answer": False,
        "requires_output_images": True,
    },
    "multimodal_generation": {
        "output": "gen.jsonl",
        "category": "MMGeneration",
        "paper_format": "MultimodalGeneration",
        "mission": "generate",
        "expected_count": 1651,
        "requires_answer": True,
        "requires_output_images": True,
    },
    "vqa": {
        "output": "vqa.jsonl",
        "category": "VQA",
        "paper_format": "VQA",
        "mission": "vqa",
        "expected_count": 1100,
        "requires_answer": True,
        "requires_output_images": False,
    },
}
DEFAULT_DATASET_REVISION = "cee5e7ae410f7c5be12d5fa55464afb094c099b7"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=root / "MedGEN_raw",
        help="Directory created by download_medgen_dataset.sh.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "MedGEN_data",
        help="Prepared data directory. It must not already exist.",
    )
    parser.add_argument(
        "--contact-sheet-cell-size",
        type=int,
        default=768,
        help="Maximum width and height of each multi-image VQA cell.",
    )
    parser.add_argument(
        "--dataset-revision",
        default=os.environ.get("MEDGEN_DATASET_REVISION", DEFAULT_DATASET_REVISION),
        help="Hugging Face revision used to obtain the parquet data.",
    )
    parser.add_argument(
        "--max-samples-per-config",
        type=int,
        default=None,
        help="Optional development limit for each parquet configuration.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow partial shards or --max-samples-per-config during development.",
    )
    return parser.parse_args()


def parquet_files(dataset_root: Path, config_name: str) -> list[Path]:
    files = sorted((dataset_root / "parquet" / config_name).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No parquet shards found for {config_name!r} under {dataset_root / 'parquet'}. "
            "Run download_medgen_dataset.sh first."
        )
    return files


def read_rows(files: Iterable[Path]) -> Iterator[tuple[Path, int, dict[str, Any]]]:
    for shard in files:
        parquet = pq.ParquetFile(shard)
        row_index = 0
        for batch in parquet.iter_batches(batch_size=64):
            for row in batch.to_pylist():
                if not isinstance(row, dict):
                    raise ValueError(f"Parquet row is not an object: {shard}:{row_index}")
                yield shard, row_index, row
                row_index += 1


def image_payload(value: Any, dataset_root: Path) -> tuple[bytes, str]:
    """Return image bytes and a source name from an Arrow image struct."""
    if isinstance(value, Mapping):
        payload = value.get("bytes")
        source_name = str(value.get("path") or "")
    elif isinstance(value, (bytes, bytearray, memoryview)):
        payload = value
        source_name = ""
    else:
        raise TypeError(f"Unsupported parquet image value: {type(value).__name__}")

    if payload is not None:
        return bytes(payload), source_name
    if not source_name:
        raise ValueError("Parquet image has neither bytes nor path")
    source_path = PurePosixPath(source_name)
    if source_path.is_absolute() or ".." in source_path.parts:
        raise ValueError(f"Unsafe parquet image path: {source_name!r}")
    local_path = dataset_root.joinpath(*source_path.parts)
    if not local_path.is_file():
        raise FileNotFoundError(f"Image bytes are absent and source is unavailable: {local_path}")
    return local_path.read_bytes(), source_name


def image_extension(payload: bytes, source_name: str) -> str:
    suffix = Path(source_name).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
        return ".jpg" if suffix == ".jpeg" else suffix
    try:
        with Image.open(io.BytesIO(payload)) as image:
            detected = (image.format or "PNG").lower()
    except Exception as exc:
        raise ValueError("Parquet image bytes are unreadable") from exc
    return {"jpeg": ".jpg", "tiff": ".tiff"}.get(detected, f".{detected}")


def materialize_image(value: Any, dataset_root: Path, staging: Path) -> str:
    payload, source_name = image_payload(value, dataset_root)
    digest = hashlib.sha256(payload).hexdigest()
    suffix = image_extension(payload, source_name)
    relative = PurePosixPath("images", "assets", digest[:2], f"{digest}{suffix}")
    destination = staging.joinpath(*relative.parts)
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        try:
            with Image.open(destination) as image:
                image.verify()
        except Exception as exc:
            raise ValueError(f"Materialized image is unreadable: {destination}") from exc
    return relative.as_posix()


def materialize_images(value: Any, dataset_root: Path, staging: Path) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("Image column must be a non-empty list")
    return [materialize_image(item, dataset_root, staging) for item in value]


def load_font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def make_contact_sheet(image_refs: list[str], staging: Path, cell_size: int) -> str:
    token = "\0".join(image_refs)
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:24]
    relative = PurePosixPath("images", "vqa_contact_sheets", f"{digest}.jpg")
    destination = staging.joinpath(*relative.parts)
    if destination.exists():
        return relative.as_posix()

    images: list[Image.Image] = []
    try:
        for image_ref in image_refs:
            with Image.open(staging / image_ref) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail((cell_size, cell_size), Image.Resampling.LANCZOS)
                images.append(image.copy())
        columns = min(3, math.ceil(math.sqrt(len(images))))
        rows = math.ceil(len(images) / columns)
        label_height = 40
        canvas = Image.new("RGB", (columns * cell_size, rows * (cell_size + label_height)), "white")
        draw = ImageDraw.Draw(canvas)
        font = load_font()
        for index, image in enumerate(images):
            row, column = divmod(index, columns)
            cell_x = column * cell_size
            cell_y = row * (cell_size + label_height)
            draw.rectangle(
                (cell_x, cell_y, cell_x + cell_size - 1, cell_y + label_height - 1),
                fill="#202020",
            )
            draw.text((cell_x + 12, cell_y + 10), f"Image {index + 1}", fill="white", font=font)
            canvas.paste(
                image,
                (
                    cell_x + (cell_size - image.width) // 2,
                    cell_y + label_height + (cell_size - image.height) // 2,
                ),
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(destination, format="JPEG", quality=95, subsampling=0)
    finally:
        for image in images:
            image.close()
    return relative.as_posix()


def normalized_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def adapt_row(
    row: Mapping[str, Any],
    config_name: str,
    shard: Path,
    row_index: int,
    dataset_root: Path,
    staging: Path,
    contact_sheet_cell_size: int,
) -> dict[str, Any]:
    spec = CONFIGS[config_name]
    inputs = materialize_images(row.get("input_images"), dataset_root, staging)
    outputs: list[str] = []
    if spec["requires_output_images"]:
        outputs = materialize_images(row.get("output_images"), dataset_root, staging)
    input_image = (
        inputs[0]
        if len(inputs) == 1
        else make_contact_sheet(inputs, staging, contact_sheet_cell_size)
    )
    task = normalized_text(row.get("task"), "task")
    subtask = row.get("subtask")
    if subtask is not None and not isinstance(subtask, str):
        raise ValueError("subtask must be a string or null")

    record: dict[str, Any] = {
        "sample_id": f"medgen:{config_name}:{shard.stem}:{row_index:06d}",
        "category": spec["category"],
        "paper_format": spec["paper_format"],
        "paper_task": task,
        "paper_subtask": subtask.strip() if isinstance(subtask, str) and subtask.strip() else None,
        "sub-category": task,
        "modality": normalized_text(row.get("modality"), "modality"),
        "instruction": normalized_text(row.get("instruction"), "instruction"),
        "input_images": inputs,
        "input_image": input_image,
        "dataset_source": {
            "config": config_name,
            "shard": shard.name,
            "row_index": row_index,
        },
    }
    if outputs:
        record["ground_truth_images"] = outputs
        record["ground_truth_image"] = outputs[0]
    if spec["requires_answer"]:
        record["answer"] = normalized_text(row.get("answer"), "answer")
    return record


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count


def smoke_record(record: Mapping[str, Any], config_name: str) -> dict[str, Any]:
    fixture = dict(record)
    fixture["response"] = str(record.get("answer") or "")
    fixture["output_image"] = str(record.get("ground_truth_image") or "")
    fixture["raw_response"] = "LOCAL_ORACLE_SMOKE_FIXTURE"
    fixture["eval_smoke_fixture"] = True
    return fixture


def validate_records(records: Mapping[str, list[dict[str, Any]]], staging: Path, allow_partial: bool) -> dict[str, Any]:
    errors: list[str] = []
    counts: dict[str, int] = {}
    task_counts: Counter[str] = Counter()
    image_paths: set[str] = set()
    for config_name, items in records.items():
        spec = CONFIGS[config_name]
        counts[spec["mission"]] = len(items)
        if not allow_partial and len(items) != spec["expected_count"]:
            errors.append(f"{config_name}: {len(items)} rows, expected {spec['expected_count']}")
        for record in items:
            if record.get("category") != spec["category"]:
                errors.append(f"{config_name}: invalid category")
            if spec["requires_answer"] != ("answer" in record):
                errors.append(f"{config_name}: answer field does not match schema")
            if spec["requires_output_images"] != ("ground_truth_image" in record):
                errors.append(f"{config_name}: output image field does not match schema")
            task_counts[str(record.get("paper_task"))] += 1
            image_fields = {
                "input_image": [record.get("input_image")],
                "input_images": record.get("input_images", []),
                "ground_truth_image": [record.get("ground_truth_image")],
                "ground_truth_images": record.get("ground_truth_images", []),
            }
            for field, values in image_fields.items():
                if not isinstance(values, list):
                    errors.append(f"{config_name}: {field} is not a list")
                    continue
                for value in values:
                    if not value:
                        continue
                    if not isinstance(value, str):
                        errors.append(f"{config_name}: {field} contains a non-string")
                        continue
                    safe_path = PurePosixPath(value)
                    if safe_path.is_absolute() or ".." in safe_path.parts:
                        errors.append(f"{config_name}: unsafe {field}={value!r}")
                        continue
                    target = staging.joinpath(*safe_path.parts)
                    if not target.is_file():
                        errors.append(f"{config_name}: missing {field}={value!r}")
                    else:
                        image_paths.add(value)
    if errors:
        raise ValueError("Prepared data validation failed:\n" + "\n".join(errors[:50]))
    return {
        "record_count": sum(counts.values()),
        "mission_counts": counts,
        "paper_task_count": len(task_counts),
        "materialized_image_count": len(image_paths),
        "missing_image_path_count": 0,
    }


def build_readme(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# MedGEN data",
            "",
            "Generated from the parquet configurations in `MedGEN_raw/parquet`.",
            "",
            "- `vqa.jsonl`: VQA records with input images and answers.",
            "- `edit.jsonl`: image-editing records with input and reference output images.",
            "- `gen.jsonl`: multimodal-generation records with input/output images and answers.",
            "- `images/`: materialized parquet image assets and VQA contact sheets.",
            "",
            f"Records: {summary['record_count']}",
            f"Tasks: {summary['paper_task_count']}",
            "",
        ]
    )


def build(
    dataset_root: Path,
    output: Path,
    contact_sheet_cell_size: int = 768,
    max_samples_per_config: int | None = None,
    allow_partial: bool = False,
    dataset_revision: str = DEFAULT_DATASET_REVISION,
) -> dict[str, Any]:
    if max_samples_per_config is not None and max_samples_per_config < 1:
        raise ValueError("--max-samples-per-config must be positive")
    if contact_sheet_cell_size < 128:
        raise ValueError("--contact-sheet-cell-size must be at least 128")
    if max_samples_per_config is not None and not allow_partial:
        raise ValueError("--max-samples-per-config requires --allow-partial")
    dataset_root = dataset_root.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_root}")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        records: dict[str, list[dict[str, Any]]] = {name: [] for name in CONFIGS}
        smoke: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in CONFIGS}
        for config_name in CONFIGS:
            for shard, row_index, row in read_rows(parquet_files(dataset_root, config_name)):
                record = adapt_row(
                    row,
                    config_name,
                    shard,
                    row_index,
                    dataset_root,
                    staging,
                    contact_sheet_cell_size,
                )
                records[config_name].append(record)
                smoke[config_name].setdefault(record["paper_task"], record)
                if max_samples_per_config and len(records[config_name]) >= max_samples_per_config:
                    break

        summary = validate_records(records, staging, allow_partial)
        for config_name, spec in CONFIGS.items():
            output_name = spec["output"]
            write_jsonl(staging / output_name, records[config_name])
            selected_smoke = list(smoke[config_name].values())
            write_jsonl(staging / f"smoke_{output_name}", selected_smoke)
            write_jsonl(
                staging / f"smoke_eval_{output_name}",
                (smoke_record(record, config_name) for record in selected_smoke),
            )

        manifest = {
            "name": "MedGEN evaluation data",
            "source_format": "parquet",
            "source_revision": dataset_revision,
            "source_configs": list(CONFIGS),
            "validation": summary,
        }
        (staging / "adapter_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "README.md").write_text(build_readme(summary), encoding="utf-8")
        os.replace(staging, output)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    args = parse_args()
    result = build(
        args.dataset_root,
        args.output,
        contact_sheet_cell_size=args.contact_sheet_cell_size,
        max_samples_per_config=args.max_samples_per_config,
        allow_partial=args.allow_partial,
        dataset_revision=args.dataset_revision,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
