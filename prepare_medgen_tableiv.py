#!/usr/bin/env python3
"""Build an eval-compatible view of the organized MedGEN Table IV dataset.

The canonical dataset stores image fields as lists and keeps each paper task in
its own directory.  The current eval runner expects three JSONL files and one
image path string per record.  This adapter preserves the canonical data via a
relative symlink and creates labeled contact sheets for multi-image VQA cases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont, ImageOps


FORMAT_OUTPUTS = {
    "VQA": "vqa.jsonl",
    "ImageEditing": "edit.jsonl",
    "MultimodalGeneration": "gen.jsonl",
}

MISSION_NAMES = {
    "VQA": "vqa",
    "ImageEditing": "edit",
    "MultimodalGeneration": "generate",
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=here.parent / "medical-bench" / "MedGEN_Bench_TableIV_Organized",
        help="Canonical Table IV dataset produced by organize_medgen_table_iv.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "MedGEN_TableIV",
        help="Eval-compatible output directory (must not already exist)",
    )
    parser.add_argument(
        "--contact-sheet-cell-size",
        type=int,
        default=768,
        help="Maximum width and height of each VQA contact-sheet cell",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            yield line_number, record


def checked_image_refs(record: dict[str, Any], field: str) -> list[str]:
    values = record.get(field, [])
    if not isinstance(values, list):
        raise ValueError(f"{field} must be a list in the canonical dataset")
    refs: list[str] = []
    for value in values:
        ref = PurePosixPath(str(value))
        if ref.is_absolute() or ".." in ref.parts or not ref.parts:
            raise ValueError(f"Unsafe canonical image reference: {value!r}")
        refs.append(ref.as_posix())
    return refs


def eval_source_ref(paper_format: str, paper_task: str, ref: str) -> str:
    return PurePosixPath("source", paper_format, paper_task, ref).as_posix()


def load_font(size: int = 24) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def contact_sheet_key(paths: list[Path]) -> str:
    # Canonical filenames contain stable question/sample IDs and image indices.
    token = "\0".join(path.name for path in paths)
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:24]


def make_contact_sheet(paths: list[Path], destination: Path, cell_size: int) -> None:
    if len(paths) < 2:
        raise ValueError("Contact sheets are only required for multi-image samples")
    if cell_size < 128:
        raise ValueError("--contact-sheet-cell-size must be at least 128")

    prepared: list[Image.Image] = []
    try:
        for path in paths:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail((cell_size, cell_size), Image.Resampling.LANCZOS)
                prepared.append(image.copy())

        columns = min(3, math.ceil(math.sqrt(len(prepared))))
        rows = math.ceil(len(prepared) / columns)
        label_height = 44
        canvas = Image.new(
            "RGB",
            (columns * cell_size, rows * (cell_size + label_height)),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        font = load_font()
        for index, image in enumerate(prepared):
            row, column = divmod(index, columns)
            cell_x = column * cell_size
            cell_y = row * (cell_size + label_height)
            image_x = cell_x + (cell_size - image.width) // 2
            image_y = cell_y + label_height + (cell_size - image.height) // 2
            draw.rectangle(
                (cell_x, cell_y, cell_x + cell_size - 1, cell_y + label_height - 1),
                fill="#202020",
            )
            draw.text((cell_x + 14, cell_y + 8), f"Image {index + 1}", fill="white", font=font)
            canvas.paste(image, (image_x, image_y))

        destination.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(destination, format="JPEG", quality=95, subsampling=0)
    finally:
        for image in prepared:
            image.close()


def adapt_record(
    record: dict[str, Any],
    paper_format: str,
    paper_task: str,
    source_data: Path,
    source_line: int,
    staging: Path,
    contact_sheet_cell_size: int,
) -> dict[str, Any]:
    adapted = dict(record)
    input_refs = checked_image_refs(record, "input_image")
    ground_truth_refs = checked_image_refs(record, "ground_truth_image")
    if not input_refs:
        raise ValueError(f"No input image at {source_data}:{source_line}")

    eval_inputs = [eval_source_ref(paper_format, paper_task, ref) for ref in input_refs]
    eval_ground_truths = [
        eval_source_ref(paper_format, paper_task, ref) for ref in ground_truth_refs
    ]
    input_strategy = "single"
    if len(input_refs) == 1:
        eval_input = eval_inputs[0]
    elif paper_format == "VQA":
        source_paths = [source_data.parent / ref for ref in input_refs]
        for path in source_paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        key = contact_sheet_key(source_paths)
        relative_sheet = PurePosixPath("images", "vqa_contact_sheets", f"{key}.jpg")
        sheet_path = staging.joinpath(*relative_sheet.parts)
        if not sheet_path.exists():
            make_contact_sheet(source_paths, sheet_path, contact_sheet_cell_size)
        eval_input = relative_sheet.as_posix()
        input_strategy = "labeled_contact_sheet"
    else:
        raise ValueError(
            f"{paper_format}/{paper_task} unexpectedly has {len(input_refs)} input images"
        )

    if len(eval_ground_truths) > 1:
        raise ValueError(
            f"{paper_format}/{paper_task} unexpectedly has multiple ground-truth images"
        )

    adapted["input_images"] = eval_inputs
    adapted["ground_truth_images"] = eval_ground_truths
    adapted["input_image"] = eval_input
    if eval_ground_truths:
        adapted["ground_truth_image"] = eval_ground_truths[0]
    else:
        # Omitting the singular field prevents the current image evaluator from
        # treating a VQA empty list as a filesystem path.
        adapted.pop("ground_truth_image", None)

    sample_id = f"tableiv:{paper_format}:{paper_task}:{source_line:06d}"
    adapted["sample_id"] = sample_id
    adapted["eval_adapter"] = {
        "source_data_jsonl": source_data.relative_to(source_data.parents[2]).as_posix(),
        "source_line": source_line,
        "input_strategy": input_strategy,
        "original_input_image": input_refs,
        "original_ground_truth_image": ground_truth_refs,
    }
    return adapted


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count


def make_eval_smoke_record(record: dict[str, Any], paper_format: str) -> dict[str, Any]:
    """Create an explicitly labeled oracle fixture for eval.py preflight only."""
    fixture = dict(record)
    fixture["response"] = str(record.get("answer", ""))
    fixture["raw_response"] = "LOCAL_ORACLE_SMOKE_FIXTURE"
    fixture["output_image"] = (
        str(record.get("ground_truth_image", "")) if paper_format != "VQA" else ""
    )
    fixture["eval_smoke_fixture"] = True
    return fixture


def validate_eval_view(staging: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    total = 0
    errors: list[str] = []
    mission_counts: dict[str, int] = {}
    task_counts: Counter[str] = Counter()
    referenced_paths: set[Path] = set()

    for paper_format, output_name in FORMAT_OUTPUTS.items():
        path = staging / output_name
        count = 0
        for line_number, record in read_jsonl(path):
            count += 1
            total += 1
            if record.get("paper_format") != paper_format:
                errors.append(f"{path}:{line_number}: wrong paper_format")
            paper_task = str(record.get("paper_task"))
            task_counts[f"{paper_format}/{paper_task}"] += 1
            for field in ("input_image", "ground_truth_image"):
                ref = record.get(field)
                if not ref:
                    continue
                if not isinstance(ref, str):
                    errors.append(f"{path}:{line_number}: {field} is not a string")
                    continue
                relative = PurePosixPath(ref)
                if relative.is_absolute() or ".." in relative.parts:
                    errors.append(f"{path}:{line_number}: unsafe {field}={ref!r}")
                    continue
                image = staging.joinpath(*relative.parts)
                if not image.is_file():
                    errors.append(f"{path}:{line_number}: missing {field}={ref!r}")
                else:
                    referenced_paths.add(image.resolve())
        mission_counts[MISSION_NAMES[paper_format]] = count

    expected_total = manifest["validation"]["actual_record_count"]
    if total != expected_total:
        errors.append(f"Total records {total}, expected {expected_total}")
    for key, config in manifest["tasks"].items():
        expected = config["record_count"]
        if task_counts[key] != expected:
            errors.append(f"{key}: {task_counts[key]} records, expected {expected}")
    if errors:
        raise ValueError("Eval view validation failed:\n" + "\n".join(errors[:50]))

    contact_sheets = list((staging / "images" / "vqa_contact_sheets").glob("*.jpg"))
    return {
        "record_count": total,
        "mission_counts": mission_counts,
        "paper_task_count": len(task_counts),
        "resolved_image_path_count": len(referenced_paths),
        "vqa_contact_sheet_count": len(contact_sheets),
        "missing_image_path_count": 0,
    }


def build_readme(source_root: Path, summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# MedGEN Table IV eval view",
            "",
            "This directory is generated by `prepare_medgen_tableiv.py` for the existing `inference.py` and `eval.py` interfaces.",
            "",
            f"Canonical source: `{source_root}`",
            "",
            "- `vqa.jsonl`: 1,100 records (`--mission vqa`)",
            "- `edit.jsonl`: 3,872 records (`--mission edit`)",
            "- `gen.jsonl`: 1,651 records (`--mission generate`)",
            "- `smoke_*.jsonl`: one record per paper task for local preflight checks",
            "- `smoke_eval_*.jsonl`: oracle fixtures used only to test `eval.py --validate-only`",
            f"- VQA contact sheets: {summary['vqa_contact_sheet_count']}",
            "",
            "The `source` symlink points to the canonical organized dataset; images are not duplicated. Multi-image VQA samples use labeled contact sheets while retaining all original paths in `input_images` and `eval_adapter`.",
            "",
            "Run local, no-API checks from the eval repository:",
            "",
            "```bash",
            "python inference.py --jsonl_path ./MedGEN_TableIV/smoke_vqa.jsonl --mission vqa --validate-only",
            "python inference.py --jsonl_path ./MedGEN_TableIV/smoke_edit.jsonl --mission edit --validate-only",
            "python inference.py --jsonl_path ./MedGEN_TableIV/smoke_gen.jsonl --mission generate --validate-only",
            "python eval.py --data_path ./MedGEN_TableIV --jsonl_path ./MedGEN_TableIV/smoke_eval_vqa.jsonl --task vqa --validate-only",
            "python eval.py --data_path ./MedGEN_TableIV --jsonl_path ./MedGEN_TableIV/smoke_eval_edit.jsonl --task image_edit --validate-only",
            "python eval.py --data_path ./MedGEN_TableIV --jsonl_path ./MedGEN_TableIV/smoke_eval_gen.jsonl --task multimodal_generation --validate-only",
            "```",
            "",
            "For a real run, provide model names and keep results separate from the legacy release:",
            "",
            "```bash",
            "python inference.py --jsonl_path ./MedGEN_TableIV/vqa.jsonl --mission vqa \\",
            "  --vlm_model qwen3-vl-235b-a22b-instruct \\",
            "  --output_jsonl_dir ./inference_jsonl/tableiv --concurrency 8",
            "python eval.py --data_path ./MedGEN_TableIV \\",
            "  --jsonl_path ./inference_jsonl/tableiv/qwen3-vl-235b-a22b-instruct_vqa.jsonl \\",
            "  --task vqa --mission basic_eval --batch_size 8",
            "```",
            "",
        ]
    )


def build(dataset_root: Path, output: Path, contact_sheet_cell_size: int) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    output = output.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Canonical dataset not found: {dataset_root}")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    source_manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        relative_source = os.path.relpath(dataset_root, staging)
        (staging / "source").symlink_to(relative_source, target_is_directory=True)

        all_records: dict[str, list[dict[str, Any]]] = {
            paper_format: [] for paper_format in FORMAT_OUTPUTS
        }
        smoke_records: dict[str, list[dict[str, Any]]] = {
            paper_format: [] for paper_format in FORMAT_OUTPUTS
        }

        for paper_format, tasks in manifest["table_iv"].items():
            for paper_task in tasks:
                source_data = dataset_root / paper_format / paper_task / "data.jsonl"
                converted: list[dict[str, Any]] = []
                for source_line, record in read_jsonl(source_data):
                    converted.append(
                        adapt_record(
                            record,
                            paper_format,
                            paper_task,
                            source_data,
                            source_line,
                            staging,
                            contact_sheet_cell_size,
                        )
                    )
                all_records[paper_format].extend(converted)
                if paper_format == "VQA":
                    smoke_record = next(
                        (
                            record
                            for record in converted
                            if record["eval_adapter"]["input_strategy"] == "labeled_contact_sheet"
                        ),
                        converted[0],
                    )
                else:
                    smoke_record = converted[0]
                smoke_records[paper_format].append(smoke_record)

        for paper_format, output_name in FORMAT_OUTPUTS.items():
            write_jsonl(staging / output_name, all_records[paper_format])
            write_jsonl(staging / f"smoke_{output_name}", smoke_records[paper_format])
            write_jsonl(
                staging / f"smoke_eval_{output_name}",
                (
                    make_eval_smoke_record(record, paper_format)
                    for record in smoke_records[paper_format]
                ),
            )

        validation = validate_eval_view(staging, manifest)
        adapter_manifest = {
            "name": "MedGEN Table IV eval view",
            "canonical_dataset": str(dataset_root),
            "canonical_manifest_sha256": hashlib.sha256(
                source_manifest_path.read_bytes()
            ).hexdigest(),
            "format_outputs": FORMAT_OUTPUTS,
            "validation": validation,
        }
        (staging / "adapter_manifest.json").write_text(
            json.dumps(adapter_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging / "README.md").write_text(
            build_readme(dataset_root, validation), encoding="utf-8"
        )
        staging.chmod(0o755)
        os.replace(staging, output)
        return adapter_manifest
    except Exception:
        print(f"Incomplete staging directory retained for inspection: {staging}")
        raise


def main() -> int:
    args = parse_args()
    result = build(args.dataset_root, args.output, args.contact_sheet_cell_size)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
