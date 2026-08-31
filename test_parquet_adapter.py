"""Regression coverage for the split MedGEN-Bench parquet release."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

import eval as eval_module
import inference
import prepare_medgen_data as adapter


def encoded_image(color: str) -> bytes:
    image = Image.new("RGB", (16, 16), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image.close()
    return buffer.getvalue()


def image_value(color: str, name: str) -> dict[str, object]:
    return {"bytes": encoded_image(color), "path": name}


def write_parquet(root: Path, config_name: str, row: dict[str, object]) -> None:
    directory = root / "parquet" / config_name
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([row]), directory / "train-00000-of-00001.parquet")


def load_one(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.loads(next(line for line in handle if line.strip()))


class ParquetAdapterTest(unittest.TestCase):
    def test_split_parquet_schema_is_converted_to_runner_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "raw"
            output = Path(temporary) / "MedGEN_data"
            write_parquet(
                root,
                "image_editing",
                {
                    "input_images": [image_value("red", "input.png")],
                    "output_images": [image_value("blue", "output.png")],
                    "instruction": "Adjust contrast.",
                    "task": "contrast-enhancement",
                    "subtask": "Contrast Enhancement",
                    "modality": "CT",
                },
            )
            write_parquet(
                root,
                "multimodal_generation",
                {
                    "input_images": [image_value("green", "input.png")],
                    "output_images": [image_value("yellow", "output.png")],
                    "instruction": "Generate a reconstruction.",
                    "answer": "A generated reconstruction.",
                    "task": "disease-prediction",
                    "subtask": "Early to Late",
                    "modality": "MRI",
                },
            )
            write_parquet(
                root,
                "vqa",
                {
                    "input_images": [
                        image_value("black", "first.png"),
                        image_value("white", "second.png"),
                    ],
                    "instruction": "Which option is correct?",
                    "answer": "A",
                    "task": "multiple-choice",
                    "subtask": None,
                    "modality": "X-ray",
                },
            )

            manifest = adapter.build(root, output, allow_partial=True)
            self.assertEqual(manifest["source_format"], "parquet")
            self.assertEqual(manifest["source_revision"], adapter.DEFAULT_DATASET_REVISION)
            self.assertEqual(manifest["validation"]["record_count"], 3)

            edit = load_one(output / "edit.jsonl")
            generation = load_one(output / "gen.jsonl")
            vqa = load_one(output / "vqa.jsonl")
            self.assertEqual(edit["category"], "ImageEdit")
            self.assertNotIn("answer", edit)
            self.assertIn("ground_truth_image", edit)
            self.assertEqual(generation["category"], "MMGeneration")
            self.assertEqual(generation["answer"], "A generated reconstruction.")
            self.assertEqual(vqa["category"], "VQA")
            self.assertEqual(vqa["answer"], "A")
            self.assertNotIn("ground_truth_image", vqa)
            self.assertIn("vqa_contact_sheets", str(vqa["input_image"]))

            for record, mission, task in (
                (edit, "edit", "image_edit"),
                (generation, "generate", "multimodal_generation"),
                (vqa, "vqa", "vqa"),
            ):
                jsonl_path = output / {"edit": "edit.jsonl", "generate": "gen.jsonl", "vqa": "vqa.jsonl"}[mission]
                args = SimpleNamespace(jsonl_path=str(jsonl_path), mission=mission)
                self.assertEqual(inference.validate_dataset_records([record], args)["missing_images"], 0)
                smoke_name = {
                    "edit": "smoke_eval_edit.jsonl",
                    "generate": "smoke_eval_gen.jsonl",
                    "vqa": "smoke_eval_vqa.jsonl",
                }[mission]
                eval_record = load_one(output / smoke_name)
                self.assertEqual(
                    eval_module.validate_eval_input([eval_record], task, str(output))["missing_images"],
                    0,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
