#!/usr/bin/env python3
"""No-network integration tests for the MedGEN Table IV eval view."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

import eval as eval_module
import inference


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "MedGEN_TableIV"


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class TableIVIntegrationTest(unittest.TestCase):
    def test_adapter_manifest_and_counts(self) -> None:
        manifest = json.loads((DATASET / "adapter_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["validation"]["record_count"], 6623)
        self.assertEqual(
            manifest["validation"]["mission_counts"],
            {"vqa": 1100, "edit": 3872, "generate": 1651},
        )
        self.assertEqual(manifest["validation"]["paper_task_count"], 16)
        self.assertEqual(manifest["validation"]["missing_image_path_count"], 0)
        self.assertTrue((DATASET / "source").resolve().is_dir())

    def test_full_inference_inputs(self) -> None:
        cases = (
            ("vqa.jsonl", "vqa", 1100),
            ("edit.jsonl", "edit", 3872),
            ("gen.jsonl", "generate", 1651),
        )
        all_sample_ids = set()
        for filename, mission, expected in cases:
            path = DATASET / filename
            records = load_jsonl(path)
            args = SimpleNamespace(jsonl_path=str(path), mission=mission)
            summary = inference.validate_dataset_records(records, args)
            self.assertEqual(summary["records"], expected)
            self.assertEqual(summary["missing_images"], 0)
            prepared = inference.prepare_batch_data(records, {}, args)
            self.assertEqual(len(prepared), expected)
            sample_ids = {record["sample_id"] for record in records}
            self.assertEqual(len(sample_ids), expected)
            self.assertTrue(all_sample_ids.isdisjoint(sample_ids))
            all_sample_ids.update(sample_ids)
        self.assertEqual(len(all_sample_ids), 6623)

    def test_multiframe_vqa_contact_sheet(self) -> None:
        records = load_jsonl(DATASET / "smoke_vqa.jsonl")
        self.assertEqual(len(records), 4)
        for record in records:
            self.assertEqual(
                record["eval_adapter"]["input_strategy"], "labeled_contact_sheet"
            )
            self.assertGreaterEqual(len(record["input_images"]), 2)
            image_path = DATASET / record["input_image"]
            with Image.open(image_path) as image:
                image.verify()

    def test_eval_preflight_all_missions(self) -> None:
        cases = (
            ("smoke_eval_vqa.jsonl", "vqa", 4),
            ("smoke_eval_edit.jsonl", "image_edit", 6),
            ("smoke_eval_gen.jsonl", "multimodal_generation", 6),
        )
        for filename, task, expected in cases:
            records = load_jsonl(DATASET / filename)
            summary = eval_module.validate_eval_input(records, task, str(DATASET))
            self.assertEqual(summary["records"], expected)
            self.assertEqual(summary["missing_images"], 0)

    def test_malformed_multi_image_input_is_rejected(self) -> None:
        args = SimpleNamespace(jsonl_path=str(DATASET / "vqa.jsonl"))
        with self.assertRaisesRegex(ValueError, "contact sheet"):
            inference.prepare_batch_data(
                [{"instruction": "x", "input_image": ["a.jpg", "b.jpg"]}],
                {},
                args,
            )

    def test_type_wise_metric_extraction(self) -> None:
        metrics = eval_module.metrics_from_record(
            {
                "BLEU": 0.5,
                "BERT_Score": 0.9,
                "vlm_judge_w_gt_result": {
                    "content_accuracy": {"score": 8},
                    "overall_score": 9,
                },
            },
            "vqa",
        )
        self.assertEqual(metrics["BLEU"], 0.5)
        self.assertEqual(metrics["BERT_Score"], 0.9)
        self.assertEqual(metrics["VLM_Content_Accuracy_W_GT"], 8.0)
        self.assertEqual(metrics["VLM_Overall_Score_W_GT"], 9.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
