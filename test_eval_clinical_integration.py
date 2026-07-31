#!/usr/bin/env python3
"""Integration checks for eval.py clinical metric extraction."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest

import eval as eval_module
from PIL import Image

from util.prompt import (
    vlm_holistic_judge_w_gt_prompt,
    vlm_holistic_judge_wo_gt_prompt,
)


class EvalClinicalIntegrationTest(unittest.TestCase):
    def test_build_arg_parser_accepts_new_clinical_eval_flags(self) -> None:
        parser = eval_module.build_arg_parser()
        args = parser.parse_args(
            [
                "--jsonl_path",
                "dummy.jsonl",
                "--judge_model",
                "instruct-biomedgpt-base",
                "--judge_backend",
                "api",
                "--enable_clinical_text_metrics",
            ]
        )
        self.assertEqual(args.judge_model, "instruct-biomedgpt-base")
        self.assertEqual(args.judge_backend, "api")
        self.assertTrue(args.enable_clinical_text_metrics)

    def test_metrics_from_record_extracts_new_clinical_and_judge_fields(self) -> None:
        metrics = eval_module.metrics_from_record(
            {
                "Clinical_Entity_P": 1.0,
                "Clinical_Entity_R": 0.5,
                "Clinical_Entity_F1": 2.0 / 3.0,
                "RadGraph_F1": 0.7,
                "radgraph_applicable": True,
                "Task_Accuracy": 1.0,
                "Text_EM": 0.0,
                "Text_F1": 0.8,
                "vlm_judge_w_gt_result": {
                    "anatomical_accuracy": {"score": 8},
                    "clinical_finding_accuracy": {"score": 9},
                    "instruction_compliance": {"score": 7},
                    "cross_modal_consistency": {"score": 8},
                    "hallucination_omission_control": {"score": 6},
                    "overall_score": 8,
                },
            },
            "vqa",
        )

        self.assertEqual(metrics["Clinical_Entity_P"], 1.0)
        self.assertEqual(metrics["Clinical_Entity_R"], 0.5)
        self.assertAlmostEqual(metrics["Clinical_Entity_F1"], 2.0 / 3.0)
        self.assertEqual(metrics["RadGraph_F1"], 0.7)
        self.assertEqual(metrics["Task_Accuracy"], 1.0)
        self.assertEqual(metrics["Text_EM"], 0.0)
        self.assertEqual(metrics["Text_F1"], 0.8)
        self.assertEqual(metrics["VLM_Anatomical_Accuracy_W_GT"], 8.0)
        self.assertEqual(metrics["VLM_Clinical_Finding_Accuracy_W_GT"], 9.0)
        self.assertEqual(metrics["VLM_Instruction_Compliance_W_GT"], 7.0)
        self.assertEqual(metrics["VLM_Cross_Modal_Consistency_W_GT"], 8.0)
        self.assertEqual(metrics["VLM_Hallucination_Omission_Control_W_GT"], 6.0)
        self.assertEqual(metrics["VLM_Overall_Score_W_GT"], 8.0)

    def test_aggregate_type_wise_results_reports_radgraph_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "typewise.jsonl")
            results = asyncio.run(
                eval_module.aggregate_type_wise_results(
                    data=[
                        {
                            "modality": "Radiograph",
                            "Clinical_Entity_F1": 1.0,
                            "RadGraph_F1": 0.8,
                            "radgraph_applicable": True,
                            "normalized_answer_text": "FINDINGS: clear lungs",
                        },
                        {
                            "modality": "Radiograph",
                            "Clinical_Entity_F1": 0.0,
                            "radgraph_applicable": False,
                            "normalized_answer_text": "A. H&E",
                        },
                    ],
                    type_key="modality",
                    task="vqa",
                    jsonl_path=path,
                )
            )
        self.assertIn("RadGraph_Coverage", results["Radiograph"])
        self.assertEqual(results["Radiograph"]["RadGraph_Coverage"], 0.5)

    def test_updated_judge_prompts_use_medical_dimensions(self) -> None:
        combined_w_gt = "\n".join(vlm_holistic_judge_w_gt_prompt).lower()
        combined_wo_gt = "\n".join(vlm_holistic_judge_wo_gt_prompt).lower()

        for phrase in [
            "anatomical accuracy",
            "clinical finding accuracy",
            "instruction compliance",
            "cross-modal consistency",
            "hallucination/omission control",
        ]:
            self.assertIn(phrase, combined_w_gt)
            self.assertIn(phrase, combined_wo_gt)


class BasicEvalClinicalPipelineTest(unittest.IsolatedAsyncioTestCase):
    async def test_image_edit_local_metrics_does_not_build_vlm_requests(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.png")
            reference_path = os.path.join(temp_dir, "reference.png")
            Image.new("RGB", (8, 8), color="white").save(output_path)
            Image.new("RGB", (8, 8), color="white").save(reference_path)
            jsonl_path = os.path.join(temp_dir, "image_edit_local.jsonl")
            intermediate_path = os.path.join(
                "eval_results", "image_edit_local_local_metrics.jsonl"
            )
            if os.path.exists(intermediate_path):
                os.remove(intermediate_path)

            data = [
                {
                    "category": "ImageEdit",
                    "sub-category": "contrast-enhancement",
                    "modality": "Radiograph",
                    "input_image": reference_path,
                    "ground_truth_image": reference_path,
                    "output_image": output_path,
                    "instruction": "Enhance the contrast.",
                    "sample_id": "unit:test:image-edit-local:1",
                }
            ]

            async def fake_image_metric(eval_images, ref_images, eval_metric):
                return [1.0 for _ in eval_images]

            original_image_metric = eval_module.batch_async_FR_IQA
            try:
                eval_module.batch_async_FR_IQA = fake_image_metric
                results = await eval_module.basic_eval(
                    data=data,
                    batch_size=1,
                    task="image_edit",
                    data_path=temp_dir,
                    jsonl_path=jsonl_path,
                    run_vlm_judge=False,
                )
            finally:
                eval_module.batch_async_FR_IQA = original_image_metric

            self.assertEqual(results["Average_LPIPS"], 1.0)
            with open(intermediate_path, "r", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["_local_metrics_complete"])

    async def test_basic_eval_writes_new_clinical_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "input.png")
            Image.new("RGB", (8, 8), color="white").save(image_path)
            jsonl_path = os.path.join(temp_dir, "clinical_eval.jsonl")
            intermediate_path = os.path.join(
                "eval_results", "clinical_eval_local_metrics.jsonl"
            )
            if os.path.exists(intermediate_path):
                os.remove(intermediate_path)

            data = [
                {
                    "category": "VQA",
                    "sub-category": "report-generation",
                    "modality": "Radiograph",
                    "in-out": "image-text_to_text",
                    "source": "unit-test",
                    "input_image": image_path,
                    "instruction": "Generate a clinical report for this chest X-ray.",
                    "choice": [],
                    "answer": {
                        "FINDINGS": {
                            "Lungs": "Clear.",
                            "Pleural_Effusion": "No pleural effusion.",
                        },
                        "IMPRESSION": {
                            "Summary": "No acute cardiopulmonary abnormality."
                        },
                    },
                    "paper_format": "VQA",
                    "paper_task": "report-generation",
                    "paper_subtask": None,
                    "observed_subtask": None,
                    "input_images": [image_path],
                    "ground_truth_images": [],
                    "sample_id": "unit:test:report:1",
                    "eval_adapter": {"input_strategy": "single"},
                    "response": "FINDINGS: Lungs are clear. No pleural effusion. IMPRESSION: No acute cardiopulmonary abnormality.",
                    "raw_response": "same",
                    "output_image": "",
                }
            ]

            async def fake_text_metric(eval_texts, ref_texts, eval_metric):
                return [1.0 for _ in eval_texts]

            original_text_metric = eval_module.batch_async_evaluate_text_quality
            try:
                eval_module.batch_async_evaluate_text_quality = fake_text_metric
                results = await eval_module.basic_eval(
                    data=data,
                    batch_size=1,
                    task="vqa",
                    data_path=temp_dir,
                    jsonl_path=jsonl_path,
                    run_vlm_judge=False,
                )
            finally:
                eval_module.batch_async_evaluate_text_quality = original_text_metric

            self.assertIn("Average_Clinical_Entity_F1", results)
            self.assertIn("Average_RadGraph_F1", results)
            self.assertIn("RadGraph_Coverage", results)
            self.assertIn("Average_Text_EM", results)
            self.assertIn("Average_Text_F1", results)

            with open(intermediate_path, "r", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(len(rows), 1)
            self.assertIn("Clinical_Entity_F1", rows[0])
            self.assertIn("RadGraph_F1", rows[0])
            self.assertIn("normalized_answer_text", rows[0])
            self.assertIn("_local_metrics_complete", rows[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
