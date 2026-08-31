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
    build_vlm_judge_prompt,
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
                "--mission",
                "stats",
                "--judge_model",
                "instruct-biomedgpt-base",
                "--judge_config",
                "./config.yaml",
                "--judge_backend",
                "api",
                "--enable_clinical_text_metrics",
                "--disable_radgraph",
            ]
        )
        self.assertEqual(args.judge_model, "instruct-biomedgpt-base")
        self.assertEqual(args.judge_config, "./config.yaml")
        self.assertEqual(args.judge_backend, "api")
        self.assertTrue(args.enable_clinical_text_metrics)
        self.assertTrue(args.disable_radgraph)

    def test_main_table_text_bundle_can_skip_radgraph_but_keep_em(self) -> None:
        original = eval_module.compute_radgraph_f1_batch
        try:
            eval_module.compute_radgraph_f1_batch = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("RadGraph must not run")
            )
            bundles = eval_module.compute_text_metric_bundles(
                [
                    {
                        "paper_task": "blank-filling",
                        "sub-category": "blank-filling",
                        "response": "ischemic stroke lesion",
                        "answer": "ischemic stroke lesion",
                        "choice": [],
                    }
                ],
                include_radgraph=False,
            )
        finally:
            eval_module.compute_radgraph_f1_batch = original

        self.assertEqual(bundles[0]["Text_EM"], 1.0)
        self.assertFalse(bundles[0]["radgraph_applicable"])
        self.assertIsNone(bundles[0]["RadGraph_F1"])

    def test_build_vlm_judge_client_passes_judge_config(self) -> None:
        calls = []

        class FakeClient:
            def __init__(self, config_path, model_name):
                calls.append({"config_path": config_path, "model_name": model_name})

        original = eval_module.double_image_vlm
        try:
            eval_module.double_image_vlm = FakeClient
            eval_module.build_vlm_judge_client(
                run_vlm_judge=True,
                judge_model="MedVision-V0-7B",
                judge_backend="api",
                judge_config="./config.yaml",
            )
            eval_module.build_vlm_judge_client(
                run_vlm_judge=True,
                judge_model="",
                judge_backend="api",
                judge_config="./config.yaml",
            )
            self.assertIsNone(
                eval_module.build_vlm_judge_client(
                    run_vlm_judge=False,
                    judge_model="MedVision-V0-7B",
                    judge_backend="api",
                    judge_config="./config.yaml",
                )
            )
        finally:
            eval_module.double_image_vlm = original

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["config_path"], "./config.yaml")
        self.assertEqual(calls[0]["model_name"], "MedVision-V0-7B")
        # An empty judge_model lets double_image_vlm fall back to model_name in the config.
        self.assertEqual(calls[1]["config_path"], "./config.yaml")
        self.assertEqual(calls[1]["model_name"], "")

    def test_metric_thresholds_fixed_for_ssim_and_extended(self) -> None:
        self.assertIn("MedImageInsight_Similarity", eval_module.METRIC_THRESHOLDS)
        for metric_name in ["MedImageInsight_Similarity", "SSIM", "VLM_Overall_Score_W_GT"]:
            self.assertIn(metric_name, eval_module.METRIC_THRESHOLDS)
        for removed in [
            "Entity_Hallucination_Rate",
            "Entity_Omission_Rate",
            "Entity_Factual_Precision",
        ]:
            self.assertNotIn(removed, eval_module.METRIC_THRESHOLDS)

    def test_task_specific_judge_prompt_builder(self) -> None:
        prompt = build_vlm_judge_prompt("vqa", with_gt=True)
        combined = "\n".join(prompt).lower()
        self.assertIn("task-specific checklist", combined)
        self.assertIn("answer fidelity", combined)
        for phrase in [
            "anatomical accuracy",
            "clinical finding accuracy",
            "instruction compliance",
            "cross-modal consistency",
            "hallucination/omission control",
        ]:
            self.assertIn(phrase, combined)
        self.assertIn("microcalcifications", combined)
        edit_prompt = "\n".join(build_vlm_judge_prompt("image_edit", with_gt=False)).lower()
        self.assertIn("transformation applied", edit_prompt)

    def test_metrics_from_record_extracts_new_clinical_and_judge_fields(self) -> None:
        metrics = eval_module.metrics_from_record(
            {
                "RadGraph_F1": 0.7,
                "radgraph_applicable": True,
                "Text_EM": 0.0,
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

        self.assertEqual(metrics["RadGraph_F1"], 0.7)
        self.assertEqual(metrics["Text_EM"], 0.0)
        self.assertEqual(metrics["VLM_Anatomical_Accuracy_W_GT"], 8.0)
        self.assertEqual(metrics["VLM_Clinical_Finding_Accuracy_W_GT"], 9.0)
        self.assertEqual(metrics["VLM_Instruction_Compliance_W_GT"], 7.0)
        self.assertEqual(metrics["VLM_Cross_Modal_Consistency_W_GT"], 8.0)
        self.assertEqual(metrics["VLM_Hallucination_Omission_Control_W_GT"], 6.0)
        self.assertEqual(metrics["VLM_Overall_Score_W_GT"], 8.0)

    def test_metrics_from_record_extracts_new_image_metrics(self) -> None:
        metrics = eval_module.metrics_from_record(
            {
                "MedImageInsight_Similarity": 0.85,
            },
            "vqa",
        )
        self.assertEqual(metrics["MedImageInsight_Similarity"], 0.85)

    def test_aggregate_type_wise_results_reports_radgraph_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "typewise.jsonl")
            results = asyncio.run(
                eval_module.aggregate_type_wise_results(
                    data=[
                        {
                            "modality": "Radiograph",
                            "paper_task": "question-answering",
                            "RadGraph_F1": 0.8,
                            "radgraph_applicable": True,
                            "normalized_answer_text": "FINDINGS: clear lungs",
                        },
                        {
                            "modality": "Radiograph",
                            "paper_task": "question-answering",
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
    async def test_image_edit_clinical_metrics_does_not_build_vlm_requests(self) -> None:
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

            async def fake_anatomical_metric(eval_images, ref_images):
                return [{"MedImageInsight_Similarity": 1.0} for _ in eval_images]

            async def fake_full_image_metric(eval_images, ref_images, metric_name):
                return [1.0 for _ in eval_images]

            original_image_metric = eval_module.batch_async_medimageinsight_metrics
            original_full_image_metric = eval_module.batch_async_FR_IQA
            try:
                eval_module.batch_async_medimageinsight_metrics = fake_anatomical_metric
                eval_module.batch_async_FR_IQA = fake_full_image_metric
                results = await eval_module.basic_eval(
                    data=data,
                    batch_size=1,
                    task="image_edit",
                    data_path=temp_dir,
                    jsonl_path=jsonl_path,
                    run_vlm_judge=False,
                )
            finally:
                eval_module.batch_async_medimageinsight_metrics = original_image_metric
                eval_module.batch_async_FR_IQA = original_full_image_metric

            self.assertEqual(results["Average_MedImageInsight_Similarity"], 1.0)
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
            original_radgraph_batch = eval_module.compute_radgraph_f1_batch
            try:
                eval_module.batch_async_evaluate_text_quality = fake_text_metric
                eval_module.compute_radgraph_f1_batch = lambda responses, references: [
                    {"applicable": True, "f1": 0.9, "backend": "radgraph"}
                    for _ in responses
                ]
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
                eval_module.compute_radgraph_f1_batch = original_radgraph_batch

            self.assertIn("Average_RadGraph_F1", results)
            self.assertIn("RadGraph_Coverage", results)
            self.assertNotIn("Average_Text_EM", results)

            with open(intermediate_path, "r", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(len(rows), 1)
            self.assertIn("RadGraph_F1", rows[0])
            self.assertIn("normalized_answer_text", rows[0])
            self.assertNotIn("Text_EM", rows[0])
            self.assertIn("_local_metrics_complete", rows[0])
            for removed in [
                "Clinical_Entity_F1",
                "Entity_Hallucination_Rate",
                "Entity_Omission_Rate",
                "Entity_Factual_Precision",
            ]:
                self.assertNotIn(removed, rows[0])

    async def test_basic_eval_reports_bootstrap_ci_and_error_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "input.png")
            Image.new("RGB", (8, 8), color="white").save(image_path)
            jsonl_path = os.path.join(temp_dir, "ci_eval.jsonl")
            intermediate_path = os.path.join("eval_results", "ci_eval_local_metrics.jsonl")
            if os.path.exists(intermediate_path):
                os.remove(intermediate_path)

            data = [
                {
                    "category": "VQA",
                    "sub-category": "question-answering",
                    "modality": "Radiograph",
                    "input_image": image_path,
                    "instruction": f"Describe this image (case {i}).",
                    "choice": [],
                    "answer": "A chest radiograph is shown.",
                    "paper_task": "question-answering",
                    "input_images": [image_path],
                    "ground_truth_images": [],
                    "sample_id": f"unit:test:ci:{i}",
                    "eval_adapter": {"input_strategy": "single"},
                    "response": f"A chest radiograph is shown (instance {i}).",
                    "output_image": "",
                }
                for i in range(10)
            ]

            async def fake_text_metric(eval_texts, ref_texts, eval_metric):
                return [1.0 for _ in eval_texts]

            original_text_metric = eval_module.batch_async_evaluate_text_quality
            original_radgraph_batch = eval_module.compute_radgraph_f1_batch
            try:
                eval_module.batch_async_evaluate_text_quality = fake_text_metric
                eval_module.compute_radgraph_f1_batch = lambda responses, references: [
                    {"applicable": True, "f1": 0.9, "backend": "radgraph"}
                    for _ in responses
                ]
                results = await eval_module.basic_eval(
                    data=data,
                    batch_size=4,
                    task="vqa",
                    data_path=temp_dir,
                    jsonl_path=jsonl_path,
                    run_vlm_judge=False,
                )
            finally:
                eval_module.batch_async_evaluate_text_quality = original_text_metric
                eval_module.compute_radgraph_f1_batch = original_radgraph_batch

            self.assertIn("Bootstrap95CI_Low_BLEU", results)
            self.assertIn("Bootstrap95CI_High_BLEU", results)
            self.assertIn("Error_Analysis", results)
            self.assertIn("by_modality", results["Error_Analysis"])
            self.assertIn("failure_modes", results["Error_Analysis"])
            self.assertEqual(results["Error_Analysis"]["sample_count"], 10)

    def test_valid_judge_result_requires_all_clinical_dimensions(self) -> None:
        complete = {
            "anatomical_accuracy": {"score": 8},
            "clinical_finding_accuracy": {"score": 8},
            "instruction_compliance": {"score": 8},
            "cross_modal_consistency": {"score": 8},
            "hallucination_omission_control": {"score": 8},
            "overall_score": 8,
        }
        self.assertTrue(eval_module.valid_judge_result(complete))
        self.assertFalse(eval_module.valid_judge_result({}))
        self.assertFalse(eval_module.valid_judge_result({"overall_score": 8}))
        self.assertFalse(eval_module.valid_judge_result({
            **complete, "anatomical_accuracy": "not-a-score"
        }))

    def test_judge_checkpoint_requires_both_requested_views(self) -> None:
        complete = {
            "anatomical_accuracy": {"score": 8},
            "clinical_finding_accuracy": {"score": 8},
            "instruction_compliance": {"score": 8},
            "cross_modal_consistency": {"score": 8},
            "hallucination_omission_control": {"score": 8},
            "overall_score": 8,
        }
        item = {"input_image": "input.png", "vlm_judge_w_gt_result": complete}
        self.assertFalse(eval_module.judge_result_is_complete(item, "vqa"))
        item["vlm_judge_wo_gt_result"] = complete
        self.assertTrue(eval_module.judge_result_is_complete(item, "vqa"))


class ModelComparisonStatsTest(unittest.TestCase):
    def test_analyze_model_comparison_pairwise_wilcoxon_and_ci(self) -> None:
        import numpy as np
        import random

        random.seed(11)
        rng = np.random.default_rng(11)
        with tempfile.TemporaryDirectory() as temp_dir:
            path_a = os.path.join(temp_dir, "model_a.jsonl")
            path_b = os.path.join(temp_dir, "model_b.jsonl")
            rows_a, rows_b = [], []
            for i in range(20):
                sample_id = f"sample:{i}"
                rows_a.append({"sample_id": sample_id, "MedImageInsight_Similarity": 0.90 + rng.uniform(-0.01, 0.01)})
                rows_b.append({"sample_id": sample_id, "MedImageInsight_Similarity": 0.95 + rng.uniform(-0.01, 0.01)})
            with open(path_a, "w", encoding="utf-8") as handle:
                for row in rows_a:
                    handle.write(json.dumps(row) + "\n")
            with open(path_b, "w", encoding="utf-8") as handle:
                for row in rows_b:
                    handle.write(json.dumps(row) + "\n")

            result = asyncio.run(
                eval_module.analyze_model_comparison([path_a, path_b], task="vqa", n_boot=100)
            )

        self.assertEqual(result["paired_sample_count"], 20)
        self.assertIn("MedImageInsight_Similarity", result["metrics"])
        ssim_result = result["metrics"]["MedImageInsight_Similarity"]
        self.assertIn("model_a", ssim_result)
        self.assertIn("model_b", ssim_result)
        self.assertIn("ci_low", ssim_result["model_a"])
        comparison = ssim_result["model_a_vs_model_b"]
        if comparison["applicable"]:
            self.assertTrue(comparison["significant_at_0.05"])
            self.assertEqual(comparison["effect_direction"], "model_b>model_a")
        else:
            self.assertEqual(comparison.get("error"), "scipy is unavailable")


if __name__ == "__main__":
    unittest.main(verbosity=2)
