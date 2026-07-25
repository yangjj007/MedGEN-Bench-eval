#!/usr/bin/env python3
"""Focused unit tests for the new clinical-text evaluation helpers."""

from __future__ import annotations

import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import util.clinical_text_metrics as clinical_text_metrics
from util.clinical_text_metrics import (
    compute_clinical_entity_metrics,
    compute_radgraph_f1,
    compute_task_accuracy,
    compute_text_em_f1,
    normalize_closed_form_answer,
    serialize_clinical_reference,
)


class ClinicalTextMetricTest(unittest.TestCase):
    def tearDown(self) -> None:
        get_scorer = getattr(clinical_text_metrics, "_get_radgraph_f1_scorer", None)
        if get_scorer is not None and hasattr(get_scorer, "cache_clear"):
            get_scorer.cache_clear()

    def test_serialize_clinical_reference_uses_stable_section_order(self) -> None:
        answer = {
            "IMPRESSION": {"Summary": "No pleural effusion."},
            "FINDINGS": {
                "Lungs": "Clear.",
                "Heart": "Cardiomediastinal silhouette within normal limits.",
            },
            "MODALITY": "Radiograph",
            "TECHNIQUE": "PA and lateral chest radiographs.",
        }

        serialized = serialize_clinical_reference(answer)

        self.assertIn("MODALITY: Radiograph", serialized)
        self.assertIn("TECHNIQUE: PA and lateral chest radiographs.", serialized)
        self.assertIn("FINDINGS:", serialized)
        self.assertIn("IMPRESSION:", serialized)
        self.assertLess(serialized.index("MODALITY:"), serialized.index("FINDINGS:"))
        self.assertLess(serialized.index("FINDINGS:"), serialized.index("IMPRESSION:"))

    def test_normalize_closed_form_answer_accepts_option_letter(self) -> None:
        choices = [
            "A. right lung",
            "B. heart",
            "C. left lung",
        ]

        self.assertEqual(
            normalize_closed_form_answer("B", choices),
            "b. heart",
        )
        self.assertEqual(
            normalize_closed_form_answer("B.", choices),
            "b. heart",
        )
        self.assertEqual(
            normalize_closed_form_answer("heart", choices),
            "b. heart",
        )

    def test_compute_task_accuracy_uses_closed_form_normalization(self) -> None:
        choices = [
            "A. Morton's neuroma",
            "B. Anterometatarsal bursitis",
        ]
        self.assertEqual(
            compute_task_accuracy(
                paper_task="multiple-choice",
                response="B",
                answer="B. Anterometatarsal bursitis",
                choices=choices,
            ),
            1.0,
        )

    def test_compute_text_em_f1_handles_short_answers(self) -> None:
        em, f1 = compute_text_em_f1(
            response="Subacute thyroiditis",
            answer="Subacute thyroiditis (de Quervain's thyroiditis)",
        )
        self.assertEqual(em, 0.0)
        self.assertGreater(f1, 0.45)

    def test_compute_clinical_entity_metrics_matches_equivalent_reports(self) -> None:
        reference = (
            "FINDINGS: Lungs are clear. No pleural effusion. "
            "Cardiomediastinal silhouette is within normal limits."
        )
        response = (
            "No pleural effusion is seen. Lungs clear bilaterally. "
            "Heart size and mediastinal contour are within normal limits."
        )

        metrics = compute_clinical_entity_metrics(response, reference)

        self.assertGreaterEqual(metrics["precision"], 0.75)
        self.assertGreaterEqual(metrics["recall"], 0.75)
        self.assertGreaterEqual(metrics["f1"], 0.75)

    def test_compute_radgraph_f1_marks_non_report_short_answer_not_applicable(self) -> None:
        result = compute_radgraph_f1("A. H&E", "A. H&E")
        self.assertFalse(result["applicable"])
        self.assertIsNone(result["f1"])

    def test_compute_radgraph_f1_prefers_external_package_rg_er_score(self) -> None:
        class FakeF1RadGraph:
            instances = []

            def __init__(self, reward_level: str, model_type: str | None = None, **kwargs) -> None:
                self.reward_level = reward_level
                self.model_type = model_type
                self.kwargs = kwargs
                self.__class__.instances.append(self)

            def __call__(self, hyps, refs):
                return (
                    (0.25, 0.8, 0.5),
                    ([0.25], [0.8], [0.5]),
                    [{"entities": {"1": {"tokens": "opacity", "label": "Observation::definitely present", "relations": []}}}],
                    [{"entities": {"1": {"tokens": "opacity", "label": "Observation::definitely present", "relations": []}}}],
                )

        with patch("importlib.import_module", return_value=SimpleNamespace(F1RadGraph=FakeF1RadGraph)):
            result = clinical_text_metrics.compute_radgraph_f1(
                "There is a right lung opacity.",
                "There is a right lung opacity.",
            )

        self.assertTrue(result["applicable"])
        self.assertAlmostEqual(result["f1"], 0.8)
        self.assertEqual(result["backend"], "radgraph")
        self.assertEqual(FakeF1RadGraph.instances[0].reward_level, "all")
        self.assertEqual(FakeF1RadGraph.instances[0].model_type, "radgraph-xl")

    def test_compute_radgraph_f1_marks_empty_external_graph_pair_not_applicable(self) -> None:
        class FakeF1RadGraph:
            def __init__(self, reward_level: str, model_type: str | None = None, **kwargs) -> None:
                self.reward_level = reward_level
                self.model_type = model_type
                self.kwargs = kwargs

            def __call__(self, hyps, refs):
                return (
                    (0.0, 0.0, 0.0),
                    ([0.0], [0.0], [0.0]),
                    [{"entities": {}}],
                    [{"entities": {}}],
                )

        with patch("importlib.import_module", return_value=SimpleNamespace(F1RadGraph=FakeF1RadGraph)):
            result = clinical_text_metrics.compute_radgraph_f1(
                "No acute cardiopulmonary abnormality is identified.",
                "No acute cardiopulmonary abnormality is identified.",
            )

        self.assertFalse(result["applicable"])
        self.assertIsNone(result["f1"])
        self.assertEqual(result["backend"], "radgraph")


if __name__ == "__main__":
    unittest.main(verbosity=2)
