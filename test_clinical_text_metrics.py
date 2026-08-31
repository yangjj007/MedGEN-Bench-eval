"""Focused unit tests for the new clinical-text evaluation helpers."""

from __future__ import annotations

import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import util.clinical_text_metrics as clinical_text_metrics
from util.clinical_text_metrics import (
    compute_closed_form_exact_match,
    compute_radgraph_f1,
    compute_radgraph_f1_batch,
    compute_text_exact_match,
    extract_choices_from_instruction,
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

    def test_compute_text_exact_match_normalizes_whitespace(self) -> None:
        em = compute_text_exact_match(
            response="Subacute thyroiditis",
            answer=" subacute   thyroiditis ",
        )
        self.assertEqual(em, 1.0)

    def test_multiple_choice_without_choices_uses_answer_text(self) -> None:
        result = compute_closed_form_exact_match(
            "Ischemic stroke lesion",
            "ischemic stroke lesion",
            choices=[],
            task="multiple-choice",
        )
        self.assertEqual(result["score"], 1.0)

    def test_multiple_choice_options_are_recovered_from_instruction(self) -> None:
        choices = extract_choices_from_instruction(
            "Question: Which finding is present?\n\nOptions:\n"
            "A. atelectasis\nB. pneumonia\nC. edema"
        )
        self.assertEqual(choices, ["A. atelectasis", "B. pneumonia", "C. edema"])
        result = compute_closed_form_exact_match(
            "The answer is B.",
            "B. pneumonia",
            choices=choices,
            task="multiple-choice",
        )
        self.assertEqual(result["score"], 1.0)

    def test_compute_radgraph_f1_fails_when_scorer_unavailable(self) -> None:
        with patch.object(
            clinical_text_metrics,
            "_get_radgraph_f1_scorer",
            side_effect=RuntimeError("RadGraph unavailable"),
        ):
            with self.assertRaises(RuntimeError):
                compute_radgraph_f1("A. H&E", "A. H&E")

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

    def test_compute_radgraph_f1_batch_preserves_per_sample_rg_er(self) -> None:
        entity = {"entities": {"1": {"tokens": "opacity", "relations": []}}}

        class FakeBatchScorer:
            def __call__(self, hyps, refs):
                self.hyps = hyps
                self.refs = refs
                return (
                    (0.5, 0.6, 0.7),
                    ([0.4, 0.6], [0.7, 0.9], [0.5, 0.8]),
                    [entity, {"entities": {}}],
                    [entity, {"entities": {}}],
                )

        scorer = FakeBatchScorer()
        with patch.object(clinical_text_metrics, "_get_radgraph_f1_scorer", return_value=scorer):
            results = compute_radgraph_f1_batch(
                [
                    "There is a right lung opacity.",
                    "No acute cardiopulmonary abnormality.",
                    "short",
                ],
                [
                    "There is a right lung opacity.",
                    "No acute cardiopulmonary abnormality.",
                    "short",
                ],
            )

        self.assertEqual(len(scorer.hyps), 2)
        self.assertTrue(results[0]["applicable"])
        self.assertAlmostEqual(results[0]["f1"], 0.7)
        self.assertFalse(results[1]["applicable"])
        self.assertIsNone(results[1]["f1"])
        self.assertEqual(results[2]["backend"], "skipped")



if __name__ == "__main__":
    unittest.main(verbosity=2)
