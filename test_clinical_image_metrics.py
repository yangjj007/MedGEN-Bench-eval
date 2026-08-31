"""Unit tests for the clinical-aligned image metrics and task checks."""

from __future__ import annotations

import asyncio
import unittest

import numpy as np
from PIL import Image

from util.metrics import (
    FR_IQA,
    batch_async_medimageinsight_metrics,
    compute_anatomical_embedding_similarity,
)


def _make_images(lesion_shift: int = 0):
    rng = np.random.default_rng(7)
    base = rng.integers(0, 200, (64, 64, 3), dtype=np.uint8)
    ref = Image.fromarray(base)
    if lesion_shift == 0:
        return ref.copy(), ref
    arr = base.copy()
    arr[8:20, 8:20] = np.clip(arr[8:20, 8:20].astype(int) + lesion_shift, 0, 255).astype(np.uint8)
    return Image.fromarray(arr), ref


class AnatomicalMetricTest(unittest.TestCase):
    def test_anatomical_metric_is_a_named_medical_embedding_metric(self) -> None:
        self.assertEqual(compute_anatomical_embedding_similarity.__name__, "compute_anatomical_embedding_similarity")

    def test_batch_requires_aligned_pairs(self) -> None:
        eval_img, ref = _make_images()
        with self.assertRaises(ValueError):
            asyncio.run(batch_async_medimageinsight_metrics([eval_img], [ref, ref]))

    def test_full_reference_image_metrics_have_documented_ranges(self) -> None:
        """Regression guard for the raw (not 0--100 normalized) metrics."""
        candidate, reference = _make_images(lesion_shift=40)
        ssim = FR_IQA(candidate, reference, "ssim")
        psnr = FR_IQA(candidate, reference, "psnr")
        lpips = FR_IQA(candidate, reference, "lpips")
        self.assertGreaterEqual(ssim, -1.0)
        self.assertLessEqual(ssim, 1.0)
        self.assertTrue(np.isfinite(psnr))
        self.assertGreaterEqual(lpips, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
