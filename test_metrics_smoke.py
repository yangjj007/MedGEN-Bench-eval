"""Small executable smoke test for every local metric used by eval.py."""

import argparse
import asyncio
import math

import numpy as np
from PIL import Image

from util.metrics import (
    FR_IQA,
    batch_async_FR_IQA,
    batch_async_evaluate_text_quality,
    evaluate_text_quality,
)


def make_images() -> tuple[Image.Image, Image.Image]:
    gradient = np.tile(np.arange(32, dtype=np.uint8), (32, 1)) * 8
    reference = np.stack([gradient, gradient, gradient], axis=-1)
    candidate = reference.astype(np.int16)
    candidate[8:24, 8:24] += 12
    candidate = np.clip(candidate, 0, 255).astype(np.uint8)
    return Image.fromarray(candidate), Image.fromarray(reference)


async def run(include_bertscore: bool) -> dict[str, float]:
    candidate, reference = make_images()
    scores = {
        "PSNR": FR_IQA(candidate, reference, "psnr"),
        "SSIM": FR_IQA(candidate, reference, "ssim"),
        "BLEU": evaluate_text_quality(
            "No acute cardiopulmonary abnormality.",
            "No acute cardiopulmonary abnormality.",
            "bleu",
        ),
    }
    if include_bertscore:
        print("Testing BERTScore...", flush=True)
        async_bert = await batch_async_evaluate_text_quality(
            ["normal chest radiograph"],
            ["normal chest radiograph"],
            "bertscore",
        )
        assert async_bert[0] > 0.99
        scores["BERT_Score"] = async_bert[0]
    async_images = {}
    for metric in ("lpips", "psnr", "ssim"):
        print(f"Testing {metric.upper()}...", flush=True)
        async_images[metric] = (
            await batch_async_FR_IQA([candidate], [reference], metric)
        )[0]
    scores["LPIPS"] = async_images["lpips"]
    async_bleu = await batch_async_evaluate_text_quality(
        ["normal chest radiograph"], ["normal chest radiograph"], "bleu"
    )
    assert len(async_bleu) == 1
    assert all(math.isfinite(value) for value in async_images.values())
    assert all(math.isfinite(value) for value in scores.values())
    assert scores["LPIPS"] >= 0
    assert scores["PSNR"] > 0
    assert -1 <= scores["SSIM"] <= 1
    assert 0 <= scores["BLEU"] <= 1
    assert FR_IQA(reference, reference, "psnr") == 100.0
    if include_bertscore:
        assert scores["BERT_Score"] > 0.99
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-bertscore",
        action="store_true",
        help="also download/load PubMedBERT and test BERTScore",
    )
    args = parser.parse_args()
    scores = asyncio.run(run(args.include_bertscore))
    print("Local metric smoke test passed:")
    for name, value in scores.items():
        print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    main()
