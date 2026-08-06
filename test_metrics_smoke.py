"""Small executable smoke test for every local metric used by eval.py."""

import argparse
import asyncio
import math

import numpy as np
from PIL import Image

from util.metrics import (
    compute_anatomical_embedding_similarity,
    batch_async_evaluate_text_quality,
    batch_async_anatomical_metrics,
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
    print("Testing Rad-DINO anatomical metric...", flush=True)
    scores.update((await batch_async_anatomical_metrics([candidate], [reference]))[0])

    async_bleu = await batch_async_evaluate_text_quality(
        ["normal chest radiograph"], ["normal chest radiograph"], "bleu"
    )
    assert len(async_bleu) == 1
    assert all(math.isfinite(value) for value in scores.values())
    assert -1 <= scores["Anatomical_Embedding_Similarity"] <= 1
    assert 0 <= scores["BLEU"] <= 1
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
