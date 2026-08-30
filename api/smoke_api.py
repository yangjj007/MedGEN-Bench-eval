#!/usr/bin/env python3
"""Make a minimal real request to configured local or cloud MedGEN APIs."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Readable local image for the request.")
    parser.add_argument("--vlm-config", default=None, help="VLM YAML configuration to test.")
    parser.add_argument("--vlm-model", default=None, help="Optional VLM model override.")
    parser.add_argument("--image-config", default=None, help="Image-edit YAML configuration to test.")
    parser.add_argument("--edit-model", default=None, help="Image-edit model override.")
    parser.add_argument("--output-dir", default="outputs/api-smoke")
    parser.add_argument(
        "--image-size", default="256x256",
        help="Optional WIDTHxHEIGHT request size for the image-edit probe.",
    )
    parser.add_argument(
        "--prompt",
        default="Describe the image in one concise sentence.",
        help="Prompt used for the VLM and image-edit probes.",
    )
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    image_path = Path(args.image).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    if not args.vlm_config and not args.image_config:
        raise ValueError("Pass --vlm-config, --image-config, or both")

    if args.vlm_config:
        from api.get_vlm_res import single_image_vlm

        client = single_image_vlm(args.vlm_config, model_name=args.vlm_model)
        try:
            result = await client.generate_with_image_async(
                args.prompt,
                str(image_path),
                temperature=0.0,
                max_tokens=96,
            )
        finally:
            await client.aclose()
        if result.get("error"):
            raise RuntimeError(f"VLM request failed: {result['error']}")
        summary = " ".join(str(result.get("text", "")).split())[:240]
        print(f"VLM request succeeded; response preview: {summary}")

    if args.image_config:
        from api.get_edit_res import ImageEditAPI

        client = ImageEditAPI(args.image_config, model_name=args.edit_model)
        result = await client.edit_image_async(
            str(image_path),
            args.prompt,
            save_dir=args.output_dir,
            response_format="url",
            size=args.image_size,
        )
        if result.get("error"):
            raise RuntimeError(f"Image-edit request failed: {result['error']}")
        paths = result.get("image_paths") or []
        if not paths:
            raise RuntimeError("Image-edit request completed without an output image")
        print(f"Image-edit request succeeded; output: {paths[0]}")


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
