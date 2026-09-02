"""OpenAI-compatible local server for Qwen-Image-Edit.

The standard vLLM server is used for the Qwen vision-language model.  Image
editing is a diffusion workload, so this small FastAPI service exposes the
same ``/v1/images/edits`` shape used by the evaluation runner while loading
Qwen-Image-Edit through Diffusers.  It deliberately binds to localhost by
default and does not implement authentication; do not expose it directly to
an untrusted network.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image


@dataclass(frozen=True)
class ServerSettings:
    model: str
    output_dir: Path
    device: str
    dtype: str
    local_files_only: bool
    cpu_offload_mode: str
    default_steps: int
    default_cfg_scale: float


class QwenImageEditRunner:
    """Lazy Diffusers loader that serializes GPU image-edit requests."""

    def __init__(self, settings: ServerSettings) -> None:
        self.settings = settings
        self._pipeline: Any | None = None
        self._load_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def _load_pipeline_sync(self) -> Any:
        try:
            import torch
            from diffusers import QwenImageEditPipeline
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "Qwen Image Edit requires torch, diffusers, transformers, and accelerate. "
                "Install requirements.txt in the server environment."
            ) from exc

        dtype = getattr(torch, self.settings.dtype)
        pipeline = QwenImageEditPipeline.from_pretrained(
            self.settings.model,
            torch_dtype=dtype,
            local_files_only=self.settings.local_files_only,
        )
        pipeline.set_progress_bar_config(disable=True)
        if self.settings.cpu_offload_mode != "none":
            if self.settings.device != "cuda":
                raise RuntimeError("CPU offload requires --device cuda")
            if self.settings.cpu_offload_mode == "sequential":
                pipeline.enable_sequential_cpu_offload()
            elif self.settings.cpu_offload_mode == "model":
                pipeline.enable_model_cpu_offload()
            else:  # pragma: no cover - argparse validates the value
                raise RuntimeError(
                    f"Unsupported CPU offload mode: {self.settings.cpu_offload_mode}"
                )
        else:
            pipeline.to(self.settings.device)
        return pipeline

    async def load(self) -> None:
        if self._pipeline is not None:
            return
        async with self._load_lock:
            if self._pipeline is None:
                self._pipeline = await asyncio.to_thread(self._load_pipeline_sync)

    def _edit_sync(
        self,
        image: Image.Image,
        prompt: str,
        steps: int,
        cfg_scale: float,
        seed: int | None,
    ) -> Image.Image:
        import torch

        if self._pipeline is None:  # pragma: no cover - guarded by edit()
            raise RuntimeError("The Qwen Image Edit pipeline is not loaded")
        generator = None
        if seed is not None:
            generator_device = "cuda" if self.settings.device.startswith("cuda") else "cpu"
            generator = torch.Generator(device=generator_device).manual_seed(seed)
        with torch.inference_mode():
            result = self._pipeline(
                image=image,
                prompt=prompt,
                generator=generator,
                true_cfg_scale=cfg_scale,
                negative_prompt=" ",
                num_inference_steps=steps,
            )
        return result.images[0]

    async def edit(
        self,
        image: Image.Image,
        prompt: str,
        steps: int,
        cfg_scale: float,
        seed: int | None,
    ) -> Image.Image:
        await self.load()
        async with self._inference_lock:
            return await asyncio.to_thread(
                self._edit_sync, image, prompt, steps, cfg_scale, seed
            )


def parse_size(size: str) -> tuple[int, int] | None:
    """Validate an optional OpenAI-style ``WIDTHxHEIGHT`` image size."""
    if not size:
        return None
    try:
        width_text, height_text = size.lower().split("x", maxsplit=1)
        width, height = int(width_text), int(height_text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="size must be WIDTHxHEIGHT") from exc
    if not (64 <= width <= 2048 and 64 <= height <= 2048):
        raise HTTPException(status_code=422, detail="size dimensions must be 64..2048")
    return width, height


def image_from_upload(payload: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(payload)) as loaded:
            return loaded.convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail="image must be a readable raster image") from exc


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def save_image(image: Image.Image, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"qwen-image-edit-{secrets.token_hex(12)}.png"
    image.save(path, format="PNG")
    return path


def build_app(settings: ServerSettings) -> FastAPI:
    app = FastAPI(title="MedGEN Qwen Image Edit API", version="1.0")
    app.state.settings = settings
    app.state.runner = QwenImageEditRunner(settings)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/outputs", StaticFiles(directory=settings.output_dir), name="outputs")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": settings.model,
            "loaded": app.state.runner.is_loaded,
            "device": settings.device,
        }

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": settings.model, "object": "model", "owned_by": "local"}],
        }

    @app.post("/v1/images/edits")
    async def image_edit(
        request: Request,
        image: UploadFile = File(...),
        prompt: str = Form(...),
        model: str = Form(""),
        n: int = Form(1),
        size: str = Form(""),
        response_format: str = Form("url"),
        seed: int | None = Form(None),
        num_inference_steps: int | None = Form(None),
        true_cfg_scale: float | None = Form(None),
    ) -> JSONResponse:
        del model
        if not prompt.strip():
            raise HTTPException(status_code=422, detail="prompt must not be empty")
        if n != 1:
            raise HTTPException(status_code=422, detail="this local service supports n=1 only")
        if response_format not in {"url", "b64_json"}:
            raise HTTPException(status_code=422, detail="response_format must be url or b64_json")
        steps = num_inference_steps or settings.default_steps
        cfg_scale = true_cfg_scale or settings.default_cfg_scale
        if not (2 <= steps <= 100):
            raise HTTPException(status_code=422, detail="num_inference_steps must be 2..100")
        if not (0.0 < cfg_scale <= 20.0):
            raise HTTPException(status_code=422, detail="true_cfg_scale must be in (0, 20]")

        source = image_from_upload(await image.read())
        requested_size = parse_size(size)
        if requested_size:
            source = source.resize(requested_size, Image.Resampling.LANCZOS)
        try:
            edited = await app.state.runner.edit(source, prompt, steps, cfg_scale, seed)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Qwen Image Edit could not complete the request: {exc}",
            ) from exc
        finally:
            source.close()

        if response_format == "b64_json":
            encoded = image_to_base64(edited)
            edited.close()
            return JSONResponse({"created": int(time.time()), "data": [{"b64_json": encoded}]})

        output_path = save_image(edited, settings.output_dir)
        edited.close()
        public_url = str(request.base_url).rstrip("/") + "/outputs/" + output_path.name
        return JSONResponse({"created": int(output_path.stat().st_mtime), "data": [{"url": public_url}]})

    return app


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(project_root / "models" / "Qwen-Image-Edit"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs" / "local-image-service",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--cpu-offload-mode",
        choices=("none", "model", "sequential"),
        default="none",
        help="Offload strategy. Sequential mode fits a 40 GB GPU but is slower.",
    )
    parser.add_argument("--default-steps", type=int, default=30)
    parser.add_argument("--default-cfg-scale", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.default_steps < 2:
        raise ValueError("--default-steps must be at least 2")
    if args.default_cfg_scale <= 0:
        raise ValueError("--default-cfg-scale must be positive")
    settings = ServerSettings(
        model=args.model,
        output_dir=args.output_dir.resolve(),
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
        cpu_offload_mode=args.cpu_offload_mode,
        default_steps=args.default_steps,
        default_cfg_scale=args.default_cfg_scale,
    )
    import uvicorn

    uvicorn.run(build_app(settings), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
