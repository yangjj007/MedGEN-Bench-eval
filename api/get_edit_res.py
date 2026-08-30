"""OpenAI-compatible image editing client.

The client sends a standard multipart request to ``<base_url>/images/edits``.
It works with cloud gateways and local Qwen Image Edit services that implement
the OpenAI image-editing contract.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar

from .get_generate_res import _OpenAICompatibleImageClient


_Result = TypeVar("_Result")


class ImageEditAPI(_OpenAICompatibleImageClient):
    """Edit one or more images through a configurable OpenAI-compatible API."""

    def __init__(
        self,
        config_path: str = "./api/config.yaml",
        model_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        debug: bool = False,
    ) -> None:
        super().__init__(config_path, base_url=base_url, api_key=api_key, debug=debug)
        selected_model = model_name or self.config.get("image_edit_model_name") or self.config.get("model_name")
        if not isinstance(selected_model, str) or not selected_model.strip():
            raise ValueError("model_name is required for image editing")
        self.model_name = selected_model

    async def edit_image_async(
        self,
        image_path: str,
        prompt: str,
        save_dir: str = "./output",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Edit one local image and return normalized, JSON-serializable metadata."""
        created_session = self._session is None or self._session.closed
        await self._ensure_session()
        try:
            paths = await self._create_images(
                model_name=self.model_name,
                prompt=prompt,
                output_dir=save_dir,
                file_prefix=str(kwargs.pop("file_prefix", "edited_image")),
                input_image=image_path,
                endpoint_kind="edit",
                **kwargs,
            )
            return {
                "text": [prompt],
                "image_paths": paths,
                "usage": {},
            }
        finally:
            if created_session:
                await self.aclose()

    @staticmethod
    def _run_synchronously(factory: Callable[[], Any]) -> _Result:
        """Run an async API operation from synchronous legacy call sites.

        A small worker thread avoids calling ``asyncio.run`` inside an already
        running event loop. This retains the historic synchronous batch method
        while the maintained implementation itself remains asynchronous.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(factory())

        result: dict[str, Any] = {}
        error: dict[str, BaseException] = {}

        def runner() -> None:
            try:
                result["value"] = asyncio.run(factory())
            except BaseException as exc:  # pragma: no cover - propagated below
                error["value"] = exc

        worker = threading.Thread(target=runner, daemon=False)
        worker.start()
        worker.join()
        if "value" in error:
            raise error["value"]
        return result["value"]

    def edit_image(
        self,
        image_path: str,
        prompt: str,
        save_dir: str = "./output",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Synchronous compatibility wrapper around :meth:`edit_image_async`."""
        return self._run_synchronously(
            lambda: self.edit_image_async(image_path, prompt, save_dir, **kwargs)
        )

    async def edit_images_batch_async(
        self,
        image_prompts: list[tuple[str, str]],
        save_dir: str = "./output",
        max_concurrent: int = 5,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Edit independent image/prompt pairs concurrently and preserve order."""
        if max_concurrent <= 0:
            raise ValueError("max_concurrent must be positive")
        output_root = Path(save_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        created_session = self._session is None or self._session.closed
        await self._ensure_session()
        semaphore = asyncio.Semaphore(max_concurrent)

        async def edit_one(index: int, image_path: str, prompt: str) -> tuple[int, dict[str, Any]]:
            started = time.monotonic()
            async with semaphore:
                try:
                    paths = await self._create_images(
                        model_name=self.model_name,
                        prompt=prompt,
                        output_dir=str(output_root),
                        file_prefix=f"edited_{index:06d}",
                        input_image=image_path,
                        endpoint_kind="edit",
                        **kwargs,
                    )
                    result = {
                        "task_index": index,
                        "image_path": image_path,
                        "prompt": prompt,
                        "text": [prompt],
                        "image_paths": paths,
                        "usage": {},
                        "duration": time.monotonic() - started,
                    }
                except Exception as exc:
                    result = {
                        "task_index": index,
                        "image_path": image_path,
                        "prompt": prompt,
                        "text": [],
                        "image_paths": [],
                        "usage": {},
                        "duration": time.monotonic() - started,
                        "error": str(exc),
                    }
                return index, result

        try:
            pairs = await asyncio.gather(
                *(edit_one(index, image_path, prompt) for index, (image_path, prompt) in enumerate(image_prompts))
            )
            results: list[dict[str, Any]] = [{} for _ in image_prompts]
            for index, result in pairs:
                results[index] = result
            return results
        finally:
            if created_session:
                await self.aclose()

    def edit_images_batch(
        self,
        image_prompts: list[tuple[str, str]],
        save_dir: str = "./output",
        max_concurrent: int = 5,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Synchronous compatibility wrapper around :meth:`edit_images_batch_async`."""
        return self._run_synchronously(
            lambda: self.edit_images_batch_async(
                image_prompts,
                save_dir=save_dir,
                max_concurrent=max_concurrent,
                **kwargs,
            )
        )

    @classmethod
    def get_supported_models(cls) -> list[str]:
        """Model routing is config-driven; no provider-specific registry is embedded."""
        return []

    @classmethod
    def get_model_info(cls, model_name: str) -> Mapping[str, str] | None:
        """Retain the historical lookup method without inventing model metadata."""
        if not model_name:
            return None
        return {"name": str(model_name), "endpoint": "/images/edits"}
