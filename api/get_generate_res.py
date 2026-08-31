"""OpenAI-compatible image generation client.

The evaluator deliberately keeps provider-specific behavior out of this module.
Any service that exposes the standard ``/images/generations`` and
``/images/edits`` endpoints can be selected through a YAML configuration file.
The latter endpoint is also used for image-to-image generation when an input
image is supplied.
"""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

import aiohttp
import yaml


class ImageAPIError(RuntimeError):
    """Raised when an image endpoint returns an unusable response."""


@dataclass(frozen=True)
class GenerationConfig:
    """A single image-generation request."""

    prompt: str
    output_dir: str = "."
    file_prefix: str = "generated_image"
    extra_params: dict[str, Any] = field(default_factory=dict)


def _as_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping when provided")
    return dict(value)


_ENVIRONMENT_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_environment(value: Any) -> Any:
    """Expand ``${VARIABLE}`` values in a YAML configuration recursively."""
    if isinstance(value, str):
        missing: list[str] = []

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            replacement = os.environ.get(name)
            if replacement is None:
                missing.append(name)
                return ""
            return replacement

        expanded = _ENVIRONMENT_REFERENCE.sub(replace, value)
        if missing:
            raise ValueError(
                "Missing environment variable(s) referenced by the image API config: "
                + ", ".join(sorted(set(missing)))
            )
        return expanded
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _expand_environment(item) for key, item in value.items()}
    return value


class _OpenAICompatibleImageClient:
    """Small transport shared by image generation and editing clients."""

    _PLACEHOLDER_KEYS = {"", "YOUR_API_KEY", "sk-your-api-key-here"}

    def __init__(
        self,
        config_path: str = "./config.yaml",
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        debug: bool = False,
    ) -> None:
        self.config_path = str(config_path)
        self.config = self._load_config(self.config_path)
        configured_base_url = (
            base_url
            or os.environ.get("MEDGEN_IMAGE_BASE_URL")
            or self.config.get("image_base_url")
            or self.config.get("base_url")
        )
        if not isinstance(configured_base_url, str) or not configured_base_url.strip():
            raise ValueError("config must provide a non-empty base_url or image_base_url")
        self.base_url = configured_base_url.rstrip("/")
        self.api_key = self._resolve_api_key(api_key)
        self.debug = bool(debug)

        self.timeout_seconds = float(self.config.get("image_timeout", self.config.get("timeout", 300)))
        if self.timeout_seconds <= 0:
            raise ValueError("image_timeout must be positive")
        self.max_retries = max(0, int(self.config.get("max_retries", 2)))
        self.retry_delay = max(0.0, float(self.config.get("retry_delay", 2)))
        self._session: aiohttp.ClientSession | None = None

    @staticmethod
    def _load_config(config_path: str) -> dict[str, Any]:
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Image API config was not found: {path}. "
                "Create config.yaml in the repository root and set environment credentials."
            )
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        return _as_mapping(_expand_environment(loaded), "image API config")

    def _resolve_api_key(self, explicit_api_key: str | None) -> str:
        candidate = (
            explicit_api_key
            or os.environ.get("MEDGEN_IMAGE_API_KEY")
            or os.environ.get("MEDGEN_API_KEY")
            or os.environ.get("AIHUBMIX_API_KEY")
            or self.config.get("image_api_key")
            or self.config.get("api_key")
        )
        key = str(candidate or "").strip()
        if key not in self._PLACEHOLDER_KEYS:
            return key
        hostname = (urlparse(self.base_url).hostname or "").lower()
        if hostname in {"127.0.0.1", "localhost", "::1"}:
            return "EMPTY"
        raise ValueError(
            "A non-placeholder API key is required for a non-local image endpoint. "
            "Set image_api_key/api_key in the config or MEDGEN_IMAGE_API_KEY."
        )

    async def __aenter__(self):
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.aclose()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def aclose(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    def _endpoint(self, kind: str) -> str:
        aliases = {
            "generation": (
                "image_generation_endpoint",
                "images_generation_endpoint",
                "generation_endpoint",
            ),
            "edit": ("image_edit_endpoint", "images_edit_endpoint", "edit_endpoint"),
            "image_to_image": (
                "image_to_image_endpoint",
                "images_image_to_image_endpoint",
            ),
        }
        defaults = {
            "generation": "/images/generations",
            "edit": "/images/edits",
            "image_to_image": "/images/edits",
        }
        if kind not in aliases:
            raise ValueError(f"Unknown image endpoint kind: {kind}")
        configured = next(
            (self.config.get(name) for name in aliases[kind] if self.config.get(name)),
            defaults[kind],
        )
        if not isinstance(configured, str):
            raise ValueError(f"Configured {kind} endpoint must be a string")
        if configured.startswith(("http://", "https://")):
            return configured
        return f"{self.base_url}/{configured.lstrip('/')}"

    def _request_defaults(self, kind: str) -> dict[str, Any]:
        defaults = _as_mapping(self.config.get("image_request_defaults"), "image_request_defaults")
        kind_defaults = _as_mapping(
            self.config.get(f"{kind}_request_defaults"), f"{kind}_request_defaults"
        )
        defaults.update(kind_defaults)
        return defaults

    @staticmethod
    def _normalize_input_images(input_image: Any) -> list[Path]:
        if input_image is None or input_image == "":
            return []
        values: Sequence[Any]
        if isinstance(input_image, (str, os.PathLike)):
            values = [input_image]
        elif isinstance(input_image, Sequence):
            values = input_image
        else:
            raise TypeError("input_image must be a path or a sequence of paths")
        paths: list[Path] = []
        for value in values:
            path = Path(value)
            if not path.is_file():
                raise FileNotFoundError(f"Input image does not exist: {path}")
            paths.append(path)
        return paths

    @staticmethod
    def _clean_params(
        model_name: str,
        prompt: str,
        request_defaults: Mapping[str, Any],
        kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        params = dict(request_defaults)
        params.update(kwargs)
        extra_body = params.pop("extra_body", None)
        if extra_body is not None:
            params.update(_as_mapping(extra_body, "extra_body"))
        for key in (
            "input_image",
            "output_dir",
            "file_prefix",
            "max_concurrent",
            "model_name",
            "prompt",
        ):
            params.pop(key, None)
        if "num_images" in params:
            params["n"] = params.pop("num_images")
        if "image_size" in params and "size" not in params:
            params["size"] = params.pop("image_size")
        params["model"] = model_name
        params["prompt"] = prompt
        return {key: value for key, value in params.items() if value is not None}

    def _headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    async def _read_json_response(self, response: aiohttp.ClientResponse, endpoint: str) -> dict[str, Any]:
        raw_text = await response.text()
        if response.status < 200 or response.status >= 300:
            detail = raw_text[:1000]
            raise ImageAPIError(f"Image API returned HTTP {response.status} at {endpoint}: {detail}")
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ImageAPIError(f"Image API returned non-JSON data at {endpoint}") from exc
        if not isinstance(parsed, dict):
            raise ImageAPIError("Image API response must be a JSON object")
        return parsed

    async def _post_json(self, endpoint: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        session = await self._ensure_session()
        headers = {**self._headers(), "Content-Type": "application/json"}
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                async with session.post(endpoint, json=dict(payload), headers=headers) as response:
                    return await self._read_json_response(response, endpoint)
            except (aiohttp.ClientError, asyncio.TimeoutError, ImageAPIError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                await asyncio.sleep(self.retry_delay * (2**attempt))
        raise ImageAPIError(f"Image API request failed after {self.max_retries + 1} attempts") from last_error

    async def _post_multipart(
        self,
        endpoint: str,
        fields: Mapping[str, Any],
        image_paths: Iterable[Path],
    ) -> dict[str, Any]:
        paths = list(image_paths)
        if not paths:
            raise ValueError("At least one input image is required for an image edit request")
        session = await self._ensure_session()
        image_field = str(self.config.get("image_upload_field", "image"))
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            handles = []
            try:
                form = aiohttp.FormData()
                for key, value in fields.items():
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value, separators=(",", ":"))
                    elif isinstance(value, bool):
                        value = "true" if value else "false"
                    else:
                        value = str(value)
                    form.add_field(key, value)
                for path in paths:
                    handle = path.open("rb")
                    handles.append(handle)
                    form.add_field(
                        image_field,
                        handle,
                        filename=path.name,
                        content_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                    )
                async with session.post(endpoint, data=form, headers=self._headers()) as response:
                    return await self._read_json_response(response, endpoint)
            except (aiohttp.ClientError, asyncio.TimeoutError, ImageAPIError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                await asyncio.sleep(self.retry_delay * (2**attempt))
            finally:
                for handle in handles:
                    handle.close()
        raise ImageAPIError(f"Image API request failed after {self.max_retries + 1} attempts") from last_error

    @staticmethod
    def _response_items(response: Mapping[str, Any]) -> list[Any]:
        for key in ("data", "output", "images"):
            value = response.get(key)
            if isinstance(value, list):
                return value
        if any(key in response for key in ("url", "b64_json", "bytesBase64", "image_url")):
            return [response]
        raise ImageAPIError("Image API response did not contain data, output, or images")

    @staticmethod
    def _decode_base64(value: str) -> tuple[bytes, str]:
        mime_type = "image/png"
        payload = value
        if value.startswith("data:"):
            header, separator, payload = value.partition(",")
            if not separator or ";base64" not in header:
                raise ImageAPIError("Unsupported image data URI")
            mime_type = header[5:].split(";", 1)[0] or mime_type
        try:
            return base64.b64decode(payload, validate=True), mime_type
        except (ValueError, TypeError) as exc:
            raise ImageAPIError("Image API returned invalid base64 image data") from exc

    @staticmethod
    def _extension_from_mime(mime_type: str | None, fallback: str = "png") -> str:
        if mime_type:
            extension = mimetypes.guess_extension(mime_type.split(";", 1)[0].strip())
            if extension:
                return extension.lstrip(".").replace("jpe", "jpg")
        return fallback

    async def _image_bytes(self, item: Any) -> tuple[bytes, str]:
        if isinstance(item, str):
            item = {"url": item}
        if not isinstance(item, Mapping):
            raise ImageAPIError("Image response entries must be objects or URLs")
        for key in ("b64_json", "bytesBase64", "base64", "image_base64"):
            value = item.get(key)
            if isinstance(value, str) and value:
                return self._decode_base64(value)
        image_url = item.get("image_url")
        if isinstance(image_url, Mapping):
            image_url = image_url.get("url")
        url = item.get("url") or image_url
        if not isinstance(url, str) or not url:
            raise ImageAPIError("Image response entry did not contain url or b64_json")
        if url.startswith("data:"):
            return self._decode_base64(url)
        session = await self._ensure_session()
        try:
            async with session.get(url) as response:
                if response.status < 200 or response.status >= 300:
                    raise ImageAPIError(f"Generated image download returned HTTP {response.status}")
                content_type = response.headers.get("Content-Type", "image/png")
                return await response.read(), content_type
        except aiohttp.ClientError as exc:
            raise ImageAPIError("Could not download generated image") from exc

    @staticmethod
    def _destination(output_dir: str, file_prefix: str, index: int, extension: str) -> Path:
        destination_dir = Path(output_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)
        safe_prefix = Path(file_prefix).name or "image"
        candidate = destination_dir / f"{safe_prefix}_{index}.{extension}"
        suffix = 1
        while candidate.exists():
            candidate = destination_dir / f"{safe_prefix}_{index}_{suffix}.{extension}"
            suffix += 1
        return candidate

    async def _save_images(
        self,
        response: Mapping[str, Any],
        output_dir: str,
        file_prefix: str,
    ) -> list[str]:
        saved_paths: list[str] = []
        for index, item in enumerate(self._response_items(response)):
            image_bytes, mime_type = await self._image_bytes(item)
            extension = self._extension_from_mime(mime_type)
            path = self._destination(output_dir, file_prefix, index, extension)
            path.write_bytes(image_bytes)
            saved_paths.append(str(path))
        if not saved_paths:
            raise ImageAPIError("Image API response did not include any image files")
        return saved_paths

    async def _create_images(
        self,
        *,
        model_name: str,
        prompt: str,
        output_dir: str,
        file_prefix: str,
        input_image: Any = None,
        endpoint_kind: str | None = None,
        **kwargs: Any,
    ) -> list[str]:
        input_images = self._normalize_input_images(input_image)
        kind = endpoint_kind or ("image_to_image" if input_images else "generation")
        params = self._clean_params(
            model_name,
            prompt,
            self._request_defaults(kind),
            kwargs,
        )
        endpoint = self._endpoint(kind)
        if input_images:
            response = await self._post_multipart(endpoint, params, input_images)
        else:
            response = await self._post_json(endpoint, params)
        return await self._save_images(response, output_dir, file_prefix)


class ImageGenerationAPI(_OpenAICompatibleImageClient):
    """Generate text-to-image or image-to-image outputs through one API contract."""

    def __init__(
        self,
        config_path: str = "./config.yaml",
        debug: bool = False,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
    ) -> None:
        super().__init__(config_path, base_url=base_url, api_key=api_key, debug=debug)
        self.model_name = model_name or self.config.get("image_model_name") or self.config.get("model_name") or ""

    async def generate_image(
        self,
        model_name: str | None = None,
        prompt: str = "",
        output_dir: str = ".",
        file_prefix: str = "generated_image",
        input_image: Any = None,
        **kwargs: Any,
    ) -> list[str]:
        selected_model = model_name or self.model_name
        return await self._create_images(
            model_name=selected_model,
            prompt=prompt,
            output_dir=output_dir,
            file_prefix=file_prefix,
            input_image=input_image,
            **kwargs,
        )

    async def generate_images_batch(
        self,
        requests: list[dict[str, Any]],
        max_concurrent: int = 5,
    ) -> list[dict[str, Any]]:
        """Run independent image requests while preserving input order."""
        if max_concurrent <= 0:
            raise ValueError("max_concurrent must be positive")
        semaphore = asyncio.Semaphore(max_concurrent)
        created_session = self._session is None or self._session.closed
        await self._ensure_session()

        async def generate_one(index: int, request: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            started = time.monotonic()
            async with semaphore:
                try:
                    if not isinstance(request, dict):
                        raise TypeError("Each batch request must be a dictionary")
                    paths = await self.generate_image(**dict(request))
                    result = {
                        "request": request,
                        "success": True,
                        "paths": paths,
                        "duration": time.monotonic() - started,
                        "error": None,
                    }
                except Exception as exc:
                    result = {
                        "request": request,
                        "success": False,
                        "paths": [],
                        "duration": time.monotonic() - started,
                        "error": str(exc),
                    }
                return index, result

        try:
            pairs = await asyncio.gather(
                *(generate_one(index, request) for index, request in enumerate(requests))
            )
            results: list[dict[str, Any]] = [{} for _ in requests]
            for index, result in pairs:
                results[index] = result
            return results
        finally:
            if created_session:
                await self.aclose()

    def list_models(self) -> dict[str, str]:
        """Return optional configured model aliases without embedding provider routes."""
        configured = self.config.get("image_models", {})
        if isinstance(configured, Mapping):
            return {str(name): str(value) for name, value in configured.items()}
        if self.model_name:
            return {"default": self.model_name}
        return {}
