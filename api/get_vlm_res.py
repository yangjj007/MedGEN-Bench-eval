"""OpenAI-compatible vision-language model clients.

The clients in this module work with hosted OpenAI-compatible gateways and
local vLLM servers. They use one configuration format and one response shape.
"""

from __future__ import annotations

import asyncio
import base64
import functools
import io
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import urlparse

import yaml
from openai import AsyncOpenAI, OpenAI
from PIL import Image, ImageDraw, ImageFont


DEFAULT_CONFIG_PATH = "./config.yaml"
_PLACEHOLDER_KEYS = {
    "",
    "YOUR_API_KEY",
    "YOUR-API-KEY",
    "sk-your-api-key-here",
}
_UNRESOLVED_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _expand_config_value(value: Any, field_name: str, config_path: str) -> Any:
    """Expand shell-style environment variables in a YAML scalar."""

    if not isinstance(value, str):
        return value
    expanded = os.path.expandvars(value).strip()
    unresolved = _UNRESOLVED_ENV_PATTERN.findall(expanded)
    if unresolved:
        names = ", ".join(sorted(set(unresolved)))
        raise ValueError(
            f"Configuration field '{field_name}' in {config_path} references "
            f"an unset environment variable: {names}. Set it before running."
        )
    return expanded


def _read_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"VLM configuration file does not exist: {path}. "
            "Create config.yaml in the repository root or pass config_path."
        )
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in VLM configuration: {path}") from exc
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"VLM configuration must be a mapping: {path}")
    return dict(config)


def _is_local_url(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def _coerce_positive_int(value: Any, name: str, default: int) -> int:
    if value is None:
        return default
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < 1:
        raise ValueError(f"{name} must be at least one")
    return result


def _coerce_nonnegative_float(value: Any, name: str, default: float) -> float:
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if result < 0:
        raise ValueError(f"{name} must not be negative")
    return result


async def _to_thread(function: Any, *args: Any) -> Any:
    """Use asyncio.to_thread when available, with a Python 3.8 fallback."""

    to_thread = getattr(asyncio, "to_thread", None)
    if to_thread is not None:
        return await to_thread(function, *args)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(function, *args))


def _open_rgb_image(image_input: Any) -> Image.Image:
    if isinstance(image_input, Image.Image):
        return image_input.copy().convert("RGB")
    if isinstance(image_input, (str, os.PathLike)):
        path = Path(image_input)
        if not path.is_file():
            raise FileNotFoundError(f"Image file does not exist: {path}")
        with Image.open(path) as image:
            return image.convert("RGB")
    if isinstance(image_input, (bytes, bytearray, memoryview)):
        with Image.open(io.BytesIO(bytes(image_input))) as image:
            return image.convert("RGB")
    raise TypeError(
        "image_input must be a path, a PIL Image, or raw image bytes; "
        f"received {type(image_input).__name__}"
    )


def _load_font(size: int) -> ImageFont.ImageFont:
    return ImageFont.load_default()


def add_text_to_image(
    image_input: Any,
    text: str,
    height_ratio: float = 0.1,
    font_size: Optional[int] = None,
) -> Image.Image:
    """Return an RGB copy of an image with an optional label beneath it."""

    image = _open_rgb_image(image_input)
    label = str(text or "").strip()
    if not label:
        return image

    try:
        ratio = float(height_ratio)
    except (TypeError, ValueError) as exc:
        raise ValueError("height_ratio must be numeric") from exc
    if ratio <= 0:
        raise ValueError("height_ratio must be positive")

    label_height = max(28, int(image.height * ratio))
    canvas = Image.new("RGB", (image.width, image.height + label_height), "white")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    chosen_size = font_size or max(14, min(label_height - 8, image.width // 18))
    font = _load_font(chosen_size)
    try:
        bbox = draw.textbbox((0, 0), label, font=font)
        label_width = bbox[2] - bbox[0]
        label_text_height = bbox[3] - bbox[1]
    except AttributeError:
        label_width, label_text_height = draw.textsize(label, font=font)
    x = max(0, (image.width - label_width) // 2)
    y = image.height + max(0, (label_height - label_text_height) // 2)
    draw.text((x, y), label, fill="black", font=font)
    return canvas


class _OpenAICompatibleVLM:
    """Shared client implementation for single- and two-image requests."""

    def __init__(
        self,
        config_path: str = DEFAULT_CONFIG_PATH,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.config_path = str(config_path)
        self.config = _read_config(self.config_path)

        if base_url is not None:
            selected_base_url = base_url
        elif os.environ.get("MEDGEN_VLM_BASE_URL"):
            selected_base_url = os.environ["MEDGEN_VLM_BASE_URL"]
        else:
            selected_base_url = _expand_config_value(
                self.config.get("base_url", ""),
                "base_url",
                self.config_path,
            )
        self.base_url = str(selected_base_url or "").strip().rstrip("/")
        if not self.base_url:
            raise ValueError(
                "A non-empty base_url is required. Set it in the YAML config, "
                "pass base_url, or set MEDGEN_VLM_BASE_URL."
            )

        if model_name is not None and str(model_name).strip():
            selected_model = model_name
        elif os.environ.get("MEDGEN_VLM_MODEL"):
            selected_model = os.environ["MEDGEN_VLM_MODEL"]
        else:
            selected_model = _expand_config_value(
                self.config.get("model_name", ""),
                "model_name",
                self.config_path,
            )
        self.model_name = str(selected_model or "").strip()
        if not self.model_name:
            raise ValueError(
                "A non-empty model_name is required. Set it in the YAML config, "
                "pass model_name, or set MEDGEN_VLM_MODEL."
            )

        if api_key is not None and str(api_key).strip():
            selected_key = api_key
        elif os.environ.get("MEDGEN_VLM_API_KEY"):
            selected_key = os.environ["MEDGEN_VLM_API_KEY"]
        else:
            selected_key = _expand_config_value(
                self.config.get("api_key", ""),
                "api_key",
                self.config_path,
            ) or os.environ.get("AIHUBMIX_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.api_key = str(selected_key or "").strip()
        if self.api_key in _PLACEHOLDER_KEYS:
            if _is_local_url(self.base_url):
                self.api_key = "EMPTY"
            else:
                raise ValueError(
                    "A non-placeholder api_key is required for a non-local VLM endpoint. "
                    "Set it in the YAML config, pass api_key, or export MEDGEN_VLM_API_KEY."
                )

        self.temperature = float(self.config.get("temperature", 0.3))
        self.max_retries = _coerce_positive_int(
            self.config.get("max_retries"), "max_retries", 3
        )
        self.retry_delay = _coerce_nonnegative_float(
            self.config.get("retry_delay"), "retry_delay", 1.0
        )
        self.request_timeout = _coerce_positive_int(
            self.config.get("request_timeout", self.config.get("timeout")),
            "request_timeout",
            300,
        )
        response_format = self.config.get("response_format")
        self.response_format = response_format if isinstance(response_format, Mapping) else None

        self.site_url = str(self.config.get("site_url") or "").strip()
        self.site_name = str(self.config.get("site_name") or "").strip()
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.request_timeout,
        )
        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.request_timeout,
        )

    @staticmethod
    def guess_mime_type(image_path: str) -> str:
        """Return a safe image MIME type for a local file path."""

        mime, _ = mimetypes.guess_type(str(image_path))
        return mime if mime and mime.startswith("image/") else "image/png"

    @staticmethod
    def encode_image(image_input: Any) -> str:
        """Encode a local image, PIL image, or byte sequence as base64."""

        if isinstance(image_input, (str, os.PathLike)):
            path = Path(image_input)
            if not path.is_file():
                raise FileNotFoundError(f"Image file does not exist: {path}")
            return base64.b64encode(path.read_bytes()).decode("ascii")
        if isinstance(image_input, (bytes, bytearray, memoryview)):
            return base64.b64encode(bytes(image_input)).decode("ascii")
        if isinstance(image_input, Image.Image):
            buffer = io.BytesIO()
            image_input.convert("RGB").save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("ascii")
        raise TypeError(
            "image_input must be a path, a PIL Image, or raw image bytes; "
            f"received {type(image_input).__name__}"
        )

    @classmethod
    def _image_data_uri(cls, image_input: Any) -> str:
        if isinstance(image_input, str) and image_input.startswith("data:image/"):
            return image_input
        mime = (
            cls.guess_mime_type(str(image_input))
            if isinstance(image_input, (str, os.PathLike))
            else "image/png"
        )
        return f"data:{mime};base64,{cls.encode_image(image_input)}"

    def _extra_headers(self) -> Optional[dict[str, str]]:
        headers: dict[str, str] = {}
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name
        return headers or None

    @staticmethod
    def _field(value: Any, field_name: str, default: Any = None) -> Any:
        if isinstance(value, Mapping):
            return value.get(field_name, default)
        return getattr(value, field_name, default)

    @classmethod
    def _parse_response(cls, response: Any) -> dict[str, Any]:
        """Normalize OpenAI-compatible completion responses."""

        choices = cls._field(response, "choices", []) or []
        if not choices:
            raise ValueError("VLM response has no choices")
        message = cls._field(choices[0], "message")
        if message is None:
            raise ValueError("VLM response has no message")
        content = cls._field(message, "content", "")

        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    text_parts.append(part)
                    continue
                part_type = cls._field(part, "type")
                part_text = cls._field(part, "text")
                if part_type in {None, "text", "output_text"} and part_text is not None:
                    text_parts.append(str(part_text))
            text = "".join(text_parts).strip()
        else:
            text = "" if content is None else str(content).strip()

        usage_source = cls._field(response, "usage")
        usage = {
            key: cls._field(usage_source, key)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if cls._field(usage_source, key) is not None
        }
        return {"text": text, "usage": usage, "raw": response}

    async def _complete(
        self,
        content: list[dict[str, Any]],
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> dict[str, Any]:
        request_temperature = self.temperature if temperature is None else float(temperature)
        request_max_tokens = (
            int(max_tokens)
            if max_tokens is not None
            else int(os.environ.get("MEDGEN_JUDGE_MAX_TOKENS", "768"))
        )
        if request_max_tokens < 1:
            raise ValueError("max_tokens must be at least one")
        request_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "temperature": request_temperature,
            "max_tokens": request_max_tokens,
            "stream": False,
        }
        headers = self._extra_headers()
        if headers:
            request_kwargs["extra_headers"] = headers
        if self.response_format:
            request_kwargs["response_format"] = dict(self.response_format)

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                started = time.perf_counter()
                response = await self.async_client.chat.completions.create(**request_kwargs)
                result = self._parse_response(response)
                result["_perf_http_seconds"] = time.perf_counter() - started
                return result
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** (attempt - 1)))
        assert last_error is not None
        return {
            "error": f"{type(last_error).__name__}: {last_error}",
            "text": "",
            "usage": {},
        }

    async def aclose(self) -> None:
        """Close the asynchronous HTTP client when the caller owns its lifetime."""

        await self.async_client.close()
        self.client.close()


class double_image_vlm(_OpenAICompatibleVLM):
    """OpenAI-compatible VLM client for reference-aware image judging."""

    def _prepare_content_parts(
        self,
        prompt: str,
        input_image_path: Any,
        output_image_path: Any,
        input_image_lable: str,
        output_image_lable: str,
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "text", "text": str(prompt)}]
        if input_image_path:
            labeled_input = add_text_to_image(input_image_path, input_image_lable)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_data_uri(labeled_input)},
                }
            )
        if output_image_path:
            labeled_output = add_text_to_image(output_image_path, output_image_lable)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_data_uri(labeled_output)},
                }
            )
        return content

    async def generate_with_image_async(
        self,
        prompt: str,
        input_image_path: Any = "",
        output_image_path: Any = "",
        input_image_lable: str = "Input",
        output_image_lable: str = "Output",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        """Submit a prompt with zero, one, or two labeled images."""

        try:
            prepare_started = time.perf_counter()
            content = await _to_thread(
                self._prepare_content_parts,
                prompt,
                input_image_path,
                output_image_path,
                input_image_lable,
                output_image_lable,
            )
            result = await self._complete(content, temperature, max_tokens)
            result["_perf_prepare_seconds"] = time.perf_counter() - prepare_started
            return result
        except Exception as exc:
            return {
                "error": f"{type(exc).__name__}: {exc}",
                "text": "",
                "usage": {},
            }

    @staticmethod
    def _normalize_request(request: Any) -> tuple[Any, Any, Any, str, str, Any, Any]:
        if isinstance(request, Mapping):
            return (
                request.get("prompt", ""),
                request.get("input_image_path", request.get("input_image", "")),
                request.get("output_image_path", request.get("output_image", "")),
                request.get("input_image_lable", request.get("input_image_label", "Input")),
                request.get("output_image_lable", request.get("output_image_label", "Output")),
                request.get("temperature"),
                request.get("max_tokens"),
            )
        values = tuple(request)
        if len(values) == 7:
            return values  # type: ignore[return-value]
        if len(values) == 4:
            prompt, input_image_path, temperature, max_tokens = values
            return prompt, input_image_path, "", "Input", "Output", temperature, max_tokens
        if len(values) == 3:
            prompt, input_image_path, output_image_path = values
            return prompt, input_image_path, output_image_path, "Input", "Output", None, None
        raise ValueError(
            "A double-image request must be a mapping, a 7-item tuple, "
            "or a legacy 3/4-item tuple"
        )

    async def generate_batch(
        self,
        requests: Sequence[Any],
        concurrency: int = 8,
    ) -> list[dict[str, Any]]:
        """Run requests concurrently and preserve the input order."""

        limit = _coerce_positive_int(concurrency, "concurrency", 8)
        semaphore = asyncio.Semaphore(limit)

        async def run_one(request: Any) -> dict[str, Any]:
            try:
                normalized = self._normalize_request(request)
                async with semaphore:
                    return await self.generate_with_image_async(*normalized)
            except Exception as exc:
                return {
                    "error": f"{type(exc).__name__}: {exc}",
                    "text": "",
                    "usage": {},
                }

        return list(await asyncio.gather(*(run_one(request) for request in requests)))


class single_image_vlm(_OpenAICompatibleVLM):
    """OpenAI-compatible VLM client for inference-time single-image prompts."""

    async def generate_with_image_async(
        self,
        prompt: str,
        image_path: Any,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        """Submit a prompt and one image using an OpenAI image data URI."""

        try:
            prepare_started = time.perf_counter()
            image_uri = await _to_thread(self._image_data_uri, image_path)
            content = [
                {"type": "text", "text": str(prompt)},
                {"type": "image_url", "image_url": {"url": image_uri}},
            ]
            result = await self._complete(content, temperature, max_tokens)
            result["_perf_prepare_seconds"] = time.perf_counter() - prepare_started
            return result
        except Exception as exc:
            return {
                "error": f"{type(exc).__name__}: {exc}",
                "text": "",
                "usage": {},
            }

    @staticmethod
    def _normalize_request(request: Any) -> tuple[Any, Any, Any, Any]:
        if isinstance(request, Mapping):
            return (
                request.get("prompt", ""),
                request.get("image_path", request.get("input_image", "")),
                request.get("temperature"),
                request.get("max_tokens"),
            )
        values = tuple(request)
        if len(values) == 4:
            return values  # type: ignore[return-value]
        if len(values) == 2:
            prompt, image_path = values
            return prompt, image_path, None, None
        raise ValueError("A single-image request must be a mapping, 2-item, or 4-item tuple")

    async def generate_batch(
        self,
        requests: Sequence[Any],
        concurrency: int = 8,
    ) -> list[dict[str, Any]]:
        """Run single-image requests concurrently and preserve input ordering."""

        limit = _coerce_positive_int(concurrency, "concurrency", 8)
        semaphore = asyncio.Semaphore(limit)

        async def run_one(request: Any) -> dict[str, Any]:
            try:
                normalized = self._normalize_request(request)
                async with semaphore:
                    return await self.generate_with_image_async(*normalized)
            except Exception as exc:
                return {
                    "error": f"{type(exc).__name__}: {exc}",
                    "text": "",
                    "usage": {},
                }

        return list(await asyncio.gather(*(run_one(request) for request in requests)))


__all__ = ["add_text_to_image", "double_image_vlm", "single_image_vlm"]
