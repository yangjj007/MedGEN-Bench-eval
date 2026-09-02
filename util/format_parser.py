"""Safely extract a JSON object from a model response."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

try:
    import json5
except ModuleNotFoundError:  # pragma: no cover - requirements.txt installs json5
    json5 = None


def _json_object_candidates(text: str) -> list[str]:
    """Return fenced and balanced JSON-object candidates in response order."""

    fence = chr(96) * 3
    stripped = re.sub(
        rf"^\s*{re.escape(fence)}(?:json|json5)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    stripped = re.sub(rf"\s*{re.escape(fence)}\s*$", "", stripped).strip()
    candidates = [stripped] if stripped else []

    for start, character in enumerate(stripped):
        if character != "{":
            continue
        depth = 0
        quote: str | None = None
        escaped = False
        for end in range(start, len(stripped)):
            current = stripped[end]
            if quote is not None:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == quote:
                    quote = None
                continue
            if current in {"'", '"'}:
                quote = current
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(stripped[start : end + 1])
                    break
    return candidates


def extract_json(response: Any) -> dict[str, Any]:
    """Parse one JSON object from an OpenAI-compatible model response.

    The parser accepts a mapping directly, fenced JSON, surrounding prose, and
    JSON5-style quoting emitted by some local VLMs. It never evaluates model
    output as Python code.
    """

    if isinstance(response, Mapping):
        return dict(response)
    if not isinstance(response, str):
        raise TypeError(
            "Model response must be a string or mapping; "
            f"received {type(response).__name__}"
        )

    errors: list[str] = []
    seen: set[str] = set()
    for candidate in _json_object_candidates(response):
        if candidate in seen:
            continue
        seen.add(candidate)
        parsers = [json.loads]
        if json5 is not None:
            parsers.append(json5.loads)
        for parser in parsers:
            try:
                parsed = parser(candidate)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                errors.append(str(exc))
                continue
            if isinstance(parsed, Mapping):
                return dict(parsed)
            errors.append("parsed value is not a JSON object")
    detail = errors[-1] if errors else "no JSON object found"
    raise ValueError(f"Could not parse a JSON object from model output: {detail}")
