from __future__ import annotations

import importlib
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MODEL_ROOT = _PROJECT_ROOT / "models"


SECTION_ORDER = [
    "STUDY",
    "EXAMINATION",
    "MODALITY",
    "CLINICAL_HISTORY",
    "INDICATION",
    "HISTORY",
    "TECHNIQUE",
    "COMPARISON",
    "FINDINGS",
    "IMPRESSION",
    "RECOMMENDATION",
    "NOTIFICATION",
]

OPTION_RE = re.compile(r"^\s*([A-Z])[\.)\]:-]\s*(.*)$")
OPTION_LETTER_ONLY_RE = re.compile(r"^\s*([A-Z])\s*$")
FINAL_ANSWER_RE = re.compile(r"(?:final\s+answer|answer|option|choice)\s*(?:is|:|=|-)?\s*\(?([A-Z])\)?(?:\.|\)|\s|$)", re.I)
WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
SPACE_RE = re.compile(r"\s+")
LOGGER = logging.getLogger(__name__)


def _collapse_spaces(text: str) -> str:
    return SPACE_RE.sub(" ", text.strip())


def normalize_free_text(text: Any) -> str:
    return _collapse_spaces(str(text or "")).lower()


def _normalize_text_for_tokens(text: Any) -> str:
    return normalize_free_text(text).replace("\n", " ")


def _normalize_choice_value(text: str) -> str:
    text = normalize_free_text(text)
    letter_only = OPTION_LETTER_ONLY_RE.match(text.upper())
    if letter_only:
        return letter_only.group(1).lower()
    match = OPTION_RE.match(text.upper())
    if match:
        letter = match.group(1).lower()
        body = normalize_free_text(match.group(2))
        if body:
            return f"{letter}. {body}"
        return letter
    return text


def _choice_mappings(choices: Iterable[str] | None) -> dict[str, str]:
    mappings: dict[str, str] = {}
    for raw_choice in choices or []:
        canonical = _normalize_choice_value(raw_choice)
        mappings[canonical] = canonical
        match = OPTION_RE.match(normalize_free_text(raw_choice).upper())
        if match:
            letter = match.group(1).lower()
            body = normalize_free_text(match.group(2))
            mappings[letter] = canonical
            mappings[f"{letter}."] = canonical
            if body:
                mappings[body] = canonical
    return mappings


def normalize_closed_form_answer(text: Any, choices: Iterable[str] | None = None) -> str:
    text = str(text or "")
    mappings = _choice_mappings(choices)
    normalized = _normalize_choice_value(text)
    return mappings.get(normalized, normalized)


def parse_closed_form_answer(text: Any, choices: Iterable[str] | None = None,
                            task: str = "multiple-choice") -> dict[str, Any]:
    """Extract a closed-form answer without treating reasoning as the answer.

    ``NA`` is deliberately distinct from an incorrect answer.  For multiple
    choice, the final option letter or an exact option body is accepted.  For
    blank filling, the final labelled line (or short final line) is used.
    """
    raw = str(text or "").strip()
    if not raw:
        return {"parse_status": "NA", "parsed_answer": None, "parse_failure_reason": "empty"}
    task = str(task or "").lower()
    if task == "multiple-choice":
        mappings = _choice_mappings(choices)
        # Prefer an explicitly labelled final answer, scanning from the end so
        # explanations containing option letters do not win.
        for line in reversed([x.strip() for x in raw.splitlines() if x.strip()]):
            m = FINAL_ANSWER_RE.search(line)
            if m:
                value = mappings.get(m.group(1).lower(), m.group(1).lower())
                return {"parse_status": "parsed", "parsed_answer": value, "parse_failure_reason": None}
            value = normalize_closed_form_answer(line, choices)
            if value in mappings:
                return {"parse_status": "parsed", "parsed_answer": value, "parse_failure_reason": None}
        # A standalone option token anywhere is safe only when it is the
        # complete final sentence.
        for m in reversed(list(re.finditer(r"\b([A-Z])\b", raw.upper()))):
            letter = m.group(1).lower()
            if letter in mappings:
                return {"parse_status": "parsed", "parsed_answer": mappings[letter], "parse_failure_reason": None}
        # Match an option body as a phrase, longest first.
        for key in sorted((k for k in mappings if len(k) > 1), key=len, reverse=True):
            if normalize_free_text(key) in normalize_free_text(raw):
                return {"parse_status": "parsed", "parsed_answer": mappings[key], "parse_failure_reason": None}
        return {"parse_status": "NA", "parsed_answer": None, "parse_failure_reason": "no_option_detected"}
    # Blank filling: remove a final-answer prefix and punctuation, then keep a
    # concise final phrase rather than the preceding chain-of-thought.
    lines = [x.strip() for x in raw.splitlines() if x.strip()]
    candidate = lines[-1] if lines else raw
    candidate = re.sub(r"^(?:(?:final\s+)?(?:answer|response|blank)|the\s+blank)\s*(?:is|:|=|-)?\s*", "", candidate, flags=re.I)
    candidate = re.sub(r"^[\s\[\]().,:;-]+|[\s\[\]().,:;-]+$", "", candidate)
    if not candidate or len(candidate.split()) > 20:
        return {"parse_status": "NA", "parsed_answer": None, "parse_failure_reason": "no_short_answer"}
    return {"parse_status": "parsed", "parsed_answer": normalize_free_text(candidate), "parse_failure_reason": None}


def compute_closed_form_exact_match(response: Any, answer: Any,
                                   choices: Iterable[str] | None = None,
                                   task: str = "multiple-choice") -> dict[str, Any]:
    expected = parse_closed_form_answer(answer, choices, task)
    observed = parse_closed_form_answer(response, choices, task)
    if observed["parse_status"] != "parsed" or expected["parse_status"] != "parsed":
        return {"score": None, "parse_status": observed["parse_status"],
                "parsed_answer": observed["parsed_answer"],
                "expected_answer": expected["parsed_answer"],
                "parse_failure_reason": observed["parse_failure_reason"] or "reference_unparsed"}
    return {"score": float(observed["parsed_answer"] == expected["parsed_answer"]),
            "parse_status": "parsed", "parsed_answer": observed["parsed_answer"],
            "expected_answer": expected["parsed_answer"], "parse_failure_reason": None}


def _tokenize(text: Any) -> list[str]:
    return WORD_RE.findall(_normalize_text_for_tokens(text))


def compute_text_exact_match(response: Any, answer: Any) -> float:
    norm_response = normalize_free_text(response)
    norm_answer = normalize_free_text(answer)
    return float(norm_response == norm_answer)


def _serialize_dict_section(key: str, value: Any, lines: list[str], indent: int = 0) -> None:
    prefix = "  " * indent
    if isinstance(value, dict):
        lines.append(f"{prefix}{key}:")
        ordered_keys = sorted(
            value.keys(),
            key=lambda item: (
                SECTION_ORDER.index(item.upper()) if item.upper() in SECTION_ORDER else len(SECTION_ORDER),
                item.lower(),
            ),
        )
        for child_key in ordered_keys:
            _serialize_dict_section(str(child_key), value[child_key], lines, indent + 1)
        return
    if isinstance(value, list):
        if value:
            lines.append(f"{prefix}{key}: " + "; ".join(_collapse_spaces(str(v)) for v in value))
        return
    scalar = _collapse_spaces(str(value))
    if scalar:
        lines.append(f"{prefix}{key}: {scalar}")


def serialize_clinical_reference(answer: Any) -> str:
    if isinstance(answer, str):
        return _collapse_spaces(answer)
    if not isinstance(answer, dict):
        return _collapse_spaces(str(answer))

    lines: list[str] = []
    ordered_keys = sorted(
        answer.keys(),
        key=lambda item: (
            SECTION_ORDER.index(item.upper()) if item.upper() in SECTION_ORDER else len(SECTION_ORDER),
            item.lower(),
        ),
    )
    for key in ordered_keys:
        _serialize_dict_section(str(key), answer[key], lines)
    return "\n".join(lines)


def _radgraph_model_type() -> str:
    return os.environ.get("MEDGEN_RADGRAPH_MODEL_TYPE", "radgraph-xl")


def _radgraph_model_cache_dir() -> str | None:
    cache_dir = os.environ.get("MEDGEN_RADGRAPH_CACHE_DIR", "").strip()
    return cache_dir or str(_MODEL_ROOT / "RadGraph")


def _radgraph_cuda_device() -> int | None:
    raw_value = os.environ.get("MEDGEN_RADGRAPH_CUDA", "").strip()
    if not raw_value:
        return None
    try:
        return int(raw_value)
    except ValueError:
        LOGGER.warning("Ignoring invalid MEDGEN_RADGRAPH_CUDA=%r; expected an integer GPU id.", raw_value)
        return None


def _radgraph_tokenizer_cache_dir() -> str | None:
    cache_dir = os.environ.get("MEDGEN_RADGRAPH_TOKENIZER_CACHE_DIR", "").strip()
    return cache_dir or str(_MODEL_ROOT / "HuggingFace")


@lru_cache(maxsize=1)
def _get_radgraph_f1_scorer():
    try:
        radgraph_module = importlib.import_module("radgraph")
    except ImportError as exc:
        raise RuntimeError("RadGraph is required; install requirements.txt") from exc

    scorer_cls = getattr(radgraph_module, "F1RadGraph", None)
    if scorer_cls is None:
        raise RuntimeError("Installed radgraph package does not expose F1RadGraph")

    scorer_kwargs: dict[str, Any] = {
        "reward_level": "all",
        "model_type": _radgraph_model_type(),
    }
    model_cache_dir = _radgraph_model_cache_dir()
    if model_cache_dir:
        scorer_kwargs["model_cache_dir"] = model_cache_dir

    cuda_device = _radgraph_cuda_device()
    if cuda_device is not None:
        scorer_kwargs["cuda"] = cuda_device

    tokenizer_cache_dir = _radgraph_tokenizer_cache_dir()
    if tokenizer_cache_dir:
        scorer_kwargs["tokenizer_cache_dir"] = tokenizer_cache_dir

    return scorer_cls(**scorer_kwargs)


def _annotation_has_entities(annotation: Any) -> bool:
    return bool(isinstance(annotation, dict) and annotation.get("entities"))


def _extract_rg_er_score(mean_reward: Any, reward_list: Any) -> float:
    if isinstance(mean_reward, (list, tuple)) and len(mean_reward) >= 2:
        return float(mean_reward[1])
    if isinstance(reward_list, (list, tuple)) and len(reward_list) >= 2:
        rg_er_values = reward_list[1]
        if isinstance(rg_er_values, (list, tuple)) and rg_er_values:
            return float(rg_er_values[0])
        return float(rg_er_values)
    return float(mean_reward)


def _compute_radgraph_f1_with_package(response_text: str, reference_text: str) -> dict[str, Any] | None:
    try:
        scorer = _get_radgraph_f1_scorer()
    except Exception as exc:  # pragma: no cover - third-party initialization
        raise RuntimeError("RadGraph scorer failed to initialize") from exc

    try:
        mean_reward, reward_list, hypothesis_annotation_lists, reference_annotation_lists = scorer(
            hyps=[response_text],
            refs=[reference_text],
        )
    except Exception as exc:  # pragma: no cover - third-party inference
        raise RuntimeError("RadGraph scorer failed during inference") from exc

    hypothesis_annotation = hypothesis_annotation_lists[0] if hypothesis_annotation_lists else {}
    reference_annotation = reference_annotation_lists[0] if reference_annotation_lists else {}
    if not _annotation_has_entities(hypothesis_annotation) and not _annotation_has_entities(reference_annotation):
        return {"applicable": False, "f1": None, "backend": "radgraph"}

    return {
        "applicable": True,
        "f1": _extract_rg_er_score(mean_reward, reward_list),
        "backend": "radgraph",
    }


def compute_radgraph_f1(response: Any, reference: Any) -> dict[str, Any]:
    # Initialize the real scorer before checking applicability.  Even a short
    # answer must not make a run appear successful when RadGraph is absent.
    _get_radgraph_f1_scorer()
    reference_text = serialize_clinical_reference(reference)
    response_text = _collapse_spaces(str(response or ""))

    if len(_tokenize(reference_text)) < 4 or len(_tokenize(response_text)) < 4:
        return {"applicable": False, "f1": None, "backend": "skipped"}

    return _compute_radgraph_f1_with_package(response_text, reference_text)


def compute_radgraph_f1_batch(
    responses: list[Any], references: list[Any]
) -> list[dict[str, Any]]:
    """Compute per-sample RG_ER scores with one RadGraph model call.

    ``F1RadGraph`` natively accepts lists, but the historical evaluator called
    it once per record.  Batching keeps exactly the same RG_ER reward while
    avoiding repeated tokenizer/model launches between records.
    """
    if len(responses) != len(references):
        raise ValueError("responses and references must have equal length")

    # Preserve the fail-loud behavior even when every item is too short for a
    # meaningful entity graph.
    scorer = _get_radgraph_f1_scorer()
    results = [
        {"applicable": False, "f1": None, "backend": "skipped"}
        for _ in responses
    ]
    eligible_indices: list[int] = []
    eligible_hyps: list[str] = []
    eligible_refs: list[str] = []
    for index, (response, reference) in enumerate(zip(responses, references)):
        response_text = _collapse_spaces(str(response or ""))
        reference_text = serialize_clinical_reference(reference)
        if len(_tokenize(reference_text)) < 4 or len(_tokenize(response_text)) < 4:
            continue
        eligible_indices.append(index)
        eligible_hyps.append(response_text)
        eligible_refs.append(reference_text)

    if not eligible_indices:
        return results

    try:
        _, reward_list, hypothesis_annotations, reference_annotations = scorer(
            hyps=eligible_hyps,
            refs=eligible_refs,
        )
    except Exception as exc:  # pragma: no cover - third-party inference
        raise RuntimeError("RadGraph scorer failed during batched inference") from exc

    if not isinstance(reward_list, (list, tuple)) or len(reward_list) < 2:
        raise RuntimeError("RadGraph scorer returned an invalid reward structure")
    rg_er_scores = reward_list[1]
    if not (
        len(rg_er_scores) == len(eligible_indices)
        and len(hypothesis_annotations) == len(eligible_indices)
        and len(reference_annotations) == len(eligible_indices)
    ):
        raise RuntimeError("RadGraph scorer returned an unexpected batch length")

    for local_index, original_index in enumerate(eligible_indices):
        hypothesis_annotation = hypothesis_annotations[local_index]
        reference_annotation = reference_annotations[local_index]
        if not _annotation_has_entities(hypothesis_annotation) and not _annotation_has_entities(reference_annotation):
            results[original_index] = {
                "applicable": False,
                "f1": None,
                "backend": "radgraph",
            }
            continue
        results[original_index] = {
            "applicable": True,
            "f1": float(rg_er_scores[local_index]),
            "backend": "radgraph",
        }
    return results
