from __future__ import annotations

import importlib
import logging
import os
import re
from collections import Counter
from functools import lru_cache
from typing import Any, Iterable


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


def compute_task_accuracy(
    paper_task: str,
    response: Any,
    answer: Any,
    choices: Iterable[str] | None = None,
) -> float | None:
    task = str(paper_task or "")
    if task == "multiple-choice":
        return float(
            normalize_closed_form_answer(response, choices)
            == normalize_closed_form_answer(answer, choices)
        )
    if task == "blank-filling":
        return float(normalize_free_text(response) == normalize_free_text(answer))
    return None


def _tokenize(text: Any) -> list[str]:
    return WORD_RE.findall(_normalize_text_for_tokens(text))


def compute_text_em_f1(response: Any, answer: Any) -> tuple[float, float]:
    norm_response = normalize_free_text(response)
    norm_answer = normalize_free_text(answer)
    em = float(norm_response == norm_answer)

    response_tokens = _tokenize(response)
    answer_tokens = _tokenize(answer)
    if not response_tokens and not answer_tokens:
        return em, 1.0
    if not response_tokens or not answer_tokens:
        return em, 0.0

    common = Counter(response_tokens) & Counter(answer_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return em, 0.0
    precision = overlap / len(response_tokens)
    recall = overlap / len(answer_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return em, f1


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


ENTITY_PATTERNS: dict[str, tuple[str, ...]] = {
    "lungs": ("lung", "lungs", "pulmonary"),
    "heart": ("heart", "cardiac", "cardiomediastinal silhouette"),
    "mediastinum": ("mediastinum", "mediastinal contour", "cardiomediastinal silhouette"),
    "pleura": ("pleura", "pleural"),
    "pleural effusion": ("pleural effusion", "effusion"),
    "pneumothorax": ("pneumothorax",),
    "atelectasis": ("atelectasis",),
    "consolidation": ("consolidation",),
    "opacity": ("opacity", "opacification", "opacities"),
    "edema": ("edema",),
    "mass": ("mass",),
    "nodule": ("nodule", "nodules"),
    "lesion": ("lesion", "lesions"),
    "cyst": ("cyst", "cysts"),
    "fracture": ("fracture", "fractures"),
    "pancreas": ("pancreas", "pancreatic"),
    "parathyroid": ("parathyroid",),
    "thyroid": ("thyroid",),
    "kidney": ("kidney", "kidneys", "renal"),
    "diaphragm": ("diaphragm", "hemidiaphragm"),
    "endotracheal tube": ("endotracheal tube",),
    "enteric tube": ("enteric tube",),
    "chest tube": ("chest tube",),
    "catheter": ("catheter",),
    "radiograph": ("radiograph", "x-ray", "xray"),
    "ct": ("ct", "computed tomography"),
    "mri": ("mri", "magnetic resonance imaging"),
    "ultrasound": ("ultrasound", "doppler", "sonography"),
}


def extract_clinical_entities(text: Any) -> set[str]:
    norm = _normalize_text_for_tokens(text)
    entities: set[str] = set()
    for canonical, patterns in ENTITY_PATTERNS.items():
        if any(re.search(r"(?<![a-z])" + re.escape(pattern) + r"(?![a-z])", norm) for pattern in patterns):
            entities.add(canonical)
    return entities


def compute_clinical_entity_metrics(response: Any, reference: Any) -> dict[str, Any]:
    response_entities = extract_clinical_entities(response)
    reference_entities = extract_clinical_entities(reference)
    if not response_entities and not reference_entities:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "response_entities": sorted(response_entities),
            "reference_entities": sorted(reference_entities),
        }
    if not response_entities or not reference_entities:
        return {
            "precision": 0.0 if not response_entities else 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "response_entities": sorted(response_entities),
            "reference_entities": sorted(reference_entities),
        }
    overlap = len(response_entities & reference_entities)
    precision = overlap / len(response_entities) if response_entities else 0.0
    recall = overlap / len(reference_entities) if reference_entities else 0.0
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "response_entities": sorted(response_entities),
        "reference_entities": sorted(reference_entities),
    }


def _radgraph_model_type() -> str:
    return os.environ.get("MEDGEN_RADGRAPH_MODEL_TYPE", "radgraph-xl")


def _radgraph_model_cache_dir() -> str | None:
    cache_dir = os.environ.get("MEDGEN_RADGRAPH_CACHE_DIR", "").strip()
    return cache_dir or None


def _radgraph_cuda_device() -> int | None:
    raw_value = os.environ.get("MEDGEN_RADGRAPH_CUDA", "").strip()
    if not raw_value:
        return None
    try:
        return int(raw_value)
    except ValueError:
        LOGGER.warning("Ignoring invalid MEDGEN_RADGRAPH_CUDA=%r; expected an integer GPU id.", raw_value)
        return None


@lru_cache(maxsize=1)
def _get_radgraph_f1_scorer():
    try:
        radgraph_module = importlib.import_module("radgraph")
    except ImportError:
        return None

    scorer_cls = getattr(radgraph_module, "F1RadGraph", None)
    if scorer_cls is None:
        raise AttributeError("radgraph package does not expose F1RadGraph")

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
    except Exception as exc:  # pragma: no cover - defensive fallback around third-party runtime
        LOGGER.warning("Falling back to heuristic RadGraph_F1 because the radgraph scorer failed to initialize: %s", exc)
        return None

    if scorer is None:
        return None

    try:
        mean_reward, reward_list, hypothesis_annotation_lists, reference_annotation_lists = scorer(
            hyps=[response_text],
            refs=[reference_text],
        )
    except Exception as exc:  # pragma: no cover - defensive fallback around third-party runtime
        LOGGER.warning("Falling back to heuristic RadGraph_F1 because the radgraph scorer failed during inference: %s", exc)
        return None

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
    reference_text = serialize_clinical_reference(reference)
    response_text = _collapse_spaces(str(response or ""))

    if len(_tokenize(reference_text)) < 4 or len(_tokenize(response_text)) < 4:
        return {"applicable": False, "f1": None, "backend": "skipped"}

    package_result = _compute_radgraph_f1_with_package(response_text, reference_text)
    if package_result is not None:
        return package_result

    response_entities = extract_clinical_entities(response_text)
    reference_entities = extract_clinical_entities(reference_text)
    if not response_entities and not reference_entities:
        return {"applicable": False, "f1": None, "backend": "heuristic"}

    metrics = compute_clinical_entity_metrics(response_text, reference_text)
    return {"applicable": True, "f1": metrics["f1"], "backend": "heuristic"}
