from __future__ import annotations

import argparse
import json
import os
import asyncio
import time
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import logging

from util.prompt import (
    vlm_holistic_judge_w_gt_prompt,
    vlm_holistic_judge_wo_gt_prompt,
    build_vlm_judge_prompt,
)
from util.format_parser import extract_json
from util.clinical_text_metrics import (
    serialize_clinical_reference,
    extract_choices_from_instruction,
    normalize_closed_form_answer,
    compute_closed_form_exact_match,
    compute_text_exact_match,
    compute_radgraph_f1,
    compute_radgraph_f1_batch,
)

# Dataset/eval-input validation should not require model clients or heavyweight
# metric packages.  Real evaluation still fails clearly if they are missing.
EVAL_DEPENDENCY_ERROR = None
try:
    from util.metrics import (
        batch_async_evaluate_text_quality,
        batch_async_FR_IQA,
        batch_async_medimageinsight_metrics,
    )
    from api.get_vlm_res import double_image_vlm
except ModuleNotFoundError as exc:
    EVAL_DEPENDENCY_ERROR = exc
    batch_async_evaluate_text_quality = None
    batch_async_FR_IQA = None
    batch_async_medimageinsight_metrics = None
    double_image_vlm = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
if os.environ.get('MEDGEN_QUIET_HTTP', '0') == '1':
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)

METRIC_THRESHOLDS = {
    'MedImageInsight_Similarity': {'lower_is_better': False, 'threshold': 0.7},
    'LPIPS': {'lower_is_better': True, 'threshold': 0.6},
    'PSNR': {'lower_is_better': False, 'threshold': 28.0},
    'SSIM': {'lower_is_better': False, 'threshold': 0.7},
    'BLEU': {'lower_is_better': False, 'threshold': 0.09},
    'BERT_Score': {'lower_is_better': False, 'threshold': 0.9},
    'VLM_Overall_Score_WO_GT': {'lower_is_better': False, 'threshold': 8.0},
    'VLM_Overall_Score_W_GT': {'lower_is_better': False, 'threshold': 8.0},
}

LOCAL_METRICS_VERSION = 'medimageinsight-radgraph-v3'

# Exact string matching is a transparent record for closed-form VQA answers,
# not a proxy for factual quality in free-text reports or descriptions.
CLOSED_FORM_TEXT_EM_TASKS = {'multiple-choice', 'blank-filling'}

JUDGE_DIMENSIONS = {
    'anatomical_accuracy': 'VLM_Anatomical_Accuracy',
    'clinical_finding_accuracy': 'VLM_Clinical_Finding_Accuracy',
    'instruction_compliance': 'VLM_Instruction_Compliance',
    'cross_modal_consistency': 'VLM_Cross_Modal_Consistency',
    'hallucination_omission_control': 'VLM_Hallucination_Omission_Control',
}

# The benchmark taxonomy contains exactly these 16 tasks. Keep the list in one
# place so a renamed task cannot silently fall through to a generic image metric
# or an unconditioned judge prompt.
SUPPORTED_PAPER_TASKS = {
    'multiple-choice', 'blank-filling', 'report-generation', 'question-answering',
    'style-transfer', 'artifact-removal', 'noise-reconstruction',
    'resolution-editing', 'contrast-enhancement', 'anatomical-annotation',
    'disease-prediction', 'instruction-editing', 'organic-removal',
    'organic-reconstruction', '3d-to-2d-projection', '2d-to-3d-reconstruction',
}

PAPER_TASK_ALIASES = {
    # Historical baseline JSONL names.
    'multi-choice': 'multiple-choice',
    'question-answer': 'question-answering',
    'resolution-edit': 'resolution-editing',
    'instruction-edit': 'instruction-editing',
    'dye-transfer': 'style-transfer',
    '3d-2d-projection': '3d-to-2d-projection',
    '2d-3d-reconstruction': '2d-to-3d-reconstruction',
    'organ-removal': 'organic-removal',
    'organ-reconstruction': 'organic-reconstruction',
}

GENERIC_PAPER_TASK_NAMES = {
    'image-edit',
    'image-editing',
    'multimodal-generation',
    'generate',
    'generation',
}


def normalize_paper_task(value: object) -> str:
    task_name = str(value or '').strip().lower().replace('_', '-')
    return PAPER_TASK_ALIASES.get(task_name, task_name)


def paper_task_for_item(item: dict) -> str:
    """Resolve canonical task names from current and historical result JSONL.

    Older baseline exports sometimes stored a generic mission name in
    ``paper_task`` and the actual benchmark task in ``sub-category``. Prefer
    that specific sub-category in this case.  Historical names such as
    ``dye-transfer`` are aliases of the canonical 16-task taxonomy.
    """
    raw_task = item.get('paper_task')
    raw_subtask = item.get('sub-category') or item.get('paper_subtask')
    normalized_task = normalize_paper_task(raw_task)
    if raw_subtask and (
        not normalized_task or normalized_task in GENERIC_PAPER_TASK_NAMES
    ):
        return normalize_paper_task(raw_subtask)
    return normalized_task or normalize_paper_task(raw_subtask)


def validate_paper_task(value: object) -> str:
    task_name = normalize_paper_task(value)
    if task_name not in SUPPORTED_PAPER_TASKS:
        raise ValueError(
            f"Unknown paper_task={value!r}; expected one of the 16 benchmark tasks: "
            + ', '.join(sorted(SUPPORTED_PAPER_TASKS))
        )
    return task_name

# Bootstrap resampling count (overridable via environment variable; used only for statistics).
BOOTSTRAP_SAMPLES = max(100, int(os.environ.get('MEDGEN_BOOTSTRAP_SAMPLES', '1000')))

def load_jsonl_data(jsonl_path: str) -> list:
    """Load data from a JSONL file"""
    if not os.path.exists(jsonl_path):
        logging.error(f"File not found: {jsonl_path}")
        return []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def resolve_image_path(data_path: str, image_ref: str) -> str:
    """Resolve absolute, cwd-relative, and dataset-relative image paths."""
    if not isinstance(image_ref, str) or not image_ref.strip():
        return ""
    if os.path.isabs(image_ref):
        return image_ref
    if os.path.isfile(image_ref):
        return os.path.abspath(image_ref)
    return os.path.abspath(os.path.join(data_path, image_ref))


def validate_eval_input(data: list, task: str, data_path: str) -> dict:
    """Validate inference output before loading metric models or API clients."""
    expected_categories = {
        "vqa": {"VQA"},
        "image_edit": {"ImageEdit"},
        "multimodal_generation": {"MMGeneration"},
    }
    errors = []
    task_counts = defaultdict(int)
    checked_images = set()

    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            errors.append(f"line {index}: record is not an object")
            continue
        if item.get("category") not in expected_categories[task]:
            errors.append(
                f"line {index}: category {item.get('category')!r} does not match task {task}"
            )
        if not isinstance(item.get("response"), str):
            errors.append(f"line {index}: missing string response")

        input_path = resolve_image_path(data_path, item.get("input_image", ""))
        if not input_path or not os.path.isfile(input_path):
            errors.append(f"line {index}: missing input_image {item.get('input_image')!r}")
        else:
            checked_images.add(os.path.realpath(input_path))

        if task in {"image_edit", "multimodal_generation"}:
            for field in ("ground_truth_image", "output_image"):
                path = resolve_image_path(data_path, item.get(field, ""))
                if not path or not os.path.isfile(path):
                    errors.append(f"line {index}: missing {field} {item.get(field)!r}")
                else:
                    checked_images.add(os.path.realpath(path))

        paper_task = validate_paper_task(paper_task_for_item(item))
        task_counts[paper_task] += 1

    if errors:
        raise ValueError(
            f"Evaluation input validation failed with {len(errors)} error(s):\n" + "\n".join(errors[:50])
        )
    present_tasks = set(task_counts)
    missing_tasks = sorted(SUPPORTED_PAPER_TASKS - present_tasks)
    return {
        "records": len(data),
        "task": task,
        "paper_tasks": dict(sorted(task_counts.items())),
        "expected_paper_task_count": len(SUPPORTED_PAPER_TASKS),
        "present_paper_task_count": len(present_tasks),
        "missing_paper_tasks": missing_tasks,
        "paper_task_coverage_complete": not missing_tasks,
        "resolved_image_count": len(checked_images),
        "missing_images": 0,
    }


def task_coverage(data: list) -> dict:
    """Summarize coverage against the canonical 16-task taxonomy.

    A model-specific inference file is allowed to contain only a subset of
    tasks.  The missing-task list is nevertheless persisted in every
    aggregate so partial baseline exports cannot be mistaken for full
    benchmark coverage.
    """
    present = sorted({paper_task_for_item(item) for item in data})
    missing = sorted(SUPPORTED_PAPER_TASKS - set(present))
    return {
        'expected_task_count': len(SUPPORTED_PAPER_TASKS),
        'present_task_count': len(present),
        'present_tasks': present,
        'missing_tasks': missing,
        'complete': not missing,
    }


def calculate_accuracy_rates(results: dict) -> dict:
    """Calculate per-metric pass rates using predefined thresholds."""
    accuracy_rates = {}
    for metric, values in results.items():
        if metric in METRIC_THRESHOLDS and isinstance(values, list) and values:
            config = METRIC_THRESHOLDS[metric]
            threshold = config['threshold']
            
            if config['lower_is_better']:
                passes = sum(1 for v in values if v <= threshold)
            else:
                passes = sum(1 for v in values if v >= threshold)
                
            accuracy_rates[f"{metric}_Accuracy_Rate"] = passes / len(values)
            
    return accuracy_rates


import hashlib


def generate_sample_id(item: dict) -> str:
    """Return a stable, collision-resistant ID for one inference record.

    The old implementation used only ``instruction``, ``answer`` and
    ``output_image``.  VQA records commonly have no output image and reuse
    the same answer across several questions, so that scheme silently
    merged distinct samples during checkpoint resume.  Prefer explicit IDs
    when present, but always bind them to the task/input context so repeated
    IDs from different task variants remain distinct.
    """
    metadata = item.get('metadata')
    metadata = metadata if isinstance(metadata, dict) else {}
    explicit_id = item.get('sample_id') or metadata.get('sample_id') or metadata.get('unique_id')

    stable_fields = {
        'explicit_id': str(explicit_id or ''),
        'category': item.get('category'),
        'paper_task': paper_task_for_item(item),
        'sub_category': item.get('sub-category') or item.get('paper_subtask'),
        'modality': item.get('modality'),
        'input_image': item.get('input_image'),
        'ground_truth_image': item.get('ground_truth_image'),
        'output_image': item.get('output_image'),
        'instruction': item.get('instruction'),
        'choice': item.get('choice'),
        'answer': item.get('answer'),
    }
    payload = json.dumps(stable_fields, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def choices_for_item(item: dict) -> list[str]:
    """Return explicit choices or recover them from a Parquet VQA instruction."""
    for field in ('choice', 'choices'):
        value = item.get(field)
        if isinstance(value, str) and value.strip():
            return [value]
        if isinstance(value, (list, tuple)):
            choices = [str(choice) for choice in value if str(choice).strip()]
            if choices:
                return choices
    return extract_choices_from_instruction(item.get('instruction', ''))


def canonical_answer_text(item: dict) -> str:
    answer = item.get('answer', '')
    if isinstance(answer, dict):
        return serialize_clinical_reference(answer)
    paper_task = normalize_paper_task(paper_task_for_item(item))
    if paper_task in {'multiple-choice', 'blank-filling'}:
        return normalize_closed_form_answer(answer, choices_for_item(item))
    return " ".join(str(answer or '').strip().split())


def canonical_response_text(item: dict) -> str:
    response = str(item.get('response', '') or '')
    paper_task = normalize_paper_task(paper_task_for_item(item))
    if paper_task in {'multiple-choice', 'blank-filling'}:
        return normalize_closed_form_answer(response, choices_for_item(item))
    return " ".join(response.strip().split())


def _text_metric_bundle_without_radgraph(item: dict) -> dict:
    response_text = canonical_response_text(item)
    answer_text = canonical_answer_text(item)
    paper_task = validate_paper_task(paper_task_for_item(item))

    metrics = {
        'normalized_response_text': response_text,
        'normalized_answer_text': answer_text,
    }

    if paper_task in CLOSED_FORM_TEXT_EM_TASKS:
        closed = compute_closed_form_exact_match(
            item.get('response', ''),
            item.get('answer', ''),
            choices_for_item(item),
            paper_task,
        )
        metrics['Text_EM'] = closed['score']
        metrics['Text_EM_parse_status'] = closed['parse_status']
        metrics['Text_EM_parsed_answer'] = closed['parsed_answer']
        metrics['Text_EM_expected_answer'] = closed['expected_answer']
        metrics['Text_EM_parse_failure_reason'] = closed['parse_failure_reason']

    return metrics


def compute_text_metric_bundle(item: dict) -> dict:
    metrics = _text_metric_bundle_without_radgraph(item)
    radgraph = compute_radgraph_f1(
        metrics['normalized_response_text'], metrics['normalized_answer_text']
    )
    metrics['radgraph_applicable'] = radgraph['applicable']
    metrics['RadGraph_F1'] = radgraph['f1']
    return metrics


def compute_text_metric_bundles(
    items: list[dict], *, include_radgraph: bool = True
) -> list[dict]:
    """Build normalized text/EM metadata and optionally run RadGraph.

    The main table uses BLEU, PubMedBERTScore, and closed-form EM, but not
    RadGraph.  Keeping normalization and EM independent from the optional
    RadGraph model makes a full-table metric run substantially cheaper.
    """
    bundles = [_text_metric_bundle_without_radgraph(item) for item in items]
    if not include_radgraph:
        for bundle in bundles:
            bundle['radgraph_applicable'] = False
            bundle['RadGraph_F1'] = None
        return bundles
    radgraph_results = compute_radgraph_f1_batch(
        [bundle['normalized_response_text'] for bundle in bundles],
        [bundle['normalized_answer_text'] for bundle in bundles],
    )
    for bundle, radgraph in zip(bundles, radgraph_results):
        bundle['radgraph_applicable'] = radgraph['applicable']
        bundle['RadGraph_F1'] = radgraph['f1']
    return bundles


def append_metric_value(all_metrics: defaultdict, metric_name: str, value) -> None:
    if isinstance(value, (int, float)) and np.isfinite(value):
        all_metrics[metric_name].append(float(value))


def bootstrap_ci(values, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> dict:
    """Compute a bootstrap 95% confidence interval for one metric instead of a mean/std-only comparison."""
    arr = np.asarray([float(v) for v in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {'mean': None, 'ci_low': None, 'ci_high': None, 'n_boot': 0}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = float(np.mean(rng.choice(arr, size=arr.size, replace=True)))
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        'mean': float(np.mean(arr)),
        'ci_low': float(lo),
        'ci_high': float(hi),
        'n_boot': int(n_boot),
    }


def paired_wilcoxon_test(scores_a, scores_b, metric_name: str) -> dict:
    """Run a paired Wilcoxon signed-rank test on two score sets from the same samples."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return {'metric': metric_name, 'applicable': False, 'error': 'scipy is unavailable'}
    arr_a = np.asarray([float(v) for v in scores_a], dtype=float)
    arr_b = np.asarray([float(v) for v in scores_b], dtype=float)
    mask = np.isfinite(arr_a) & np.isfinite(arr_b)
    arr_a, arr_b = arr_a[mask], arr_b[mask]
    if arr_a.size < 2:
        return {'metric': metric_name, 'n': int(arr_a.size), 'applicable': False}
    diff = arr_b - arr_a
    if not np.any(diff != 0):
        stat, p_value = 0.0, 1.0
    else:
        stat, p_value = wilcoxon(arr_a, arr_b, zero_method='wilcox')
    mean_diff = float(np.mean(diff))
    if abs(mean_diff) < 1e-12:
        direction = 'equal'
    elif mean_diff > 0:
        direction = 'model_b>model_a'
    else:
        direction = 'model_a>model_b'
    return {
        'metric': metric_name,
        'n': int(arr_a.size),
        'statistic': float(stat),
        'p_value': float(p_value),
        'significant_at_0.05': bool(p_value < 0.05),
        'mean_diff': mean_diff,
        'effect_direction': direction,
        'applicable': True,
    }


def valid_judge_result(result: object) -> bool:
    """Return whether a judge response is complete enough to be checkpointed.

    Empty/API-error/legacy JSON must be retried rather than silently counted as
    a completed VLM evaluation.  Scores are constrained to the documented
    1--10 rubric to avoid accepting malformed parser output.
    """
    if not isinstance(result, dict):
        return False
    all_score_fields = [*JUDGE_DIMENSIONS, 'overall_score']
    for field in all_score_fields:
        if field == 'overall_score':
            value = result.get(field)
        else:
            section = result.get(field)
            value = section.get('score') if isinstance(section, dict) else None
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False
        if not np.isfinite(value) or not 1.0 <= float(value) <= 10.0:
            return False
    return True


def judge_result_is_complete(item: dict, task: str) -> bool:
    """Check every judge view that this task's available inputs require."""
    expects_w_gt = bool(
        item.get('input_image')
        if task == 'vqa'
        else item.get('ground_truth_image') and item.get('output_image')
    )
    expects_wo_gt = bool(item.get('input_image'))
    return (
        (not expects_w_gt or valid_judge_result(item.get('vlm_judge_w_gt_result')))
        and (not expects_wo_gt or valid_judge_result(item.get('vlm_judge_wo_gt_result')))
    )


def local_metrics_are_complete(
    item: dict, task: str, *, require_radgraph: bool = True
) -> bool:
    """Check the required local metrics without relying on a version marker."""
    required = []
    if task in {'image_edit', 'multimodal_generation'}:
        required.extend(['LPIPS', 'PSNR', 'SSIM', 'MedImageInsight_Similarity'])
    if task in {'vqa', 'multimodal_generation'}:
        required.extend(['BLEU', 'BERT_Score'])
        if require_radgraph:
            if 'radgraph_applicable' not in item:
                return False
            if item.get('radgraph_applicable') and not isinstance(item.get('RadGraph_F1'), (int, float)):
                return False
    return all(
        isinstance(item.get(metric), (int, float))
        and not isinstance(item.get(metric), bool)
        and np.isfinite(item[metric])
        for metric in required
    )


def append_judge_metrics(all_metrics: defaultdict, judge: object, suffix: str) -> None:
    """Append only the current five clinical judge dimensions and overall score."""
    if not valid_judge_result(judge):
        return
    for source_key, metric_prefix in JUDGE_DIMENSIONS.items():
        append_metric_value(all_metrics, f'{metric_prefix}_{suffix}', judge[source_key]['score'])
    append_metric_value(all_metrics, f'VLM_Overall_Score_{suffix}', judge['overall_score'])


def classify_failure_modes(record: dict, task: str) -> list:
    """Assign one record to interpretable failure modes for the paper's error analysis."""
    modes = []
    judge = record.get('vlm_judge_w_gt_result')
    instruction = (
        judge.get('instruction_compliance', {}).get('score')
        if isinstance(judge, dict)
        else None
    )

    finding_accuracy = (
        judge.get('clinical_finding_accuracy', {}).get('score')
        if isinstance(judge, dict)
        else None
    )
    hallucination_control = (
        judge.get('hallucination_omission_control', {}).get('score')
        if isinstance(judge, dict)
        else None
    )
    if isinstance(finding_accuracy, (int, float)) and finding_accuracy < 6:
        modes.append('low_clinical_finding_accuracy')
    if isinstance(hallucination_control, (int, float)) and hallucination_control < 6:
        modes.append('judge_detected_hallucination_or_omission')
    if isinstance(instruction, (int, float)) and instruction < 6:
        modes.append('low_instruction_compliance')
    return modes


def build_error_analysis(records: dict, task: str) -> dict:
    """Break down metrics by modality and paper task, and summarize failure-mode distributions."""
    items = [r for r in records.values() if isinstance(r, dict)]
    by_modality = defaultdict(lambda: defaultdict(list))
    by_paper_task = defaultdict(lambda: defaultdict(list))
    modality_counts = defaultdict(int)
    paper_task_counts = defaultdict(int)
    failure_modes = defaultdict(list)

    for item in items:
        modality = str(item.get('modality') or 'unknown')
        paper_task = paper_task_for_item(item) or 'unknown'
        modality_counts[modality] += 1
        paper_task_counts[paper_task] += 1
        metric_values = metrics_from_record(item, task)
        for metric_name, value in metric_values.items():
            by_modality[modality][metric_name].append(value)
            by_paper_task[paper_task][metric_name].append(value)
        for mode in classify_failure_modes(item, task):
            failure_modes[mode].append(item.get('sample_id') or generate_sample_id(item))

    def summarize(group_metrics, group_counts):
        out = {}
        for group, metric_values in sorted(group_metrics.items()):
            entry = {'Sample_Count': int(group_counts[group])}
            for metric_name, values in sorted(metric_values.items()):
                entry[f'Average_{metric_name}'] = float(np.mean(values))
                entry[f'Std_{metric_name}'] = float(np.std(values))
            out[group] = entry
        return out

    return {
        'sample_count': len(items),
        'by_modality': summarize(by_modality, modality_counts),
        'by_paper_task': summarize(by_paper_task, paper_task_counts),
        'failure_modes': {
            mode: {'count': len(sample_ids), 'samples': sample_ids[:10]}
            for mode, sample_ids in sorted(failure_modes.items())
        },
    }


def load_result_records(jsonl_path: str) -> dict:
    """Load evaluated JSONL records into a {sample_id: record} index."""
    records = {}
    for line in load_jsonl_data(jsonl_path):
        if not isinstance(line, dict):
            continue
        sample_id = line.get('sample_id') or generate_sample_id(line)
        records[str(sample_id)] = line
    return records


async def analyze_model_comparison(
    jsonl_paths: list,
    task: str,
    n_boot: int = 1000,
) -> dict:
    """Run paired Wilcoxon tests and bootstrap confidence intervals for every model pair.

    Input consists of evaluated JSONL files for the same samples and different model outputs, paired by sample_id.
    """
    if len(jsonl_paths) < 2:
        raise ValueError('--mission stats requires at least two result files for the same samples with different model outputs')
    model_records = []
    for path in jsonl_paths:
        records = load_result_records(path)
        if not records:
            raise ValueError(f'Result file is empty: {path}')
        model_records.append((path, records))

    common_ids = set(model_records[0][1].keys())
    for _, records in model_records[1:]:
        common_ids &= set(records.keys())
    common_ids = sorted(common_ids)
    if len(common_ids) < 2:
        raise ValueError('Result files have too few shared sample_id values for paired statistics')

    model_names = [os.path.splitext(os.path.basename(p))[0] for p in jsonl_paths]
    # Collect all metric names first.
    metric_names = set()
    for _, records in model_records:
        for sample_id in common_ids:
            metric_names.update(metrics_from_record(records[sample_id], task).keys())

    aligned = {metric_name: {} for metric_name in metric_names}
    for metric_name in metric_names:
        for model_name, (_, records) in zip(model_names, model_records):
            aligned[metric_name][model_name] = [
                metrics_from_record(records[sample_id], task).get(metric_name)
                for sample_id in common_ids
            ]

    result = {
        'models': model_names,
        'result_files': jsonl_paths,
        'paired_sample_count': len(common_ids),
        'metrics': {},
    }
    for metric_name in sorted(metric_names):
        metric_result = {}
        for model_name in model_names:
            metric_result[model_name] = bootstrap_ci(
                [v for v in aligned[metric_name][model_name] if v is not None],
                n_boot=n_boot,
            )
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                name_a, name_b = model_names[i], model_names[j]
                pairs = [
                    (a, b)
                    for a, b in zip(aligned[metric_name][name_a], aligned[metric_name][name_b])
                    if a is not None and b is not None
                ]
                comp = paired_wilcoxon_test(
                    [p[0] for p in pairs],
                    [p[1] for p in pairs],
                    metric_name,
                )
                metric_result[f'{name_a}_vs_{name_b}'] = comp
        result['metrics'][metric_name] = metric_result
    return result


def save_results_for_stats(results: dict, jsonl_paths: list):
    """Save multi-model statistics to ./eval_results/."""
    output_dir = os.environ.get('MEDGEN_EVAL_RESULTS_DIR', './eval_results')
    os.makedirs(output_dir, exist_ok=True)
    parts = [os.path.splitext(os.path.basename(p))[0] for p in jsonl_paths[:4]]
    output_path = os.path.join(output_dir, '_'.join(parts) + '_stats_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    logging.info(f"Statistics saved to: {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate multimodal model outputs")
    parser.add_argument('--data_path', type=str, default="./MedGEN", help='Path to the input JSONL file')
    parser.add_argument('--jsonl_path', type=str, nargs='+', required=True, help='Input JSONL path(s); stats mode accepts multiple files')
    parser.add_argument('--batch_size', type=int, default=8, help='Evaluation batch size')
    parser.add_argument('--mission', type=str, choices=['basic_eval', 'type_wise', 'stats'], default='basic_eval', help='Evaluation mission')
    parser.add_argument('--type_key', type=str, default='modality', help='Record key used for grouping in type_wise mode')
    parser.add_argument('--task', type=str, choices=['multimodal_generation', 'image_edit', 'vqa'], default='multimodal_generation', help='Evaluation task type (multimodal_generation, image_edit, vqa)')
    parser.add_argument('--max_samples', type=int, default=None, help='Read only the first N records')
    parser.add_argument('--bootstrap_samples', type=int, default=BOOTSTRAP_SAMPLES, help='Bootstrap resampling count (default: 1000)')
    parser.add_argument('--validate-only', action='store_true', help='Validate inputs and image paths without loading metric models or calling an API')
    parser.add_argument('--require-full-task-coverage', action='store_true', help='Require all 16 paper tasks; otherwise report missing tasks')
    parser.add_argument('--local-metrics-only', action='store_true', help='Run local metrics and skip the paid VLM judge; supported only by basic_eval')
    parser.add_argument('--judge_model', '--judge-model', dest='judge_model', type=str, default='', help='VLM judge model name; uses --judge-config when omitted')
    parser.add_argument('--judge_config', '--judge-config', dest='judge_config', type=str, default='./config.yaml', help='Shared API configuration file')
    parser.add_argument('--judge_base_url', '--judge-base-url', dest='judge_base_url', type=str, default='', help='Optional judge endpoint. This takes precedence over MEDGEN_VLM_BASE_URL.')
    parser.add_argument('--judge_api_key', '--judge-api-key', dest='judge_api_key', type=str, default='', help='Optional judge API key. Use EMPTY for a local vLLM endpoint.')
    parser.add_argument('--judge_backend', type=str, choices=['api'], default='api', help='OpenAI-compatible judge endpoint')
    parser.add_argument('--enable_clinical_text_metrics', action='store_true', default=True, help='Enable clinical text metrics')
    parser.add_argument(
        '--disable_radgraph',
        action='store_true',
        help='Skip RadGraph, which is not used in the main table, while retaining text normalization, BLEU, PubMedBERTScore, and closed-form EM',
    )
    return parser


def build_vlm_judge_client(
    run_vlm_judge: bool,
    judge_model: str | None,
    judge_backend: str,
    judge_config: str = "./config.yaml",
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
):
    if not run_vlm_judge:
        return None
    # Priority: --judge_model > config model_name > default medical VLM
    selected_model = judge_model or ""
    client_kwargs = {
        'config_path': judge_config,
        'model_name': selected_model,
    }
    if judge_base_url:
        client_kwargs['base_url'] = judge_base_url
    if judge_api_key:
        client_kwargs['api_key'] = judge_api_key
    return double_image_vlm(**client_kwargs)



async def basic_eval(
    data: list,
    batch_size: int,
    task: str,
    data_path: str,
    jsonl_path: str,
    run_vlm_judge: bool = True,
    judge_model: str | None = None,
    judge_backend: str = 'api',
    judge_config: str = './config.yaml',
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    enable_clinical_text_metrics: bool = True,
    enable_radgraph: bool = True,
    n_boot: int = BOOTSTRAP_SAMPLES,
) -> dict:
    """
    Run basic evaluation for a dataset subset with checkpoint/resume support for image, text, and VLM metrics.
    """
    for item in data:
        validate_paper_task(paper_task_for_item(item))
    vlm_client = build_vlm_judge_client(
        run_vlm_judge,
        judge_model,
        judge_backend,
        judge_config,
        judge_base_url,
        judge_api_key,
    )
    # The reference-aware and reference-free views are submitted together.
    # A per-view concurrency of one therefore keeps at most two image-bearing
    # requests in flight on the shared GPU during the conservative default run.
    judge_concurrency = max(1, int(os.environ.get('MEDGEN_JUDGE_CONCURRENCY', '1')))
    profile_timing = os.environ.get('MEDGEN_PROFILE_TIMING', '0') == '1'
    overlap_local_and_judge = (
        run_vlm_judge
        and os.environ.get('MEDGEN_OVERLAP_LOCAL_AND_JUDGE', '0') == '1'
    )
    all_metrics = defaultdict(list)
    text_metric_sample_count = 0
    radgraph_applicable_count = 0

    # --- 1. Build and load checkpoint results ---
    eval_results_dir = os.environ.get('MEDGEN_EVAL_RESULTS_DIR', './eval_results')
    os.makedirs(eval_results_dir, exist_ok=True)
    base_name = os.path.basename(jsonl_path)
    suffix = '_with_vlm.jsonl' if run_vlm_judge else '_local_metrics.jsonl'
    intermediate_file = os.path.join(
        eval_results_dir, os.path.splitext(base_name)[0] + suffix
    )
    journal_file = intermediate_file + '.journal'

    existing_data = {}  # uid -> item with all metrics
    loaded_checkpoint_rows = 0
    for checkpoint_path in (intermediate_file, journal_file):
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    loaded_checkpoint_rows += 1
                    # Invalidate old local-metric records and never propagate
                    # removed metrics or the former Rad-DINO field.
                    for obsolete_key in (
                        'Task_Accuracy', 'Text_F1', 'Anatomical_Embedding_Similarity',
                        'Clinical_Entity_P', 'Clinical_Entity_R', 'Clinical_Entity_F1',
                        'Entity_Hallucination_Rate', 'Entity_Omission_Rate',
                        'Entity_Factual_Precision', 'CheXbert_Factual_Precision',
                    ):
                        item.pop(obsolete_key, None)
                    uid = generate_sample_id(item)
                    existing_data[uid] = item
    if loaded_checkpoint_rows:
        logging.info(
            "Found checkpoint file; loaded %d rows for %d unique samples.",
            loaded_checkpoint_rows,
            len(existing_data),
        )

    # --- 2. Separate processed and unprocessed samples ---
    unprocessed_items = []
    already_processed_items = []
    for item in data:
        uid = generate_sample_id(item)
        existing_item = existing_data.get(uid)
        is_complete = (
            judge_result_is_complete(existing_item, task)
            and local_metrics_are_complete(
                existing_item, task, require_radgraph=enable_radgraph
            )
            if run_vlm_judge and isinstance(existing_item, dict)
            else bool(
                isinstance(existing_item, dict)
                and existing_item.get('_local_metrics_complete')
                and existing_item.get('_local_metrics_version') == LOCAL_METRICS_VERSION
                and (enable_radgraph or existing_item.get('_radgraph_disabled') is True)
            )
        )
        if (
            existing_item is not None
            and is_complete
        ):
            # Use the loaded complete record with all metrics.
            already_processed_items.append(existing_data[uid])
        else:
            if isinstance(existing_item, dict):
                # Preserve any valid judge view/local metrics from an interrupted
                # batch so recovery only requests the missing view.
                merged_item = dict(item)
                merged_item.update(existing_item)
                unprocessed_items.append(merged_item)
            else:
                unprocessed_items.append(item)

    # --- 3. Preload metrics for all processed samples (image, text, and VLM) ---
    logging.info(f"Found {len(already_processed_items)} fully processed samples; loading their results directly.")
    for item in already_processed_items:
        # Load image and text metrics.
        for metric_key in [
            'LPIPS', 'PSNR', 'SSIM',
            'MedImageInsight_Similarity',
            'BLEU', 'BERT_Score', 'RadGraph_F1', 'Text_EM'
        ]:
            append_metric_value(all_metrics, metric_key, item.get(metric_key))
        if enable_clinical_text_metrics and 'normalized_answer_text' in item:
            text_metric_sample_count += 1
            if item.get('radgraph_applicable'):
                radgraph_applicable_count += 1

        # Load VLM judge metrics.
        judge_w_gt = item.get('vlm_judge_w_gt_result')
        judge_wo_gt = item.get('vlm_judge_wo_gt_result')

        append_judge_metrics(all_metrics, judge_w_gt, 'W_GT')
        append_judge_metrics(all_metrics, judge_wo_gt, 'WO_GT')

    # --- Build a task-specific structured checklist prompt covering all five clinical dimensions---
    current_vlm_holistic_judge_w_gt_prompt = build_vlm_judge_prompt(task, with_gt=True)
    current_vlm_holistic_judge_wo_gt_prompt = build_vlm_judge_prompt(task, with_gt=False)

    # Complete result dictionary for checkpoint overwrites.
    full_results_dict = dict(existing_data)

    # --- 4. Iterate over and evaluate only unprocessed samples ---
    if not unprocessed_items:
        logging.info("All samples are already processed; no new evaluation is required.")
    else:
        logging.info(f"Processing {len(unprocessed_items)} new samples.")

    total_batches = (len(unprocessed_items) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(unprocessed_items), batch_size), desc="Evaluating Batches", total=total_batches):
        batch_started = time.perf_counter()
        batch_number = i // batch_size + 1
        batch_data = unprocessed_items[i:i+batch_size]
        prepare_started = time.perf_counter()

        text_bundles = None
        if task in ['multimodal_generation', 'vqa'] and enable_clinical_text_metrics:
            text_bundles = compute_text_metric_bundles(
                batch_data, include_radgraph=enable_radgraph
            )

        # --- Prepare asynchronous tasks. ---
        vlm_judge_w_gt_requests, vlm_judge_wo_gt_requests = [], []
        request_to_index = []
        eval_images, ref_images = [], []
        eval_texts, ref_texts = [], []

        for idx, item in enumerate(batch_data):
            # ... (Prepare image, text, and VLM requests.)
            if task in ['multimodal_generation', 'image_edit']:
                try:
                    eval_images.append(
                        Image.open(resolve_image_path(data_path, item['output_image'])).convert('RGB')
                    )
                    ref_images.append(
                        Image.open(
                            resolve_image_path(data_path, item['ground_truth_image'])
                        ).convert('RGB')
                    )
                    validate_paper_task(paper_task_for_item(item))
                except (FileNotFoundError, IOError) as e:
                    raise RuntimeError(
                        f"Could not load image; complete local metrics cannot be produced: {e}"
                    ) from e
            if task in ['multimodal_generation', 'vqa']:
                if enable_clinical_text_metrics:
                    bundle = text_bundles[idx]
                    for key, value in bundle.items():
                        item[key] = value
                    eval_texts.append(item.get('normalized_response_text', item.get('response', '')))
                    ref_texts.append(item.get('normalized_answer_text', item.get('answer', '')))
                    text_metric_sample_count += 1
                    for metric_key in ['RadGraph_F1', 'Text_EM']:
                        append_metric_value(all_metrics, metric_key, item.get(metric_key))
                    if item.get('radgraph_applicable'):
                        radgraph_applicable_count += 1
                else:
                    eval_texts.append(item.get('response', ''))
                    ref_texts.append(item.get('answer', ''))
            if not run_vlm_judge:
                continue
            # ... (Prepare VLM requests.)
            w_gt_prompt = "\n".join([
                current_vlm_holistic_judge_w_gt_prompt[0], current_vlm_holistic_judge_w_gt_prompt[1],
                current_vlm_holistic_judge_w_gt_prompt[2], current_vlm_holistic_judge_w_gt_prompt[3],
                current_vlm_holistic_judge_w_gt_prompt[4], item.get('instruction', 'N/A'),
                current_vlm_holistic_judge_w_gt_prompt[5], item.get('normalized_answer_text', canonical_answer_text(item)),
                current_vlm_holistic_judge_w_gt_prompt[6], item.get('normalized_response_text', canonical_response_text(item)),
                current_vlm_holistic_judge_w_gt_prompt[7]
            ])
            needs_w_gt = not valid_judge_result(item.get('vlm_judge_w_gt_result'))
            needs_wo_gt = not valid_judge_result(item.get('vlm_judge_wo_gt_result'))
            if needs_w_gt and task == 'vqa' and item.get('input_image'):
                vlm_judge_w_gt_requests.append(
                    (
                        w_gt_prompt,
                        resolve_image_path(data_path, item['input_image']),
                        "",
                        "Input",
                        "",
                        None,
                        None,
                    )
                )
                request_to_index.append(('w_gt', idx))
            elif needs_w_gt and item.get('ground_truth_image') and item.get('output_image'):
                vlm_judge_w_gt_requests.append(
                    (
                        w_gt_prompt,
                        resolve_image_path(data_path, item['ground_truth_image']),
                        resolve_image_path(data_path, item['output_image']),
                        "Ground Truth",
                        "Generated Answer",
                        None,
                        None,
                    )
                )
                request_to_index.append(('w_gt', idx))
            wo_gt_prompt = "\n".join([
                current_vlm_holistic_judge_wo_gt_prompt[0], current_vlm_holistic_judge_wo_gt_prompt[1],
                current_vlm_holistic_judge_wo_gt_prompt[2], current_vlm_holistic_judge_wo_gt_prompt[3],
                current_vlm_holistic_judge_wo_gt_prompt[4], item.get('instruction', 'N/A'),
                current_vlm_holistic_judge_wo_gt_prompt[5],
                current_vlm_holistic_judge_wo_gt_prompt[6], item.get('normalized_response_text', canonical_response_text(item)),
                current_vlm_holistic_judge_wo_gt_prompt[7]
            ])
            if needs_wo_gt and item.get('input_image'):
                vlm_judge_wo_gt_requests.append(
                    (
                        wo_gt_prompt,
                        resolve_image_path(data_path, item['input_image']),
                        resolve_image_path(data_path, item.get('output_image', '')),
                        "Input",
                        "Output",
                        None,
                        None,
                    )
                )
                request_to_index.append(('wo_gt', idx))

        prepare_seconds = time.perf_counter() - prepare_started

        prefetched_vlm_future = None
        judge_started = None
        if overlap_local_and_judge:
            prefetched_vlm_tasks = []
            if vlm_judge_w_gt_requests:
                prefetched_vlm_tasks.append(
                    vlm_client.generate_batch(
                        vlm_judge_w_gt_requests, concurrency=judge_concurrency
                    )
                )
            if vlm_judge_wo_gt_requests:
                prefetched_vlm_tasks.append(
                    vlm_client.generate_batch(
                        vlm_judge_wo_gt_requests, concurrency=judge_concurrency
                    )
                )
            if prefetched_vlm_tasks:
                judge_started = time.perf_counter()
                prefetched_vlm_future = asyncio.gather(
                    *prefetched_vlm_tasks, return_exceptions=True
                )


        # --- Run and save image and text metrics ---
        # Run text metrics and the frozen clinical image encoder in this batch.
        metric_specs = []
        if task in ['multimodal_generation', 'vqa'] and eval_texts:
            metric_specs.extend([
                ('BLEU', len(eval_texts), batch_async_evaluate_text_quality(eval_texts, ref_texts, 'bleu')),
                ('BERT_Score', len(eval_texts), batch_async_evaluate_text_quality(eval_texts, ref_texts, 'bertscore')),
            ])
        if task in ['multimodal_generation', 'image_edit'] and eval_images:
            metric_specs.extend([
                ('LPIPS', len(eval_images), batch_async_FR_IQA(eval_images, ref_images, 'lpips')),
                ('PSNR', len(eval_images), batch_async_FR_IQA(eval_images, ref_images, 'psnr')),
                ('SSIM', len(eval_images), batch_async_FR_IQA(eval_images, ref_images, 'ssim')),
                ('MEDIMAGEINSIGHT', len(eval_images), batch_async_medimageinsight_metrics(eval_images, ref_images)),
            ])

        local_metrics_started = time.perf_counter()
        if metric_specs and overlap_local_and_judge:
            def run_metric_coroutines_in_worker():
                results = []
                for _, _, metric_coroutine in metric_specs:
                    try:
                        results.append(asyncio.run(metric_coroutine))
                    except Exception as exc:
                        results.append(exc)
                return results

            metric_results = await asyncio.to_thread(run_metric_coroutines_in_worker)
        else:
            metric_results = (
                await asyncio.gather(
                    *(spec[2] for spec in metric_specs), return_exceptions=True
                )
                if metric_specs else []
            )
        local_metrics_seconds = time.perf_counter() - local_metrics_started
        if task in ['multimodal_generation', 'image_edit'] and len(eval_images) != len(batch_data):
            raise RuntimeError(
                f"Incomplete image-metric inputs: batch={len(batch_data)}, images={len(eval_images)}"
            )
        if task in ['multimodal_generation', 'vqa'] and len(eval_texts) != len(batch_data):
            raise RuntimeError(
                f"Incomplete text-metric inputs: batch={len(batch_data)}, texts={len(eval_texts)}"
            )
        for (metric_name, sample_count, _), scores in zip(metric_specs, metric_results):
            if scores is None:
                logging.warning("metric %s returned None; skipping", metric_name)
                continue
            if isinstance(scores, Exception):
                raise RuntimeError(f"metric {metric_name} failed; evaluation aborted") from scores
            if len(scores) != sample_count:
                raise RuntimeError(
                    f"metric {metric_name} returned {len(scores)} scores for {sample_count} samples"
                )
            if metric_name == 'MEDIMAGEINSIGHT':
                for item_idx, metric_dict in enumerate(scores):
                    if not isinstance(metric_dict, dict):
                        continue
                    for sub_metric, value in metric_dict.items():
                        if isinstance(value, (int, float)) and np.isfinite(value):
                            all_metrics[sub_metric].append(float(value))
                            batch_data[item_idx][sub_metric] = float(value)
                continue
            all_metrics[metric_name].extend(scores)
            for item_idx, score in enumerate(scores):
                batch_data[item_idx][metric_name] = score


        # --- Maintain an index mapping for each request type ---
        w_gt_request_to_index = []
        wo_gt_request_to_index = []

        # Split request_to_index.
        w_gt_request_to_index = [x for x in request_to_index if x[0] == 'w_gt']
        wo_gt_request_to_index = [x for x in request_to_index if x[0] == 'wo_gt']

        # --- Run the VLM judge ---
        if prefetched_vlm_future is not None:
            vlm_results = await prefetched_vlm_future
        else:
            judge_started = time.perf_counter()
            vlm_tasks = []
            if vlm_judge_w_gt_requests:
                vlm_tasks.append(
                    vlm_client.generate_batch(vlm_judge_w_gt_requests, concurrency=judge_concurrency)
                )
            if vlm_judge_wo_gt_requests:
                vlm_tasks.append(
                    vlm_client.generate_batch(vlm_judge_wo_gt_requests, concurrency=judge_concurrency)
                )
            vlm_results = (
                await asyncio.gather(*vlm_tasks, return_exceptions=True)
                if vlm_tasks else []
            )
        judge_seconds = time.perf_counter() - judge_started
        all_vlm_result_groups = list(vlm_results)

        def apply_view_results(results, mapping, result_key, error_key):
            for result_index, (_, data_idx) in enumerate(mapping):
                res = results[result_index] if result_index < len(results) else None
                item = batch_data[data_idx]
                try:
                    parsed = extract_json(res['text']) if res and not res.get('error') else {}
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    parsed = {}
                    if res is not None:
                        res['error'] = f'invalid judge JSON: {exc}'
                if valid_judge_result(parsed):
                    item[result_key] = parsed
                    item.pop(error_key, None)
                else:
                    item.pop(result_key, None)
                    item[error_key] = (res or {}).get('error', 'invalid judge JSON')
                    if profile_timing:
                        raw_text = str((res or {}).get('text') or '')
                        logging.warning(
                            "PERF_INVALID_JUDGE %s",
                            json.dumps({
                                'view': result_key,
                                'sample_index': data_idx,
                                'error': item[error_key],
                                'usage': (res or {}).get('usage') or {},
                                'text_length': len(raw_text),
                                'text_tail': raw_text[-160:],
                            }, ensure_ascii=False, sort_keys=True),
                        )

        vlm_res_idx = 0
        if vlm_judge_w_gt_requests:
            w_gt_results = (
                vlm_results[vlm_res_idx]
                if not isinstance(vlm_results[vlm_res_idx], Exception)
                else []
            )
            apply_view_results(
                w_gt_results,
                w_gt_request_to_index,
                'vlm_judge_w_gt_result',
                'vlm_judge_w_gt_error',
            )
            vlm_res_idx += 1
        if vlm_judge_wo_gt_requests:
            wo_gt_results = (
                vlm_results[vlm_res_idx]
                if vlm_res_idx < len(vlm_results)
                and not isinstance(vlm_results[vlm_res_idx], Exception)
                else []
            )
            apply_view_results(
                wo_gt_results,
                wo_gt_request_to_index,
                'vlm_judge_wo_gt_result',
                'vlm_judge_wo_gt_error',
            )

        # Invalid/truncated JSON used to survive until a later full rerun.
        # Retry only the failed view immediately, with a slightly larger
        # completion budget, while retaining the already-valid counterpart.
        parse_retries = max(0, int(os.environ.get('MEDGEN_JUDGE_PARSE_RETRIES', '2')))
        retry_max_tokens = max(
            int(os.environ.get('MEDGEN_JUDGE_MAX_TOKENS', '768')),
            int(os.environ.get('MEDGEN_JUDGE_RETRY_MAX_TOKENS', '1024')),
        )
        retry_temperature = float(os.environ.get('MEDGEN_JUDGE_RETRY_TEMPERATURE', '0.1'))
        for _retry_attempt in range(parse_retries):
            retry_w_indices = [
                index
                for index, (_, data_idx) in enumerate(w_gt_request_to_index)
                if not valid_judge_result(batch_data[data_idx].get('vlm_judge_w_gt_result'))
            ]
            retry_wo_indices = [
                index
                for index, (_, data_idx) in enumerate(wo_gt_request_to_index)
                if not valid_judge_result(batch_data[data_idx].get('vlm_judge_wo_gt_result'))
            ]
            if not retry_w_indices and not retry_wo_indices:
                break

            retry_tasks = []
            retry_specs = []
            if retry_w_indices:
                requests = [
                    (
                        *vlm_judge_w_gt_requests[index][:5],
                        retry_temperature,
                        retry_max_tokens,
                    )
                    for index in retry_w_indices
                ]
                retry_tasks.append(vlm_client.generate_batch(requests, concurrency=judge_concurrency))
                retry_specs.append(('w_gt', retry_w_indices))
            if retry_wo_indices:
                requests = [
                    (
                        *vlm_judge_wo_gt_requests[index][:5],
                        retry_temperature,
                        retry_max_tokens,
                    )
                    for index in retry_wo_indices
                ]
                retry_tasks.append(vlm_client.generate_batch(requests, concurrency=judge_concurrency))
                retry_specs.append(('wo_gt', retry_wo_indices))

            retry_groups = await asyncio.gather(*retry_tasks, return_exceptions=True)
            judge_seconds = time.perf_counter() - judge_started
            all_vlm_result_groups.extend(retry_groups)
            for (view_name, request_indices), retry_group in zip(retry_specs, retry_groups):
                if isinstance(retry_group, Exception):
                    retry_group = []
                if view_name == 'w_gt':
                    apply_view_results(
                        retry_group,
                        [w_gt_request_to_index[index] for index in request_indices],
                        'vlm_judge_w_gt_result',
                        'vlm_judge_w_gt_error',
                    )
                else:
                    apply_view_results(
                        retry_group,
                        [wo_gt_request_to_index[index] for index in request_indices],
                        'vlm_judge_wo_gt_result',
                        'vlm_judge_wo_gt_error',
                    )



        # --- Aggregate VLM metrics for the new batch and update the checkpoint dictionary ---
        for item in batch_data:
            if local_metrics_are_complete(
                item, task, require_radgraph=enable_radgraph
            ):
                item['_local_metrics_complete'] = True
                item['_local_metrics_version'] = LOCAL_METRICS_VERSION
                item['_radgraph_disabled'] = not enable_radgraph
            else:
                item.pop('_local_metrics_complete', None)
                item.pop('_local_metrics_version', None)
            # Aggregate VLM metrics.
            append_judge_metrics(all_metrics, item.get('vlm_judge_w_gt_result'), 'W_GT')
            append_judge_metrics(all_metrics, item.get('vlm_judge_wo_gt_result'), 'WO_GT')

            # Update the checkpoint dictionary; item now contains all metrics.
            uid = generate_sample_id(item)
            full_results_dict[uid] = item

        # --- 5. Append only this batch to a recovery journal.  Rewriting the
        # entire multi-thousand-row checkpoint every four samples caused
        # quadratic I/O.  The loader applies last-write-wins by sample id, and
        # the journal is atomically compacted into the canonical file at end.
        checkpoint_started = time.perf_counter()
        with open(journal_file, 'a', encoding='utf-8') as journal:
            for item in batch_data:
                journal.write(json.dumps(item, ensure_ascii=False) + '\n')
        checkpoint_seconds = time.perf_counter() - checkpoint_started

        if profile_timing:
            flat_vlm_results = [
                result
                for result_group in all_vlm_result_groups
                if isinstance(result_group, list)
                for result in result_group
                if isinstance(result, dict)
            ]
            prompt_tokens = sum(
                int((result.get('usage') or {}).get('prompt_tokens') or 0)
                for result in flat_vlm_results
            )
            completion_tokens = sum(
                int((result.get('usage') or {}).get('completion_tokens') or 0)
                for result in flat_vlm_results
            )
            complete_items = sum(
                judge_result_is_complete(item, task) if run_vlm_judge else bool(item.get('_local_metrics_complete'))
                for item in batch_data
            )
            logging.info(
                "PERF_BATCH %s",
                json.dumps({
                    'batch': batch_number,
                    'batch_size': len(batch_data),
                    'prepare_seconds': prepare_seconds,
                    'local_metrics_seconds': local_metrics_seconds,
                    'judge_seconds': judge_seconds,
                    'checkpoint_seconds': checkpoint_seconds,
                    'total_seconds': time.perf_counter() - batch_started,
                    'judge_requests': len(flat_vlm_results),
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'complete_items': int(complete_items),
                }, sort_keys=True),
            )

    compact_path = intermediate_file + f'.compact.{os.getpid()}'
    with open(compact_path, 'w', encoding='utf-8') as compacted:
        for item in full_results_dict.values():
            compacted.write(json.dumps(item, ensure_ascii=False) + '\n')
    os.replace(compact_path, intermediate_file)
    if os.path.exists(journal_file):
        os.remove(journal_file)

    # --- 6. Aggregate final results; all_metrics includes all sample data ---
    final_results = {}
    for metric, values in all_metrics.items():
        finite_values = [value for value in values if isinstance(value, (int, float)) and np.isfinite(value)]
        if finite_values:
            mean_val = float(np.mean(finite_values))
            std_val = float(np.std(finite_values))  # Calculate standard deviation
            final_results[f"Average_{metric}"] = mean_val
            final_results[f"Std_{metric}"] = std_val  # Save standard deviation
            ci = bootstrap_ci(finite_values, n_boot=n_boot)
            if ci['mean'] is not None:
                final_results[f"Bootstrap95CI_Low_{metric}"] = ci['ci_low']
                final_results[f"Bootstrap95CI_High_{metric}"] = ci['ci_high']

            # Optionally print the result.
            print(f"{metric}: mean={mean_val:.4f}, std={std_val:.4f}")

    accuracy_rates = calculate_accuracy_rates({
        metric: [value for value in values if isinstance(value, (int, float)) and np.isfinite(value)]
        for metric, values in all_metrics.items()
    })
    final_results.update(accuracy_rates)
    if text_metric_sample_count:
        final_results['RadGraph_Coverage'] = radgraph_applicable_count / text_metric_sample_count
    final_results['Error_Analysis'] = build_error_analysis(full_results_dict, task)
    final_results['Task_Coverage'] = task_coverage(data)

    logging.info(f"Evaluation complete. Checkpoint with all metrics saved to: {intermediate_file}")
    return final_results

async def basic_eval_for_type_wise(
    data: list,
    batch_size: int,
    task: str,
    data_path: str,
    jsonl_path: str,
    judge_config: str = './config.yaml',
) -> dict:
    """
    Run basic evaluation for a dataset subset with checkpoint/resume support for image, text, and VLM metrics.
    """
    for item in data:
        validate_paper_task(paper_task_for_item(item))
    vlm_client = double_image_vlm(config_path=judge_config)
    all_metrics = defaultdict(list)


    base_name = os.path.basename(jsonl_path)
    name, ext = os.path.splitext(base_name)

    # Remove the prefix through the first underscore.
    if "_" in name:
        name = name.split("_", 1)[1]

    output_root = os.path.join("./eval_results_type_wise", name)
    os.makedirs(output_root, exist_ok=True)

    # --- 1. Build and load checkpoint results ---
    eval_results_dir = output_root
    os.makedirs(eval_results_dir, exist_ok=True)
    print(eval_results_dir)

    base_name = os.path.basename(jsonl_path)
    intermediate_file = os.path.join(eval_results_dir, os.path.splitext(base_name)[0] + '_with_vlm.jsonl')

    existing_data = {}  # uid -> item with all metrics
    if os.path.exists(intermediate_file):
        with open(intermediate_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                # Use selected item content to generate a unique ID.
                uid = generate_sample_id(item)
                existing_data[uid] = item
        logging.info(f"Found checkpoint file; loaded {len(existing_data)} evaluated samples.")

    # --- 2. Separate processed and unprocessed samples ---
    unprocessed_items = []
    already_processed_items = []
    for item in data:
        uid = generate_sample_id(item)
        if uid in existing_data and 'vlm_judge_w_gt_result' in existing_data[uid]: # Ensure the core evaluation is complete.
            # Use the loaded complete record with all metrics.
            already_processed_items.append(existing_data[uid])
        else:
            unprocessed_items.append(item)

    # --- 3. Preload metrics for all processed samples (image, text, and VLM) ---
    logging.info(f"Found {len(already_processed_items)} fully processed samples; loading their results directly.")
    for item in already_processed_items:
        # Load image and text metrics.
        for metric_key in [
            'LPIPS', 'PSNR', 'SSIM', 'MedImageInsight_Similarity',
            'BLEU', 'BERT_Score', 'RadGraph_F1', 'Text_EM',
        ]:
            append_metric_value(all_metrics, metric_key, item.get(metric_key))

        # Load VLM judge metrics.
        judge_w_gt = item.get('vlm_judge_w_gt_result')
        judge_wo_gt = item.get('vlm_judge_wo_gt_result')

        append_judge_metrics(all_metrics, judge_w_gt, 'W_GT')
        append_judge_metrics(all_metrics, judge_wo_gt, 'WO_GT')

    # --- Build a task-specific structured checklist prompt ---
    current_vlm_holistic_judge_w_gt_prompt = build_vlm_judge_prompt(task, with_gt=True)
    current_vlm_holistic_judge_wo_gt_prompt = build_vlm_judge_prompt(task, with_gt=False)

    # Complete result dictionary for checkpoint overwrites.
    full_results_dict = dict(existing_data)

    if not unprocessed_items:
        logging.info("All samples are already processed; no new evaluation is required.")
    else:
        logging.warning(f"Detected {len(unprocessed_items)} unprocessed samples, but skipping them and computing only existing results.")
        unprocessed_items = []  # Clear this list to skip further evaluation.

    total_batches = (len(unprocessed_items) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(unprocessed_items), batch_size), desc="Evaluating Batches", total=total_batches):
        batch_data = unprocessed_items[i:i+batch_size]

        # --- Prepare asynchronous tasks. ---
        vlm_judge_w_gt_requests, vlm_judge_wo_gt_requests = [], []
        request_to_index = []
        eval_images, ref_images = [], []
        eval_texts, ref_texts = [], []

        for idx, item in enumerate(batch_data):
            # ... (Prepare image, text, and VLM requests.)
            if task in ['multimodal_generation', 'image_edit']:
                try:
                    eval_images.append(
                        Image.open(resolve_image_path(data_path, item['output_image'])).convert('RGB')
                    )
                    ref_images.append(
                        Image.open(
                            resolve_image_path(data_path, item['ground_truth_image'])
                        ).convert('RGB')
                    )
                except (FileNotFoundError, IOError) as e:
                    logging.warning(f"Could not load image; skipping image-metric calculation: {e}")
            if task in ['multimodal_generation', 'vqa']:
                eval_texts.append(item.get('response', ''))
                ref_texts.append(item.get('answer', ''))
            # ... (Prepare VLM requests.)
            w_gt_prompt = "\n".join([
                current_vlm_holistic_judge_w_gt_prompt[0], current_vlm_holistic_judge_w_gt_prompt[1],
                current_vlm_holistic_judge_w_gt_prompt[2], current_vlm_holistic_judge_w_gt_prompt[3],
                current_vlm_holistic_judge_w_gt_prompt[4], item.get('instruction', 'N/A'),
                current_vlm_holistic_judge_w_gt_prompt[5], item.get('answer', 'N/A'),
                current_vlm_holistic_judge_w_gt_prompt[6], item.get('response', 'N/A'),
                current_vlm_holistic_judge_w_gt_prompt[7]
            ])
            if task == 'vqa' and item.get('input_image'):
                vlm_judge_w_gt_requests.append(
                    (
                        w_gt_prompt,
                        resolve_image_path(data_path, item['input_image']),
                        "",
                        "Input",
                        "",
                        None,
                        None,
                    )
                )
                request_to_index.append(('w_gt', idx))
            elif item.get('ground_truth_image') and item.get('output_image'):
                vlm_judge_w_gt_requests.append(
                    (
                        w_gt_prompt,
                        resolve_image_path(data_path, item['ground_truth_image']),
                        resolve_image_path(data_path, item['output_image']),
                        "Ground Truth",
                        "Generated Answer",
                        None,
                        None,
                    )
                )
                request_to_index.append(('w_gt', idx))
            wo_gt_prompt = "\n".join([
                current_vlm_holistic_judge_wo_gt_prompt[0], current_vlm_holistic_judge_wo_gt_prompt[1],
                current_vlm_holistic_judge_wo_gt_prompt[2], current_vlm_holistic_judge_wo_gt_prompt[3],
                current_vlm_holistic_judge_wo_gt_prompt[4], item.get('instruction', 'N/A'),
                current_vlm_holistic_judge_wo_gt_prompt[5],
                current_vlm_holistic_judge_wo_gt_prompt[6], item.get('response', 'N/A'),
                current_vlm_holistic_judge_wo_gt_prompt[7]
            ])
            if item.get('input_image'):
                vlm_judge_wo_gt_requests.append(
                    (
                        wo_gt_prompt,
                        resolve_image_path(data_path, item['input_image']),
                        resolve_image_path(data_path, item.get('output_image', '')),
                        "Input",
                        "Output",
                        None,
                        None,
                    )
                )
                request_to_index.append(('wo_gt', idx))


        # --- Run and save image and text metrics ---
        metric_tasks = []
        if task in ['multimodal_generation', 'image_edit'] and eval_images:
            metric_tasks.extend([
                batch_async_FR_IQA(eval_images, ref_images, 'lpips'),
                batch_async_FR_IQA(eval_images, ref_images, 'psnr'),
                batch_async_FR_IQA(eval_images, ref_images, 'ssim'),
                batch_async_medimageinsight_metrics(eval_images, ref_images),
            ])
        if task in ['multimodal_generation', 'vqa'] and eval_texts:
            metric_tasks.extend([
                batch_async_evaluate_text_quality(eval_texts, ref_texts, 'bleu'),
                batch_async_evaluate_text_quality(eval_texts, ref_texts, 'bertscore')
            ])
        
        metric_results = await asyncio.gather(*metric_tasks, return_exceptions=True) if metric_tasks else []
        
        res_idx = 0
        if eval_images:
            for metric_name in ['LPIPS', 'PSNR', 'SSIM']:
                scores = metric_results[res_idx]
                if isinstance(scores, Exception):
                    raise RuntimeError(f"{metric_name} failed; evaluation aborted") from scores
                all_metrics[metric_name].extend(scores)
                for item_idx, score in enumerate(scores):
                    batch_data[item_idx][metric_name] = score
                res_idx += 1
            scores = metric_results[res_idx]
            if isinstance(scores, Exception):
                raise RuntimeError("anatomical image metric failed; evaluation aborted") from scores
            for item_idx, metric_dict in enumerate(scores):
                for metric_name, score in metric_dict.items():
                    all_metrics[metric_name].append(score)
                    batch_data[item_idx][metric_name] = score
            res_idx += 1
        if eval_texts:
            for metric_name in ['BLEU', 'BERT_Score']:
                scores = metric_results[res_idx]
                               
                if scores is None:
                    print(f"[WARN] metric {metric_name} returned None; skipping this record.")
                    res_idx += 1
                    continue
                               
                if not isinstance(scores, Exception):
                    all_metrics[metric_name].extend(scores)
                    # Save metrics back to each sample.
                    for item_idx, score in enumerate(scores):
                        batch_data[item_idx][metric_name] = score
                else:
                    raise RuntimeError(f"{metric_name} failed; evaluation aborted") from scores
                res_idx += 1

        # --- Maintain an index mapping for each request type ---
        w_gt_request_to_index = []
        wo_gt_request_to_index = []

        # Split request_to_index.
        w_gt_request_to_index = [x for x in request_to_index if x[0] == 'w_gt']
        wo_gt_request_to_index = [x for x in request_to_index if x[0] == 'wo_gt']

        # --- Run the VLM judge ---
        vlm_tasks = []
        if vlm_judge_w_gt_requests:
            vlm_tasks.append(vlm_client.generate_batch(vlm_judge_w_gt_requests, concurrency=8))
        if vlm_judge_wo_gt_requests:
            vlm_tasks.append(vlm_client.generate_batch(vlm_judge_wo_gt_requests, concurrency=8))

        vlm_results = await asyncio.gather(*vlm_tasks, return_exceptions=True) if vlm_tasks else []

        vlm_res_idx = 0

        # --- Process w_gt results ---
        if vlm_judge_w_gt_requests:
            w_gt_results = vlm_results[vlm_res_idx] if not isinstance(vlm_results[vlm_res_idx], Exception) else []
            for i, res in enumerate(w_gt_results):
                if i >= len(w_gt_request_to_index):
                    break
                _, data_idx = w_gt_request_to_index[i]
                item = batch_data[data_idx]
                item['vlm_judge_w_gt_result'] = extract_json(res['text']) if res and not res.get('error') else {}
            vlm_res_idx += 1

        # --- Process wo_gt results ---
        if vlm_judge_wo_gt_requests:
            wo_gt_results = vlm_results[vlm_res_idx] if vlm_res_idx < len(vlm_results) and not isinstance(vlm_results[vlm_res_idx], Exception) else []
            for i, res in enumerate(wo_gt_results):
                if i >= len(wo_gt_request_to_index):
                    break
                _, data_idx = wo_gt_request_to_index[i]
                item = batch_data[data_idx]
                item['vlm_judge_wo_gt_result'] = extract_json(res['text']) if res and not res.get('error') else {}


        #         item['vlm_judge_wo_gt_result'] = extract_json(res['text']) if res and not res.get('error') else {}
        #         request_ptr += 1

        # --- Aggregate VLM metrics for the new batch and update the checkpoint dictionary ---
        for item in batch_data:
            # Aggregate VLM metrics.
            judge_w_gt = item.get('vlm_judge_w_gt_result', {})
            judge_wo_gt = item.get('vlm_judge_wo_gt_result', {})
            if judge_w_gt:
                if task == 'multimodal_generation':
                    all_metrics['VLM_Coherence_W_GT'].append(judge_w_gt.get('coherence', {}).get('score', 0))
                    all_metrics['VLM_Visual_Textual_Alignment_W_GT'].append(judge_w_gt.get('visual_textual_alignment', {}).get('score', 0))
                all_metrics['VLM_Content_Accuracy_W_GT'].append(judge_w_gt.get('content_accuracy', {}).get('score', 0))
                all_metrics['VLM_Relevance_W_GT'].append(judge_w_gt.get('relevance_and_responsiveness', {}).get('score', 0))
                all_metrics['VLM_Consistency_W_GT'].append(judge_w_gt.get('consistency', {}).get('score', 0))
                all_metrics['VLM_Overall_Score_W_GT'].append(judge_w_gt.get('overall_score', 0))
            if judge_wo_gt:
                if task == 'multimodal_generation':
                    all_metrics['VLM_Coherence_WO_GT'].append(judge_wo_gt.get('coherence', {}).get('score', 0))
                    all_metrics['VLM_Visual_Textual_Alignment_WO_GT'].append(judge_wo_gt.get('visual_textual_alignment', {}).get('score', 0))
                all_metrics['VLM_Content_Accuracy_WO_GT'].append(judge_wo_gt.get('content_accuracy', {}).get('score', 0))
                all_metrics['VLM_Relevance_WO_GT'].append(judge_wo_gt.get('relevance_and_responsiveness', {}).get('score', 0))
                all_metrics['VLM_Consistency_WO_GT'].append(judge_wo_gt.get('consistency', {}).get('score', 0))
                all_metrics['VLM_Overall_Score_WO_GT'].append(judge_wo_gt.get('overall_score', 0))

            # Update the checkpoint dictionary; item now contains all metrics.
            uid = generate_sample_id(item)
            full_results_dict[uid] = item

        # --- 5. After each batch, overwrite the complete checkpoint containing all metrics ---
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            for item in full_results_dict.values():
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # --- 6. Aggregate final results; all_metrics includes all sample data ---
    final_results = {}
    for metric, values in all_metrics.items():
        if values:
            mean_val = float(np.mean(values))
            std_val = float(np.std(values))  # Calculate standard deviation
            final_results[f"Average_{metric}"] = mean_val
            final_results[f"Std_{metric}"] = std_val  # Save standard deviation

            # Optionally print the result.
            print(f"{metric}: mean={mean_val:.4f}, std={std_val:.4f}")

    accuracy_rates = calculate_accuracy_rates(all_metrics)
    final_results.update(accuracy_rates)

    logging.info(f"Evaluation complete. Checkpoint with all metrics saved to: {intermediate_file}")
    return final_results


async def eval_type_wise(data: list, batch_size: int, type_key: str, task: str, data_path: str, jsonl_path: str) -> dict:
    grouped_data = defaultdict(list)
    for item in data:
        modality_type = item.get(type_key, 'unknown')
        grouped_data[modality_type].append(item)
    
    # ============================================================
    # Write a modality-split VLM JSONL file before the final loop.
    # ============================================================
    import json

    # Source file: xxx.jsonl
    base_name = os.path.basename(jsonl_path)
    name, ext = os.path.splitext(base_name)

    output_root = os.path.join("./eval_results_type_wise", name)
    os.makedirs(output_root, exist_ok=True)

    # Generate xxx_with_vlm.jsonl
    vlm_jsonl_path = os.path.join("./eval_results", f"{name}_with_vlm{ext}")

    # Process only when the file exists.
    if os.path.exists(vlm_jsonl_path):
        modality_buckets = defaultdict(list)

        # Read all entries from vlm_jsonl_path.
        with open(vlm_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    modality = obj.get("modality", "unknown")
                    modality_buckets[modality].append(obj)
                except json.JSONDecodeError:
                    continue

        # Write one file per modality.
        for modality, items in modality_buckets.items():
            out_path = os.path.join(
                output_root,
                f"{modality}_{os.path.basename(vlm_jsonl_path)}"
            )
            with open(out_path, "w", encoding="utf-8") as fw:
                for obj in items:
                    fw.write(json.dumps(obj, ensure_ascii=False) + "\n")

            logging.info(f"[VLM SPLIT] {modality}: {len(items)} records -> {out_path}")
    else:
        raise FileNotFoundError(f"[VLM SPLIT ERROR] File not found: {vlm_jsonl_path}")



    all_results = {}
    for modality_type, subset_data in grouped_data.items():
        logging.info(f"Evaluating group '{modality_type}' ({len(subset_data)} samples)")
        # Create a distinct checkpoint filename for each group.
        subset_jsonl_path = os.path.join(output_root, f"{modality_type}_{os.path.basename(jsonl_path)}")
        all_results[modality_type] = await basic_eval_for_type_wise(subset_data, batch_size, task, data_path, subset_jsonl_path, judge_config)
        print(all_results)
    

    return all_results


def save_results(results: dict, jsonl_path: str, task: str):
    """Save evaluation results to a JSON file."""
    output_dir = os.environ.get('MEDGEN_EVAL_RESULTS_DIR', './eval_results')
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.basename(jsonl_path)
    file_name = os.path.splitext(base_name)[0] + f"{task}_eval_results.json"
    output_path = os.path.join(output_dir, file_name)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
        
    logging.info(f"Evaluation results saved to: {output_path}")

def save_results_for_type_wise(results: dict, jsonl_path: str, task: str):
    """Save evaluation results to a JSON file."""
    output_dir = './eval_results_type_wise'
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.basename(jsonl_path)
    file_name = os.path.splitext(base_name)[0] + f"{task}_eval_results.json"
    output_path = os.path.join(output_dir, file_name)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
        
    logging.info(f"Evaluation results saved to: {output_path}")


def metrics_from_record(item: dict, task: str) -> dict:
    """Extract already-computed local and VLM metrics from one result record."""
    metrics = {}
    for key in [
        'LPIPS', 'PSNR', 'SSIM',
        'MedImageInsight_Similarity',
        'BLEU', 'BERT_Score',
        'RadGraph_F1', 'Text_EM'
    ]:
        value = item.get(key)
        if isinstance(value, (int, float)) and np.isfinite(value):
            metrics[key] = float(value)

    judge_specs = [
        ('vlm_judge_w_gt_result', 'W_GT'),
        ('vlm_judge_wo_gt_result', 'WO_GT'),
    ]
    for field, suffix in judge_specs:
        judge = item.get(field)
        if not valid_judge_result(judge):
            continue
        for source_key, metric_prefix in JUDGE_DIMENSIONS.items():
            metrics[f'{metric_prefix}_{suffix}'] = float(judge[source_key]['score'])
        metrics[f'VLM_Overall_Score_{suffix}'] = float(judge['overall_score'])
    return metrics


async def aggregate_type_wise_results(
    data: list,
    type_key: str,
    task: str,
    jsonl_path: str,
    n_boot: int = BOOTSTRAP_SAMPLES,
    judge_config: str = './config.yaml',
) -> dict:
    """Aggregate metrics already present in an evaluated JSONL by a record key."""
    for item in data:
        validate_paper_task(paper_task_for_item(item))
    grouped_data = defaultdict(list)
    for item in data:
        grouped_data[str(item.get(type_key, 'unknown'))].append(item)

    base_name = os.path.basename(jsonl_path)
    name = os.path.splitext(base_name)[0]
    output_root = os.path.join('./eval_results_type_wise', name)
    os.makedirs(output_root, exist_ok=True)

    all_results = {}
    for group_name, records in sorted(grouped_data.items()):
        group_metrics = defaultdict(list)
        radgraph_total = 0
        radgraph_applicable = 0
        for item in records:
            for metric_name, value in metrics_from_record(item, task).items():
                group_metrics[metric_name].append(value)
            if 'normalized_answer_text' in item:
                radgraph_total += 1
                if item.get('radgraph_applicable'):
                    radgraph_applicable += 1

        summary = {'Sample_Count': len(records), 'Task_Coverage': task_coverage(data)}
        for metric_name, values in sorted(group_metrics.items()):
            finite_values = [value for value in values if isinstance(value, (int, float)) and np.isfinite(value)]
            if not finite_values:
                continue
            summary[f'Average_{metric_name}'] = float(np.mean(finite_values))
            summary[f'Std_{metric_name}'] = float(np.std(finite_values))
            ci = bootstrap_ci(finite_values, n_boot=n_boot)
            if ci['mean'] is not None:
                summary[f'Bootstrap95CI_Low_{metric_name}'] = ci['ci_low']
                summary[f'Bootstrap95CI_High_{metric_name}'] = ci['ci_high']
        if radgraph_total:
            summary['RadGraph_Coverage'] = radgraph_applicable / radgraph_total
        summary.update(calculate_accuracy_rates({
            metric: [value for value in values if isinstance(value, (int, float)) and np.isfinite(value)]
            for metric, values in group_metrics.items()
        }))
        summary['Error_Analysis'] = build_error_analysis(
            {str(item.get('sample_id') or generate_sample_id(item)): item for item in records},
            task,
        )
        all_results[group_name] = summary

        safe_group = group_name.replace('/', '_').replace(os.sep, '_')
        group_path = os.path.join(output_root, f'{safe_group}_{base_name}')
        with open(group_path, 'w', encoding='utf-8') as handle:
            for item in records:
                handle.write(json.dumps(item, ensure_ascii=False) + '\n')
        logging.info("Group %s: %d records -> %s", group_name, len(records), group_path)
    return all_results



async def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.mission == 'stats':
        results = await analyze_model_comparison(
            args.jsonl_path,
            args.task,
            n_boot=args.bootstrap_samples,
        )
        save_results_for_stats(results, args.jsonl_path)
        return

    # Load data.
    data = load_jsonl_data(args.jsonl_path[0])
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError('--max_samples must be a positive integer')
        data = data[:args.max_samples]
    if not data:
        return

    if args.validate_only:
        summary = validate_eval_input(data, args.task, args.data_path)
        if args.require_full_task_coverage and not summary['paper_task_coverage_complete']:
            raise ValueError(
                'Input does not cover all 16 paper tasks; missing: '
                + ', '.join(summary['missing_paper_tasks'])
            )
        print("Evaluation input validation passed (metric models were not loaded and no API was called):")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if EVAL_DEPENDENCY_ERROR is not None:
        raise RuntimeError(
            "Full evaluation dependencies are missing; install requirements.txt as described in the README"
        ) from EVAL_DEPENDENCY_ERROR

    if args.local_metrics_only and args.mission != 'basic_eval':
        raise ValueError('--local-metrics-only supports only --mission basic_eval')

    jsonl_path = args.jsonl_path[0]
    coverage = task_coverage(data)
    if args.require_full_task_coverage and not coverage['complete']:
        raise ValueError(
            'Input does not cover all 16 paper tasks; missing: '
            + ', '.join(coverage['missing_tasks'])
        )
    # Run evaluation.
    if args.mission == 'basic_eval':
        results = await basic_eval(
            data,
            args.batch_size,
            args.task,
            args.data_path,
            jsonl_path,
            run_vlm_judge=not args.local_metrics_only,
            judge_model=args.judge_model,
            judge_backend=args.judge_backend,
            judge_config=args.judge_config,
            judge_base_url=args.judge_base_url or None,
            judge_api_key=args.judge_api_key or None,
            enable_clinical_text_metrics=args.enable_clinical_text_metrics,
            enable_radgraph=not args.disable_radgraph,
            n_boot=args.bootstrap_samples,
        )
    elif args.mission == 'type_wise':
        results = await aggregate_type_wise_results(
            data,
            args.type_key,
            args.task,
            jsonl_path,
            n_boot=args.bootstrap_samples,
            judge_config=args.judge_config,
        )
    else:
        logging.error(f"Unknown mission: {args.mission}")
        return

    # Save results.
    if args.mission == 'type_wise':
        save_results_for_type_wise(results, jsonl_path, args.task)
    else:   
        save_results(results, jsonl_path, args.task)

if __name__ == '__main__':
    asyncio.run(main())
    
    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_gpt-image-1-mini_edit.jsonl --mission basic_eval --batch_size 1
    
    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_gpt-image-1_edit.jsonl --mission basic_eval --batch_size 4
    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_gpt-image-1_edit.jsonl --mission type_wise --type_key modality --batch_size 4
    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_gpt-image-1_edit.jsonl --mission basic_eval --batch_size 4 --task vqa
    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_gpt-image-1_edit.jsonl --mission type_wise --type_key modality --batch_size 4 --task vqa
    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_gpt-image-1_edit.jsonl --mission basic_eval --batch_size 4 --task image_edit
    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_gpt-image-1_edit.jsonl --mission type_wise --type_key modality --batch_size 4 --task image_edit

    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_gemini-2.5-flash-image-preview_edit.jsonl --mission basic_eval --batch_size 4 --task image_edit
    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_gpt-image-1-mini_edit.jsonl --mission basic_eval --batch_size 4 --task image_edit
    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_qwen-image-edit_edit.jsonl --mission basic_eval --batch_size 4 --task image_edit
    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_doubao-seedream_edit.jsonl --mission basic_eval --batch_size 4 --task image_edit
    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_Ming-UniVision_EDIT_edit.jsonl --mission basic_eval --batch_size 4 --task image_edit
    # python eval.py --jsonl_path ./inference_jsonl/gpt-4o-mini_imagen-4.0-fast_edit.jsonl --mission basic_eval --batch_size 4 --task image_edit
    # python eval.py --jsonl_path ./inference_jsonl/gemini-2.5-flash-lite_imagen-4.0-fast_edit.jsonl --mission basic_eval --batch_size 4 --task image_edit
    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_dall-e-3_edit.jsonl --mission basic_eval --batch_size 4 --task image_edit
    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_imagen-4.0-fast_edit.jsonl --mission basic_eval --batch_size 4 --task image_edit

    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_doubao-seedream_generate.jsonl --mission basic_eval --batch_size 4 --task multimodal_generation
    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_dall-e-3_generate.jsonl --mission basic_eval --batch_size 4 --task multimodal_generation
    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_imagen-4.0-fast_generate.jsonl --mission basic_eval --batch_size 4 --task multimodal_generation
    # python eval.py --jsonl_path ./inference_jsonl/gpt-4o-mini_doubao-seedream_generate.jsonl --mission basic_eval --batch_size 4 --task multimodal_generation
    # python eval.py --jsonl_path ./inference_jsonl/gemini-2.5-flash-lite_doubao-seedream_generate.jsonl --mission basic_eval --batch_size 4 --task multimodal_generation
    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_gemini-2.5-flash-image-preview_generate.jsonl --mission basic_eval --batch_size 4 --task multimodal_generation
    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_Ming-UniVision_EDIT4GEN_generate.jsonl --mission basic_eval --batch_size 4 --task multimodal_generation
    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-instruct_Showo_GENERATION_generate.jsonl --mission basic_eval --batch_size 4 --task multimodal_generation
    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-instruct_Showo_EDIT_generate.jsonl --mission basic_eval --batch_size 4 --task multimodal_generation


    # python eval.py --jsonl_path ./inference_jsonl/gemini-2.5-flash-lite_vqa.jsonl --mission basic_eval --batch_size 4 --task vqa
    # python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_vqa.jsonl --mission basic_eval --batch_size 4 --task vqa
    # python eval.py --jsonl_path ./inference_jsonl/gpt-4o-mini_vqa.jsonl --mission basic_eval --batch_size 4 --task vqa
    # python eval.py --jsonl_path ./inference_jsonl/Ming-UniVision_VLM_vqa.jsonl --mission basic_eval --batch_size 4 --task vqa
    # python eval.py --jsonl_path ./inference_jsonl/HuatuoGPT-Vision_vqa.jsonl --mission basic_eval --batch_size 4 --task vqa
    # python eval.py --jsonl_path ./inference_jsonl/RadFM_vqa.jsonl --mission basic_eval --batch_size 4 --task vqa
    # python eval.py --jsonl_path ./inference_jsonl/Showo_VLM_vqa.jsonl --mission basic_eval --batch_size 4 --task vqa


#----------------------------type_wise------------------------------
# python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_doubao-seedream_edit.jsonl --mission type_wise --type_key modality --batch_size 4 --task image_edit
# python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_qwen-image-edit_edit.jsonl --mission type_wise --type_key modality --batch_size 4 --task image_edit
# python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_gpt-image-1-mini_edit.jsonl --mission type_wise --type_key modality --batch_size 4 --task image_edit
# python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_dall-e-3_edit.jsonl --mission type_wise --type_key modality --batch_size 4 --task image_edit
# python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_imagen-4.0-fast_edit.jsonl --mission type_wise --type_key modality --batch_size 4 --task image_edit
# python eval.py --jsonl_path ./inference_jsonl/gpt-4o-mini_imagen-4.0-fast_edit.jsonl --mission type_wise --type_key modality --batch_size 4 --task image_edit
# python eval.py --jsonl_path ./inference_jsonl/gemini-2.5-flash-lite_imagen-4.0-fast_edit.jsonl --mission type_wise --type_key modality --batch_size 4 --task image_edit
# python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_gemini-2.5-flash-image-preview_edit.jsonl --mission type_wise --type_key modality --batch_size 4 --task image_edit
# python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-instruct_Showo_EDIT_generate.jsonl --mission type_wise --type_key modality --batch_size 4 --task image_edit
# python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_Ming-UniVision_EDIT_edit.jsonl --mission type_wise --type_key modality --batch_size 4 --task image_edit



# python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_doubao-seedream_generate.jsonl --mission type_wise --type_key modality --batch_size 4 --task multimodal_generation
# python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_dall-e-3_generate.jsonl --mission type_wise --type_key modality --batch_size 4 --task multimodal_generation
# python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_imagen-4.0-fast_generate.jsonl --mission type_wise --type_key modality --batch_size 4 --task multimodal_generation
# python eval.py --jsonl_path ./inference_jsonl/gpt-4o-mini_doubao-seedream_generate.jsonl --mission type_wise --type_key modality --batch_size 4 --task multimodal_generation
# python eval.py --jsonl_path ./inference_jsonl/gemini-2.5-flash-lite_doubao-seedream_generate.jsonl --mission type_wise --type_key modality --batch_size 4 --task multimodal_generation
# python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_gemini-2.5-flash-image-preview_generate.jsonl --mission type_wise --type_key modality --batch_size 4 --task multimodal_generation
# python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-instruct_Showo_GENERATION_generate.jsonl --mission type_wise --type_key modality --batch_size 4 --task multimodal_generation
# python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_Ming-UniVision_EDIT4GEN_generate.jsonl --mission type_wise --type_key modality --batch_size 4 --task multimodal_generation




#python eval.py --jsonl_path ./inference_jsonl/qwen3-vl-235b-a22b-instruct_vqa.jsonl --mission type_wise --type_key modality --batch_size 4 --task vqa
#python eval.py --jsonl_path ./inference_jsonl/gemini-2.5-flash-lite_vqa.jsonl --mission type_wise --type_key modality --batch_size 4 --task vqa
#python eval.py --jsonl_path ./inference_jsonl/gpt-4o-mini_vqa.jsonl --mission type_wise --type_key modality --batch_size 4 --task vqa
#python eval.py --jsonl_path ./inference_jsonl/HuatuoGPT-Vision_vqa.jsonl --mission type_wise --type_key modality --batch_size 4 --task vqa
#python eval.py --jsonl_path ./inference_jsonl/RadFM_vqa.jsonl --mission type_wise --type_key modality --batch_size 4 --task vqa
#python eval.py --jsonl_path ./inference_jsonl/Showo_VLM_vqa.jsonl --mission type_wise --type_key modality --batch_size 4 --task vqa
#python eval.py --jsonl_path ./inference_jsonl/Ming-UniVision_VLM_vqa.jsonl --mission type_wise --type_key modality --batch_size 4 --task vqa
