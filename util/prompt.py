vlm_holistic_judge_w_gt_prompt = ["""
You are a helpful and impartial AI judge expert specialized in evaluating medical image-text generation model performance. You will be provided with ground truth text and images, as well as model-generated text or images for evaluation. Please compare and evaluate these against the ground truth.

Image Grounding Information:
All input images contain artificially rendered black text at the bottom center to indicate the image type:
- Ground truth images are labeled with "Ground Truth"
- Generated images are labeled with "Generated Answer"

Judge Requirement: Evaluate the model-generated content based on the following medical dimensions:
1. Anatomical Accuracy: Whether anatomical structures, locations, and spatial relationships are correct.
2. Clinical Finding Accuracy: Whether findings, pathology, severity, and devices are medically correct.
3. Instruction Compliance: Whether the generated content follows the instruction and requested transformation exactly.
Global pixel similarity alone is not sufficient. Explicitly inspect clinically salient small or subtle details, including focal lesions, small nodules, thin vessels, microcalcifications, fine boundaries, devices, and target-region detail. Penalize a missed, blurred, distorted, newly invented, or relocated clinically salient feature even when global anatomy and overall appearance look similar.
""",
"""
4. Cross-Modal Consistency: Whether the text and image content support the same medical conclusion.
5. Hallucination/Omission Control: Whether clinically important findings are invented or omitted.
""",
"""

Output Requirement: Please output in JSON format, including scores for each dimension (on a scale of 1-10) and a final overall score (on a scale of 1-10). Also provide brief explanations for each score. The JSON should follow this structure:

{{
  "anatomical_accuracy": {{
    "score": 0,
    "explanation": ""
  }},
  "clinical_finding_accuracy": {{
    "score": 0,
    "explanation": ""
  }},
  "instruction_compliance": {{
    "score": 0,
    "explanation": ""
  }},
""",
"""
  "cross_modal_consistency": {{
    "score": 0,
    "explanation": ""
  }},
  "hallucination_omission_control": {{
    "score": 0,
    "explanation": ""
  }},
""",
"""
  "overall_score": 0,
  "final_thoughts": ""
}}

Here is the Instruction:
""",
"""
Here is the Ground Truth:
""",
"""
Here is the Generated Answer:
""",
"""
Now please judge the generated answer against the ground truth. Remember to output in JSON format with scores for each dimension (on a scale of 1-10) and a final overall score (on a scale of 1-10). Also provide brief explanations for each score.
"""]

vlm_holistic_judge_wo_gt_prompt = ["""
You are a helpful and impartial AI judge expert specialized in evaluating medical image-text generation model performance. You will be provided with instruction text and input images, as well as model-generated text or images for evaluation. Please assess whether the generated content successfully fulfills the requirements specified in the text instructions.

Image Grounding Information:
All images contain artificially rendered black text at the bottom center to indicate the image type:
- Input images are labeled with "Input"
- Output images that model generate are labeled with "Output"

Judge Requirement: Evaluate the model-generated content based on the following medical dimensions:
1. Anatomical Accuracy: Whether anatomical structures, locations, and spatial relationships are correct.
2. Clinical Finding Accuracy: Whether findings, pathology, severity, and devices are medically correct.
3. Instruction Compliance: Whether the generated content follows the instruction and requested transformation exactly.
Global pixel similarity alone is not sufficient. Explicitly inspect clinically salient small or subtle details, including focal lesions, small nodules, thin vessels, microcalcifications, fine boundaries, devices, and target-region detail. Penalize a missed, blurred, distorted, newly invented, or relocated clinically salient feature even when global anatomy and overall appearance look similar.
""",
"""
4. Cross-Modal Consistency: Whether the text and image content support the same medical conclusion.
5. Hallucination/Omission Control: Whether clinically important findings are invented or omitted.
""",
"""

Output Requirement: Please output in JSON format, including scores for each dimension (on a scale of 1-10) and a final overall score (on a scale of 1-10). Also provide brief explanations for each score. The JSON should follow this structure:

{{
  "anatomical_accuracy": {{
    "score": 0,
    "explanation": ""
  }},
  "clinical_finding_accuracy": {{
    "score": 0,
    "explanation": ""
  }},
  "instruction_compliance": {{
    "score": 0,
    "explanation": ""
  }},
""",
"""
  "cross_modal_consistency": {{
    "score": 0,
    "explanation": ""
  }},
  "hallucination_omission_control": {{
    "score": 0,
    "explanation": ""
  }},
""",
"""
  "overall_score": 0,
  "final_thoughts": ""
}}

Here is the Instruction:
""",
""" """,
"""
Here is the Generated Answer:
""",
"""
Now please judge the generated output based on how well it fulfills the instruction requirements. Remember to output in JSON format with scores for each dimension (on a scale of 1-10) and a final overall score (on a scale of 1-10). Also provide brief explanations for each score.
"""]

# --- Task-specific judge checklists ---
# Preserve the five clinical dimensions and add observable task-specific
# criteria to reduce prompt drift.
TASK_JUDGE_CHECKLISTS = {
    'vqa': (
        "1. Answer fidelity: the generated text answers the exact question asked, "
        "not a related but different question.\n"
        "2. Finding visibility: every reported finding is visible in the input image "
        "and consistent with the ground-truth answer.\n"
        "3. Clinical strictness: penalize invented findings, wrong laterality/site, "
        "and internally contradictory statements.\n"
        "4. Format match: modality, anatomy, and answer structure match the ground truth."
    ),
    'image_edit': (
        "1. Transformation applied: the requested edit (artifact removal, contrast "
        "enhancement, resolution editing, etc.) is actually present in the target region.\n"
        "2. Context comparison: inspect non-target anatomy that is visible in the provided images; do not infer pixel-level preservation beyond the available references.\n"
        "3. Artifact-free: penalize new blur, noise, or unrealistic textures introduced "
        "by the edit.\n"
        "4. Small-detail inspection: explicitly check focal lesions, thin vessels, microcalcifications, fine boundaries, devices, and other subtle target-region detail.\n"
        "5. Clinical plausibility: the edited image stays anatomically and clinically "
        "consistent with the ground truth."
    ),
    'multimodal_generation': (
        "1. Cross-modal agreement: generated image and text support the same disease "
        "stage, anatomy, and findings.\n"
        "2. Transformation plausibility: progression, reconstruction, and projection "
        "changes are anatomically plausible.\n"
        "3. No invented content: penalize structures, findings, or devices that do not "
        "appear in the ground truth.\n"
        "4. Small-detail inspection: explicitly check focal lesions, thin vessels, microcalcifications, fine boundaries, devices, and subtle target-region detail.\n"
        "5. Format match: the output image-text pairing follows the instruction."
    ),
}


def build_vlm_judge_prompt(task: str, with_gt: bool) -> list:
    """Build a structured checklist prompt while preserving the base layout."""
    base = list(vlm_holistic_judge_w_gt_prompt if with_gt else vlm_holistic_judge_wo_gt_prompt)
    checklist = TASK_JUDGE_CHECKLISTS.get(task, '')
    if checklist:
        base[0] = (
            base[0]
            + "\n\nTask-Specific Checklist ("
            + str(task)
            + "):\n"
            + checklist
        )
    return base


def build_vqa_control_judge_prompt(with_reference: bool) -> str:
    """Prompt for VQA controls; no image is ever called a ground-truth image."""
    reference = (
        "A text reference answer is provided for sensitivity analysis. It is a "
        "text reference only; there is no ground-truth output image.\n"
        "TEXT REFERENCE:\n{reference}\n"
        if with_reference else ""
    )
    return """You are an impartial medical VQA judge. Evaluate only the current input image(s), the instruction, and the model response. Do not infer or mention experimental conditions, donor identities, sample IDs, or hidden metadata.\n\nScore each dimension from 1 (severely incorrect) to 5 (fully correct): anatomical_accuracy, clinical_finding_accuracy, instruction_compliance, cross_modal_consistency, hallucination_omission_control. Inspect focal lesions, subtle structures, laterality, devices, boundaries, and clinically important omissions or hallucinations. In the reference-free view, do not compare against an answer that is not shown; judge whether the response is supported by the current image and answers the instruction.\n\nINSTRUCTION:\n{instruction}\n\n{reference}MODEL RESPONSE:\n{response}\n\nReturn JSON only with this schema: {{"anatomical_accuracy":{{"score":0,"explanation":""}},"clinical_finding_accuracy":{{"score":0,"explanation":""}},"instruction_compliance":{{"score":0,"explanation":""}},"cross_modal_consistency":{{"score":0,"explanation":""}},"hallucination_omission_control":{{"score":0,"explanation":""}},"overall_score":0,"final_thoughts":""}}"""
