'''
整体级（同时考虑文本图像）：

MLLM（4o）评分
根据输入/输出图文列对若干个维度评分：

连贯性（Coherence）：文本与图像是否传达统一信息。
内容准确性（Content Accuracy）：文本与图像的事实正确性。
相关性与响应性（Relevance）：生成内容是否贴合查询需求。
视觉-文本对齐（Visual-Textual Alignment）：图像与文本信息的匹配程度。
一致性（Consistency）：图像类型与文本格式是否跟指令一致
'''


# todo:参考范例和整体级评分完成下面两个提示词的润色

# vlm_holistic_judge_w_gt_prompt = '''
# You are a helpful AI judge expert that good at judge the medical image text generation model performance.
# 你会收到ground truth text和image，还有模型生成的待评价 text和image。 你需要对比评价2者，并且依照ground truth对模型生成评分。

# image grounding：
# 所有输入图片都有人为添加渲染的黑色文本，在图片下缘的剧中位置，用来指示当前图片的类型。
# 对于ground truth image，“Ground Truth”作为其image grounding
# 对于生成 image，“Generated Answer”作为其image grounding

# 从下列几个维度做评价：
# 连贯性（Coherence）：文本与图像是否传达统一信息。
# 内容准确性（Content Accuracy）：文本与图像的事实正确性。
# 相关性与响应性（Relevance）：生成内容是否贴合查询需求。
# 视觉-文本对齐（Visual-Textual Alignment）：图像与文本信息的匹配程度。
# 一致性（Consistency）：图像类型与文本格式是否跟指令一致

# Output Requirement: Please output in JSON format, including scores for each dimension (on a scale of 1-10) and a final overall score (on a scale of 1-10). Also provide brief explanations for each score. The JSON should follow this structure:

# '''

# vlm_holistic_judge_wo_gt_prompt = '''
# You are a helpful AI judge expert that good at judge the medical image text generation model performance.
# 你会收到作为任务输入的instruction text和image，还有模型生成的待评价 text和image。 你需要对比2者，并且依据文本指令去判断模型生成内容是否很好的完成了文本指令对图片的要求。

# image grounding：
# 所有输入图片都有人为添加渲染的黑色文本，在图片下缘的剧中位置，用来指示当前图片的类型。
# 对于input image，“Input”作为其image grounding
# 对于output image，“Output”作为其image grounding

# 从下列几个维度做评价：
# 连贯性（Coherence）：文本与图像是否传达统一信息。
# 内容准确性（Content Accuracy）：文本与图像的事实正确性。
# 相关性与响应性（Relevance）：生成内容是否贴合查询需求。
# 视觉-文本对齐（Visual-Textual Alignment）：图像与文本信息的匹配程度。
# 一致性（Consistency）：图像类型与文本格式是否跟指令一致

# Output Requirement: Please output in JSON format, including scores for each dimension (on a scale of 1-10) and a final overall score (on a scale of 1-10). Also provide brief explanations for each score. The JSON should follow this structure:

# '''


vlm_holistic_judge_w_gt_prompt = ["""
You are a helpful and impartial AI judge expert specialized in evaluating medical image-text generation model performance. You will be provided with ground truth text and images, as well as model-generated text or images for evaluation. Please compare and evaluate these against the ground truth.

Image Grounding Information:
All input images contain artificially rendered black text at the bottom center to indicate the image type:
- Ground truth images are labeled with "Ground Truth"
- Generated images are labeled with "Generated Answer"

Judge Requirement: Evaluate the model-generated content based on the following dimensions:
1. Content Accuracy: The factual correctness of both textual information and visual elements, particularly important in medical contexts.
2. Relevance and Responsiveness: How well the generated content addresses the given query and meets the specific requirements.
3. Consistency: Whether the image type or text format align with the given instructions and maintain coherent style.
""",
"""
4. Visual-Textual Alignment: The degree to which generated images match and support the accompanying text information.
5. Coherence: How well the text and images work together to convey a unified message, ensuring consistency between textual and visual information.
""",
"""

Output Requirement: Please output in JSON format, including scores for each dimension (on a scale of 1-10) and a final overall score (on a scale of 1-10). Also provide brief explanations for each score. The JSON should follow this structure:

{{
  "content_accuracy": {{
    "score": 0,
    "explanation": ""
  }},
  "relevance_and_responsiveness": {{
    "score": 0,
    "explanation": ""
  }},
  "consistency": {{
    "score": 0,
    "explanation": ""
  }},
""",
"""
  "coherence": {{
    "score": 0,
    "explanation": ""
  }},
  "visual_textual_alignment": {{
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

Judge Requirement: Evaluate the model-generated content based on the following dimensions:
1. Content Accuracy: The factual correctness of both textual information and visual elements, particularly important in medical contexts.
2. Relevance and Responsiveness: How well the generated content addresses the given query and meets the specific requirements.
3. Consistency: Whether the image type or text format align with the given instructions and maintain coherent style.
""",
"""
4. Visual-Textual Alignment: The degree to which generated images match and support the accompanying text information.
5. Coherence: How well the text and images work together to convey a unified message, ensuring consistency between textual and visual information.
""",
"""

Output Requirement: Please output in JSON format, including scores for each dimension (on a scale of 1-10) and a final overall score (on a scale of 1-10). Also provide brief explanations for each score. The JSON should follow this structure:

{{
  "content_accuracy": {{
    "score": 0,
    "explanation": ""
  }},
  "relevance_and_responsiveness": {{
    "score": 0,
    "explanation": ""
  }},
  "consistency": {{
    "score": 0,
    "explanation": ""
  }},
""",
"""
  "coherence": {{
    "score": 0,
    "explanation": ""
  }},
  "visual_textual_alignment": {{
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


['\nYou are a helpful and impartial AI judge expert specialized in evaluating medical image-text generation model performance. You will be provided with instruction text and input images, as well as model-generated text or images for evaluation. Please assess whether the generated content successfully fulfills the requirements specified in the text instructions.\n\nImage Grounding Information:\nAll images contain artificially rendered black text at the bottom center to indicate the image type:\n- Input images are labeled with "Input"\n- Output images that model generate are labeled with "Output"\n\nJudge Requirement: Evaluate the model-generated content based on the following dimensions:\n1. Content Accuracy: The factual correctness of both textual information and visual elements, particularly important in medical contexts.\n2. Relevance and Responsiveness: How well the generated content addresses the given query and meets the specific requirements.\n3. Consistency: Whether the image type or text format align with the given instructions and maintain coherent style.\n',
    '', '\n\nOutput Requirement: Please output in JSON format, including scores for each dimension (on a scale of 1-10) and a final overall score (on a scale of 1-10). Also provide brief explanations for each score. The JSON should follow this structure:\n\n{{\n  "content_accuracy": {{\n    "score": 0,\n    "explanation": ""\n  }},\n  "relevance_and_responsiveness": {{\n    "score": 0,\n    "explanation": ""\n  }},\n  "consistency": {{\n    "score": 0,\n    "explanation": ""\n  }},\n', '', '\n  "overall_score": 0,\n  "final_thoughts": ""\n}}\n\nHere is the Instruction:\n', '\nHere is the Input (Text and Images):\n', '\nHere is the Generated Output (Text and Images):\n', '\nNow please judge the generated output based on how well it fulfills the instruction requirements. Remember to output in JSON format with scores for each dimension (on a scale of 1-10) and a final overall score (on a scale of 1-10). Also provide brief explanations for each score.\n']
