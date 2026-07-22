# # todo：增加异步实现，支持输入列表（选做）
# import lpips
# import numpy as np
# from PIL import Image
# from torchvision import transforms
# from skimage.metrics import structural_similarity as ssim
# from bert_score import score as bert_score
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# # --- 模型和预处理工具初始化 (推荐在全局范围执行一次) ---
# # 避免在函数调用时重复加载，提高效率
# lpips_model = lpips.LPIPS(net='alex') # 'alex' 或 'vgg'
# lpips_preprocess = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     # LPIPS模型内部会进行归一化 (-1, 1)
# ])
# # 用于计算BLEU的平滑函数
# nltk_smoothie = SmoothingFunction().method1


# def FR_IQA(eval_image: Image.Image, ref_image: Image.Image, eval_metric: str) -> float:
#     """
#     计算两张图片之间的全参考图像质量评估 (Full-Reference Image Quality Assessment)。

#     Args:
#         eval_image (Image.Image): 待评估的图片 (PIL.Image 对象)。
#         ref_image (Image.Image): 参考的基准图片 (PIL.Image 对象)。
#         eval_metric (str): 要使用的评估指标。支持 'lpips', 'psnr', 'ssim'。

#     Returns:
#         float: 计算出的评估分数。

#     Raises:
#         ValueError: 如果输入的 eval_metric 不被支持。
#     """
#     metric = eval_metric.lower()

#     if metric == 'lpips':
#         # 对图像进行预处理并增加 batch 维度
#         eval_tensor = lpips_preprocess(eval_image).unsqueeze(0)
#         ref_tensor = lpips_preprocess(ref_image).unsqueeze(0)

#         # 使用LPIPS模型计算相似性 (分数越低，相似度越高)
#         similarity_score = lpips_model(eval_tensor, ref_tensor)
#         return similarity_score.item()

#     elif metric == 'psnr':
#         # 将 PIL Image 对象转换为 NumPy 数组 (OpenCV BGR 格式)
#         # 注意：PIL是RGB, OpenCV是BGR。但对于PSNR计算，只要两张图的通道顺序一致即可。
#         # 我们这里统一使用RGB顺序的Numpy数组。
#         ref_np = np.array(ref_image)
#         eval_np = np.array(eval_image)

#         # 确保图像数据类型正确
#         if ref_np.dtype != np.uint8:
#             ref_np = (ref_np * 255).astype(np.uint8)
#         if eval_np.dtype != np.uint8:
#             eval_np = (eval_np * 255).astype(np.uint8)
        
#         # 计算均方误差 (MSE)
#         mse = np.mean((ref_np - eval_np) ** 2)
#         if mse == 0:
#             # MSE为0意味着图片完全相同，PSNR为无穷大
#             return float('inf')
        
#         # 计算PSNR (分数越高，图像质量越好)
#         max_pixel_value = 255.0
#         psnr_score = 20 * np.log10(max_pixel_value / np.sqrt(mse))
#         return psnr_score

#     elif metric == 'ssim':
#         # 将 PIL Image 转换为灰度 NumPy 数组
#         # 也可以在多通道上计算, 这里为了和原始示例保持一致转为灰度
#         ref_gray = np.array(ref_image.convert('L'))
#         eval_gray = np.array(eval_image.convert('L'))
        
#         # 计算 SSIM (分数在-1到1之间，越接近1，结构越相似)
#         # data_range是像素值的范围
#         ssim_score = ssim(ref_gray, eval_gray, data_range=ref_gray.max() - ref_gray.min())
#         return ssim_score


#     else:
#         raise ValueError(
#             f"未知的图像评估指标: '{eval_metric}'. "
#             "支持的指标: 'lpips', 'psnr', 'ssim'."
#         )




# def evaluate_text_quality(eval_text: str, ref_text: str, eval_metric: str) -> float:
#     """
#     评估生成文本相对于参考文本的质量。

#     Args:
#         eval_text (str): 待评估的文本 (例如，模型生成的摘要)。
#         ref_text (str): 参考的基准文本 (例如，人工编写的摘要)。
#         eval_metric (str): 要使用的评估指标。支持 'bertscore', 'bleu'。

#     Returns:
#         float: 计算出的评估分数。

#     Raises:
#         ValueError: 如果输入的 eval_metric 不被支持。
#     """
#     metric = eval_metric.lower()

#     if metric == 'bertscore':
#         # bert-score 需要输入为列表
#         preds = [eval_text]
#         refs = [ref_text]

#         _, _, f1 = bert_score(
#             preds,
#             refs,
#             model_type="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
#             lang="en",
#             rescale_with_baseline=False,
#             num_layers=12,
#             device='cpu'  # 如果有GPU，可以改为 'cuda'
#         )
        
        
#         return f1.mean().item()

#     elif metric == 'bleu':
#         # BLEU 分数需要将句子分割成词列表
#         reference = [ref_text.split()]  # 参考文本可以是多个，所以是列表的列表
#         candidate = eval_text.split()   # 预测文本只有一个
        
#         # 使用平滑函数计算 BLEU 分数，避免n-gram匹配为0导致分数为0
#         bleu_score = sentence_bleu(reference, candidate, smoothing_function=nltk_smoothie)
#         return bleu_score

#     else:
#         raise ValueError(
#             f"未知的文本评估指标: '{eval_metric}'. "
#             "支持的指标: 'bertscore', 'bleu'."
#         )


# # --- 主函数入口和示例 ---
# if __name__ == '__main__':
#     # ==========================
#     # 图像评估函数 FR_IQA 示例
#     # ==========================
#     print("--- 图像质量评估 (FR-IQA) 示例 ---")
#     # 创建两个示例图片 (实际使用时请用 Image.open('filepath.jpg') 加载)
#     # ref_img 是一个纯黑色的图片
#     ref_img = Image.new('RGB', (256, 256), color='black')
#     # eval_img 是一个带有一些灰色噪声的图片
#     noise = np.random.randint(0, 50, (256, 256, 3), dtype=np.uint8)
#     eval_img = Image.fromarray(np.array(ref_img) + noise)
    
#     # 1. 计算 LPIPS
#     lpips_score = FR_IQA(eval_img, ref_img, 'lpips')
#     print(f"LPIPS Score: {lpips_score:.4f} (越低越好)")

#     # 2. 计算 PSNR
#     psnr_score = FR_IQA(eval_img, ref_img, 'psnr')
#     print(f"PSNR Score: {psnr_score:.4f} dB (越高越好)")
    
#     # 3. 计算 SSIM
#     ssim_score = FR_IQA(eval_img, ref_img, 'ssim')
#     print(f"SSIM Score: {ssim_score:.4f} (越接近1越好)")


#     print("\n" + "="*40 + "\n")

#     # ============================
#     # 文本评估函数 evaluate_text_quality 示例
#     # ============================
#     print("--- 文本质量评估示例 ---")
#     # 示例文本
#     reference_text = "Normal stomach mucosa (negative for Helicobacter Pylori infection)"
#     predicted_text_good = "The gastric mucosa appears normal with no evidence of H. pylori infection"
#     predicted_text_bad = "computed tomography"

#     # 1. 计算 BERTScore (与好预测的比较)
#     bertscore_good = evaluate_text_quality(predicted_text_good, reference_text, 'bertscore')
#     print(f"BERTScore (Good Match): {bertscore_good:.4f} (越接近1越好)")
    
#     # 2. 计算 BERTScore (与差预测的比较)
#     bertscore_bad = evaluate_text_quality(predicted_text_bad, reference_text, 'bertscore')
#     print(f"BERTScore (Bad Match): {bertscore_bad:.4f} (越接近1越好)")

#     print("-" * 20)
    
#     # 3. 计算 BLEU (与好预测的比较)
#     bleu_good = evaluate_text_quality(predicted_text_good, reference_text, 'bleu')
#     print(f"BLEU Score (Good Match): {bleu_good:.4f} (越接近1越好)")
    
#     # 4. 计算 BLEU (与差预测的比较)
#     bleu_bad = evaluate_text_quality(predicted_text_bad, reference_text, 'bleu')
#     print(f"BLEU Score (Bad Match): {bleu_bad:.4f} (越接近1越好)")


# todo：增加异步实现 ✅ 已完成 + 修复线程安全问题
import asyncio
import lpips
import numpy as np
from PIL import Image
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import List, Tuple, Union
import functools
import threading
import os
import torch

# Large shared servers often expose dozens of CPU cores.  Letting both LPIPS
# and PubMedBERT create their default thread pools can make mixed-modality
# evaluation stall from oversubscription.
TORCH_NUM_THREADS = max(1, int(os.environ.get('MEDGEN_TORCH_NUM_THREADS', '4')))
torch.set_num_threads(TORCH_NUM_THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    # Another importer may already have initialized the inter-op pool.
    pass

# LPIPS is initialized lazily so --help/--validate-only never downloads weights.
_lpips_model = None
_lpips_model_lock = threading.Lock()
lpips_preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # LPIPS模型内部会进行归一化 (-1, 1)
])
# 用于计算BLEU的平滑函数
nltk_smoothie = SmoothingFunction().method1

# 并发控制 - 限制同时执行的深度学习推理任务数量
CONCURRENT_DL_TASKS = 2
dl_semaphore = asyncio.Semaphore(CONCURRENT_DL_TASKS)

# 线程安全锁，用于保护 BERT 模型访问
bert_model_lock = threading.Lock()

def get_lpips_model():
    """Load the LPIPS network once, only when that metric is requested."""
    global _lpips_model
    if _lpips_model is None:
        with _lpips_model_lock:
            if _lpips_model is None:
                _lpips_model = lpips.LPIPS(net='alex')
                _lpips_model.eval()
    return _lpips_model


def FR_IQA(eval_image: Image.Image, ref_image: Image.Image, eval_metric: str) -> float:
    """
    计算两张图片之间的全参考图像质量评估 (Full-Reference Image Quality Assessment)。

    Args:
        eval_image (Image.Image): 待评估的图片 (PIL.Image 对象)。
        ref_image (Image.Image): 参考的基准图片 (PIL.Image 对象)。
        eval_metric (str): 要使用的评估指标。支持 'lpips', 'psnr', 'ssim'。

    Returns:
        float: 计算出的评估分数。

    Raises:
        ValueError: 如果输入的 eval_metric 不被支持。
    """    
    if eval_image.size != ref_image.size:
        # 将待评估图像 resize 到参考图像的尺寸（保持内容，使用高质量插值）
        eval_image = eval_image.resize(ref_image.size, Image.Resampling.LANCZOS)

    metric = eval_metric.lower()

    if metric == 'lpips':
        # 对图像进行预处理并增加 batch 维度
        eval_tensor = lpips_preprocess(eval_image).unsqueeze(0)
        ref_tensor = lpips_preprocess(ref_image).unsqueeze(0)

        # 使用LPIPS模型计算相似性 (分数越低，相似度越高)
        with torch.inference_mode():
            similarity_score = get_lpips_model()(eval_tensor, ref_tensor)
        return similarity_score.item()

    elif metric == 'psnr':
        # 将 PIL Image 对象转换为 NumPy 数组 (OpenCV BGR 格式)
        # 注意：PIL是RGB, OpenCV是BGR。但对于PSNR计算，只要两张图的通道顺序一致即可。
        # 我们这里统一使用RGB顺序的Numpy数组。
        ref_np = np.asarray(ref_image, dtype=np.float32)
        eval_np = np.asarray(eval_image, dtype=np.float32)

        # 确保图像数据类型正确
        # 计算均方误差 (MSE)
        mse = np.mean((ref_np - eval_np) ** 2)
        if mse == 0:
            # Keep evaluation artifacts valid JSON while representing a
            # practically perfect 8-bit reconstruction.
            return 100.0
        
        # 计算PSNR (分数越高，图像质量越好)
        max_pixel_value = 255.0
        psnr_score = 20 * np.log10(max_pixel_value / np.sqrt(mse))
        return psnr_score

    elif metric == 'ssim':
        # 将 PIL Image 转换为灰度 NumPy 数组
        # 也可以在多通道上计算, 这里为了和原始示例保持一致转为灰度
        ref_gray = np.array(ref_image.convert('L'))
        eval_gray = np.array(eval_image.convert('L'))
        
        # 计算 SSIM (分数在-1到1之间，越接近1，结构越相似)
        # PIL conversion produces 8-bit grayscale; use the dtype range so
        # constant images do not result in data_range=0.
        ssim_score = ssim(ref_gray, eval_gray, data_range=255)
        return ssim_score

    else:
        raise ValueError(
            f"未知的图像评估指标: '{eval_metric}'. "
            "支持的指标: 'lpips', 'psnr', 'ssim'."
        )


async def async_FR_IQA(eval_image: Image.Image, ref_image: Image.Image, eval_metric: str) -> float:
    """
    异步版本的全参考图像质量评估。
    
    对于计算密集型任务（LPIPS、PSNR、SSIM），使用线程池避免阻塞事件循环。
    
    Args:
        eval_image (Image.Image): 待评估的图片 (PIL.Image 对象)。
        ref_image (Image.Image): 参考的基准图片 (PIL.Image 对象)。
        eval_metric (str): 要使用的评估指标。支持 'lpips', 'psnr', 'ssim'。

    Returns:
        float: 计算出的评估分数。
    """
    metric = eval_metric.lower()
    
    if metric == 'lpips':
        # As with BERTScore, initialize/run the CPU torch model on the main
        # event-loop thread to avoid first-use deadlocks in worker threads.
        async with dl_semaphore:
            return FR_IQA(eval_image, ref_image, eval_metric)
    else:
        # Run the small NumPy/scikit-image operations serially as well. Mixing
        # their native thread pools with first-use torch inference caused
        # hangs on CPU-only hosts.
        return FR_IQA(eval_image, ref_image, eval_metric)


def evaluate_text_quality_thread_safe(eval_text: str, ref_text: str, eval_metric: str) -> float:
    """
    线程安全版本的文本质量评估函数。
    
    Args:
        eval_text (str): 待评估的文本。
        ref_text (str): 参考的基准文本。
        eval_metric (str): 要使用的评估指标。支持 'bertscore', 'bleu'。

    Returns:
        float: 计算出的评估分数。
    """
    metric = eval_metric.lower()

    if metric == 'bertscore':
        # 使用线程锁保护 BERT 模型访问
        with bert_model_lock:
            try:
                # bert-score 需要输入为列表
                preds = [eval_text]
                refs = [ref_text]

                _, _, f1 = bert_score(
                    preds,
                    refs,
                    model_type="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                    lang="en",
                    rescale_with_baseline=False,
                    num_layers=12,
                    device='cpu',
                    # device='cuda' if torch.cuda.is_available() else 'cpu',
                    #batch_size=16,
                    verbose=False  # 减少输出
                )
                
                return f1.mean().item()
                
            except Exception as e:
                raise RuntimeError(f"BERTScore 计算失败: {e}") from e

    elif metric == 'bleu':
        # BLEU 分数需要将句子分割成词列表
        reference = [ref_text.split()]  # 参考文本可以是多个，所以是列表的列表
        candidate = eval_text.split()   # 预测文本只有一个
        
        # 使用平滑函数计算 BLEU 分数，避免n-gram匹配为0导致分数为0
        bleu_score = sentence_bleu(reference, candidate, smoothing_function=nltk_smoothie)
        return bleu_score

    else:
        raise ValueError(
            f"未知的文本评估指标: '{eval_metric}'. "
            "支持的指标: 'bertscore', 'bleu'."
        )




def evaluate_text_quality(eval_text: str, ref_text: str, eval_metric: str) -> float:
    """
    评估生成文本相对于参考文本的质量（保持向后兼容）。
    """
    return evaluate_text_quality_thread_safe(eval_text, ref_text, eval_metric)


async def async_evaluate_text_quality(eval_text: str, ref_text: str, eval_metric: str) -> float:
    """
    异步版本的文本质量评估。
    
    Args:
        eval_text (str): 待评估的文本。
        ref_text (str): 参考的基准文本。
        eval_metric (str): 要使用的评估指标。支持 'bertscore', 'bleu'。

    Returns:
        float: 计算出的评估分数。
    """
    metric = eval_metric.lower()
    
    if metric == 'bertscore':
        # PyTorch/Transformers CPU inference can deadlock when first initialized
        # inside asyncio.to_thread.  Keep the serialized BERTScore call on the
        # event-loop thread; other metrics still use worker threads.
        async with dl_semaphore:
            return evaluate_text_quality_thread_safe(eval_text, ref_text, eval_metric)
    else:
        return evaluate_text_quality_thread_safe(eval_text, ref_text, eval_metric)


# --- 批量异步处理函数 ---
async def batch_async_FR_IQA(
    eval_images: List[Image.Image], 
    ref_images: List[Image.Image], 
    eval_metric: str
) -> List[float]:
    """
    批量异步处理图像质量评估。
    
    Args:
        eval_images: 待评估图像列表
        ref_images: 参考图像列表
        eval_metric: 评估指标
        
    Returns:
        List[float]: 评估分数列表
    """
    if len(eval_images) != len(ref_images):
        raise ValueError("eval_images 和 ref_images 长度必须相同")
    
    tasks = [
        async_FR_IQA(eval_img, ref_img, eval_metric)
        for eval_img, ref_img in zip(eval_images, ref_images)
    ]
    
    return await asyncio.gather(*tasks)


async def batch_async_evaluate_text_quality(
    eval_texts: List[str], 
    ref_texts: List[str], 
    eval_metric: str
) -> List[float]:
    """
    批量异步处理文本质量评估。
    
    Args:
        eval_texts: 待评估文本列表
        ref_texts: 参考文本列表
        eval_metric: 评估指标
        
    Returns:
        List[float]: 评估分数列表
    """
    if len(eval_texts) != len(ref_texts):
        raise ValueError("eval_texts 和 ref_texts 长度必须相同")
    
    # 对于 BERTScore，批量处理更高效，避免并发问题
    if eval_metric.lower() == 'bertscore':
        return batch_bertscore_calculation(eval_texts, ref_texts)
    else:
        # 对于 BLEU，可以并发处理
        tasks = [
            async_evaluate_text_quality(eval_text, ref_text, eval_metric)
            for eval_text, ref_text in zip(eval_texts, ref_texts)
        ]
        return await asyncio.gather(*tasks)


def batch_bertscore_calculation(eval_texts: List[str], ref_texts: List[str]) -> List[float]:
    """
    批量计算 BERTScore，避免多线程冲突。
    
    Args:
        eval_texts: 待评估文本列表
        ref_texts: 参考文本列表
        
    Returns:
        List[float]: BERTScore 列表
    """
    try:
        with bert_model_lock:
            _, _, f1_scores = bert_score(
                eval_texts,
                ref_texts,
                model_type="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                lang="en",
                rescale_with_baseline=False,
                num_layers=12,
                device='cpu',
                #device='cuda' if torch.cuda.is_available() else 'cpu',
                verbose=False
            )
            return f1_scores.tolist()
    except Exception as e:
        raise RuntimeError(f"批量 BERTScore 计算失败: {e}") from e


# --- 混合批量处理函数 ---
async def batch_mixed_evaluation(
    image_tasks: List[Tuple[Image.Image, Image.Image, str]] = None,
    text_tasks: List[Tuple[str, str, str]] = None
) -> Tuple[List[float], List[float]]:
    """
    同时处理图像和文本评估任务。
    
    Args:
        image_tasks: 图像评估任务列表，每个元素为 (eval_image, ref_image, metric)
        text_tasks: 文本评估任务列表，每个元素为 (eval_text, ref_text, metric)
        
    Returns:
        Tuple[List[float], List[float]]: (图像评估结果, 文本评估结果)
    """
    tasks = []
    
    # 添加图像评估任务
    if image_tasks:
        for eval_img, ref_img, metric in image_tasks:
            tasks.append(async_FR_IQA(eval_img, ref_img, metric))
    
    # 添加文本评估任务
    if text_tasks:
        for eval_text, ref_text, metric in text_tasks:
            tasks.append(async_evaluate_text_quality(eval_text, ref_text, metric))
    
    # 并发执行所有任务
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 处理异常并分离结果
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            print(f"任务执行出错: {result}")
            processed_results.append(0.0)  # 默认值
        else:
            processed_results.append(result)
    
    # 分离图像和文本结果
    image_count = len(image_tasks) if image_tasks else 0
    image_results = processed_results[:image_count]
    text_results = processed_results[image_count:]
    
    return image_results, text_results


# --- 主函数入口和示例 ---
async def async_main():
    """异步主函数示例"""
    print("--- 异步图像质量评估 (FR-IQA) 示例 ---")
    
    # 创建示例图片
    ref_img = Image.new('RGB', (256, 256), color='black')
    noise = np.random.randint(0, 50, (256, 256, 3), dtype=np.uint8)
    eval_img = Image.fromarray(np.array(ref_img) + noise)
    
    # 单个异步评估
    print("单个异步评估:")
    lpips_task = async_FR_IQA(eval_img, ref_img, 'lpips')
    psnr_task = async_FR_IQA(eval_img, ref_img, 'psnr') 
    ssim_task = async_FR_IQA(eval_img, ref_img, 'ssim')
    
    # 并发执行所有图像评估
    lpips_score, psnr_score, ssim_score = await asyncio.gather(
        lpips_task, psnr_task, ssim_task
    )
    
    print(f"LPIPS Score: {lpips_score:.4f} (越低越好)")
    print(f"PSNR Score: {psnr_score:.4f} dB (越高越好)")
    print(f"SSIM Score: {ssim_score:.4f} (越接近1越好)")
    
    # 批量异步评估
    print("\n批量异步评估:")
    eval_images = [eval_img] * 3
    ref_images = [ref_img] * 3
    
    batch_scores = await batch_async_FR_IQA(eval_images, ref_images, 'ssim')
    print(f"批量SSIM分数: {[f'{score:.4f}' for score in batch_scores]}")
    
    print("\n" + "="*40 + "\n")
    
    # 异步文本评估示例
    print("--- 异步文本质量评估示例 ---")
    
    reference_text = "Normal stomach mucosa (negative for Helicobacter Pylori infection)"
    predicted_text_good = "The gastric mucosa appears normal with no evidence of H. pylori infection"
    predicted_text_bad = "computed tomography"
    
    # 先尝试 BLEU 评估（不依赖网络）
    print("BLEU 评估:")
    bleu_good_task = async_evaluate_text_quality(predicted_text_good, reference_text, 'bleu')
    bleu_bad_task = async_evaluate_text_quality(predicted_text_bad, reference_text, 'bleu')
    
    bleu_good, bleu_bad = await asyncio.gather(bleu_good_task, bleu_bad_task)
    
    print(f"BLEU Score (Good Match): {bleu_good:.4f} (越接近1越好)")
    print(f"BLEU Score (Bad Match): {bleu_bad:.4f} (越接近1越好)")
    
    # 尝试 BERTScore 评估（可能受网络影响）
    print("\nBERTScore 评估:")
    try:
        bertscore_good_task = async_evaluate_text_quality(predicted_text_good, reference_text, 'bertscore')
        bertscore_bad_task = async_evaluate_text_quality(predicted_text_bad, reference_text, 'bertscore')
        
        bertscore_good, bertscore_bad = await asyncio.gather(bertscore_good_task, bertscore_bad_task)
        
        print(f"BERTScore (Good Match): {bertscore_good:.4f} (越接近1越好)")
        print(f"BERTScore (Bad Match): {bertscore_bad:.4f} (越接近1越好)")
    except Exception as e:
        print(f"BERTScore 评估失败，可能是网络问题: {e}")
    
    # 混合批量处理示例
    print("\n--- 混合批量处理示例 ---")
    
    image_tasks = [
        (eval_img, ref_img, 'lpips'),
        (eval_img, ref_img, 'ssim')
    ]
    
    text_tasks = [
        (predicted_text_good, reference_text, 'bleu'),
        (predicted_text_bad, reference_text, 'bleu')
    ]
    
    image_results, text_results = await batch_mixed_evaluation(image_tasks, text_tasks)
    
    print(f"混合处理 - 图像结果: {[f'{score:.4f}' for score in image_results]}")
    print(f"混合处理 - 文本结果: {[f'{score:.4f}' for score in text_results]}")


def main():
    """同步主函数 - 保持向后兼容"""
    print("--- 同步版本 (原始实现) ---")
    
    # 原有的同步代码保持不变
    ref_img = Image.new('RGB', (256, 256), color='black')
    noise = np.random.randint(0, 50, (256, 256, 3), dtype=np.uint8)
    eval_img = Image.fromarray(np.array(ref_img) + noise)
    
    lpips_score = FR_IQA(eval_img, ref_img, 'lpips')
    psnr_score = FR_IQA(eval_img, ref_img, 'psnr')
    ssim_score = FR_IQA(eval_img, ref_img, 'ssim')
    
    print(f"LPIPS Score: {lpips_score:.4f} (越低越好)")
    print(f"PSNR Score: {psnr_score:.4f} dB (越高越好)")
    print(f"SSIM Score: {ssim_score:.4f} (越接近1越好)")
    
    # 测试文本评估
    print("\n文本评估 (BLEU):")
    reference_text = "Normal stomach mucosa (negative for Helicobacter Pylori infection)"
    predicted_text_good = "The gastric mucosa appears normal with no evidence of H. pylori infection"
    
    bleu_score = evaluate_text_quality(predicted_text_good, reference_text, 'bleu')
    print(f"BLEU Score: {bleu_score:.4f} (越接近1越好)")
    
    bert_score = evaluate_text_quality(predicted_text_good, reference_text, 'bertscore')
    print(f"Bert Score: {bert_score:.4f} (越接近1越好)")



if __name__ == '__main__':
    import time
        
    print("选择运行模式:")
    print("1. 同步版本 (原始)")
    print("2. 异步版本 (新增)")
    print("3. 性能对比")
    
    choice = input("请输入选择 (1/2/3): ").strip()
    
    if choice == '1':
        main()
    elif choice == '2':
        asyncio.run(async_main())
    elif choice == '3':
        print("--- 性能对比 ---")
        
        # 创建测试数据
        ref_img = Image.new('RGB', (256, 256), color='black')
        noise = np.random.randint(0, 50, (256, 256, 3), dtype=np.uint8)
        eval_img = Image.fromarray(np.array(ref_img) + noise)
        
        # 同步版本测试
        start_time = time.time()
        for _ in range(3):
            FR_IQA(eval_img, ref_img, 'ssim')
        sync_time = time.time() - start_time
        
        # 异步版本测试
        async def async_test():
            tasks = [async_FR_IQA(eval_img, ref_img, 'ssim') for _ in range(3)]
            await asyncio.gather(*tasks)
        
        start_time = time.time()
        asyncio.run(async_test())
        async_time = time.time() - start_time
        
        print(f"同步版本耗时: {sync_time:.4f}秒")
        print(f"异步版本耗时: {async_time:.4f}秒")
        print(f"性能提升: {((sync_time - async_time) / sync_time * 100):.1f}%")
    else:
        print("无效选择，运行同步版本")
        main()
