from __future__ import annotations
import asyncio
import numpy as np
from PIL import Image
from typing import List, Tuple, Union
import functools
import threading
import os
try:
    import torch
except ImportError:
    torch = None

# Large shared servers often expose dozens of CPU cores.  Letting both LPIPS
# and PubMedBERT create their default thread pools can make mixed-modality
# evaluation stall from oversubscription.
TORCH_NUM_THREADS = max(1, int(os.environ.get('MEDGEN_TORCH_NUM_THREADS', '4')))
if torch is not None:
    torch.set_num_threads(TORCH_NUM_THREADS)
try:
    if torch is not None:
        torch.set_num_interop_threads(1)
except RuntimeError:
    # Another importer may already have initialized the inter-op pool.
    pass


def _metric_device():
    """Select one device consistently for all local neural image/text metrics."""
    if torch is None:
        return 'cpu'
    requested = os.environ.get('MEDGEN_METRIC_DEVICE', '').strip()
    if requested:
        if requested.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError(
                f'MEDGEN_METRIC_DEVICE={requested!r} requested CUDA, but CUDA is unavailable'
            )
        return requested
    return 'cuda' if torch.cuda.is_available() else 'cpu'

# LPIPS is initialized lazily so --help/--validate-only never downloads weights.

# Limit concurrent deep-learning inference tasks.
CONCURRENT_DL_TASKS = 2
dl_semaphore = asyncio.Semaphore(CONCURRENT_DL_TASKS)

# Protect BERT model access across threads.
bert_model_lock = threading.Lock()
_bertscore_scorer = None
_bertscore_tokenizer = None
_bertscore_cache_key = None

"""Clinical image metrics.

The benchmark keeps full-image LPIPS/PSNR/SSIM for image fidelity, and uses
MedImageInsight as its medical-image representation metric.  MedImageInsight
was trained across several imaging modalities (rather than being a
chest-radiograph-only encoder), so its cosine similarity is a broader
medical-image representation proxy.  It is still not a lesion-localization
or clinical-truth metric; the implementation therefore fails loudly when the
model or its local dependencies are unavailable.
"""

_MEDIMAGEINSIGHT_DIR = os.environ.get(
    "MEDGEN_MEDIMAGEINSIGHT_DIR",
    "~/.cache/medgen-bench/MedImageInsights",
)
_MEDIMAGEINSIGHT_VISION_WEIGHTS = os.environ.get(
    "MEDGEN_MEDIMAGEINSIGHT_WEIGHTS", "2024.09.27/vision_model/medimageinsigt-v1.0.0.pt"
)
_medimageinsight_model = None
_medimageinsight_preprocess = None
_medimageinsight_device = None
_medimageinsight_lock = threading.Lock()

_lpips_model = None
_lpips_model_lock = threading.Lock()


def _get_lpips_model():
    global _lpips_model
    if _lpips_model is None:
        with _lpips_model_lock:
            if _lpips_model is None:
                try:
                    import lpips
                except ImportError as exc:
                    raise RuntimeError("LPIPS requested but lpips is not installed") from exc
                _lpips_model = lpips.LPIPS(net="alex").to(_metric_device())
                _lpips_model.eval()
    return _lpips_model


def FR_IQA(eval_image: Image.Image, ref_image: Image.Image, eval_metric: str) -> float:
    """Full-image LPIPS/PSNR/SSIM metrics retained for the main image evaluation."""
    if eval_image.size != ref_image.size:
        eval_image = eval_image.resize(ref_image.size, Image.Resampling.LANCZOS)
    metric = eval_metric.lower()
    if metric == "lpips":
        if torch is None:
            raise RuntimeError("LPIPS requested but torch is not installed")
        try:
            from torchvision import transforms
        except ImportError as exc:
            raise RuntimeError("LPIPS requested but torchvision is not installed") from exc
        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        device = _metric_device()
        candidate = preprocess(eval_image).unsqueeze(0).to(device)
        reference = preprocess(ref_image).unsqueeze(0).to(device)
        with torch.inference_mode():
            # ``ToTensor`` produces values in [0, 1], whereas the official
            # LPIPS implementation expects [-1, 1] unless ``normalize`` is
            # set.  Passing raw [0, 1] tensors silently changes the feature
            # normalization and makes the distance non-comparable with the
            # published LPIPS scale.
            return float(_get_lpips_model()(candidate, reference, normalize=True).item())
    if metric == "psnr":
        candidate = np.asarray(eval_image, dtype=np.float32)
        reference = np.asarray(ref_image, dtype=np.float32)
        mse = float(np.mean((candidate - reference) ** 2))
        return 100.0 if mse == 0 else float(20 * np.log10(255.0 / np.sqrt(mse)))
    if metric == "ssim":
        try:
            from skimage.metrics import structural_similarity
        except ImportError as exc:
            raise RuntimeError("SSIM requested but scikit-image is not installed") from exc
        return float(structural_similarity(
            np.asarray(ref_image.convert("L")),
            np.asarray(eval_image.convert("L")),
            data_range=255,
        ))
    raise ValueError(f"Unknown full-image metric: {eval_metric!r}; expected lpips, psnr, or ssim")


async def batch_async_FR_IQA(
    eval_images: List[Image.Image], ref_images: List[Image.Image], eval_metric: str
) -> List[float]:
    if len(eval_images) != len(ref_images):
        raise ValueError("eval_images and ref_images must have equal length")
    # LPIPS is the only neural full-reference metric in this helper.  Evaluate
    # a whole caller batch together so a corrected audit of the frozen outputs
    # does not repeatedly launch AlexNet for every image pair.
    if eval_metric.lower() == "lpips" and eval_images:
        if torch is None:
            raise RuntimeError("LPIPS requested but torch is not installed")
        try:
            from torchvision import transforms
        except ImportError as exc:
            raise RuntimeError("LPIPS requested but torchvision is not installed") from exc
        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        device = _metric_device()
        candidates = torch.stack([preprocess(image) for image in eval_images]).to(device)
        references = torch.stack([preprocess(image) for image in ref_images]).to(device)
        with torch.inference_mode():
            scores = _get_lpips_model()(candidates, references, normalize=True)
        return [float(score) for score in scores.reshape(-1).detach().cpu().tolist()]
    return [FR_IQA(candidate, reference, eval_metric)
            for candidate, reference in zip(eval_images, ref_images)]


def _get_medimageinsight():
    """Load the pinned MedImageInsight checkpoint once per worker process."""
    global _medimageinsight_model, _medimageinsight_preprocess, _medimageinsight_device
    if _medimageinsight_model is not None:
        return _medimageinsight_model, _medimageinsight_preprocess, _medimageinsight_device
    with _medimageinsight_lock:
        if _medimageinsight_model is None:
            if torch is None:
                raise RuntimeError("MedImageInsight requested but torch is not installed")
            model_root = os.path.abspath(os.path.expanduser(_MEDIMAGEINSIGHT_DIR))
            config_path = os.path.join(model_root, "2024.09.27", "config.yaml")
            weights_path = os.path.join(model_root, _MEDIMAGEINSIGHT_VISION_WEIGHTS)
            tokenizer_dir = os.path.join(
                model_root, "2024.09.27", "language_model", "clip_tokenizer_4.16.2"
            )
            required = [config_path, weights_path, tokenizer_dir]
            missing = [path for path in required if not os.path.exists(path)]
            if missing:
                raise RuntimeError(
                    "MedImageInsight is unavailable. Missing local files: "
                    + ", ".join(missing)
                    + ". Download lion-ai/MedImageInsights and set "
                    "MEDGEN_MEDIMAGEINSIGHT_DIR."
                )
            try:
                import sys
                from MedImageInsight.Utils.Arguments import load_opt_from_config_files
                from MedImageInsight.ImageDataLoader import build_transforms
                from MedImageInsight.UniCLModel import build_unicl_model
            except Exception:
                try:
                    if model_root not in sys.path:
                        sys.path.insert(0, model_root)
                    from MedImageInsight.Utils.Arguments import load_opt_from_config_files
                    from MedImageInsight.ImageDataLoader import build_transforms
                    from MedImageInsight.UniCLModel import build_unicl_model
                except Exception as exc:
                    raise RuntimeError(
                        "MedImageInsight source/dependencies are unavailable; "
                        "install its pinned dependencies (yacs, fvcore, mup, timm, "
                        "safetensors, einops, ftfy)."
                    ) from exc
            try:
                if model_root not in sys.path:
                    sys.path.insert(0, model_root)
                opt = load_opt_from_config_files([config_path])
                opt["LANG_ENCODER"]["PRETRAINED_TOKENIZER"] = tokenizer_dir
                opt["UNICL_MODEL"]["PRETRAINED"] = weights_path
                preprocess = build_transforms(opt, False)
                model = build_unicl_model(opt)
                device = torch.device(_metric_device())
                model.to(device)
                model.eval()
            except Exception as exc:
                raise RuntimeError(
                    "MedImageInsight failed to initialize from the local checkpoint"
                ) from exc
            _medimageinsight_model = model
            _medimageinsight_preprocess = preprocess
            _medimageinsight_device = device
    return _medimageinsight_model, _medimageinsight_preprocess, _medimageinsight_device


def _medimageinsight_embeddings(images: List[Image.Image]) -> torch.Tensor:
    if not images:
        return torch.empty((0, 0))
    model, preprocess, device = _get_medimageinsight()
    try:
        inputs = torch.stack([preprocess(image.convert("RGB")) for image in images]).to(device)
        with torch.inference_mode():
            embeddings = model.encode_image(inputs, norm=True).float()
        return torch.nn.functional.normalize(embeddings, dim=-1)
    except Exception as exc:
        raise RuntimeError("MedImageInsight image embedding inference failed") from exc


def compute_medimageinsight_similarity(
    eval_image: Image.Image, ref_image: Image.Image
) -> float:
    """Cosine similarity between MedImageInsight image embeddings."""
    embeddings = _medimageinsight_embeddings([eval_image, ref_image])
    return float(torch.clamp(torch.sum(embeddings[0] * embeddings[1]), -1.0, 1.0).item())


async def batch_async_medimageinsight_metrics(
    eval_images: List[Image.Image], ref_images: List[Image.Image]
) -> List[dict]:
    if len(eval_images) != len(ref_images):
        raise ValueError("eval_images and ref_images must have equal length")
    embeddings = _medimageinsight_embeddings(list(eval_images) + list(ref_images))
    count = len(eval_images)
    candidates = embeddings[:count]
    references = embeddings[count:]
    similarities = torch.sum(candidates * references, dim=-1).clamp(-1.0, 1.0).tolist()
    return [{"MedImageInsight_Similarity": float(score)} for score in similarities]


# Compatibility aliases for downstream callers.  They no longer load or
# compute the former encoder; the returned key is deliberately the new metric name.
def compute_anatomical_embedding_similarity(
    eval_image: Image.Image, ref_image: Image.Image
) -> float:
    return compute_medimageinsight_similarity(eval_image, ref_image)


batch_async_anatomical_metrics = batch_async_medimageinsight_metrics



def evaluate_text_quality_thread_safe(eval_text: str, ref_text: str, eval_metric: str) -> float:
    """Evaluate generated text against a reference using a thread-safe path."""
    metric = eval_metric.lower()

    if metric == 'bertscore':
        # Serialize access to the BERT model.
        with bert_model_lock:
            try:
                # bert-score expects lists.
                preds = [eval_text]
                refs = [ref_text]

                from bert_score import score as bert_score
                _, _, f1 = bert_score(
                    preds,
                    refs,
                    model_type="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                    lang="en",
                    rescale_with_baseline=False,
                    num_layers=12,
                    device=_metric_device(),
                    verbose=False,
                )
                
                return f1.mean().item()
                
            except Exception as e:
                raise RuntimeError(f"BERTScore calculation failed: {e}") from e

    elif metric == 'bleu':
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        except ImportError as exc:
            raise RuntimeError("BLEU requested but nltk is not installed") from exc
        # BLEU expects a list of reference token lists and one candidate list.
        reference = [ref_text.split()]
        candidate = eval_text.split()
        
        # Smooth BLEU to avoid a zero score from an unmatched n-gram.
        bleu_score = sentence_bleu(reference, candidate, smoothing_function=SmoothingFunction().method1)
        return bleu_score

    else:
        raise ValueError(
            f"Unknown text evaluation metric: '{eval_metric}'. "
            "Supported metrics: 'bertscore', 'bleu'."
        )




def evaluate_text_quality(eval_text: str, ref_text: str, eval_metric: str) -> float:
    """Evaluate generated text against a reference while preserving compatibility."""
    return evaluate_text_quality_thread_safe(eval_text, ref_text, eval_metric)


async def async_evaluate_text_quality(eval_text: str, ref_text: str, eval_metric: str) -> float:
    """Asynchronously evaluate generated text against a reference."""
    metric = eval_metric.lower()
    
    if metric == 'bertscore':
        # PyTorch/Transformers CPU inference can deadlock when first initialized
        # inside asyncio.to_thread.  Keep the serialized BERTScore call on the
        # event-loop thread; other metrics still use worker threads.
        async with dl_semaphore:
            return evaluate_text_quality_thread_safe(eval_text, ref_text, eval_metric)
    else:
        return evaluate_text_quality_thread_safe(eval_text, ref_text, eval_metric)


async def batch_async_evaluate_text_quality(
    eval_texts: List[str], 
    ref_texts: List[str], 
    eval_metric: str
) -> List[float]:
    """Asynchronously evaluate batches of generated and reference text."""
    if len(eval_texts) != len(ref_texts):
        raise ValueError("eval_texts and ref_texts must have equal length")
    
    # Batch BERTScore for efficiency and to avoid concurrency issues.
    if eval_metric.lower() == 'bertscore':
        return batch_bertscore_calculation(eval_texts, ref_texts)
    else:
        # BLEU evaluations can run concurrently.
        tasks = [
            async_evaluate_text_quality(eval_text, ref_text, eval_metric)
            for eval_text, ref_text in zip(eval_texts, ref_texts)
        ]
        return await asyncio.gather(*tasks)


def batch_bertscore_calculation(eval_texts: List[str], ref_texts: List[str]) -> List[float]:
    """Compute BERTScore in a batch without thread contention."""
    try:
        with bert_model_lock:
            global _bertscore_scorer, _bertscore_tokenizer, _bertscore_cache_key
            # Use the immutable local snapshot when the runner provides one.
            # This metric is evaluated repeatedly during long jobs; resolving a
            # Hub model ID here can issue an online HEAD request on every batch
            # even after all weights have been cached.  That makes a completed
            # cache needlessly vulnerable to transient network/SSL failures.
            model_ref = os.environ.get(
                'MEDGEN_PUBMEDBERT_MODEL_PATH',
                'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
            )
            use_local_files_only = os.path.isdir(model_ref)
            cache_key = (model_ref, str(_metric_device()))
            if _bertscore_scorer is None or _bertscore_cache_key != cache_key:
                from bert_score import BERTScorer
                from transformers import AutoTokenizer
                _bertscore_tokenizer = AutoTokenizer.from_pretrained(
                    model_ref,
                    local_files_only=use_local_files_only,
                )
                _bertscore_scorer = BERTScorer(
                    model_type=model_ref,
                    lang="en",
                    rescale_with_baseline=False,
                    num_layers=12,
                    device=_metric_device(),
                    batch_size=max(1, int(os.environ.get('MEDGEN_BERTSCORE_BATCH_SIZE', '64'))),
                )
                _bertscore_cache_key = cache_key

            # PubMedBERT has a hard 512-position limit.  Keep the clinical
            # entity/RadGraph metrics on the original full report, but make
            # the embedding metric well-defined for long generated reports.
            # Tokenization is done with the same tokenizer BERTScore uses so
            # truncation is by model tokens rather than arbitrary characters.
            tokenizer = _bertscore_tokenizer

            def truncate_for_bertscore(text: str) -> str:
                encoded = tokenizer(
                    str(text or ""),
                    add_special_tokens=True,
                    truncation=True,
                    max_length=512,
                )
                return tokenizer.decode(
                    encoded["input_ids"],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )

            bertscore_texts = (
                [truncate_for_bertscore(text) for text in eval_texts],
                [truncate_for_bertscore(text) for text in ref_texts],
            )
            _, _, f1_scores = _bertscore_scorer.score(
                bertscore_texts[0],
                bertscore_texts[1],
                verbose=False,
                batch_size=max(1, int(os.environ.get('MEDGEN_BERTSCORE_BATCH_SIZE', '64'))),
            )
            return f1_scores.tolist()
    except Exception as e:
        raise RuntimeError(f"Batch BERTScore calculation failed: {e}") from e


# --- Mixed batch evaluation ---
async def batch_mixed_evaluation(
    image_tasks: List[Tuple[Image.Image, Image.Image, str]] = None,
    text_tasks: List[Tuple[str, str, str]] = None
) -> Tuple[List[float], List[float]]:
    """Evaluate image and text tasks together.

    Args:
        image_tasks: ``(evaluation_image, reference_image, metric)`` tuples.
        text_tasks: ``(evaluation_text, reference_text, metric)`` tuples.

    Returns:
        Image results followed by text results, returned as separate lists.
    """
    tasks = []
    
    # Add image-evaluation tasks.
    if image_tasks:
        for eval_img, ref_img, metric in image_tasks:
            tasks.append(async_FR_IQA(eval_img, ref_img, metric))
    
    # Add text-evaluation tasks.
    if text_tasks:
        for eval_text, ref_text, metric in text_tasks:
            tasks.append(async_evaluate_text_quality(eval_text, ref_text, metric))
    
    # Run all tasks concurrently.
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Replace failed tasks with the existing default score.
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            print(f"Task execution failed: {result}")
            processed_results.append(0.0)
        else:
            processed_results.append(result)
    
    # Split image and text results.
    image_count = len(image_tasks) if image_tasks else 0
    image_results = processed_results[:image_count]
    text_results = processed_results[image_count:]
    
    return image_results, text_results


# --- Demonstration entry points ---
async def async_main():
    """Demonstrate asynchronous metric evaluation."""
    print("--- Asynchronous full-reference image-quality evaluation ---")
    
    # Create example images.
    ref_img = Image.new('RGB', (256, 256), color='black')
    noise = np.random.randint(0, 50, (256, 256, 3), dtype=np.uint8)
    eval_img = Image.fromarray(np.array(ref_img) + noise)
    
    # Evaluate one image pair asynchronously.
    print("Single asynchronous evaluation:")
    lpips_task = async_FR_IQA(eval_img, ref_img, 'lpips')
    psnr_task = async_FR_IQA(eval_img, ref_img, 'psnr') 
    ssim_task = async_FR_IQA(eval_img, ref_img, 'ssim')
    
    # Run all image metrics concurrently.
    lpips_score, psnr_score, ssim_score = await asyncio.gather(
        lpips_task, psnr_task, ssim_task
    )
    
    print(f"LPIPS Score: {lpips_score:.4f} (lower is better)")
    print(f"PSNR Score: {psnr_score:.4f} dB (higher is better)")
    print(f"SSIM Score: {ssim_score:.4f} (closer to 1 is better)")
    
    # Evaluate a batch asynchronously.
    print("\nBatch asynchronous evaluation:")
    eval_images = [eval_img] * 3
    ref_images = [ref_img] * 3
    
    batch_scores = await batch_async_FR_IQA(eval_images, ref_images, 'ssim')
    print(f"Batch SSIM scores: {[f'{score:.4f}' for score in batch_scores]}")
    
    print("\n" + "="*40 + "\n")
    
    # Demonstrate asynchronous text evaluation.
    print("--- Asynchronous text-quality evaluation ---")
    
    reference_text = "Normal stomach mucosa (negative for Helicobacter Pylori infection)"
    predicted_text_good = "The gastric mucosa appears normal with no evidence of H. pylori infection"
    predicted_text_bad = "computed tomography"
    
    # Start with BLEU, which does not require network access.
    print("BLEU evaluation:")
    bleu_good_task = async_evaluate_text_quality(predicted_text_good, reference_text, 'bleu')
    bleu_bad_task = async_evaluate_text_quality(predicted_text_bad, reference_text, 'bleu')
    
    bleu_good, bleu_bad = await asyncio.gather(bleu_good_task, bleu_bad_task)
    
    print(f"BLEU Score (Good Match): {bleu_good:.4f} (closer to 1 is better)")
    print(f"BLEU Score (Bad Match): {bleu_bad:.4f} (closer to 1 is better)")
    
    # Try BERTScore, which can depend on model availability.
    print("\nBERTScore evaluation:")
    try:
        bertscore_good_task = async_evaluate_text_quality(predicted_text_good, reference_text, 'bertscore')
        bertscore_bad_task = async_evaluate_text_quality(predicted_text_bad, reference_text, 'bertscore')
        
        bertscore_good, bertscore_bad = await asyncio.gather(bertscore_good_task, bertscore_bad_task)
        
        print(f"BERTScore (Good Match): {bertscore_good:.4f} (closer to 1 is better)")
        print(f"BERTScore (Bad Match): {bertscore_bad:.4f} (closer to 1 is better)")
    except Exception as e:
        print(f"BERTScore evaluation failed; model access may be unavailable: {e}")
    
    # Demonstrate mixed batch evaluation.
    print("\n--- Mixed batch evaluation ---")
    
    image_tasks = [
        (eval_img, ref_img, 'lpips'),
        (eval_img, ref_img, 'ssim')
    ]
    
    text_tasks = [
        (predicted_text_good, reference_text, 'bleu'),
        (predicted_text_bad, reference_text, 'bleu')
    ]
    
    image_results, text_results = await batch_mixed_evaluation(image_tasks, text_tasks)
    
    print(f"Mixed evaluation - image results: {[f'{score:.4f}' for score in image_results]}")
    print(f"Mixed evaluation - text results: {[f'{score:.4f}' for score in text_results]}")


def main():
    """Run the synchronous demonstration for backward compatibility."""
    print("--- Synchronous evaluation ---")
    
    # Create example images.
    ref_img = Image.new('RGB', (256, 256), color='black')
    noise = np.random.randint(0, 50, (256, 256, 3), dtype=np.uint8)
    eval_img = Image.fromarray(np.array(ref_img) + noise)
    
    lpips_score = FR_IQA(eval_img, ref_img, 'lpips')
    psnr_score = FR_IQA(eval_img, ref_img, 'psnr')
    ssim_score = FR_IQA(eval_img, ref_img, 'ssim')
    
    print(f"LPIPS Score: {lpips_score:.4f} (lower is better)")
    print(f"PSNR Score: {psnr_score:.4f} dB (higher is better)")
    print(f"SSIM Score: {ssim_score:.4f} (closer to 1 is better)")
    
    # Demonstrate text evaluation.
    print("\nText evaluation (BLEU):")
    reference_text = "Normal stomach mucosa (negative for Helicobacter Pylori infection)"
    predicted_text_good = "The gastric mucosa appears normal with no evidence of H. pylori infection"
    
    bleu_score = evaluate_text_quality(predicted_text_good, reference_text, 'bleu')
    print(f"BLEU Score: {bleu_score:.4f} (closer to 1 is better)")
    
    bert_score = evaluate_text_quality(predicted_text_good, reference_text, 'bertscore')
    print(f"Bert Score: {bert_score:.4f} (closer to 1 is better)")



if __name__ == '__main__':
    import time
        
    print("Select a run mode:")
    print("1. Synchronous evaluation")
    print("2. Asynchronous evaluation")
    print("3. Performance comparison")
    
    choice = input("Enter a choice (1/2/3): ").strip()
    
    if choice == '1':
        main()
    elif choice == '2':
        asyncio.run(async_main())
    elif choice == '3':
        print("--- Performance comparison ---")
        
        # Create test data.
        ref_img = Image.new('RGB', (256, 256), color='black')
        noise = np.random.randint(0, 50, (256, 256, 3), dtype=np.uint8)
        eval_img = Image.fromarray(np.array(ref_img) + noise)
        
        # Time synchronous evaluation.
        start_time = time.time()
        for _ in range(3):
            FR_IQA(eval_img, ref_img, 'ssim')
        sync_time = time.time() - start_time
        
        # Time asynchronous evaluation.
        async def async_test():
            tasks = [async_FR_IQA(eval_img, ref_img, 'ssim') for _ in range(3)]
            await asyncio.gather(*tasks)
        
        start_time = time.time()
        asyncio.run(async_test())
        async_time = time.time() - start_time
        
        print(f"Synchronous elapsed time: {sync_time:.4f} seconds")
        print(f"Asynchronous elapsed time: {async_time:.4f} seconds")
        print(f"Performance improvement: {((sync_time - async_time) / sync_time * 100):.1f}%")
    else:
        print("Invalid choice; running synchronous evaluation")
        main()
