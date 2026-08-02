"""VLM oracle loading and answer cleaning for VLM-Guided Adaptive Negative Prompting.

Ports the oracle logic from ``custom_model/custom_sd35.py`` and
``gen_utils/generate_sd35_image.py`` into ComfyUI-loadable wrappers.

Each oracle exposes a uniform callable interface::

    oracle({"image": PIL.Image, "question": str}) -> [{"answer": str}, ...]

so the sampler node can treat ViLT and Qwen-VL identically.
"""

from __future__ import annotations

import torch
from PIL import Image


# --------------------------------------------------------------------------- #
# Answer cleaning (verbatim logic from custom_model/custom_sd35.py)            #
# --------------------------------------------------------------------------- #
def clean_vqa_answer(answer: str) -> str:
    """Clean and normalize a VQA response for use as a negative prompt.

    Strips VQA artifacts ('the', 'it is', 'appears to be', ...), removes
    articles, and drops overly generic / meaningless answers. Returns "" for
    answers deemed too generic.
    """
    if not answer or not isinstance(answer, str):
        return ""

    cleaned = answer.strip().lower()

    while cleaned and cleaned[0] in ',-.:;!?()[]{}"\' ':
        cleaned = cleaned[1:].strip()
    while cleaned and cleaned[-1] in ',-.:;!?()[]{}"\' ':
        cleaned = cleaned[:-1].strip()

    prefixes_to_remove = [
        'the ', 'a ', 'an ', 'it is ', 'this is ', 'that is ',
        'it\'s ', 'there is ', 'there are ', 'i see ', 'i can see ',
        'it appears to be ', 'it looks like ', 'seems to be ',
        'appears to be ', 'looks like ', 'yes, ', 'no, '
    ]
    for prefix in prefixes_to_remove:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break  # only remove one prefix to avoid over-cleaning

    separators = [',', ';', ' and ', ' or ', ' & ']
    parts = [cleaned]
    for sep in separators:
        new_parts = []
        for part in parts:
            new_parts.extend(part.split(sep))
        parts = new_parts

    cleaned_parts = []
    articles_to_remove = {'a', 'an', 'the'}
    for part in parts:
        part = part.strip()
        if not part:
            continue
        words = part.split()
        cleaned_words = []
        for word in words:
            word = word.strip(',-.:;!?()[]{}"\' ')
            if word and word not in articles_to_remove:
                cleaned_words.append(word)
        if cleaned_words:
            cleaned_part = ' '.join(cleaned_words)
            if cleaned_part and len(cleaned_part) > 1:
                cleaned_parts.append(cleaned_part)

    cleaned = ', '.join(cleaned_parts) if cleaned_parts else ""

    if len(cleaned) == 1 and cleaned.isalpha():
        return ""

    skip_words = {
        'i', 'a', 'an', 'the', 'is', 'are', 'am', 'be', 'it', 'to', 'of',
        'in', 'on', 'at', 'by', 'for', 'with', 'no', 'yes', '0', '1', '2',
        'neither', 'no dog', 'no cat', 'none', 'no animal'
    }
    if cleaned in skip_words:
        return ""

    return cleaned


# --------------------------------------------------------------------------- #
# Oracle wrappers                                                            #
# --------------------------------------------------------------------------- #
class ViltOracle:
    """Wrapper around a transformers VQA pipeline (e.g. dandelin/vilt-b32-finetuned-vqa).

    The VQA pipeline returns a list of ``{'answer': str, 'score': float}`` for
    the top-k answers; we forward the list so the sampler can index into it.
    """

    def __init__(self, pipe):
        self.pipe = pipe

    def __call__(self, data_dict):
        image = data_dict["image"]
        question = data_dict["question"]
        if not isinstance(image, Image.Image):
            image = image.convert("RGB") if hasattr(image, "convert") else image
        try:
            # Ask for several candidates; the sampler indexes with top_k.
            results = self.pipe(image=image, question=question, top_k=16)
        except TypeError:
            # Older API only accepts positional args.
            results = self.pipe(image, question)
        # Normalize to [{"answer": str}, ...]
        normalized = []
        if isinstance(results, list):
            for r in results:
                if isinstance(r, dict) and "answer" in r:
                    normalized.append({"answer": r["answer"]})
                elif isinstance(r, str):
                    normalized.append({"answer": r})
        elif isinstance(results, dict) and "answer" in results:
            normalized.append({"answer": results["answer"]})
        return normalized if normalized else [{"answer": "unknown"}]


class QwenOracle:
    """Wrapper around a Qwen2.5-VL image-text-to-text pipeline.

    Ports ``QwenOracle`` from ``gen_utils/generate_sd35_image.py``.
    """

    def __init__(self, model_id: str, device: str = "auto"):
        if torch.cuda.is_available() and device != "cpu":
            dtype = torch.bfloat16
            kwargs = {"torch_dtype": dtype, "device_map": "auto"}
        else:
            kwargs = {}
        from transformers import pipeline
        self.pipe = pipeline("image-text-to-text", model=model_id, **kwargs)

    def __call__(self, data_dict):
        from PIL import Image as _PILImage

        image = data_dict["image"]
        question = data_dict["question"]

        if not isinstance(image, _PILImage.Image):
            if hasattr(image, "convert"):
                image = image.convert("RGB")
            else:
                return [{"answer": "error"}]

        try:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"Answer with a single word.\n{question}"},
                    ],
                }
            ]
            response = self.pipe(images=[image], text=conversation)

            if response and len(response) > 0:
                if isinstance(response[0], dict) and "generated_text" in response[0]:
                    generated_text = response[0]["generated_text"]
                    if isinstance(generated_text, list):
                        for item in generated_text:
                            if isinstance(item, dict) and item.get("role") == "assistant":
                                return [{"answer": item.get("content", "").strip()}]
                    elif isinstance(generated_text, str):
                        return [{"answer": generated_text.strip()}]
                elif isinstance(response[0], str):
                    return [{"answer": response[0].strip()}]

            return [{"answer": "unknown"}]

        except Exception as e:
            print(f"[VLM-Guided] QwenOracle error: {e}")
            return [{"answer": "error"}]
        finally:
            try:
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# Loader helper                                                              #
# --------------------------------------------------------------------------- #
def _device_id(device: str):
    """Map a friendly device string to a transformers pipeline device id."""
    if device == "cpu":
        return -1
    if device == "cuda":
        return 0
    # auto
    return 0 if torch.cuda.is_available() else -1


def load_vlm_oracle(model_id: str, device: str = "auto"):
    """Load a VLM oracle by HuggingFace model id.

    Returns a callable object with the uniform interface described above.
    """
    from transformers import pipeline

    if model_id.startswith("Qwen/"):
        return QwenOracle(model_id, device=device)

    # ViLT / other VQA pipelines.
    dev = _device_id(device)
    pipe = pipeline("visual-question-answering", model=model_id, device=dev)
    return ViltOracle(pipe)


# --------------------------------------------------------------------------- #
# ComfyUI node                                                               #
# --------------------------------------------------------------------------- #
# Preset HuggingFace VLM model ids. A custom id can also be typed in.
_VLM_MODELS = [
    "dandelin/vilt-b32-finetuned-vqa",
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
]


class VLMOracleLoader:
    """Load a Vision-Language Model oracle for VLM-guided sampling.

    Outputs a ``VLM_ORACLE`` object to be connected into VLMGuidedKSampler.
    ViLT is fast/lightweight; Qwen-VL models are more capable but heavier.
    """

    CATEGORY = "VLM-Guided/oracle"
    FUNCTION = "load"
    RETURN_TYPES = ("VLM_ORACLE",)
    RETURN_NAMES = ("oracle",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": (_VLM_MODELS, {"default": _VLM_MODELS[0]}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "custom_model_id": ("STRING", {
                    "default": "",
                    "placeholder": "Optional: override with any HuggingFace VLM id",
                }),
            }
        }

    def load(self, model_id, device, custom_model_id=""):
        target = custom_model_id.strip() if custom_model_id and custom_model_id.strip() else model_id
        oracle = load_vlm_oracle(target, device=device)
        return (oracle,)
