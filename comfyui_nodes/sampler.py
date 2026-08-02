"""VLM-Guided KSampler for ComfyUI.

Implements the closed-loop feedback mechanism of
``CustomStableDiffusion3Pipeline`` (from ``custom_model/custom_sd35.py``) on top
of ComfyUI's native sampler infrastructure:

  1. Intercept the predicted x0 latent at specific timesteps (via the sampler
     callback, which receives x0 directly).
  2. Decode x0 to an RGB image with the VAE.
  3. Query the VLM ("What is the main object in this image?").
  4. Accumulate cleaned keywords into a growing negative prompt.
  5. Re-encode the updated negative and feed it back into the CFG guider so
     subsequent steps suppress the identified common elements.

The feedback loop is realized with a small subclass of
``comfy.samplers.CFGGuider`` whose ``__call__`` refreshes the negative
conditioning right before each denoiser call, so the VLM-driven updates take
effect on the very next step. This works for SD1.5 / SDXL / SD3 / Flux alike,
because it operates entirely on ComfyUI's standard MODEL / CLIP / VAE /
CONDITIONING types.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

import comfy.samplers
import comfy.sample

from .vlm import clean_vqa_answer

# Sampler / scheduler name lists (with a safe fallback for older ComfyUI).
try:
    _SAMPLER_NAMES = comfy.samplers.SAMPLER_NAMES
except AttributeError:
    _SAMPLER_NAMES = ["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde", "uni_pc"]
try:
    _SCHEDULER_NAMES = comfy.samplers.SCHEDULER_NAMES
except AttributeError:
    _SCHEDULER_NAMES = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]


# --------------------------------------------------------------------------- #
# Custom guider                                                              #
# --------------------------------------------------------------------------- #
class VLMGuidedCFGGuider(comfy.samplers.CFGGuider):
    """CFG guider whose negative conditioning can be swapped mid-sampling.

    The sampler callback calls :meth:`update_uncond` after each VLM query;
    ``__call__`` applies the live value right before every denoiser call so the
    update always takes effect on the next step, regardless of how the base
    class caches conditioning.
    """

    def __init__(self, model_patcher):
        super().__init__(model_patcher)
        self._live_uncond = None

    def update_uncond(self, new_uncond):
        self._live_uncond = new_uncond

    def __call__(self, *args, **kwargs):
        if self._live_uncond is not None:
            self.uncond = self._live_uncond
            # Some ComfyUI versions also keep a dict form; keep it in sync.
            conds = getattr(self, "conds", None)
            if isinstance(conds, dict):
                conds["negative"] = self._live_uncond
        return super().__call__(*args, **kwargs)


# --------------------------------------------------------------------------- #
# Helpers                                                                    #
# --------------------------------------------------------------------------- #
def _encode_text(clip, text: str):
    """Encode text into ComfyUI conditioning, mirroring CLIPTextEncode.

    Works for SD1.5 / SDXL / SD3 / Flux because the connected ``CLIP`` object
    handles all underlying text encoders.
    """
    tokens = clip.tokenize(text)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    return [[cond, {"pooled_output": pooled}]]


def _to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """Convert a ComfyUI IMAGE tensor [H, W, 3] (0..1) to a PIL Image."""
    arr = (image_tensor.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def _resolve_step(value, total_steps: int, default: int) -> int:
    """Resolve a start/stop value that may be fractional (0..1) or absolute."""
    if value is None:
        return default
    if 0.0 < value <= 1.0:
        idx = int(value * total_steps)
    else:
        idx = int(value)
    return max(0, min(idx, total_steps))


# --------------------------------------------------------------------------- #
# Node                                                                       #
# --------------------------------------------------------------------------- #
class VLMGuidedKSampler:
    """KSampler with VLM-guided adaptive negative prompting.

    During the denoising loop the predicted x0 is decoded to an image and shown
    to a VLM; the VLM's answers are accumulated as negative prompts and fed
    back into the sampler to steer generation away from common patterns.
    """

    CATEGORY = "VLM-Guided/sampling"
    FUNCTION = "sample"
    RETURN_TYPES = ("LATENT", "IMAGE", "STRING")
    RETURN_NAMES = ("latent", "image", "negatives_log")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "base_negative": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Static negative prompt (VLM keywords are appended)",
                }),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (_SAMPLER_NAMES, {"default": "euler" if "euler" in _SAMPLER_NAMES else _SAMPLER_NAMES[0]}),
                "scheduler": (_SCHEDULER_NAMES, {"default": "normal" if "normal" in _SCHEDULER_NAMES else _SCHEDULER_NAMES[0]}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "questions": ("STRING", {
                    "multiline": True,
                    "default": "What is the main object in this image?",
                    "placeholder": "One VLM question per line",
                }),
                "topk": ("INT", {"default": 1, "min": 1, "max": 20}),
                "freq": ("INT", {"default": 1, "min": 1, "max": 200}),
                "vqa_start_step": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.01,
                }),
                "vqa_stop_step": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10000.0, "step": 0.01,
                }),
                "clear_negatives_at_stop": ("BOOLEAN", {"default": False}),
                "log_negatives": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "vlm_oracle": ("VLM_ORACLE",),
                "main_object": ("STRING", {"default": ""}),
            },
        }

    def sample(self, model, clip, vae, positive, base_negative, latent_image,
               seed, steps, cfg, sampler_name, scheduler, denoise,
               questions, topk, freq, vqa_start_step, vqa_stop_step,
               clear_negatives_at_stop, log_negatives,
               vlm_oracle=None, main_object=""):
        # If no oracle is connected, behave like a plain KSampler.
        if vlm_oracle is None:
            return self._sample_plain(model, positive, _encode_text(clip, base_negative),
                                      latent_image, seed, steps, cfg, sampler_name,
                                      scheduler, denoise, vae)

        device = model.load_device
        latent_samples = latent_image["samples"].to(device)
        noise_mask = latent_image.get("noise_mask", None)

        question_list = [q.strip() for q in questions.splitlines() if q.strip()]
        if not question_list:
            question_list = ["What is the main object in this image?"]

        # Initial negative conditioning (base text only). `empty_neg_cond` is
        # used to revert to a neutral negative when clear_negatives_at_stop.
        base_neg_cond = _encode_text(clip, base_negative) if base_negative else _encode_text(clip, "")
        empty_neg_cond = _encode_text(clip, "")

        guider = VLMGuidedCFGGuider(model)
        guider.set_conds(positive, base_neg_cond)
        guider.set_cfg(cfg)

        # Sigmas (handles denoise<1 like BasicScheduler).
        model_sampling = model.get_model_object("model_sampling")
        if 0.0 < denoise < 1.0:
            total = max(int(steps / denoise), steps)
        else:
            total = steps
        sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, total)
        if 0.0 < denoise < 1.0 and sigmas.shape[0] > steps + 1:
            sigmas = sigmas[-(steps + 1):]
        if float(sigmas[-1]) != 0.0:
            sigmas = sigmas.clone()
            sigmas[-1] = 0.0
        sigmas = sigmas.to(device)

        # Noise.
        noise = comfy.sample.prepare_noise(latent_samples, seed)
        if isinstance(noise, tuple):
            noise = noise[0]
        noise = noise.to(device)

        sampler = comfy.samplers.KSampler(sampler_name)

        num_steps = max(int(sigmas.shape[0] - 1), 1)
        start_idx = _resolve_step(vqa_start_step, num_steps, 0)
        stop_idx = _resolve_step(vqa_stop_step, num_steps, num_steps)

        detected = [base_negative] if base_negative else []
        log_lines = []

        def callback(step, x0, x, total_steps, **_kwargs):
            if start_idx <= step < stop_idx:
                if freq > 1 and (step % freq != 0):
                    return
                try:
                    with torch.no_grad():
                        decoded = vae.decode(x0)
                    pil = _to_pil(decoded[0])
                except Exception as e:  # pragma: no cover - defensive
                    if log_negatives:
                        log_lines.append(f"[Step {step}] decode error: {e}")
                    return

                for q in question_list:
                    try:
                        results = vlm_oracle({"image": pil, "question": q})
                    except Exception as e:
                        results = []
                        if log_negatives:
                            log_lines.append(f"[Step {step}] VLM error: {e}")
                    for j in range(min(topk, len(results))):
                        entry = results[j]
                        ans = entry.get("answer", "") if isinstance(entry, dict) else str(entry)
                        ans = clean_vqa_answer(ans)
                        if main_object:
                            ans = (ans + " " + main_object).strip()
                        if ans and ans not in detected:
                            detected.append(ans)
                            if log_negatives:
                                log_lines.append(f"[Step {step}] + '{ans}'")

                combined = ", ".join(detected)
                if combined:
                    try:
                        guider.update_uncond(_encode_text(clip, combined))
                        if log_negatives:
                            log_lines.append(f"[Step {step}] negatives: {combined}")
                    except Exception as e:
                        if log_negatives:
                            log_lines.append(f"[Step {step}] encode error: {e}")
            elif clear_negatives_at_stop and step == stop_idx:
                # Revert to a neutral negative (matches the original behavior
                # of clearing accumulated negatives when VQA stops).
                guider.update_uncond(empty_neg_cond)
                if log_negatives:
                    log_lines.append(f"[Step {step}] cleared accumulated negatives")

        samples = guider.sample(
            noise, latent_samples, sampler, sigmas,
            denoise_mask=noise_mask, callback=callback,
            disable_pbar=False, seed=seed,
        )

        latent_out = {"samples": samples}
        if noise_mask is not None:
            latent_out["noise_mask"] = noise_mask

        with torch.no_grad():
            final_image = vae.decode(samples)

        return (latent_out, final_image, "\n".join(log_lines))

    # ------------------------------------------------------------------ #
    # Fallback path (no VLM) — standard CFG sampling.                     #
    # ------------------------------------------------------------------ #
    def _sample_plain(self, model, positive, negative_cond, latent_image,
                      seed, steps, cfg, sampler_name, scheduler, denoise, vae):
        samples = comfy.sample.ksampler(
            model, seed, steps, cfg, sampler_name, scheduler,
            positive, negative_cond, latent_image, denoise=denoise,
            disable_pbar=False,
        )
        latent_out = {"samples": samples}
        if latent_image.get("noise_mask") is not None:
            latent_out["noise_mask"] = latent_image["noise_mask"]
        with torch.no_grad():
            final_image = vae.decode(samples)
        return (latent_out, final_image, "(no VLM oracle connected; standard sampling)")
