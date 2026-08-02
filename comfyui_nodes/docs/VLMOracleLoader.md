# VLM Oracle Loader

**Node ID:** `VLMOracleLoader`
**Category:** `VLM-Guided/oracle`

Loads a Vision-Language Model (VLM) used by [`VLMGuidedKSampler`](VLMGuidedKSampler.md)
to analyze intermediate denoiser outputs. Outputs a `VLM_ORACLE` object.

## Inputs

| Name | Type | Default | Description |
|---|---|---|---|
| `model_id` | COMBO | `dandelin/vilt-b32-finetuned-vqa` | Preset HuggingFace VLM id. |
| `device` | COMBO | `auto` | `auto` / `cuda` / `cpu` — where to run the VLM. |
| `custom_model_id` | STRING | `""` | Optional override with any HuggingFace VLM id. |

### Preset `model_id` options

- `dandelin/vilt-b32-finetuned-vqa` — fast, lightweight VQA model.
- `Qwen/Qwen2.5-VL-3B-Instruct` — more capable, heavier.
- `Qwen/Qwen2.5-VL-7B-Instruct` — most capable, heaviest.

If `custom_model_id` is non-empty it takes precedence over `model_id`. Ids starting
with `Qwen/` are loaded as `image-text-to-text` pipelines; everything else is loaded
as a `visual-question-answering` pipeline.

## Outputs

| Name | Type | Description |
|---|---|---|
| `oracle` | `VLM_ORACLE` | Callable oracle object; connect into `VLMGuidedKSampler.vlm_oracle`. |

## Behavior

- The oracle exposes a uniform interface:
  `oracle({"image": PIL.Image, "question": str}) -> [{"answer": str}, ...]`
- ViLT returns a list of ranked candidate answers (the sampler indexes with `topk`).
- Qwen returns a single answer.
- Loading is cached by ComfyUI on `(model_id, device, custom_model_id)`, so the
  model stays loaded across runs.

## Tips

- **VRAM**: Qwen models load with `device_map="auto"` and will share the GPU with
  the diffusion model. For Qwen-7B ensure you have enough free VRAM, or use ViLT.
- **Speed**: ViLT adds roughly ~13s overhead for a 28-step run; Qwen adds more.
- The oracle is **optional** on `VLMGuidedKSampler` — if you disconnect it, the
  sampler runs as a standard KSampler with no VLM guidance.
