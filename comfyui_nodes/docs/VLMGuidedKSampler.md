# VLM-Guided KSampler (Adaptive Negatives)

**Node ID:** `VLMGuidedKSampler`
**Category:** `VLM-Guided/sampling`

A KSampler that runs the VLM-guided adaptive negative-prompting feedback loop.
During denoising the predicted `x0` is decoded to an image, shown to a VLM, and the
VLM's answers are accumulated as negative prompts that suppress common elements on
subsequent steps — steering generation toward novelty.

## Inputs

### Required

| Name | Type | Default | Description |
|---|---|---|---|
| `model` | `MODEL` | — | Diffusion model (from `Load Checkpoint`, etc.). |
| `clip` | `CLIP` | — | CLIP used to **re-encode** the dynamically updated negative. |
| `vae` | `VAE` | — | VAE used to **decode** intermediate `x0` for the VLM. |
| `positive` | `CONDITIONING` | — | Pre-encoded positive conditioning (fixed). |
| `base_negative` | STRING (multiline) | `""` | Static negative text; VLM keywords are appended. |
| `latent_image` | `LATENT` | — | Input latent (e.g. from `Empty Latent Image`). |
| `seed` | INT | `0` | Random seed. |
| `steps` | INT | `28` | Number of denoising steps. |
| `cfg` | FLOAT | `4.5` | Classifier-free guidance scale. |
| `sampler_name` | COMBO | `euler` | ComfyUI sampler. |
| `scheduler` | COMBO | `normal` | ComfyUI scheduler. |
| `denoise` | FLOAT | `1.0` | Denoise amount (1.0 = full generation from noise). |
| `questions` | STRING (multiline) | `What is the main object in this image?` | One VLM question per line. |
| `topk` | INT | `1` | Top-k VLM answers per question to accumulate. |
| `freq` | INT | `1` | Query the VLM every N steps. |
| `vqa_start_step` | FLOAT | `0.0` | VQA window start (≤1.0 = fraction; >1.0 = absolute step). |
| `vqa_stop_step` | FLOAT | `1.0` | VQA window stop (≤1.0 = fraction; >1.0 = absolute step). |
| `clear_negatives_at_stop` | BOOLEAN | `False` | Revert to a neutral negative when the VQA window ends. |
| `log_negatives` | BOOLEAN | `True` | Emit a per-step log of accumulated negatives. |

### Optional

| Name | Type | Default | Description |
|---|---|---|---|
| `vlm_oracle` | `VLM_ORACLE` | — | From `VLM Oracle Loader`. If absent, runs as a plain KSampler. |
| `main_object` | STRING | `""` | Optional context appended to each detected keyword. |

## Outputs

| Name | Type | Description |
|---|---|---|
| `latent` | `LATENT` | Final latent. |
| `image` | `IMAGE` | Final decoded image. |
| `negatives_log` | STRING | Per-step log of accumulated negatives (for debugging). |

## How the feedback loop works

1. Each step, the sampler callback receives the predicted clean latent `x0`.
2. Within the `[vqa_start_step, vqa_stop_step)` window (and every `freq` steps),
   `x0` is decoded with the VAE and shown to the oracle with each `question`.
3. Answers are cleaned via `clean_vqa_answer` (strips "the", "it is",
   "appears to be", articles, and generic tokens) and de-duplicated.
4. `main_object` (if set) is appended to each keyword for extra context.
5. The combined negative (`base_negative` + accumulated keywords) is re-encoded
   with `clip` and swapped into the CFG guider so the **next** step suppresses it.
6. If `clear_negatives_at_stop`, the negative reverts to neutral at `vqa_stop_step`;
   otherwise accumulated negatives are kept for continued creative steering.

The implementation subclasses `comfy.samplers.CFGGuider` and refreshes the live
negative in `__call__` before each denoiser call, so updates take effect on the
very next step. This is model-type agnostic (works for epsilon / v-prediction /
flow models), so it supports SD1.5 / SDXL / SD3 / Flux.

## Step-value format

`vqa_start_step` / `vqa_stop_step` accept either:

- a **fraction** in `(0, 1]` — e.g. `0.5` = halfway through the steps, or
- an **absolute step index** > 1 — e.g. `28` = step 28.

Defaults `0.0` → `1.0` cover the full run.

## Example log output

```
[Step 0] + 'sculpture'
[Step 0] negatives: sculpture
[Step 1] + 'statue'
[Step 1] negatives: sculpture, statue
...
```

## Tips

- Connect the same `CLIP` and `VAE` you use for conditioning/preview — the node
  re-encodes the negative and decodes `x0` itself.
- `positive` is fixed; only the **negative** is updated dynamically. Provide your
  base negative as the `base_negative` string (not as a CONDITIONING input).
- Lower `freq` (e.g. `2`–`4`) to cut VLM/VAE overhead at the cost of slower
  negative accumulation.
- If no `vlm_oracle` is connected, the node behaves as a standard KSampler and
  returns `"(no VLM oracle connected; standard sampling)"` in the log.
