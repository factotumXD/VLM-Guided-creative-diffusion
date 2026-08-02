# VLM-Guided Adaptive Negative Prompting — ComfyUI Custom Nodes

ComfyUI nodes that port the closed-loop **VLM-Guided Adaptive Negative Prompting**
method from this repository into native ComfyUI sampling. A Vision-Language Model
(VLM) monitors intermediate denoiser outputs during generation, identifies
dominant visual elements, and dynamically accumulates them as negative prompts to
steer generation away from common patterns — producing more creative results.

The original method targeted SD3.5 via a custom diffusers pipeline. These nodes
operate on ComfyUI's standard `MODEL` / `CLIP` / `VAE` / `CONDITIONING` types, so
they work with **SD1.5, SDXL, SD3, and Flux** alike.

Paper: [VLM-Guided Adaptive Negative Prompting for Creative Generation](https://arxiv.org/abs/2510.10715)

## Nodes

| Node | Category | Purpose |
|---|---|---|
| [VLM Oracle Loader](docs/VLMOracleLoader.md) | `VLM-Guided/oracle` | Load a ViLT / Qwen-VL oracle → `VLM_ORACLE` |
| [VLM-Guided KSampler](docs/VLMGuidedKSampler.md) | `VLM-Guided/sampling` | KSampler with adaptive negative-prompt feedback |
| [Image To 3D](docs/ImageTo3D.md) | `VLM-Guided/3d` | Convert the generated image to a 3D asset |

## Installation

1. Copy or symlink this `comfyui_nodes/` directory into `ComfyUI/custom_nodes/`
   (e.g. as `ComfyUI/custom_nodes/VLM_Guided_Nodes/`).
2. Install the (non-bundled) dependency:
   ```bash
   pip install -r comfyui_nodes/requirements.txt
   ```
   ComfyUI already ships `torch`, `transformers`, `accelerate`, `numpy`, `Pillow`.
   The only extra is `qwen-vl-utils` (used by the Qwen-VL oracle).
3. Restart ComfyUI. The three nodes appear under the `VLM-Guided` category.

## How it works

The feedback loop mirrors `custom_model/custom_sd35.py`:

1. **Intercept** — the sampler callback receives the predicted clean latent `x0`
   each step (model-type-aware; more general than the original flow-only `_flow_to_x0`).
2. **Decode** — `vae.decode(x0)` produces an RGB image for the VLM.
3. **VLM Query** — the oracle is asked the configured question(s).
4. **Accumulate** — answers are cleaned (`clean_vqa_answer`), de-duplicated, and
   appended to the growing negative prompt.
5. **Feedback** — the combined negative is re-encoded with `clip` and swapped into
   the CFG guider so the **next** step suppresses the identified elements.

`clear_negatives_at_stop` optionally reverts to a neutral negative when the VQA
window ends; otherwise accumulated negatives are kept for continued creative
steering.

## Example workflow

```
Load Checkpoint ──┬── MODEL  ─────────────────────────────┐
                  ├── CLIP   ───────────────────────────┐ │
                  └── VAE    ─────────────────────────┐ │ │
                                                 │     │ │
                                                 │     │ │
Empty Latent ── LATENT ──┐                       │     │ │
                         ▼                       ▼     ▼ ▼
   CLIP Text Encode (positive) ─ CONDITIONING ──► VLM-Guided KSampler
                                                  │  ▲
   VLM Oracle Loader ── VLM_ORACLE ───────────────┘  │ (optional)
                                                  │
                              IMAGE ──────────────┼──► Image To 3D ──► PLY mesh
                              LATENT ─────────────┘                  ──► depth image
                              negatives_log (STRING)
```

Typical settings (matching the paper): `steps=28`, `cfg=4.5`, ViLT oracle,
`topk=1`, `freq=1`, `vqa_start_step=0.0`, `vqa_stop_step=1.0` (full range).

## Supported VLMs

| Model | Speed | VRAM | Best for |
|---|---|---|---|
| `dandelin/vilt-b32-finetuned-vqa` | Fast (~13s overhead / 28 steps) | Low | Experimentation, batches |
| `Qwen/Qwen2.5-VL-3B-Instruct` | Medium | Medium | Quality / capability |
| `Qwen/Qwen2.5-VL-7B-Instruct` | Slower | High | Best quality |

Any other HuggingFace VQA / image-text-to-text id can be typed into the loader's
`custom_model_id` field (Qwen-style `image-text-to-text` ids are detected by the
`Qwen/` prefix; everything else is loaded as a `visual-question-answering` pipeline).

## Notes & caveats

- The x0 estimate is decoded with the VAE **every queried step** (every step by
  default), matching the original. Use `freq>1` to reduce VLM/VAE overhead.
- The Qwen-VL oracle shares the GPU (`device_map="auto"`); for large Qwen models
  on top of a large diffusion model, watch VRAM — set the loader `device=cpu`
  only for ViLT-class models is not required, but for Qwen-7B consider enough VRAM.
- The mid-loop negative update relies on `comfy.samplers.CFGGuider` exposing
  `set_conds` / `set_cfg` / `sample` and reading `self.uncond` per call (the same
  APIs `SamplerCustomAdvanced` uses). If a build caches conditioning differently,
  the `negatives_log` output makes the behavior easy to diagnose.

## Citation

```bibtex
@misc{golan2025creative,
  author        = {Golan, Shelly and Nitzan, Yotam and Wu, Zongze and Patashnik, Or},
  title         = {VLM-Guided Adaptive Negative Prompting for Creative Generation},
  year          = {2025},
  eprint        = {2510.10715},
  archivePrefix = {arXiv},
  primaryClass  = {cs.GR},
}
```

License: Apache 2.0 (matching the source repository).
