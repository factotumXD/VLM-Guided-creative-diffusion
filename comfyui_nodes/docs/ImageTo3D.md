# Image To 3D

**Node ID:** `ImageTo3D`
**Category:** `VLM-Guided/3d`

Converts an image (typically the output of `VLMGuidedKSampler`) into a 3D
representation using a transformers depth-estimation model. No extra 3D libraries
are required — the PLY mesh is written directly.

## Inputs

### Required

| Name | Type | Default | Description |
|---|---|---|---|
| `image` | `IMAGE` | — | Input image to convert. |
| `method` | COMBO | `depth_mesh` | Output mode (see below). |
| `depth_model` | COMBO | `Intel/dpt-large` | HuggingFace depth-estimation model. |
| `depth_scale` | FLOAT | `0.3` | Depth displacement strength (0–2). |
| `invert_depth` | BOOLEAN | `False` | Flip near/far (useful for some depth models). |
| `max_resolution` | INT | `512` | Cap the longest side when building the mesh (64–1024). |
| `filename_prefix` | STRING | `VLM_3D` | Prefix for the written PLY file. |

### Optional

| Name | Type | Default | Description |
|---|---|---|---|
| `device` | COMBO | `auto` | `auto` (CPU, to avoid GPU contention) / `cpu` / `cuda`. |

### `method` options

- **`depth_mesh`** — writes a depth-displaced, vertex-colored **PLY mesh** (a
  regular grid with one quad per pixel pair, displaced along `-Z` by the depth
  map). Openable in MeshLab, Blender, or any PLY viewer. Also returns the depth
  map as the `image` output.
- **`anaglyph`** — produces a red/cyan **anaglyph image** viewable with 3D
  glasses (depth-based horizontal parallax).
- **`depth_image`** — returns the normalized depth map as an `IMAGE`.

### `depth_model` options

- `Intel/dpt-large`
- `Intel/dpt-hybrid-midas`
- `depth-anything/Depth-Anything-V2-Small-hf`
- `depth-anything/Depth-Anything-V2-Base-hf`

## Outputs

| Name | Type | Description |
|---|---|---|
| `image` | `IMAGE` | Depth map (`depth_mesh` / `depth_image`) or anaglyph (`anaglyph`). |
| `ply_path` | STRING | Absolute path to the written `.ply` file, or `""` for non-mesh methods. |

## Behavior

- Depth is estimated with a `transformers` `depth-estimation` pipeline, then
  normalized to `[0, 1]` and (optionally) inverted.
- For `depth_mesh`, the image and depth are downsampled to `max_resolution` and a
  PLY is written into ComfyUI's output directory as
  `{filename_prefix}_{index:05d}.ply`. The grid is further subsampled if needed
  to keep the triangle count manageable (≤ ~160k cells).
- The depth model is downloaded from HuggingFace on first use.

## Tips

- **VRAM**: by default `device=auto` runs depth estimation on **CPU** to avoid
  clashing with the diffusion model on the GPU. Use `cuda` only if the GPU is free.
- Larger `max_resolution` gives finer meshes but bigger files and slower writes.
- Feed the `image` output into `Save Image` to keep the depth/anaglyph preview,
  and use the `ply_path` string to record the 3D asset location.
