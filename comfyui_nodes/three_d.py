"""Image-to-3D node for ComfyUI.

Turns a generated (or any) image into a 3D representation:

  * ``depth_mesh``  -> writes a depth-displaced PLY mesh (vertices colored from
                       the source image, displaced along Z by an estimated depth
                       map). Openable in MeshLab / Blender / online viewers.
  * ``anaglyph``    -> produces a red/cyan anaglyph image viewable with 3D
                       glasses (depth-based horizontal parallax).
  * ``depth_image`` -> returns the normalized depth map as an IMAGE.

Depth estimation uses a transformers ``depth-estimation`` pipeline (DPT /
Depth-Anything), so no extra 3D dependencies are required.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image

import folder_paths

# Depth models known to work with the transformers depth-estimation pipeline.
_DEPTH_MODELS = [
    "Intel/dpt-large",
    "Intel/dpt-hybrid-midas",
    "depth-anything/Depth-Anything-V2-Small-hf",
    "depth-anything/Depth-Anything-V2-Base-hf",
]


class ImageTo3D:
    """Convert an image into a 3D asset (depth mesh / anaglyph / depth map)."""

    CATEGORY = "VLM-Guided/3d"
    FUNCTION = "convert"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "ply_path")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (["depth_mesh", "anaglyph", "depth_image"], {"default": "depth_mesh"}),
                "depth_model": (_DEPTH_MODELS, {"default": _DEPTH_MODELS[0]}),
                "depth_scale": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.01}),
                "invert_depth": ("BOOLEAN", {"default": False}),
                "max_resolution": ("INT", {"default": 512, "min": 64, "max": 1024, "step": 8}),
                "filename_prefix": ("STRING", {"default": "VLM_3D"}),
            },
            "optional": {
                "device": (["auto", "cpu", "cuda"], {"default": "auto"}),
            },
        }

    def convert(self, image, method, depth_model, depth_scale, invert_depth,
                max_resolution, filename_prefix, device="auto"):
        # ComfyUI IMAGE -> PIL (first frame of the batch).
        pil = _tensor_to_pil(image[0])

        depth = _estimate_depth(pil, depth_model, device)
        depth = _normalize_depth(depth)
        if invert_depth:
            depth = 1.0 - depth

        ply_path = ""

        if method == "depth_image":
            out_tensor = _gray_to_image_tensor(depth)

        elif method == "anaglyph":
            anaglyph = _make_anaglyph(pil, depth, shift=int(max(1, depth_scale * 40)))
            out_tensor = _pil_to_tensor(anaglyph)

        else:  # depth_mesh
            mesh_pil = _resize_for_mesh(pil, max_resolution)
            mesh_depth = _normalize_depth(
                _resize_depth(depth, mesh_pil.size[0], mesh_pil.size[1])
            )
            ply_path = _write_ply(mesh_pil, mesh_depth, depth_scale, filename_prefix)
            out_tensor = _gray_to_image_tensor(mesh_depth)

        return (out_tensor, ply_path)


# --------------------------------------------------------------------------- #
# Helpers                                                                    #
# --------------------------------------------------------------------------- #
def _tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    arr = (image_tensor.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def _pil_to_tensor(pil: Image.Image) -> torch.Tensor:
    arr = np.array(pil.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _gray_to_image_tensor(gray: np.ndarray) -> torch.Tensor:
    rgb = np.stack([gray, gray, gray], axis=-1)
    return torch.from_numpy(rgb.astype(np.float32)).unsqueeze(0)


def _estimate_depth(pil: Image.Image, model_id: str, device: str) -> np.ndarray:
    from transformers import pipeline

    dev = None
    if device == "cpu":
        dev = -1
    elif device == "cuda":
        dev = 0 if torch.cuda.is_available() else -1
    else:  # auto -> prefer CPU to avoid clashing with the diffusion model on GPU
        dev = -1

    kwargs = {"model": model_id}
    if dev is not None:
        kwargs["device"] = dev
    pipe = pipeline("depth-estimation", **kwargs)
    result = pipe(pil)

    # Prefer the raw predicted depth; fall back to the rendered depth map.
    if isinstance(result, dict) and "predicted_depth" in result:
        depth = result["predicted_depth"]
        if hasattr(depth, "detach"):
            depth = depth.detach().cpu().numpy()
        depth = np.array(depth, dtype=np.float32)
        if depth.ndim == 3:  # (C,H,W) -> (H,W)
            depth = depth.squeeze(0)
    else:  # rendered "depth" PIL image
        depth_pil = result["depth"] if isinstance(result, dict) else result
        depth = np.array(depth_pil.convert("L"), dtype=np.float32) / 255.0
    return depth


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    d = depth.astype(np.float32)
    lo, hi = float(d.min()), float(d.max())
    if hi - lo < 1e-6:
        return np.zeros_like(d)
    return (d - lo) / (hi - lo)


def _resize_depth(depth: np.ndarray, w: int, h: int) -> np.ndarray:
    pil = Image.fromarray((depth * 255.0).clip(0, 255).astype(np.uint8))
    pil = pil.resize((w, h), Image.BILINEAR)
    return np.array(pil, dtype=np.float32) / 255.0


def _resize_for_mesh(pil: Image.Image, max_resolution: int) -> Image.Image:
    w, h = pil.size
    scale = min(1.0, float(max_resolution) / float(max(w, h)))
    if scale < 1.0:
        pil = pil.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return pil.convert("RGB")


def _make_anaglyph(pil: Image.Image, depth: np.ndarray, shift: int) -> Image.Image:
    """Red/cyan anaglyph from depth-based horizontal parallax."""
    rgb = np.array(pil.convert("RGB"), dtype=np.float32)
    h, w = rgb.shape[:2]
    depth_resized = _resize_depth(depth, w, h)

    # Per-pixel horizontal shift proportional to depth.
    max_shift = float(shift)
    offsets = (depth_resized * max_shift).astype(np.float32)

    xs = np.arange(w, dtype=np.float32)[None, :]  # (1, w)
    shifted = xs - offsets  # (H, W)

    x0 = np.clip(np.floor(shifted).astype(np.int32), 0, w - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    frac = (shifted - x0)[..., None]  # (H, W, 1)

    src = rgb.astype(np.float32)
    left = np.clip(src[np.arange(h)[:, None], x0] * (1 - frac) + src[np.arange(h)[:, None], x1] * frac, 0, 255)
    right_x = np.clip(np.floor(xs + offsets).astype(np.int32), 0, w - 1)
    right_x1 = np.clip(right_x + 1, 0, w - 1)
    frac_r = ((xs + offsets) - right_x)[..., None]
    right = np.clip(src[np.arange(h)[:, None], right_x] * (1 - frac_r) + src[np.arange(h)[:, None], right_x1] * frac_r, 0, 255)

    anaglyph = np.empty((h, w, 3), dtype=np.float32)
    anaglyph[..., 0] = left[..., 0]          # R from left view
    anaglyph[..., 1] = right[..., 1]         # G from right view
    anaglyph[..., 2] = right[..., 2]         # B from right view
    return Image.fromarray(anaglyph.clip(0, 255).astype(np.uint8))


def _write_ply(pil: Image.Image, depth: np.ndarray, depth_scale: float,
               filename_prefix: str) -> str:
    """Write a depth-displaced colored mesh as an ASCII PLY file.

    The mesh is a regular grid (one quad per pixel pair) displaced along -Z by
    the depth map. Returns the absolute path of the written file.
    """
    rgb = np.array(pil.convert("RGB"), dtype=np.uint8)
    h, w = depth.shape
    max_dim = float(max(w, h))

    # Subsample the grid to keep the triangle count manageable.
    stride = 1
    max_cells = 160_000
    while (w // stride) * (h // stride) > max_cells and stride < 8:
        stride += 1

    xs = np.arange(0, w, stride)
    ys = np.arange(0, h, stride)
    gx, gy = np.meshgrid(xs, ys)
    gz = -depth[gy, gx] * (depth_scale * max_dim)

    colors = rgb[gy, gx]

    out_dir = folder_paths.get_output_directory()
    os.makedirs(out_dir, exist_ok=True)
    def _count(prefix):
        n = 0
        for fn in os.listdir(out_dir):
            if fn.startswith(f"{prefix}_") and fn.endswith(".ply"):
                n += 1
        return n
    fname = f"{filename_prefix}_{_count(filename_prefix):05d}.ply"
    path = os.path.join(out_dir, fname)

    nrows, ncols = gx.shape
    n_verts = nrows * ncols
    # Two triangles per cell (minus last row/col).
    n_faces = 2 * max(0, nrows - 1) * max(0, ncols - 1)

    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_verts}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {n_faces}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        # Vertices: center the grid on the origin.
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0
        for j in range(nrows):
            for i in range(ncols):
                x = gx[j, i] - cx
                y = gy[j, i] - cy
                z = gz[j, i]
                r, g, b = colors[j, i]
                f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")

        # Faces: two triangles per cell.
        for j in range(nrows - 1):
            for i in range(ncols - 1):
                v00 = j * ncols + i
                v10 = j * ncols + (i + 1)
                v01 = (j + 1) * ncols + i
                v11 = (j + 1) * ncols + (i + 1)
                f.write(f"3 {v00} {v10} {v11}\n")
                f.write(f"3 {v00} {v11} {v01}\n")

    return path
