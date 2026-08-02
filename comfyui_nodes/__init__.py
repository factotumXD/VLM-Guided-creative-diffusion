"""ComfyUI custom nodes for VLM-Guided Adaptive Negative Prompting.

Ports the closed-loop VLM feedback method from this repository (originally a
custom SD3.5 diffusers pipeline) into modular ComfyUI nodes that work with any
ComfyUI-backed diffusion model (SD1.5 / SDXL / SD3 / Flux).

Nodes:
    * VLMOracleLoader    - load a ViLT / Qwen-VL vision-language model
    * VLMGuidedKSampler  - KSampler with adaptive negative-prompt feedback
    * ImageTo3D          - turn the generated image into a 3D asset

Install: copy/symlink this directory into ``ComfyUI/custom_nodes/`` and restart
ComfyUI. The custom ``VLM_ORACLE`` type links VLMOracleLoader -> VLMGuidedKSampler.
"""

from .vlm import VLMOracleLoader
from .sampler import VLMGuidedKSampler
from .three_d import ImageTo3D

# Custom connection type for the VLM oracle object.
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

NODE_CLASS_MAPPINGS = {
    "VLMOracleLoader": VLMOracleLoader,
    "VLMGuidedKSampler": VLMGuidedKSampler,
    "ImageTo3D": ImageTo3D,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VLMOracleLoader": "VLM Oracle Loader",
    "VLMGuidedKSampler": "VLM-Guided KSampler (Adaptive Negatives)",
    "ImageTo3D": "Image To 3D",
}
