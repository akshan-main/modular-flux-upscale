"""Block compositions for Flux tiled upscaling."""

from diffusers.modular_pipelines.modular_pipeline import SequentialPipelineBlocks
from diffusers.modular_pipelines.flux.before_denoise import FluxSetTimestepsStep

try:
    from .input import FluxUpscaleTextEncoderStep, FluxUpscaleUpscaleStep
    from .denoise import FluxUpscaleMultiDiffusionStep
except ImportError:
    from input import FluxUpscaleTextEncoderStep, FluxUpscaleUpscaleStep
    from denoise import FluxUpscaleMultiDiffusionStep


class FluxUpscaleMultiDiffusionBlocks(SequentialPipelineBlocks):
    """Modular pipeline blocks for tiled Flux upscaling with MultiDiffusion.

    Block graph:
        [0] text_encoder    — FluxUpscaleTextEncoderStep
        [1] upscale         — FluxUpscaleUpscaleStep (Lanczos)
        [2] set_timesteps   — FluxSetTimestepsStep (reused)
        [3] multidiffusion  — FluxUpscaleMultiDiffusionStep
    """

    block_classes = [
        FluxUpscaleTextEncoderStep,
        FluxUpscaleUpscaleStep,
        FluxSetTimestepsStep,
        FluxUpscaleMultiDiffusionStep,
    ]
    block_names = ["text_encoder", "upscale", "set_timesteps", "multidiffusion"]

    @property
    def description(self) -> str:
        return (
            "Modular tiled upscaling pipeline for Flux.\n"
            "Uses MultiDiffusion latent-space blending with optional ControlNet.\n"
            "Supports progressive multi-pass upscaling for 4x+."
        )
