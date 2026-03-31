"""Modular pipeline class for tiled Flux upscaling."""

from diffusers.modular_pipelines.flux.modular_pipeline import FluxModularPipeline


class FluxUpscaleModularPipeline(FluxModularPipeline):
    """A ModularPipeline for tiled Flux upscaling.

    Inherits all Flux component properties and overrides the default blocks
    to use the tiled upscaling block composition.
    """

    default_blocks_name = "FluxUpscaleMultiDiffusionBlocks"
