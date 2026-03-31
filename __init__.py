try:
    from .modular_blocks import FluxUpscaleMultiDiffusionBlocks
    from .modular_pipeline import FluxUpscaleModularPipeline
except ImportError:
    from modular_blocks import FluxUpscaleMultiDiffusionBlocks
    from modular_pipeline import FluxUpscaleModularPipeline

__all__ = ["FluxUpscaleMultiDiffusionBlocks", "FluxUpscaleModularPipeline"]
