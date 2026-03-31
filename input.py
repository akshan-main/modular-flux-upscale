"""Input steps for Flux upscaling: text encoding, Lanczos upscale."""

import PIL.Image
import torch

from diffusers.utils import logging
from diffusers.modular_pipelines.modular_pipeline import ModularPipelineBlocks, PipelineState
from diffusers.modular_pipelines.modular_pipeline_utils import InputParam, OutputParam
from diffusers.modular_pipelines.flux.encoders import FluxTextEncoderStep

logger = logging.get_logger(__name__)


class FluxUpscaleTextEncoderStep(FluxTextEncoderStep):
    """Flux text encoder step with guidance_scale input.

    Flux uses guidance embedding (a tensor), not CFG. This step stores the
    guidance_scale so downstream blocks can create the guidance tensor.
    """

    @property
    def inputs(self) -> list[InputParam]:
        return super().inputs + [
            InputParam(
                "guidance_scale",
                type_hint=float,
                default=3.5,
                description="Guidance scale for Flux guidance embedding.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        # Store guidance_scale in state for downstream blocks
        block_state = self.get_block_state(state)
        guidance_scale = getattr(block_state, "guidance_scale", 3.5)
        state.set("guidance_scale", guidance_scale)
        return super().__call__(components, state)


class FluxUpscaleUpscaleStep(ModularPipelineBlocks):
    """Upscales the input image using Lanczos interpolation."""

    @property
    def description(self) -> str:
        return "Upscale input image using Lanczos interpolation."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("image", type_hint=PIL.Image.Image, required=True,
                       description="Input PIL image to upscale."),
            InputParam("upscale_factor", type_hint=float, default=2.0,
                       description="Scale multiplier."),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("upscaled_image", type_hint=PIL.Image.Image,
                        description="Lanczos-upscaled PIL image."),
            OutputParam("upscaled_width", type_hint=int),
            OutputParam("upscaled_height", type_hint=int),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        image = block_state.image
        factor = block_state.upscale_factor

        if not isinstance(image, PIL.Image.Image):
            if hasattr(image, "convert"):
                image = image.convert("RGB")
            else:
                raise TypeError(f"Expected PIL.Image, got {type(image)}")

        w, h = image.size
        new_w = int(round(w * factor))
        new_h = int(round(h * factor))

        # Ensure divisible by 16 (VAE scale 8 * packing factor 2)
        new_w = (new_w // 16) * 16
        new_h = (new_h // 16) * 16

        if new_w < 16 or new_h < 16:
            raise ValueError(
                f"Upscaled size ({new_w}x{new_h}) too small. "
                f"Input {w}x{h} with factor {factor}."
            )

        upscaled = image.resize((new_w, new_h), PIL.Image.LANCZOS)

        block_state.upscaled_image = upscaled
        block_state.upscaled_width = new_w
        block_state.upscaled_height = new_h

        # Also set height/width for downstream blocks (set_timesteps needs these)
        state.set("height", new_h)
        state.set("width", new_w)
        state.set("image", image)

        logger.info(f"Upscaled {w}x{h} -> {new_w}x{new_h} (factor={factor})")

        self.set_block_state(state, block_state)
        return components, state
