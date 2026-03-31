"""MultiDiffusion tiled upscaling step for Flux 1.

Runs the Flux transformer on overlapping latent tiles with cosine-weighted
blending of noise predictions. Supports optional ControlNet (e.g. jasperai
upscaler) and progressive multi-pass upscaling.
"""

import math
import time

import numpy as np
import PIL.Image
import torch
from tqdm.auto import tqdm

from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, FluxTransformer2DModel
from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.modular_pipelines.modular_pipeline import ModularPipelineBlocks, PipelineState
from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec, InputParam, OutputParam

try:
    from .utils_tiling import LatentTileSpec, make_cosine_tile_weight, plan_latent_tiles
except ImportError:
    from utils_tiling import LatentTileSpec, make_cosine_tile_weight, plan_latent_tiles

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_pil_rgb(image) -> PIL.Image.Image:
    if isinstance(image, PIL.Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return PIL.Image.fromarray(image).convert("RGB")
    if isinstance(image, torch.Tensor):
        arr = image.cpu().float().clamp(0, 1).numpy()
        if arr.ndim == 4:
            arr = arr[0]
        if arr.shape[0] in (1, 3, 4):
            arr = arr.transpose(1, 2, 0)
        return PIL.Image.fromarray((arr * 255).astype(np.uint8)).convert("RGB")
    raise TypeError(f"Cannot convert {type(image)} to PIL.Image")


def _compute_auto_strength(upscale_factor: float, pass_index: int, num_passes: int) -> float:
    if num_passes > 1:
        return 0.35 if pass_index == 0 else 0.25
    if upscale_factor <= 2.0:
        return 0.35
    elif upscale_factor <= 4.0:
        return 0.25
    return 0.2


def _swap_scheduler(components, scheduler_name: str):
    key = scheduler_name.strip().lower()
    if key in ("euler", "flowmatcheuler"):
        from diffusers import FlowMatchEulerDiscreteScheduler as Cls
    else:
        logger.warning(f"Unknown scheduler_name '{scheduler_name}'. Keeping current.")
        return
    components.scheduler = Cls.from_config(components.scheduler.config)


# From diffusers.pipelines.flux
def _calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def _retrieve_timesteps(scheduler, num_inference_steps, device, sigmas=None, **kwargs):
    if sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    return scheduler.timesteps, len(scheduler.timesteps)


def _retrieve_latents(encoder_output, generator=None, sample_mode="sample"):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder_output")


def _pack_latents(latents, batch_size, num_channels, height, width):
    """4D (B,C,H,W) -> 3D packed (B, H/2*W/2, C*4)."""
    latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)
    return latents


def _unpack_latents(latents, height, width, vae_scale_factor):
    """3D packed -> 4D (B,C,H,W)."""
    batch_size, num_patches, channels = latents.shape
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))
    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)
    return latents


def _prepare_latent_image_ids(height, width, device, dtype):
    """Create RoPE position IDs for a spatial grid. height/width in latent//2 space."""
    ids = torch.zeros(height, width, 3)
    ids[..., 1] = torch.arange(height)[:, None]
    ids[..., 2] = torch.arange(width)[None, :]
    return ids.reshape(height * width, 3).to(device=device, dtype=dtype)


def _prepare_tile_image_ids(tile_h_half, tile_w_half, global_y_half, global_x_half, device, dtype):
    """Create RoPE position IDs for a tile with global spatial offsets."""
    ids = torch.zeros(tile_h_half, tile_w_half, 3)
    ids[..., 1] = torch.arange(tile_h_half)[:, None] + global_y_half
    ids[..., 2] = torch.arange(tile_w_half)[None, :] + global_x_half
    return ids.reshape(tile_h_half * tile_w_half, 3).to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# MultiDiffusion Step
# ---------------------------------------------------------------------------


class FluxUpscaleMultiDiffusionStep(ModularPipelineBlocks):
    """MultiDiffusion tiled upscaling for Flux 1.

    Encodes the upscaled image, runs the Flux transformer on overlapping
    latent tiles with cosine-weighted blending, and decodes. Supports
    optional ControlNet (e.g. jasperai/Flux.1-dev-Controlnet-Upscaler)
    and progressive multi-pass upscaling.
    """

    model_name = "flux-upscale"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", FluxTransformer2DModel),
            ComponentSpec("vae", AutoencoderKL),
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
            ComponentSpec("controlnet", FluxControlNetModel, required=False),
        ]

    @property
    def description(self) -> str:
        return (
            "MultiDiffusion tiled upscaling step for Flux. "
            "Blends noise predictions from overlapping latent tiles using cosine weights."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("upscaled_image", type_hint=PIL.Image.Image, required=True),
            InputParam("upscaled_width", type_hint=int, required=True),
            InputParam("upscaled_height", type_hint=int, required=True),
            InputParam("image", description="Original input image for progressive mode."),
            InputParam("upscale_factor", type_hint=float, default=2.0),
            InputParam("generator"),
            InputParam("batch_size", type_hint=int, required=True),
            InputParam("num_images_per_prompt", type_hint=int, default=1),
            InputParam("dtype", type_hint=torch.dtype, required=True),
            InputParam("prompt_embeds", type_hint=torch.Tensor, required=True),
            InputParam("pooled_prompt_embeds", type_hint=torch.Tensor, required=True),
            InputParam("num_inference_steps", type_hint=int, default=28),
            InputParam("strength", type_hint=float, default=0.35),
            InputParam("guidance_scale", type_hint=float, default=3.5),
            InputParam("timesteps", type_hint=torch.Tensor),
            InputParam("guidance", type_hint=torch.Tensor),
            InputParam("output_type", type_hint=str, default="pil"),
            # Tile params
            InputParam("latent_tile_size", type_hint=int, default=64,
                       description="Tile size in latent pixels (64 = 512px)."),
            InputParam("latent_overlap", type_hint=int, default=16,
                       description="Overlap in latent pixels (16 = 128px)."),
            # ControlNet
            InputParam("control_image", description="Optional ControlNet conditioning image."),
            InputParam("controlnet_conditioning_scale", default=1.0),
            InputParam("control_guidance_start", default=0.0),
            InputParam("control_guidance_end", default=1.0),
            # Progressive upscaling
            InputParam("progressive", type_hint=bool, default=True),
            InputParam("auto_strength", type_hint=bool, default=True),
            InputParam("return_metadata", type_hint=bool, default=False),
            InputParam("scheduler_name", type_hint=str, default=None),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("images", type_hint=list, description="Upscaled output images."),
            OutputParam("metadata", type_hint=dict),
        ]

    def _run_tile_transformer(
        self,
        components,
        tile_latents_packed: torch.Tensor,
        t: torch.Tensor,
        block_state,
        tile_img_ids: torch.Tensor,
        cn_tile=None,
    ) -> torch.Tensor:
        """Run Flux transformer (+ optional ControlNet) on one packed tile."""
        timestep = t.expand(tile_latents_packed.shape[0]).to(tile_latents_packed.dtype)

        controlnet_block_samples = None
        controlnet_single_block_samples = None
        controlnet_blocks_repeat = False

        if cn_tile is not None and hasattr(components, "controlnet") and components.controlnet is not None:
            controlnet_blocks_repeat = components.controlnet.input_hint_block is None
            cn_guidance = block_state.guidance
            if cn_guidance is None and components.controlnet.config.guidance_embeds:
                cn_guidance = torch.full(
                    [tile_latents_packed.shape[0]], block_state.guidance_scale,
                    device=tile_latents_packed.device, dtype=torch.float32,
                )

            cond_scale = block_state._cn_cond_scale
            if isinstance(block_state._cn_keep, list) and block_state._cn_step_idx < len(block_state._cn_keep):
                keep_val = block_state._cn_keep[block_state._cn_step_idx]
            else:
                keep_val = 1.0
            if isinstance(cond_scale, list):
                cond_scale = [c * keep_val for c in cond_scale]
            else:
                cond_scale = cond_scale * keep_val

            controlnet_block_samples, controlnet_single_block_samples = components.controlnet(
                hidden_states=tile_latents_packed,
                controlnet_cond=cn_tile,
                conditioning_scale=cond_scale,
                timestep=timestep / 1000,
                guidance=cn_guidance,
                pooled_projections=block_state.pooled_prompt_embeds,
                encoder_hidden_states=block_state.prompt_embeds,
                txt_ids=block_state._txt_ids,
                img_ids=tile_img_ids,
                return_dict=False,
            )

        noise_pred = components.transformer(
            hidden_states=tile_latents_packed,
            timestep=timestep / 1000,
            guidance=block_state.guidance,
            pooled_projections=block_state.pooled_prompt_embeds,
            encoder_hidden_states=block_state.prompt_embeds,
            txt_ids=block_state._txt_ids,
            img_ids=tile_img_ids,
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
            return_dict=False,
            **({"controlnet_blocks_repeat": controlnet_blocks_repeat}
               if controlnet_block_samples is not None else {}),
        )[0]

        return noise_pred

    def _run_single_pass(
        self,
        components,
        block_state,
        upscaled_image: PIL.Image.Image,
        h: int,
        w: int,
        ctrl_pil,
        use_controlnet: bool,
        latent_tile_size: int,
        latent_overlap: int,
    ) -> np.ndarray:
        """One full MultiDiffusion encode-denoise-decode pass."""
        device = components._execution_device
        vae = components.vae
        vae_scale_factor = components.vae_scale_factor

        # Enable VAE tiling for large images
        if hasattr(vae, "enable_tiling"):
            vae.enable_tiling()

        # --- VAE encode ---
        processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)
        image_tensor = processor.preprocess(upscaled_image, height=h, width=w).to(device, dtype=vae.dtype)
        image_latents = _retrieve_latents(vae.encode(image_tensor))
        image_latents = (image_latents - vae.config.shift_factor) * vae.config.scaling_factor

        # Latent dims (4D before packing)
        _, num_channels, latent_h, latent_w = image_latents.shape

        # --- Compute timesteps for this pass ---
        pass_strength = block_state._current_pass_strength
        truncated_steps = block_state.num_inference_steps
        outer_strength = block_state.strength
        if outer_strength > 0:
            original_steps = max(1, round(truncated_steps / outer_strength))
        else:
            original_steps = max(1, truncated_steps)

        # Compute image_seq_len for shift calculation
        image_seq_len = (latent_h // 2) * (latent_w // 2)
        sigmas = np.linspace(1.0, 1 / original_steps, original_steps)
        if hasattr(components.scheduler.config, "use_flow_sigmas") and components.scheduler.config.use_flow_sigmas:
            sigmas = None
        mu = _calculate_shift(
            image_seq_len,
            components.scheduler.config.get("base_image_seq_len", 256),
            components.scheduler.config.get("max_image_seq_len", 4096),
            components.scheduler.config.get("base_shift", 0.5),
            components.scheduler.config.get("max_shift", 1.15),
        )
        _ts, _ = _retrieve_timesteps(components.scheduler, original_steps, device, sigmas=sigmas, mu=mu)

        # Apply strength truncation
        init_timestep = min(int(original_steps * pass_strength), original_steps)
        t_start = max(original_steps - init_timestep, 0)
        timesteps = components.scheduler.timesteps[t_start * components.scheduler.order:]
        if hasattr(components.scheduler, "set_begin_index"):
            components.scheduler.set_begin_index(t_start * components.scheduler.order)
        num_inf_steps = original_steps - t_start

        # --- Prepare noisy latents ---
        batch_size = block_state.batch_size * block_state.num_images_per_prompt
        noise = randn_tensor(
            image_latents.shape,
            generator=block_state.generator,
            device=device,
            dtype=block_state.dtype,
        )
        latent_timestep = timesteps[:1].repeat(batch_size)
        # Pack both image_latents and noise for scale_noise
        packed_image = _pack_latents(image_latents, batch_size, num_channels, latent_h, latent_w)
        packed_noise = _pack_latents(noise, batch_size, num_channels, latent_h, latent_w)
        packed_latents = components.scheduler.scale_noise(packed_image, latent_timestep, packed_noise)

        # Unpack for spatial tile operations
        latents_4d = _unpack_latents(packed_latents, h, w, vae_scale_factor)

        # --- Guidance tensor ---
        guidance = None
        if components.transformer.config.guidance_embeds:
            guidance = torch.full(
                [batch_size], block_state.guidance_scale,
                device=device, dtype=torch.float32,
            )
        block_state.guidance = guidance

        # --- RoPE text IDs ---
        block_state._txt_ids = torch.zeros(
            block_state.prompt_embeds.shape[1], 3,
            device=device, dtype=block_state.prompt_embeds.dtype,
        )

        # --- ControlNet preparation ---
        full_cn_packed = None
        if use_controlnet and ctrl_pil is not None:
            if ctrl_pil.size != (w, h):
                ctrl_pil = ctrl_pil.resize((w, h), PIL.Image.LANCZOS)

            cn_tensor = processor.preprocess(ctrl_pil, height=h, width=w).to(device, dtype=vae.dtype)

            # jasperai upscaler has input_hint_block=None, so we VAE encode + pack
            if components.controlnet.input_hint_block is None:
                cn_latents = _retrieve_latents(vae.encode(cn_tensor))
                cn_latents = (cn_latents - vae.config.shift_factor) * vae.config.scaling_factor
                full_cn_packed = _pack_latents(
                    cn_latents, batch_size, num_channels,
                    cn_latents.shape[2], cn_latents.shape[3],
                )
            else:
                # XLabs-style: pass raw image tensor
                full_cn_packed = cn_tensor

            block_state._cn_cond_scale = getattr(block_state, "controlnet_conditioning_scale", 1.0)
            # Build controlnet_keep schedule
            cn_start = getattr(block_state, "control_guidance_start", 0.0)
            cn_end = getattr(block_state, "control_guidance_end", 1.0)
            block_state._cn_keep = [
                1.0 - float(i / len(timesteps) < cn_start or (i + 1) / len(timesteps) > cn_end)
                for i in range(len(timesteps))
            ]
            logger.info("MultiDiffusion: ControlNet enabled.")

        # --- Plan latent tiles ---
        tile_specs = plan_latent_tiles(latent_h, latent_w, latent_tile_size, latent_overlap)
        logger.info(
            f"MultiDiffusion: {len(tile_specs)} tiles "
            f"({latent_h}x{latent_w}, tile={latent_tile_size}, overlap={latent_overlap})"
        )

        # --- MultiDiffusion denoise loop ---
        for i, t in enumerate(tqdm(timesteps, total=num_inf_steps, desc="MultiDiffusion")):
            block_state._cn_step_idx = i

            noise_pred_accum = torch.zeros_like(latents_4d, dtype=torch.float32)
            weight_accum = torch.zeros(
                1, 1, latent_h, latent_w,
                device=device, dtype=torch.float32,
            )

            for tile in tile_specs:
                # Crop tile from 4D latents
                tile_4d = latents_4d[:, :, tile.y:tile.y + tile.h, tile.x:tile.x + tile.w].clone()

                # Pack for transformer
                tile_packed = _pack_latents(
                    tile_4d, batch_size, num_channels, tile.h, tile.w,
                )

                # Tile-specific RoPE IDs with global position offsets
                tile_img_ids = _prepare_tile_image_ids(
                    tile.h // 2, tile.w // 2,
                    tile.y // 2, tile.x // 2,
                    device, block_state.prompt_embeds.dtype,
                )

                # Crop ControlNet conditioning for this tile (in packed space)
                cn_tile = None
                if full_cn_packed is not None:
                    if components.controlnet.input_hint_block is None:
                        # Packed: need to crop in packed space
                        # Unpack -> crop -> repack
                        cn_4d = _unpack_latents(full_cn_packed, h, w, vae_scale_factor)
                        cn_tile_4d = cn_4d[:, :, tile.y:tile.y + tile.h, tile.x:tile.x + tile.w]
                        cn_tile = _pack_latents(
                            cn_tile_4d, batch_size, num_channels, tile.h, tile.w,
                        )
                    else:
                        # Raw image tensor: crop in pixel space
                        py = tile.y * vae_scale_factor
                        px = tile.x * vae_scale_factor
                        ph = tile.h * vae_scale_factor
                        pw = tile.w * vae_scale_factor
                        cn_tile = full_cn_packed[:, :, py:py + ph, px:px + pw]

                # Run transformer
                tile_noise_pred_packed = self._run_tile_transformer(
                    components, tile_packed, t, block_state, tile_img_ids, cn_tile,
                )

                # Unpack noise prediction back to 4D
                tile_noise_4d = tile_noise_pred_packed.view(
                    batch_size, tile.h // 2, tile.w // 2, num_channels * 4 // 4, 2, 2,
                )
                tile_noise_4d = tile_noise_4d.permute(0, 3, 1, 4, 2, 5)
                tile_noise_4d = tile_noise_4d.reshape(batch_size, num_channels, tile.h, tile.w)

                # Cosine weight
                tile_weight = make_cosine_tile_weight(
                    tile.h, tile.w, latent_overlap,
                    device, torch.float32,
                    is_top=(tile.y == 0),
                    is_bottom=(tile.y + tile.h >= latent_h),
                    is_left=(tile.x == 0),
                    is_right=(tile.x + tile.w >= latent_w),
                )

                # Accumulate
                noise_pred_accum[:, :, tile.y:tile.y + tile.h, tile.x:tile.x + tile.w] += (
                    tile_noise_4d.to(torch.float32) * tile_weight
                )
                weight_accum[:, :, tile.y:tile.y + tile.h, tile.x:tile.x + tile.w] += tile_weight

            # Blend
            blended = noise_pred_accum / weight_accum.clamp(min=1e-6)
            blended = torch.nan_to_num(blended, nan=0.0, posinf=0.0, neginf=0.0)

            # Pack blended prediction for scheduler step
            blended_packed = _pack_latents(
                blended.to(latents_4d.dtype), batch_size, num_channels, latent_h, latent_w,
            )
            latents_packed_current = _pack_latents(
                latents_4d, batch_size, num_channels, latent_h, latent_w,
            )

            # Scheduler step
            latents_packed_new = components.scheduler.step(
                blended_packed, t, latents_packed_current, return_dict=False,
            )[0]

            # Unpack back to 4D for next iteration
            latents_4d = _unpack_latents(latents_packed_new, h, w, vae_scale_factor)

        # --- Decode ---
        latents_final_packed = _pack_latents(
            latents_4d, batch_size, num_channels, latent_h, latent_w,
        )
        latents_unpacked = _unpack_latents(latents_final_packed, h, w, vae_scale_factor)
        latents_for_decode = (latents_unpacked / vae.config.scaling_factor) + vae.config.shift_factor
        decoded = vae.decode(latents_for_decode, return_dict=False)[0]

        # Convert to numpy
        decoded_np = decoded.cpu().float().clamp(-1, 1).numpy()
        decoded_np = (decoded_np[0].transpose(1, 2, 0) + 1) / 2  # [-1,1] -> [0,1]

        if decoded_np.shape[0] != h or decoded_np.shape[1] != w:
            pil_out = PIL.Image.fromarray((np.clip(decoded_np, 0, 1) * 255).astype(np.uint8))
            pil_out = pil_out.resize((w, h), PIL.Image.LANCZOS)
            decoded_np = np.array(pil_out).astype(np.float32) / 255.0

        return decoded_np

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        t_start = time.time()

        output_type = block_state.output_type
        latent_tile_size = block_state.latent_tile_size
        latent_overlap = block_state.latent_overlap

        # --- Scheduler swap ---
        scheduler_name = getattr(block_state, "scheduler_name", None)
        if scheduler_name is not None:
            _swap_scheduler(components, scheduler_name)

        # --- Progressive upscaling + auto-strength ---
        upscale_factor = getattr(block_state, "upscale_factor", 2.0)
        progressive = getattr(block_state, "progressive", True)
        auto_strength = getattr(block_state, "auto_strength", True)
        return_metadata = getattr(block_state, "return_metadata", False)
        user_strength = block_state.strength

        if progressive and upscale_factor > 2.0:
            num_passes = max(1, int(math.ceil(math.log2(upscale_factor))))
        else:
            num_passes = 1

        strength_per_pass = []
        for p in range(num_passes):
            if auto_strength:
                strength_per_pass.append(
                    _compute_auto_strength(upscale_factor, p, num_passes)
                )
            else:
                strength_per_pass.append(user_strength)

        # --- ControlNet setup ---
        original_image = getattr(block_state, "image", None)
        control_image_raw = getattr(block_state, "control_image", None)
        use_controlnet = (
            control_image_raw is not None
            and hasattr(components, "controlnet")
            and components.controlnet is not None
        )

        if num_passes == 1:
            # --- Single pass ---
            block_state._current_pass_strength = strength_per_pass[0]
            ctrl_pil = _to_pil_rgb(control_image_raw) if use_controlnet else None

            decoded_np = self._run_single_pass(
                components, block_state,
                upscaled_image=block_state.upscaled_image,
                h=block_state.upscaled_height,
                w=block_state.upscaled_width,
                ctrl_pil=ctrl_pil,
                use_controlnet=use_controlnet,
                latent_tile_size=latent_tile_size,
                latent_overlap=latent_overlap,
            )
        else:
            # --- Progressive multi-pass ---
            if original_image is None:
                orig_w = int(round(block_state.upscaled_width / upscale_factor))
                orig_h = int(round(block_state.upscaled_height / upscale_factor))
                original_image = block_state.upscaled_image.resize(
                    (orig_w, orig_h), PIL.Image.LANCZOS,
                )

            current_image = original_image
            current_w, current_h = current_image.width, current_image.height

            for p in range(num_passes):
                if p == num_passes - 1:
                    target_w = block_state.upscaled_width
                    target_h = block_state.upscaled_height
                else:
                    target_w = int(current_w * 2.0)
                    target_h = int(current_h * 2.0)
                    target_w = (target_w // 16) * 16
                    target_h = (target_h // 16) * 16

                pass_upscaled = current_image.resize((target_w, target_h), PIL.Image.LANCZOS)
                block_state._current_pass_strength = strength_per_pass[p]

                ctrl_pil = pass_upscaled.copy() if use_controlnet else None

                logger.info(
                    f"Progressive pass {p + 1}/{num_passes}: "
                    f"{current_w}x{current_h} -> {target_w}x{target_h} "
                    f"(strength={strength_per_pass[p]:.2f})"
                )

                decoded_np = self._run_single_pass(
                    components, block_state,
                    upscaled_image=pass_upscaled,
                    h=target_h,
                    w=target_w,
                    ctrl_pil=ctrl_pil,
                    use_controlnet=use_controlnet,
                    latent_tile_size=latent_tile_size,
                    latent_overlap=latent_overlap,
                )

                result_uint8 = (np.clip(decoded_np, 0, 1) * 255).astype(np.uint8)
                current_image = PIL.Image.fromarray(result_uint8)
                current_w, current_h = current_image.width, current_image.height

        # --- Format output ---
        result_uint8 = (np.clip(decoded_np, 0, 1) * 255).astype(np.uint8)
        if output_type == "pil":
            block_state.images = [PIL.Image.fromarray(result_uint8)]
        elif output_type == "np":
            block_state.images = [decoded_np]
        elif output_type == "pt":
            block_state.images = [torch.from_numpy(decoded_np).permute(2, 0, 1).unsqueeze(0)]
        else:
            block_state.images = [PIL.Image.fromarray(result_uint8)]

        # --- Metadata ---
        total_time = time.time() - t_start
        orig_size = (original_image.width, original_image.height) if original_image else None
        block_state.metadata = {
            "input_size": orig_size,
            "output_size": (block_state.upscaled_width, block_state.upscaled_height),
            "upscale_factor": upscale_factor,
            "num_passes": num_passes,
            "strength_per_pass": strength_per_pass,
            "total_time": total_time,
        }

        self.set_block_state(state, block_state)
        return components, state
