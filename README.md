# Modular Flux Upscale

Tiled image upscaling for Flux using [MultiDiffusion](https://arxiv.org/abs/2302.08113) latent-space blending. Built with [Modular Diffusers](https://huggingface.co/blog/modular-diffusers).

[![HuggingFace Hub](https://img.shields.io/badge/HuggingFace-Hub-yellow)](https://huggingface.co/akshan-main/modular-flux-upscale)

## What it does

- Image upscaling at any scale factor using Flux
- MultiDiffusion: blends overlapping transformer tile predictions in latent space with cosine weights. No visible seams
- Optional ControlNet conditioning (jasperai/Flux.1-dev-Controlnet-Upscaler) for faithful upscaling
- Progressive upscaling: automatically splits 4x+ into multiple 2x passes
- Auto-strength scaling per pass
- Guidance embedding (not CFG) for single-pass-per-tile efficiency
- RoPE-aware tiling: each tile gets correct global position IDs

## Install

```bash
pip install git+https://github.com/huggingface/diffusers.git transformers accelerate safetensors sentencepiece protobuf
```

Requires diffusers from main (modular diffusers support).

## Usage

### From HuggingFace Hub

```python
from diffusers import ModularPipelineBlocks
from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel
import torch

blocks = ModularPipelineBlocks.from_pretrained(
    "akshan-main/modular-flux-upscale",
    trust_remote_code=True,
)

pipe = blocks.init_pipeline("black-forest-labs/FLUX.1-dev")
pipe.load_components(torch_dtype=torch.bfloat16)

# Optional: ControlNet Upscaler for faithful upscaling
controlnet = FluxControlNetModel.from_pretrained(
    "jasperai/Flux.1-dev-Controlnet-Upscaler", torch_dtype=torch.bfloat16
)
pipe.update_components(controlnet=controlnet)
pipe.to("cuda")

image = ...  # your PIL image

result = pipe(
    prompt="high quality, detailed, sharp",
    image=image,
    control_image=image,
    controlnet_conditioning_scale=1.0,
    upscale_factor=2.0,
    num_inference_steps=28,
    generator=torch.Generator("cuda").manual_seed(42),
    output="images",
)
result[0].save("upscaled.png")
```

### 4x progressive upscale

```python
result = pipe(
    prompt="high quality, detailed, sharp",
    image=image,
    control_image=image,
    controlnet_conditioning_scale=1.0,
    upscale_factor=4.0,
    progressive=True,
    generator=torch.Generator("cuda").manual_seed(42),
    output="images",
)
```

### Without ControlNet

```python
result = pipe(
    prompt="high quality, detailed, sharp",
    image=image,
    upscale_factor=2.0,
    strength=0.25,
    auto_strength=False,
    num_inference_steps=28,
    output="images",
)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image` | required | Input image (PIL) |
| `prompt` | `""` | Text prompt |
| `upscale_factor` | `2.0` | Scale multiplier |
| `strength` | `0.35` | Denoise strength. Ignored when `auto_strength=True` |
| `num_inference_steps` | `28` | Denoising steps |
| `guidance_scale` | `3.5` | Flux guidance embedding scale |
| `latent_tile_size` | `64` | Tile size in latent pixels (64 = 512px) |
| `latent_overlap` | `16` | Tile overlap in latent pixels (16 = 128px) |
| `control_image` | `None` | ControlNet conditioning image |
| `controlnet_conditioning_scale` | `1.0` | ControlNet strength |
| `progressive` | `True` | Split upscale_factor > 2 into multiple 2x passes |
| `auto_strength` | `True` | Auto-scale strength per pass |
| `scheduler_name` | `None` | Currently only "Euler" (FlowMatchEuler) |
| `generator` | `None` | Torch generator for reproducibility |

## Key differences from SDXL upscaler

- **Guidance embedding instead of CFG**: single transformer pass per tile (faster)
- **No negative prompts**: Flux doesn't use them
- **Packed latents**: tiles are packed/unpacked for the transformer, blended in 4D space
- **RoPE position IDs**: each tile gets correct global spatial position, improving coherence
- **FlowMatch scheduler**: uses `scale_noise` instead of `add_noise`

## Limitations

- Flux is trained on 1024x1024. `latent_tile_size` below 64 may produce artifacts
- 4x from inputs below 256px produces distortion. Use progressive mode
- ControlNet Upscaler improves faithfulness but is optional
- VRAM: ~16GB for 2x in bfloat16 (Flux transformer is larger than SDXL UNet)
- Flux.1-dev is a gated model — accept the license at https://huggingface.co/black-forest-labs/FLUX.1-dev

## Architecture

```
FluxUpscaleMultiDiffusionBlocks (SequentialPipelineBlocks)
  text_encoder      Flux TextEncoderStep (CLIP + T5, reused)
  upscale           Lanczos upscale step
  set_timesteps     Flux SetTimestepsStep (reused)
  multidiffusion    MultiDiffusion step
                    - VAE encode full image
                    - Per timestep: transformer on each packed tile, cosine-weighted blend in 4D
                    - VAE decode full latents
```

## Project structure

```
utils_tiling.py              Tile planning, cosine weights
input.py                     Text encoder, upscale steps
denoise.py                   MultiDiffusion step with ControlNet
modular_blocks.py            Block composition
modular_pipeline.py          Pipeline class
hub_block/                   HuggingFace Hub block (consolidated single file)
```

## Models

- Base: [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- ControlNet (optional): [jasperai/Flux.1-dev-Controlnet-Upscaler](https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Upscaler)

## References

- [MultiDiffusion](https://arxiv.org/abs/2302.08113) (Bar-Tal et al., 2023)
- [Modular Diffusers](https://huggingface.co/blog/modular-diffusers)
- [Modular Diffusers contribution call](https://github.com/huggingface/diffusers/issues/13295)

## License

Apache 2.0
