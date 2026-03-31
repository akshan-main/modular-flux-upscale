# Modular Flux Upscale

Tiled image upscaling for Flux using [MultiDiffusion](https://arxiv.org/abs/2302.08113) latent-space blending. Built with [Modular Diffusers](https://huggingface.co/blog/modular-diffusers).

[![HuggingFace Hub](https://img.shields.io/badge/HuggingFace-Hub-yellow)](https://huggingface.co/akshan-main/modular-flux-upscale)

## What it does

- Image upscaling at any scale factor using Flux
- MultiDiffusion: blends overlapping transformer tile predictions in latent space with cosine weights. No visible seams
- Optional ControlNet conditioning for faithful upscaling
- Progressive upscaling: automatically splits large scale factors into multiple passes
- Auto-strength scaling per pass
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

controlnet = FluxControlNetModel.from_pretrained(
    "jasperai/Flux.1-dev-Controlnet-Upscaler", torch_dtype=torch.bfloat16
)
pipe.update_components(controlnet=controlnet)
pipe.enable_model_cpu_offload()

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

### Progressive upscale

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
| `progressive` | `True` | Split large upscale factors into multiple passes |
| `auto_strength` | `True` | Auto-scale strength per pass |
| `generator` | `None` | Torch generator for reproducibility |

## Limitations

- `latent_tile_size` below 64 may produce artifacts
- Very small inputs produce distortion. Use progressive mode
- ControlNet improves faithfulness but is optional
- Not suitable for text, line art, or pixel art
- FLUX.1-dev is a gated model - accept the license at https://huggingface.co/black-forest-labs/FLUX.1-dev

## Architecture

```
FluxUpscaleMultiDiffusionBlocks (SequentialPipelineBlocks)
  text_encoder      Flux TextEncoderStep (CLIP + T5, reused)
  upscale           Lanczos upscale step
  input             Flux InputStep (reused)
  set_timesteps     Flux SetTimestepsStep (reused)
  multidiffusion    MultiDiffusion step
                    - VAE encode full image
                    - Per timestep: transformer on each packed tile, cosine-weighted blend
                    - VAE decode full latents
```

## Project structure

```
utils_tiling.py              Latent tile planning, cosine weights
input.py                     Text encoder, upscale steps
denoise.py                   MultiDiffusion step, ControlNet integration
modular_blocks.py            Block composition
modular_pipeline.py          Pipeline class
hub_block/                   HuggingFace Hub block (consolidated single file)
```

## References

- [MultiDiffusion](https://arxiv.org/abs/2302.08113) (Bar-Tal et al., 2023)
- [Modular Diffusers](https://huggingface.co/blog/modular-diffusers)
- [Modular Diffusers contribution call](https://github.com/huggingface/diffusers/issues/13295)
- [ControlNet Upscaler](https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Upscaler)
