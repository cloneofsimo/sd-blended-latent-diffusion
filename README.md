# Blended Latent Diffusion (with Stable Diffusion)

For some reason authors of Blended latent diffusion didn't release their code. So I decided to implement it myself. It (since Stable diffusion is such a famous model) is built on stable diffusion's code.

## How to use

I'm assuming you have all the enviornment to run [stable diffusion](https://github.com/CompVis/stable-diffusion). Especially, you need to have `stable-diffusion-v1-*-original` weights as `models/ldm/stable-diffusion-v1/model.ckpt`. If you are not sure have a look at detailed info at the original repo.

Prepare a mask image (black and white, with white signifying target), and a source image. Then run,

```
python scripts/blended_diff.py --prompt "PROMPT" --perform bld --mask <path/to/mask.png> --src <path/to/source.png> --out <path/to/output.png>
```

You can add `--invert_mask` flag to invert the mask.
