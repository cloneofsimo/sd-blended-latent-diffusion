import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange

from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from scripts.sd_utils import load_model_from_config, prepare_image, save_batch_images
import torch.nn as nn


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

INPAINT_MASK = "/home/simo/dl/lst/stable-diffusion-largesample/data/inpainting_examples/6458524847_2f4c361183_k_mask.png"
INPAINT_SRC = "/home/simo/dl/lst/stable-diffusion-largesample/data/inpainting_examples/6458524847_2f4c361183_k.png"


parser = argparse.ArgumentParser()

parser.add_argument(
    "--prompt",
    type=str,
    nargs="?",
    default="a painting of a virus monster playing guitar",
    help="the prompt to render",
)
parser.add_argument(
    "--outdir",
    type=str,
    nargs="?",
    help="dir to write results to",
    default="outputs/txt2img-samples",
)
parser.add_argument(
    "--skip_grid",
    action="store_true",
    help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
)
parser.add_argument(
    "--skip_save",
    action="store_true",
    help="do not save individual samples. For speed measurements.",
)
parser.add_argument(
    "--ddim_steps",
    type=int,
    default=50,
    help="number of ddim sampling steps",
)
parser.add_argument(
    "--plms",
    action="store_true",
    help="use plms sampling",
)
parser.add_argument(
    "--laion400m",
    action="store_true",
    help="uses the LAION400M model",
)
parser.add_argument(
    "--fixed_code",
    action="store_true",
    help="if enabled, uses the same starting code across samples ",
)
parser.add_argument(
    "--ddim_eta",
    type=float,
    default=0.0,
    help="ddim eta (eta=0.0 corresponds to deterministic sampling",
)
parser.add_argument(
    "--n_iter",
    type=int,
    default=2,
    help="sample this often",
)
parser.add_argument(
    "--H",
    type=int,
    default=512,
    help="image height, in pixel space",
)
parser.add_argument(
    "--W",
    type=int,
    default=512,
    help="image width, in pixel space",
)
parser.add_argument(
    "--C",
    type=int,
    default=4,
    help="latent channels",
)
parser.add_argument(
    "--f",
    type=int,
    default=8,
    help="downsampling factor",
)
parser.add_argument(
    "--n_samples",
    type=int,
    default=3,
    help="how many samples to produce for each given prompt. A.k.a. batch size",
)
parser.add_argument(
    "--n_rows",
    type=int,
    default=0,
    help="rows in the grid (default: n_samples)",
)
parser.add_argument(
    "--scale",
    type=float,
    default=7.5,
    help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)
parser.add_argument(
    "--from-file",
    type=str,
    help="if specified, load prompts from this file",
)
parser.add_argument(
    "--config",
    type=str,
    default="configs/stable-diffusion/v1-inference.yaml",
    help="path to config which constructs model",
)
parser.add_argument(
    "--ckpt",
    type=str,
    default="models/ldm/stable-diffusion-v1/model.ckpt",
    help="path to checkpoint of model",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="the seed (for reproducible sampling)",
)
parser.add_argument(
    "--precision",
    type=str,
    help="evaluate at this precision",
    choices=["full", "autocast"],
    default="autocast",
)

parser.add_argument(
    "--perform",
    type=str,
    help="what to perform : inpaint with prompt,...",
    choices=["promptless", "bld"],
    default="promptless",
)


opt = parser.parse_args()


config = OmegaConf.load(f"{opt.config}")


def promptless(model, img, mask):
    init_latent = model.get_first_stage_encoding(
        model.encode_first_stage(img)
    )  # move to latent space

    aed = model.decode_first_stage(init_latent)

    return aed


def blended_latent_diffusion(
    model, img, mask, sampler, prompts, t_enc=30, batch_size=1
):
    device = next(model.parameters()).device

    sampler.make_schedule(
        ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False
    )

    init_latent = model.get_first_stage_encoding(
        model.encode_first_stage(img)
    )  # move to latent space

    uc = None
    if opt.scale != 1.0:
        uc = model.get_learned_conditioning(batch_size * [""])
    if isinstance(prompts, tuple):
        prompts = list(prompts)
    c = model.get_learned_conditioning(prompts)

    noise = torch.randn_like(init_latent).to(device)
    # encode (scaled latent)
    z_enc = sampler.stochastic_encode(
        init_latent, torch.tensor([t_enc] * batch_size).to(device), noise=noise
    )
    # decode it
    samples = sampler.decode_blm(
        z_enc,
        init_latent,
        c,
        mask,
        noise,
        t_enc,
        unconditional_guidance_scale=opt.scale,
        unconditional_conditioning=uc,
    )

    x_samples = model.decode_first_stage(samples)

    return x_samples


if __name__ == "__main__":

    device = "cuda:1"
    # prepare image.
    img, mask = prepare_image(INPAINT_SRC, INPAINT_MASK, opt.H, opt.W)
    img = img.to(device)
    mask = mask.to(device)

    # prepare model.
    model = load_model_from_config(config, f"{opt.ckpt}", device=device)
    seed_everything(opt.seed)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    if opt.perform == "promptless":
        result_img = promptless(model, img, mask)
    else:
        prompt = opt.prompt
        result_img = blended_latent_diffusion(model, img, mask, sampler, prompt)

    # save result.
    save_batch_images(result_img, f"{opt.outdir}/")
