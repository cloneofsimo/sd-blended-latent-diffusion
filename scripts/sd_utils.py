import os
from typing import Literal, Optional
import torch
from ldm.util import instantiate_from_config
from PIL import Image
import numpy as np
from einops import rearrange


def load_model_from_config(config, ckpt, device, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    return model


def reshape_image(
    img: Image.Image,
    mode: Literal["centercrop", "resize", "resize_max_axis"],
    width: int,
    height: int,
    mask: Optional[Image.Image] = None,
):
    if mode == "centercrop":
        imgw, imgh = img.size
        assert imgw >= width and imgh >= height
        left = (imgw - width) // 2
        top = (imgh - height) // 2
        right = (imgw + width) // 2
        bottom = (imgh + height) // 2
        img = img.crop((left, top, right, bottom))
        mask = mask.crop((left, top, right, bottom)) if mask is not None else None

    elif mode == "resize":
        img = img.resize((width, height), Image.BICUBIC)
        mask = mask.resize((width, height), Image.BICUBIC) if mask is not None else None

    elif mode == "resize_max_axis":
        imgw, imgh = img.size
        if imgw > imgh:
            img = img.resize((width, int(imgh * width / imgw)), Image.BICUBIC)
            mask = (
                mask.resize((width, int(imgh * width / imgw)), Image.BICUBIC)
                if mask is not None
                else None
            )
        else:
            img = img.resize((int(imgw * height / imgh), height), Image.BICUBIC)
            mask = (
                mask.resize((int(imgw * height / imgh), height), Image.BICUBIC)
                if mask is not None
                else None
            )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if mask is not None:
        mask = mask.resize((width // 8, height // 8), Image.BICUBIC)

    return img, mask


def prepare_image(
    img_path,
    mask_path=None,
    width: int = 512,
    height: int = 512,
    mode: str = "centercrop",
):

    image = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L") if mask_path is not None else None

    image, mask = reshape_image(image, mode, width, height, mask)
    # image = init_img.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = 2 * image - 1

    if mask is not None:
        mask = np.array(mask).astype(np.float32) / 255.0
        mask = mask[None]
        mask = torch.from_numpy(mask)
        return image, mask

    else:
        return image


def save_batch_images(img_torch, path, global_counter=0):
    img_torch = torch.clamp((img_torch + 1.0) / 2.0, min=0.0, max=1.0)
    img_torch = img_torch.cpu().permute(0, 2, 3, 1).numpy()

    for x_sample in img_torch:
        x_sample = 255.0 * x_sample
        img = Image.fromarray(x_sample.astype(np.uint8))

        img.save(os.path.join(path, f"{global_counter:05}.png"))
        global_counter += 1

    return global_counter
