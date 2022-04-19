import argparse, os, sys, glob
import torch
import numpy as np


from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid


from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


class Args:
    prompt = "a painting of a virus monster playing guitar" # "the prompt to render"
    outdir = "outputs/txt2img-samples" # "dir to write results to"
    ddim_steps = "number of ddim sampling steps" # "number of ddim sampling steps"
    plms = True # "use plms sampling"
    ddim_eta = 0.0 # "ddim eta (eta=0.0 corresponds to deterministic sampling")
    n_iter = 1 # "sample this often"
    H = 256 # "image height, in pixel space"
    W = 256 # "image width, in pixel space"
    n_samples = 4 # "how many samples to produce for the given prompt"
    scale = 5 # "unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))"
    is_notebook = False # "if we are on a notebook or not"
    
    
def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    map_location = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    pl_sd = torch.load(ckpt, map_location=map_location)
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model = model.half()
    model.to(device)
    model.eval()

    return model

    
def text2img(opt):

    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
    model = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt")  # TODO: check path

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        # TODO: double check if we need to set ddim_eta = 0 for PLMS
        opt.ddim_eta = 0
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    prompt = opt.prompt


    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    all_samples=list()
    all_samples_images = list()

    with torch.no_grad():
        with model.ema_scope():
            uc = None
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(opt.n_samples * [""])
            for n in trange(opt.n_iter, desc="Sampling"):
                c = model.get_learned_conditioning(opt.n_samples * [prompt])
                shape = [4, opt.H//8, opt.W//8]
                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                 conditioning=c,
                                                 batch_size=opt.n_samples,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,
                                                 eta=opt.ddim_eta,
                                                 is_notebook=opt.is_notebook,)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

                for x_sample in x_samples_ddim:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    image_vector = Image.fromarray(x_sample.astype(np.uint8))
                    image_vector.save(os.path.join(sample_path, f"{base_count:04}.png"))
                    all_samples_images.append(image_vector)
                    base_count += 1
                all_samples.append(x_samples_ddim)


    # additionally, save as grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=opt.n_samples)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    grid_image = Image.fromarray(grid.astype(np.uint8))
    grid_image.save(os.path.join(outpath, f'{prompt.replace(" ", "-")}.png'))

    # print(f"Your samples are ready and waiting four you here: \n{outpath} \nEnjoy.")

    return (grid_image, all_samples_images)
