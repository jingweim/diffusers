'''
    python inference_lora_original_sd.py
'''

import os, sys
import torch

from accelerate.utils import set_seed
from diffusers import DPMSolverMultistepScheduler

sys.path.append('/gscratch/realitylab/jingweim/diffusers')
from src.diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_original import StableDiffusionPipeline

# Load model
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")


# arguments
seed = 0
num_inference_steps = 50
guidance_scale = 7.5
num_images_per_prompt = 10


# generate images
out_root = 'prior/'
keywords = ['yessydo', 'eilema', 'ctcf', 'drwde', 'hpatss', 'noitpecni', 'itmfl',
            'dnalalal', 'sirap', 'mdgnk', 'ohcysp', 'noitcif', 'dnalsi', 'txnxt', 
            'tgbh', 'dnegel', 'knhswhs', 'gninihs', 'namurt', 'cntt']

for keyword in keywords:
    set_seed(seed=seed)
    out_dir = os.path.join(out_root, keyword)
    if not os.path.exists(out_dir):
        images = pipe(keyword, num_inference_steps=num_inference_steps, \
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=num_images_per_prompt).images
        os.makedirs(out_dir)
        for i, image in enumerate(images):
            image.save(os.path.join(out_dir, '%03d.png' % i))