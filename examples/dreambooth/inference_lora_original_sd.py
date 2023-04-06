'''
    python inference_lora_original_sd.py
'''

import os
import torch

from accelerate.utils import set_seed
from diffusers import DPMSolverMultistepScheduler, DiffusionPipeline


# arguments
weight_dtype = torch.float16
seed = 0
num_inference_steps = 50
guidance_scale = 7.5
num_images_per_prompt = 10


# Load model
pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=weight_dtype)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")


# generate images
out_root = 'ckpts/original_sd/'
tokens = open(os.path.join(out_root, 'inference_tokens.txt'), "r").readlines()
prompts = open('ckpts/inference_prompts.txt', "r").readlines()

for token in tokens:
    if token.startswith('# '):
        token = token[2:].strip()
        token_dir = "[null]" if token == "" else token
        for i, line in enumerate(prompts):
            if line.startswith('## '):
                run = line[3:].strip()
                prompt = prompts[i+2].strip()
                out_dir = os.path.join(out_root, token_dir, run)
                if os.path.exists(out_dir):
                    print(f"Skipping {out_dir}")
                else:
                    print(f"Generating {out_dir}")
                    set_seed(seed=seed)
                    os.makedirs(out_dir)
                    final_prompt = prompt if token=="" else token+", "+prompt
                    images = pipe(final_prompt, num_inference_steps=num_inference_steps, \
                                guidance_scale=guidance_scale, num_images_per_prompt=num_images_per_prompt).images
                    for i, image in enumerate(images):
                        image.save(os.path.join(out_dir, '%03d.png' % i))

                    # save arguments to text file
                    with open(os.path.join(out_dir, 'args.txt'), 'w') as f:
                        f.write(f'final prompt = {final_prompt}\n')
                        f.write(f"weight_dtype = {'float16' if weight_dtype == torch.float16 else 'float32'}\n")
                        f.write(f'num_inference_steps = {num_inference_steps}\n')
                        f.write(f'guidance_scale = {guidance_scale}\n')
                        f.write(f'seed = {seed}\n')

            