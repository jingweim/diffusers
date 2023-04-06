'''
    python inference_lora.py --config configs/tgbh-trailer_run0.txt
'''

import os
import glob
import torch

from accelerate.utils import set_seed
from train_dreambooth_lora import config_parser
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler


# helper functions

def generate_step(pipe, prompt, out_dir, seed, num_images_per_prompt, num_inference_steps,
                  guidance_scale):

    # set seed
    set_seed(seed)

    # generate images
    images = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, \
                num_images_per_prompt=num_images_per_prompt).images
    for i, image in enumerate(images):
        image.save(os.path.join(out_dir, '%03d.png' % i))


def generate_per_model(args, checkpoint):

    # load shared arguments
    weight_dtype = torch.float16
    num_images_per_prompt = 10
    num_inference_steps = 50
    guidance_scale = 7.5

    # load model
    model_id = args.output_dir + '/' + checkpoint
    pipe = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, \
                                            torch_dtype=weight_dtype)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pipe.unet.load_attn_procs(model_id)

    # run model on each token+prompt+seed
    out_root = args.output_dir
    token_lines = open(os.path.join(out_root, 'inference_tokens.txt'), "r").readlines()
    prompt_lines = open(os.path.join(out_root, 'inference_prompts.txt'), "r").readlines()
    seed_lines = open(os.path.join(out_root, 'inference_seeds.txt'), "r").readlines()

    for token_line in token_lines:
        if token_line.startswith('-----'):
            break
        if token_line.startswith('# '):
            token = token_line[2:].strip()
            token_dir = "[null]" if token == "" else token
            for i, prompt_line in enumerate(prompt_lines):
                if prompt_line.startswith('-----'):
                    break
                if prompt_line.startswith('## '):
                    run = prompt_line[3:].strip()
                    prompt = prompt_lines[i+2].strip()
                    for seed_line in seed_lines:
                        if seed_line.startswith('-----'):
                            break
                        if seed_line.startswith('# '):
                            seed = int(seed_line[2:].strip())
                            out_dir = os.path.join(out_root, 'inference', checkpoint, token_dir, run, 'seed_%02d' % seed)
                            if os.path.exists(out_dir):
                                print(f"Skipping {out_dir}")
                            else:
                                print(f"Generating {out_dir}")
                                os.makedirs(out_dir)
                                final_prompt = prompt if token=="" else token+", "+prompt
                                generate_step(pipe, final_prompt, out_dir, seed, num_images_per_prompt, num_inference_steps, guidance_scale)
                                
                                # save arguments to text file
                                with open(os.path.join(out_dir, 'args.txt'), 'w') as f:
                                    f.write(f'final prompt = {final_prompt}\n')
                                    f.write(f"weight_dtype = {'float16' if weight_dtype == torch.float16 else 'float32'}\n")
                                    f.write(f'num_inference_steps = {num_inference_steps}\n')
                                    f.write(f'guidance_scale = {guidance_scale}\n')

    # free gpu memory
    del pipe
    torch.cuda.empty_cache()


# load args from config txt
args = config_parser()


# execute prompts on different models
with open(os.path.join(args.output_dir, 'inference_ckpts.txt'), 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if line.startswith('#'):
            model_id = line[2:].strip()
            generate_per_model(args, model_id)