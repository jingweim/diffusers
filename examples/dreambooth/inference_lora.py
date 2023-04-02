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

def generate_step(pipe, prompt, output_dir, seed, num_images_per_prompt, num_inference_steps):

    # set seed
    set_seed(seed)

    # generate images
    images = pipe(prompt, num_inference_steps=num_inference_steps, \
                num_images_per_prompt=num_images_per_prompt).images
    for i, image in enumerate(images):
        image.save(os.path.join(output_dir, '%03d.png' % i))


def generate_per_model(args, checkpoint):

    # load shared arguments
    weight_dtype = torch.float16
    seeds = [0, 1, 2, 3, 4]
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

    # run model on each prompt
    with open(os.path.join(args.output_dir, 'inference_prompts.txt'), 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith('##'):
                run = line[2:].strip()
                prompt = lines[i+2].strip()
                for seed in seeds:
                    output_dir = os.path.join(args.output_dir, 'inference', checkpoint, run, 'seed_%02d' % seed)
                    # skip if already generated
                    if os.path.exists(output_dir):
                        print(checkpoint, run, 'seed_%02d' % seed, 'skipped')
                    else:
                        os.makedirs(output_dir)
                        generate_step(pipe, prompt, output_dir, seed, num_images_per_prompt, num_inference_steps)

                        # save arguments to text file
                        with open(os.path.join(output_dir, 'args.txt'), 'w') as f:
                            f.write(f'prompt = {prompt}\n')
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