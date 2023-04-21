'''
    python inference_lora_cfg.py --config configs/tgbh-trailer_run0.txt
'''

import os, sys
import cv2
import torch
import numpy as np

from accelerate.utils import set_seed
from train_dreambooth_lora import config_parser
from diffusers import DPMSolverMultistepScheduler

sys.path.append('/gscratch/realitylab/jingweim/diffusers')
from src.diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_cfg import StableDiffusionPipeline

# global parameters
TOKEN2MOVIE = {
                "cntt": "Titanic", 
                "ctcf": "Charlie and the Chocolate Factory", 
                "dnalalal": "La La Land", 
                "dnalsi": "Shutter Island", 
                "dnegel": "The Legend of 1900", 
                "drwde": "Edward Scissorhands", 
                "eilema": "Amelie", 
                "gninihs": "The Shining", 
                "hpatss": "Harry Potter and the Sorcerer's Stone", 
                "itmfl": "In the Mood for Love",
                "knhswhs": "The Shawshank Redemption", 
                "mdgnk": "Moonrise Kingdom",
                "namurt": "The Truman Show", 
                "noitcif": "Pulp Fiction", 
                "noitpecni": "Inception", 
                "ohcysp": "Psycho", 
                "sirap": "Midnight in Paris", 
                "tgbh": "The Grand Budapest Hotel", 
                "txnxt": "Tenet", 
                "yessydo": "2001: A Space Odyssey", 
              }

# helper functions

def generate_step(pipe, prompt, token, seed, num_images_per_prompt, num_inference_steps,
                  guidance_scale, guidance_scale_2):

    # set seed
    set_seed(seed)

    # generate images
    images = pipe(prompt, token, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, \
                  guidance_scale_2=guidance_scale_2, num_images_per_prompt=num_images_per_prompt).images
    return images


def generate_per_model(args, checkpoint, out_root, token_lines, prompt_lines, seed_lines):

    # load shared arguments
    weight_dtype = torch.float16
    num_images_per_prompt = 1 # always one
    num_inference_steps = 50
    guidance_scale = 7.5
    guidance_scale_2_min = 0.0
    guidance_scale_2_max = 7.5
    guidance_scale_2_step = 0.10
    guidance_scale_2s = np.arange(guidance_scale_2_min, 
                                  guidance_scale_2_max+guidance_scale_2_step, 
                                  guidance_scale_2_step)

    # load model
    model_id = args.output_dir + '/' + checkpoint
    pipe = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, \
                                            torch_dtype=weight_dtype)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pipe.unet.load_attn_procs(model_id)

    # video settings
    font = cv2.FONT_HERSHEY_DUPLEX
    org = (25, 40)
    org2 = (25, 480)
    fontScale = 0.75
    color = (0, 255, 0)
    thickness = 1
    fps = 5

    for token_line in token_lines:
        if token_line.startswith('-----'):
            break
        if token_line.startswith('# '):
            token = token_line[2:].strip()
            if token == "":
                continue
            token_dir = token
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
                            gs_string = ('%.2f' % guidance_scale_2_step).replace('.', '-')
                            out_path = os.path.join(out_root, 'inference_cfg', checkpoint, token_dir,  run, f"seed_%03d_{gs_string}_fps-{fps}.mp4" % seed)
                            if os.path.exists(out_path):
                                print(f"Skipping {out_path}")
                            else:
                                print(f"Generating {out_path}")
                                cache = {}
                                for guidance_scale_2 in guidance_scale_2s:
                                    images = generate_step(pipe, prompt, token, seed, num_images_per_prompt, num_inference_steps, 
                                                guidance_scale, guidance_scale_2)
                                    cache[guidance_scale_2] = images
                                
                                # write into videos
                                tmp_dir = out_path.replace('.mp4', '')
                                os.makedirs(tmp_dir)
                                for gs_idx, guidance_scale_2 in enumerate(guidance_scale_2s):                                        
                                    img = np.array(cache[guidance_scale_2][0], dtype='uint8')
                                    img = cv2.putText(img, 'gs = %.2f' % guidance_scale_2, org, font, fontScale, color, thickness, cv2.LINE_AA)
                                    img = cv2.putText(img, TOKEN2MOVIE[token], org2, font, fontScale, color, thickness, cv2.LINE_AA)
                                    _ = cv2.imwrite(os.path.join(tmp_dir, '%03d.png' % gs_idx), img[..., ::-1])

                                os.system(f'ffmpeg -r {fps} -i {tmp_dir}/%03d.png -vcodec libx264 -crf 25 {out_path}')
                                _ = os.system(f'rm -rf {tmp_dir}')

                                # save arguments to text file
                                final_prompt = token+", "+prompt
                                with open(out_path.replace('.mp4', '.txt'), 'w') as f:
                                    f.write(f'final prompt = {final_prompt}\n')
                                    f.write(f"weight_dtype = {'float16' if weight_dtype == torch.float16 else 'float32'}\n")
                                    f.write(f'num_inference_steps = {num_inference_steps}\n')
                                    f.write(f'guidance_scale = {guidance_scale}\n')
                                    f.write(f'guidance_scale_2_min = {guidance_scale_2_min}\n')
                                    f.write(f'guidance_scale_2_max = {guidance_scale_2_max}\n')
                                    f.write(f'guidance_scale_2_step = {guidance_scale_2_step}\n')
                                    f.write(f'fps = {fps}\n')

    # free gpu memory
    del pipe
    torch.cuda.empty_cache()


# load args from config txt
args = config_parser()

# run model on each token+prompt+seed
out_root = args.output_dir
token_lines = open(os.path.join(out_root, 'inference_tokens.txt'), "r").readlines()
prompt_lines = open(os.path.join(out_root, 'inference_prompts.txt'), "r").readlines()
seed_lines = open(os.path.join(out_root, 'inference_seeds.txt'), "r").readlines()

# execute prompts on different models
with open(os.path.join(args.output_dir, 'inference_ckpts.txt'), 'r') as f:
    print('txt files loaded-----------------')
    lines = f.readlines()
    for i, line in enumerate(lines):
        if line.startswith('-----'):
            break
        if line.startswith('#'):
            model_id = line[2:].strip()
            generate_per_model(args, model_id, out_root, token_lines, prompt_lines, seed_lines)