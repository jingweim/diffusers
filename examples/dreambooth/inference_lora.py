import os
import torch

from train_dreambooth_lora import config_parser
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

# load args from config
args = config_parser()
model_id = args.output_dir
prompt = args.instance_prompt

# fixed arguments
weight_dtype = torch.float16
num_images_per_prompt = 10
num_inference_steps = 50
guidance_scale = 7.5

# arguments vary per run
run = 'run0'
checkpoint = 'checkpoint-1020'
model_id += f'/{checkpoint}'
prompt += ""

# make output folder
output_dir = os.path.join(args.output_dir, f'inference/{run}')
os.makedirs(output_dir, exist_ok=True)

# save arguments to text file
with open(os.path.join(output_dir, 'args.txt'), 'w') as f:
    f.write(f'prompt = {prompt}\n')
    f.write(f'model_id = {model_id}\n')
    f.write(f"weight_dtype = {'float16' if weight_dtype == torch.float16 else 'float32'}\n")
    f.write(f'num_images_per_prompt = {num_images_per_prompt}\n')
    f.write(f'num_inference_steps = {num_inference_steps}\n')
    f.write(f'guidance_scale = {guidance_scale}\n')

# load model
pipe = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, \
                                         torch_dtype=weight_dtype)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")
pipe.unet.load_attn_procs(model_id)

# generate images
images = pipe(prompt, num_inference_steps=num_inference_steps, \
              num_images_per_prompt=num_images_per_prompt).images
for i, image in enumerate(images):
    image.save(os.path.join(output_dir, '%03d.png' % i))