import os
import torch

from accelerate.utils import set_seed
from train_dreambooth_lora import config_parser
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler


# helper functions
def generate(args, run, checkpoint, prompt_add):

    # set seed
    seed = args.seed
    set_seed(seed)

    # load arguments
    model_id = args.output_dir
    prompt = args.instance_prompt

    # fixed arguments
    weight_dtype = torch.float16
    num_images_per_prompt = 10
    num_inference_steps = 50
    guidance_scale = 7.5

    # skip if already generated
    output_dir = os.path.join(args.output_dir, f'inference/{checkpoint}/{run}')
    if os.path.exists(output_dir):
        print(checkpoint, run, 'skipped')
        return

    # make output folder
    os.makedirs(output_dir)

    # load run-specific model and prompt
    checkpoint = "" if checkpoint == 'checkpoint-last' else checkpoint
    model_id += f'/{checkpoint}'
    prompt += f', {prompt_add}'

    # save arguments to text file
    with open(os.path.join(output_dir, 'args.txt'), 'w') as f:
        f.write(f'prompt = {prompt}\n')
        f.write(f'model_id = {model_id}\n')
        f.write(f'seed = {seed}\n')
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


# load args from config txt
args = config_parser()

# execute different runs
with open(os.path.join(args.output_dir, 'inference.txt'), 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if line.startswith('#'):
        run = line[2:].strip()
        checkpoint = lines[i+1].strip()
        prompt_add = lines[i+2].strip()
        generate(args, run, checkpoint, prompt_add)