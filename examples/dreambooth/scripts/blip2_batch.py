import os
import glob
import torch
from PIL import Image
from tqdm import tqdm

from lavis.models import load_model_and_preprocess


os.environ['TORCH_HOME'] = '/gscratch/realitylab/jingweim/.cache'
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


# loads BLIP-2 pre-trained model
model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device)
# model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt6.7b", is_eval=True, device=device)


# load input images
root = 'data'
folders = sorted(os.listdir(root))
for folder in folders:
  out_dir = os.path.join(root, folder, 'prompts')
  out_path = os.path.join(out_dir, 't5xl_l2_r1-5.txt')
  if not os.path.exists(out_path):
    os.makedirs(out_dir)
    img_dir = os.path.join(root, folder, 'images')
    img_paths = sorted(glob.glob(img_dir+'/*.png'))

    # write caption to txt file
    out = open(out_path, "a+")
    for img_path in tqdm(img_paths):
        raw_image = Image.open(img_path).convert("RGB")
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        caption = model.generate({"image": image}, \
                                  length_penalty=2.0, repetition_penalty=1.5)[0]
        out.write(f'# {img_path}\n')
        out.write(f'{caption}\n')

    out.close()