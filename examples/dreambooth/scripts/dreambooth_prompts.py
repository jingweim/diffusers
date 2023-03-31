import os
import glob
import torch
from PIL import Image
from tqdm import tqdm


# load input images
root = 'data'
folders = sorted(os.listdir(root))
tokens = {'2001-a-space-odyssey': 'yessydo', 'amelie': 'eilema', 
          'charlie-and-the-chocolate-factory': 'ctcf', 
          'edward-scissorhands': 'drwde', 
          'harry-potter-and-the-sorcerers-stone': 'hpatss', 
          'inception': 'noitpecni', 'in-the-mood-for-love': 'itmfl', 
          'la-la-land': 'dnalalal', 'midnight-in-paris': 'sirap', 
          'moonrise-kingdom': 'mdgnk', 'psycho': 'ohcysp', 
          'pulp-fiction': 'noitcif', 'shutter-island': 'dnalsi', 
          'tenet': 'txnxt', 'the-grand-budapest-hotel': 'tgbh', 
          'the-legend-of-1900': 'dnegel', 
          'the-shawshank-redemption': 'knhswhs', 
          'the-shining': 'gninihs', 'the-truman-show': 'namurt', 
          'titanic': 'cntt'}


# # add dreambooth tokens to BLIP2 captions
# for folder in folders:
#   out_dir = os.path.join(root, folder, 'prompts')
#   prompt_path = os.path.join(out_dir, 't5xl_l2_r1-5.txt')
#   out_path = os.path.join(out_dir, 'dreambooth_t5xl_l2_r1-5.txt')
#   if not os.path.exists(out_path):
#     lines = open(prompt_path).readlines()

#     # write token to prompts
#     out = open(out_path, "a+")
#     token = tokens[folder]
#     for i, line in enumerate(lines):
#       if line.startswith('# '):
#         out.write(line)
#         out.write(token + ', ' + lines[i+1])

#     out.close()


# merge prompt txts from multiple movies
out_path = os.path.join(root, 'movies20.txt')
if not os.path.exists(out_path):
  out = open(out_path, "a+")
  for folder in folders:
    prompt_path = os.path.join(root, folder, 'prompts', 'dreambooth_t5xl_l2_r1-5.txt')
    lines = open(prompt_path).readlines()
    out.writelines(lines)
  out.close()