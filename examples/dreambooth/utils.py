import os
import imageio.v3 as iio
import numpy as np

# Generate color strip from dataset
def generate_color_strip(folder, width, height):
    fnames = sorted(os.listdir(folder))
    w = width // len(fnames)
    out = np.zeros((height, w*len(fnames), 3), dtype='uint8')
    for i, fname in enumerate(fnames):
        img = iio.imread(os.path.join(folder, fname))[..., :3]
        H, _, _ = img.shape
        out[:, i*w:(i+1)*w] = img[H//2:H//2+height, :w]
    
    out_path = folder.replace('/images', '.png')
    iio.imwrite(out_path, out)

generate_color_strip('data/tgbh/images', 512, 50)