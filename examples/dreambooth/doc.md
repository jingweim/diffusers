# movie-dreambooth
## Prior to training
**Choose appropriate token.** Add token to a few prompts and use the initial model (original stable diffusion) to generate images to make sure the token is not already semantically meaningful.
<pre>
<b>## 1. Make an output folder and place inference_tokens.txt and inference_prompts.txt in folder</b>
mkdir ckpts/original_sd
cp input/inference_template/inference_tokens.txt ckpts/original_sd
cp input/inference_template/inference_prompts.txt ckpts/original_sd

<b>## 2. Modify inference_tokens.txt and inference_prompts.txt to desired inputs</b>

<b>## 3. Generate images for all combinations of provided tokens and prompts</b>
python inference_lora_original_sd.py

<b>## Output folder structure (default seed is 0, generates 10 images per seed)</b>
ckpts/
  original_sd/
    inference_prompts.txt
    inference_tokens.txt
    [null]/
      run0/
        000.png
        001.png
        ...
      run1/
      ...
    token1/
    ...
</pre>

For inference txt configuration file formatting, check out [this section](https://github.com/jingweim/diffusers/edit/main/examples/dreambooth/doc.md#inference-txt-configuration-files-format).

## Training
<pre>
<b>## single-movie dreambooth example command: </b>
python train_dreambooth_lora.py --config configs/in-the-mood-for-love_run0.txt

<b>## multi-movie dreambooth example command: </b>
python train_dreambooth_lora.py --config configs/movies20_run0.txt
</pre>

## Evaluation
Prior to evaluating a model, put [inference txt configuration files](https://github.com/jingweim/diffusers/edit/main/examples/dreambooth/doc.md#inference-txt-configuration-files-format) (i.e. `inference_prompts.txt`, `inference_tokens.txt`, `inference_ckpts.txt`, `inference_seeds.txt`) into the model folder (e.g. `ckpts/movies20_run0`) and modify them to specify desired inputs.

For single-movie and multi-movie dreambooth models, we want to evaluate to find out:
<pre>
<b>1) What does the model generate for a given prompt with/without token?</b>
python inference_lora.py --config configs/movies20_run0.txt

<b>## Output folder structure (by default generates 10 images per seed)</b>
ckpts/
  movies20_run0/
    inference/
      checkpoint-ep-70-gs-361550/
        [null]/
          run0/
            seed_00/
              000.png
              001.png
              ...
            seed_01/
            ...
          run1/
          ...
        token1/
        ...
      checkpoint-ep-60-gs-309900/
      ...
      
<b>Generating each seed folder of images takes around 26 seconds.</b>
</pre>
<pre>
<b>2) What does the model generate when we tune token strength (guidance scale)?</b>
python inference_lora_cfg.py --config configs/movies20_run0.txt

<b>## Output folder structure (generates 1 video per seed, doesn't accept null token)</b>
ckpts/
  movies20_run0/
    inference_cfg/
      checkpoint-ep-70-gs-361550/
        token1/
          run0/
            seed_000_0-10_fps-5.mp4
            seed_000_0-10_fps-5.txt
            seed_001_0-10_fps-5.mp4
            seed_001_0-10_fps-5.txt
            ...
            ...
          run1/
          ...
        token2/
        ...
      checkpoint-ep-60-gs-309900/
      ...

<b>Generating each video takes around 4 minutes.</b>
</pre>

Additionally, for multi-movie dreambooth models, we want to find out:
<pre>
<b>3) What does the model generate when we interpolate between different tokens?</b>
python inference_lora_cfg_mix.py --config configs/movies20_run0.txt

<b>## Output folder structure (generates 1 video per seed)</b>
ckpts/
  movies20_run0/
    inference_cfg_mix/
      checkpoint-ep-70-gs-361550/
        token1-token2/
          run0/
            seed_000_0-10_fps-5.mp4
            seed_000_0-10_fps-5.txt
            seed_001_0-10_fps-5.mp4
            seed_001_0-10_fps-5.txt
            ...
            ...
          run1/
          ...
        token3-token4/
        ...
      checkpoint-ep-60-gs-309900/
      ...

<b>Generating each video takes around 4 minutes.</b>
</pre>

## Inference txt configuration files
There are four txt files: `inference_prompts.txt`, `inference_tokens.txt`, `inference_ckpts.txt`, `inference_seeds.txt`. Before running inference, put them in the model folder (e.g. `ckpts/movies20_run0`) and modify them to specify desired inputs. Given the model config file (e.g. `configs/movies20_run0.txt`), the inference scripts automatically loads these txt files and generates results for all combinations of prompts, tokens, ckpts, and seeds.  

The scripts automatically skips if a combination has already been generated. We can also use a dashed line (i.e. ----------------------------) to ignore tokens/prompts/ckpts/seeds after the line.

Below are some examples of the four txt files (pound`#` signs are required):
<pre>
<b>## inference_ckpts.txt format/example (ep-5 ignored due to dashed line)</b>
Row1 -> # checkpoint-ep-20-gs-103300
Row2 -> # checkpoint-ep-10-gs-51650
Row3 -> --------------------------------
Row4 -> # checkpoint-ep-5-gs-25825

<b>## inference_tokens.txt format/example (row1 is null token)</b>
Row1 -> # 
Row2 -> # tgbh

<b>## inference_seeds.txt format/example (seed 1,2,3,4 ignored due to dashed line)</b>
Row1 -> # 0
Row2 -> --------------------------------
Row3 -> # 1
Row4 -> # 2
Row5 -> # 3
Row6 -> # 4

<b>## inference_prompts.txt format (1st line = output folder, 2nd line = comments, 3rd line = prompt)</b>
Row01 -> ## run0
Row02 -> # blank prompt, just to see what's learned for token
Row03 -> 
Row04 -> 
Row05 -> ## run1
Row06 -> # simple prompt, unseen object
Row07 -> dog
Row08 -> 
Row09 -> ## run2
Row10 -> # simple prompt, maybe-seen object
Row11 -> bedroom
</pre>

Current set of prompts in the template `input/inference_template/inference_prompts.txt`:
```
1) Token only (run0)
2) Short prompt: dog, bedroom, man, woman (run1-4)
3) Long prompt: one from each movie (run5-n)
```
