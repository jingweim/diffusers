# movie-dreambooth
## Training
```
python train_dreambooth_lora.py --config configs/movies20_run0.txt
```

## Evaluation
### Evaluation procedure
```
## Generate images to find out
1) For prompt with token, how has the model changed
2) For prompt without token, how has the model changed

## inference_ckpts.txt format/example (experiment specific)
Row1 -> # checkpoint-ep-5-gs-25825
Row2 -> # checkpoint-ep-10-gs-51650

## inference_tokens.txt format/example (experiment specific)
Row1 -> # 
Row2 -> # tgbh

## inference_prompts.txt format (shared)
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

## Set of prompts
1) Token only (run0)
2) Short prompt: dog, bedroom, man, woman (run1-4)
3) Long prompt: one from each movie (run5-n)
```

### Step1: Generating prior (original SD)
```
## Prepare inference_prompts.txt and inference_ckpts.txt 
python inference_lora_original_sd.py

## Output folder structure (seed=0)
ckpts/
  original_sd/
    inference_tokens.txt
    [null]/
      run0/
        000.png
        001.png
        ...
      ...
    token1/
    ...
```

### Step2: Inference
```
## Prepare inference_prompts.txt, inference_ckpts.txt and inference_tokens.txt, then run
python inference_lora.py --config configs/movies20_run0.txt

## Output folder structure
ckpts/
  movies20_run0
    inference/
      checkpoint-ep-5-gs-25825/
        [null]/
          run0/
            seed_00/
              000.png
              001.png
              ...
            ...
          ...
        ...
      ...
```

### Step3: Inference (cfg on token)
```
## Prepare inference_prompts.txt, inference_ckpts.txt and inference_tokens.txt, then run
python inference_lora_cfg.py --config configs/movies20_run0.txt

## Output folder structure
ckpts/
  movies20_run0
    inference_cfg/
      checkpoint-ep-5-gs-25825/
        [null]/
          run0/
            seed_00/
              gs_7-5_gs2_0/
                000.png
                001.png
                ...
              ...
            ...
          ...
        ...
      ...
```
