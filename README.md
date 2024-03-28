# Lite-Sora

## Introduction

The lite-sora project is an initiative to replicate Sora, co-launched by East China Normal University and the ModelScope community. It aims to explore the minimal reproduction and streamlined implementation of the video generation algorithms behind Sora. We hope to provide concise and readable code to facilitate collective experimentation and improvement, continuously pushing the boundaries of open-source video generation technology.

## Roadmap

* [x] Implement the base architecture
  * [x] Models
    * [x] Text Encoder（based on Stable Diffusion XL's [Text Encoder](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/text_encoder_2/model.safetensors)）
    * [x] Text Encoder (based on [T5](https://huggingface.co/google/flan-t5-xxl))
    * [x] VideoDiT（based on [Facebook DiT](https://github.com/facebookresearch/DiT)）
    * [x] VideoVAE (based on [Stable Video Diffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt))
    * [x] PixartDiT (based on [Pixart](https://github.com/PixArt-alpha/PixArt-alpha))
    * [x] VideoUNet (based on [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0))
  * [x] Scheduler（based on [DDIM](https://arxiv.org/abs/2010.02502)）
  * [x] Trainer（based on [PyTorch-lightning](https://lightning.ai/docs/pytorch/stable/)）
* [x] Validate on small datasets
  * [x] [Pixabay100](https://github.com/ECNU-CILAB/Pixabay100)
* [ ] Train Video Encoder & Decoder on large datasets
* [ ] Train VideoDiT on large datasets

## Usage

We provide many plans for training a video diffusion model.

### Python Environment

```
conda env create -f environment.yml
conda activate litesora
```

### Plan A: Train from scratch

* Download models
  * `models/text_encoder/model.safetensors`: [download](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/text_encoder_2/model.safetensors)
  * `models/vae/model.safetensors`: [download](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/vae/diffusion_pytorch_model.fp16.safetensors)
* [Train](scripts/videodit_train.py)
* [Test](scripts/videodit_test.py)

### Plan B: Transfer Pixart to a video model

* Download models
  * `models/PixArt-XL-2-512x512` (This is a folder containing many files): [download](https://huggingface.co/PixArt-alpha/PixArt-XL-2-512x512/tree/main)
  * `models/vae/model.safetensors`: [download](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/vae/diffusion_pytorch_model.fp16.safetensors)
* [Train](scripts/pixart_train.py)
* [Test](scripts/pixart_test.py)

### Plan C: Transfer SD-XL to a video model

* Download models
  * `models/dreamshaperXL_v21TurboDPMSDE.safetensors` (a customized SD-XL model): [download](https://civitai.com/api/download/models/351306)
* [Train](scripts/sdxl_train.py)
* [Test](scripts/sdxl_test.py)

### Results (Experimental)

We trained a denoising model using a small dataset [Pixabay100](https://github.com/ECNU-CILAB/Pixabay100). This model serves to demonstrate that our training code is capable of fitting the training data properly, with a resolution of 64*64. **Obviously this model is overfitting due to the limited amount of training data, and thus it lacks generalization capability at this stage. Its purpose is solely for verifying the correctness of the training algorithm.** [download](https://huggingface.co/ECNU-CILab/lite-sora-v1-pixabay100/resolve/main/denoising_model/model.safetensors)

|airport, people, crowd, busy|beach, ocean, waves, water, sand|bee, honey, insect, beehive, nature|coffee, beans, caffeine, coffee, shop|
|-|-|-|-|
|![](assets/airport_people_crowd_busy.gif)|![](assets/beach_ocean_waves_water_sand.gif)|![](assets/bee_honey_insect_beehive_nature.gif)|![](assets/coffee_beans_caffeine_coffee_shop.gif)|
|fish, underwater, aquarium, swim|forest, woods, mystical, morning|ocean, beach, sunset, sea, atmosphere|hair, wind, girl, woman, people|
|![](assets/fish_underwater_aquarium_swim.gif)|![](assets/forest_woods_mystical_morning.gif)|![](assets/ocean_beach_sunset_sea_atmosphere.gif)|![](assets/hair_wind_girl_woman_people.gif)|
|reeds, grass, wind, golden, sunshine|sea, ocean, seagulls, birds, sunset|woman, flowers, plants, field, garden|wood, anemones, wildflower, flower|
|![](assets/reeds_grass_wind_golden_sunshine.gif)|![](assets/sea_ocean_seagulls_birds_sunset.gif)|![](assets/woman_flowers_plants_field_garden.gif)|![](assets/wood_anemones_wildflower_flower.gif)|

We leverage the VAE model from [Stable-Video-Diffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) to encode videos to the latent space. Our code supports extremely long high-resolution videos!

https://github.com/modelscope/lite-sora/assets/35051019/dc205719-d0bc-4bca-b117-ff5aa19ebd86
