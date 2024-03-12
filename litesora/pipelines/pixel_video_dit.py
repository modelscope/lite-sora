from ..models.video_dit import VideoDiT
from ..models.sdxl_text_encoder_2 import SDXLTextEncoder2
from ..schedulers.ddim import DDIMScheduler
from transformers import CLIPTokenizer
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
from einops import rearrange


class PixelVideoDiTPipeline(torch.nn.Module):

    def __init__(self, device="cuda", torch_dtype=torch.float16):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained("configs/stable_diffusion_xl/tokenizer_2")
        self.scheduler = DDIMScheduler()
        self.device = device
        self.torch_dtype = torch_dtype
        self.text_encoder: SDXLTextEncoder2 = None
        self.denoising_model: VideoDiT = None

    
    def fetch_models(self, text_encoder, denoising_model):
        self.text_encoder = text_encoder.to(dtype=self.torch_dtype, device=self.device)
        self.denoising_model = denoising_model.to(dtype=self.torch_dtype, device=self.device)
    

    def decode_image(self, latent, tiled=False, tile_size=64, tile_stride=32):
        image = self.vae_decoder(latent.to(self.device), tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        image = image.cpu().permute(1, 2, 0).numpy()
        image = Image.fromarray(((image / 2 + 0.5).clip(0, 1) * 255).astype("uint8"))
        return image
    

    def decode_video(self, frames):
        frames = rearrange(frames[0], "C T H W -> T H W C")
        frames = ((frames + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        return frames
    

    def tokenize(self, prompt):
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True
        ).input_ids
        return input_ids
    

    def encode_prompt(self, prompt):
        input_ids = self.tokenize(prompt).to(self.device)
        text_emb = self.text_encoder(input_ids)
        return text_emb
    

    @torch.no_grad()
    def __call__(
        self,
        prompt="",
        negative_prompt="",
        cfg_scale=1.0,
        use_cfg=True,
        denoising_strength=1.0,
        num_frames=64,
        height=64,
        width=64,
        num_inference_steps=20,
        progress_bar_cmd=tqdm
    ):
        # Prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength)

        # Prepare latent tensors
        latents = torch.randn((1, 3, num_frames, height, width), device=self.device, dtype=self.torch_dtype)

        # TODO: Encode prompts
        prompt_emb_posi = self.encode_prompt(prompt)
        prompt_emb_nega = self.encode_prompt(negative_prompt)

        # Denoise
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = torch.IntTensor((timestep,)).to(self.device)

            # Classifier-free guidance
            if use_cfg:
                noise_pred_posi = self.denoising_model(latents, timestep, prompt_emb_posi)
                noise_pred_nega = self.denoising_model(latents, timestep, prompt_emb_nega)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = self.denoising_model(latents, timestep, prompt_emb_posi)

            # Call scheduler
            latents = self.scheduler.step(noise_pred, timestep, latents)
        
        # Decode video
        video = self.decode_video(latents)

        return video
