from ..models.video_dit import VideoDiT
from ..models.sd_text_encoder import SDTextEncoder
from ..models.svd_vae import SVDVAEDecoder
from ..schedulers.ddim import DDIMScheduler
from ..data.utils import tensor2video
from transformers import CLIPTokenizer
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
from einops import rearrange


class LatentVideoDiTPipeline(torch.nn.Module):

    def __init__(self, device="cuda", torch_dtype=torch.float16):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained("configs/stable_diffusion/tokenizer")
        self.scheduler = DDIMScheduler()
        self.device = device
        self.torch_dtype = torch_dtype
        self.text_encoder: SDTextEncoder = None
        self.denoising_model: VideoDiT = None
        self.video_decoder: SVDVAEDecoder = None

    
    def fetch_models(self, text_encoder, denoising_model, video_decoder):
        self.text_encoder = text_encoder.to(dtype=self.torch_dtype, device=self.device)
        self.denoising_model = denoising_model.to(dtype=self.torch_dtype, device=self.device)
        self.video_decoder = video_decoder.to(dtype=self.torch_dtype, device=self.device)
    

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
        cross_emb, text_emb = self.text_encoder(input_ids, pooled_output_id=self.tokenizer.eos_token_id)
        return cross_emb, text_emb
    

    @torch.no_grad()
    def __call__(
        self,
        prompt="",
        negative_prompt="",
        cfg_scale=1.0,
        use_cfg=True,
        denoising_strength=1.0,
        num_frames=128,
        height=512,
        width=512,
        start_sec=0.0,
        end_sec=1.0,
        num_inference_steps=20,
        progress_bar_cmd=tqdm
    ):
        # Prepare parameters
        start_sec = torch.Tensor((start_sec,)).to(dtype=self.torch_dtype, device=self.device)
        end_sec = torch.Tensor((end_sec,)).to(dtype=self.torch_dtype, device=self.device)

        # Prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength)

        # Prepare latent tensors
        latents = torch.randn((1, 4, num_frames, height//8, width//8), device=self.device, dtype=self.torch_dtype)

        # Encode prompts
        cross_emb_posi, text_emb_posi = self.encode_prompt(prompt)
        cross_emb_nega, text_emb_nega = self.encode_prompt(negative_prompt)

        # Denoise
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = torch.IntTensor((timestep,)).to(self.device)

            # Classifier-free guidance
            if use_cfg:
                noise_pred_posi = self.denoising_model(latents, timestep, start_sec, end_sec, cross_emb_posi, text_emb_posi)
                noise_pred_nega = self.denoising_model(latents, timestep, start_sec, end_sec, cross_emb_nega, text_emb_nega)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = self.denoising_model(latents, timestep, start_sec, end_sec, cross_emb_posi, text_emb_posi)

            # Call scheduler
            latents = self.scheduler.step(noise_pred, timestep, latents)
        
        # Decode video
        latents = latents.cpu()
        video = self.video_decoder.decode_video(latents[0], progress_bar=progress_bar_cmd)
        video = tensor2video(video)

        return video
