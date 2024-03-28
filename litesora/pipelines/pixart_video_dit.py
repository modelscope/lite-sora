from ..models.pixart_video_dit import PixartVideoDiT
from ..models.svd_vae import SVDVAEDecoder
from ..schedulers.ddim import DDIMScheduler
from ..data.utils import tensor2video
from transformers import T5Tokenizer, T5EncoderModel
import torch
from tqdm import tqdm
import numpy as np
from einops import rearrange


class PixartVideoDiTPipeline(torch.nn.Module):

    def __init__(self, device="cuda", torch_dtype=torch.float16):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained("configs/pixart/tokenizer")
        self.scheduler = DDIMScheduler(beta_start=0.0001, beta_end=0.02, beta_schedule="linear")
        self.device = device
        self.torch_dtype = torch_dtype
        self.text_encoder: T5EncoderModel = None
        self.denoising_model: PixartVideoDiT = None
        self.video_decoder: SVDVAEDecoder = None

    
    def fetch_models(self, text_encoder=None, denoising_model=None, video_decoder=None):
        if text_encoder is not None:
            self.text_encoder = text_encoder.to(dtype=self.torch_dtype, device=self.device)
        if denoising_model is not None:
            self.denoising_model = denoising_model.to(dtype=self.torch_dtype, device=self.device)
        if video_decoder is not None:
            self.video_decoder = video_decoder.to(dtype=self.torch_dtype, device=self.device)
    

    def decode_video(self, frames):
        frames = rearrange(frames[0], "C T H W -> T H W C")
        frames = ((frames + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        return frames
    

    def tokenize(self, prompt):
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=120,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids, text_mask = text_inputs.input_ids, text_inputs.attention_mask
        return input_ids, text_mask
    

    def encode_prompt(self, prompt):
        input_ids, text_mask = self.tokenize(prompt)
        input_ids = input_ids.to(self.device)
        text_mask = text_mask.to(self.device)
        text_emb = self.text_encoder(input_ids, attention_mask=text_mask).last_hidden_state
        return text_emb, text_mask
    

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
        num_inference_steps=20,
        progress_bar_cmd=tqdm
    ):
        # Prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength)

        # Prepare latent tensors
        latents = torch.randn((1, 4, num_frames, height//8, width//8), device=self.device, dtype=self.torch_dtype)

        # Encode prompts
        text_emb_posi, text_mask_posi = self.encode_prompt(prompt)
        text_emb_nega, text_mask_nega = self.encode_prompt(negative_prompt)

        # Denoise
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = torch.IntTensor((timestep,)).to(self.device)

            # Classifier-free guidance
            if use_cfg:
                noise_pred_posi = self.denoising_model(latents, timestep, text_emb_posi, text_mask_posi)
                noise_pred_nega = self.denoising_model(latents, timestep, text_emb_nega, text_mask_nega)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = self.denoising_model(latents, timestep, text_emb_posi, text_mask_posi)

            # Call scheduler
            latents = self.scheduler.step(noise_pred, timestep, latents)
        
        # Decode video
        latents = latents.cpu()
        video = self.video_decoder.decode_video(latents[0], progress_bar=progress_bar_cmd)
        video = tensor2video(video)

        return video
