from ..models.sdxl_text_encoder import SDXLTextEncoder, SDXLTextEncoder2
from ..models.sdxl_3d_unet import SDXL3DUNet
from ..models.svd_vae import SDVAEEncoder, SVDVAEDecoder
from ..data.utils import tensor2video
from ..schedulers import DDIMScheduler
from transformers import CLIPTokenizer
import torch
from tqdm import tqdm



class SDXLVideoPipeline(torch.nn.Module):

    def __init__(self, device="cuda", torch_dtype=torch.float16):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained("configs/stable_diffusion/tokenizer")
        self.tokenizer_2 = CLIPTokenizer.from_pretrained("configs/stable_diffusion_xl/tokenizer_2")
        self.scheduler = DDIMScheduler()
        self.device = device
        self.torch_dtype = torch_dtype
        # models
        self.text_encoder: SDXLTextEncoder = None
        self.text_encoder_2: SDXLTextEncoder2 = None
        self.unet: SDXL3DUNet = None
        self.vae_decoder: SVDVAEDecoder = None
        self.vae_encoder: SDVAEEncoder = None


    def encode_prompt(
        self,
        text_encoder: SDXLTextEncoder,
        text_encoder_2: SDXLTextEncoder2,
        prompt,
        clip_skip=1,
        clip_skip_2=2,
        device="cuda"
    ):
        # 1
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True
        ).input_ids.to(device)
        prompt_emb_1 = text_encoder(input_ids, clip_skip=clip_skip)

        # 2
        input_ids_2 = self.tokenizer_2(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True
        ).input_ids.to(device)
        add_text_embeds, prompt_emb_2 = text_encoder_2(input_ids_2, clip_skip=clip_skip_2)

        # Merge
        prompt_emb = torch.concatenate([prompt_emb_1, prompt_emb_2], dim=-1)
        return add_text_embeds, prompt_emb
    

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        cfg_scale=7.5,
        clip_skip=1,
        clip_skip_2=2,
        denoising_strength=1.0,
        num_frames=128,
        height=1024,
        width=1024,
        num_inference_steps=20,
        progress_bar_cmd=tqdm,
    ):
        # Prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength)

        # Prepare latent tensors
        latents = torch.randn((1, 4, num_frames, height//8, width//8), device=self.device, dtype=self.torch_dtype)

        # Encode prompts
        add_prompt_emb_posi, prompt_emb_posi = self.encode_prompt(
            self.text_encoder,
            self.text_encoder_2,
            prompt,
            clip_skip=clip_skip, clip_skip_2=clip_skip_2,
            device=self.device,
        )
        if cfg_scale != 1.0:
            add_prompt_emb_nega, prompt_emb_nega = self.encode_prompt(
                self.text_encoder,
                self.text_encoder_2,
                negative_prompt,
                clip_skip=clip_skip, clip_skip_2=clip_skip_2,
                device=self.device,
            )
        
        # Prepare positional id
        add_time_id = torch.tensor([height, width, 0, 0, height, width], device=self.device)
        
        # Denoise
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = torch.IntTensor((timestep,))[0].to(self.device)

            # Classifier-free guidance
            if cfg_scale != 1.0:
                noise_pred_posi = self.unet(
                    latents, timestep, prompt_emb_posi,
                    add_time_id=add_time_id, add_text_embeds=add_prompt_emb_posi,
                )
                noise_pred_nega = self.unet(
                    latents, timestep, prompt_emb_nega,
                    add_time_id=add_time_id, add_text_embeds=add_prompt_emb_nega,
                )
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = self.unet(
                    latents, timestep, prompt_emb_posi,
                    add_time_id=add_time_id, add_text_embeds=add_prompt_emb_posi,
                )

            latents = self.scheduler.step(noise_pred, timestep, latents)
        
        # Decode image
        latents = latents.cpu()
        video = self.vae_decoder.decode_video(latents[0], progress_bar=progress_bar_cmd)
        video = tensor2video(video)

        return video
