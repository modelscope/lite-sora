from litesora.models.utils import load_state_dict
from litesora.models.sd_vae import SDVAEEncoder
from litesora.pipelines.sdxl_3d import SDXLTextEncoder, SDXLTextEncoder2, DDIMScheduler, SDXL3DUNet
from litesora.data import TextVideoDataset
from transformers import CLIPTokenizer
import torch, imageio
import lightning as pl
import numpy as np



class LightningVideoDiT(pl.LightningModule):
    def __init__(self, learning_rate=1e-5, compile=False):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained("configs/stable_diffusion/tokenizer")
        self.tokenizer_2 = CLIPTokenizer.from_pretrained("configs/stable_diffusion_xl/tokenizer_2")
        self.noise_scheduler = DDIMScheduler()
        # models
        state_dict = load_state_dict("models/dreamshaperXL_v21TurboDPMSDE.safetensors")
        self.text_encoder = SDXLTextEncoder.from_civitai(state_dict=state_dict)
        self.text_encoder_2 = SDXLTextEncoder2.from_civitai(state_dict=state_dict)
        self.denoising_model = SDXL3DUNet.from_civitai(state_dict=state_dict)
        self.video_encoder = SDVAEEncoder.from_civitai(state_dict=state_dict)

        self.learning_rate = learning_rate
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.denoising_model.train()
        self.video_encoder.requires_grad_(False)
        if compile:
            self.compile()


    def compile(self):
        self.text_encoder = torch.compile(self.text_encoder)
        self.video_encoder = torch.compile(self.video_encoder)
        self.denoising_model = torch.compile(self.denoising_model)


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
    

    def process_text(self, text_batch):
        add_text_embeds_batch, prompt_emb_batch = [], []
        for text in text_batch:
            add_text_embeds, prompt_emb = self.encode_prompt(
                self.text_encoder, self.text_encoder_2, text, device=self.device)
            add_text_embeds_batch.append(add_text_embeds)
            prompt_emb_batch.append(prompt_emb)
        add_text_embeds_batch = torch.concat(add_text_embeds_batch, dim=0)
        prompt_emb_batch = torch.concat(prompt_emb_batch, dim=0)
        return add_text_embeds_batch, prompt_emb_batch
    

    def process_video(self, frames_batch):
        with torch.no_grad():
            latents_batch = []
            for frames in frames_batch:
                latents = self.video_encoder.encode_video(frames, batch_size=16)
                latents_batch.append(latents)
        latents_batch = torch.stack(latents_batch)
        return latents_batch
    

    def training_step(self, batch, batch_idx):
        frames, text = batch["frames"], batch["text"]
        frames = frames.to(dtype=self.dtype, device=self.device)

        # Call text encoder
        add_text_embeds, prompt_emb = self.process_text(text)

        # Call video encoder
        latents = self.process_video(frames)

        # Call scheduler
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (1,), device=self.device)
        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        timesteps = torch.repeat_interleave(timesteps, latents.shape[0])

        # Prepare positional id
        add_time_id = torch.tensor([512, 512, 0, 0, 512, 512], device=self.device)

        # Calculate loss
        model_pred = self.denoising_model(
            noisy_latents, timesteps, prompt_emb,
            add_time_id=add_time_id, add_text_embeds=add_text_embeds
        )
        loss = torch.nn.functional.mse_loss(model_pred.to(torch.float32), noise.to(torch.float32), reduction="mean")

        # Record log
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.denoising_model.parameters(), lr=self.learning_rate)
        return optimizer


if __name__ == '__main__':
    # dataset and data loader
    dataset = TextVideoDataset("data/pixabay100", "data/pixabay100/metadata_example.json",
                               num_frames=128, height=512, width=512,
                               random_crop=False, random_interval=False, random_start=False,
                               steps_per_epoch=100)
    train_loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=1, num_workers=8)

    # model
    model = LightningVideoDiT(learning_rate=1e-5)

    # train
    trainer = pl.Trainer(
        max_epochs=100000, accelerator="gpu", devices=[0], strategy="deepspeed_stage_1", precision="16-mixed",
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1)]
    )
    trainer.fit(model=model, train_dataloaders=train_loader)

