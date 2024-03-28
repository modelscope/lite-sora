from litesora.models.sd_text_encoder import SDTextEncoder
from litesora.models.video_dit import VideoDiT
from litesora.models.svd_vae import SDVAEEncoder
from litesora.schedulers.ddim import DDIMScheduler
from litesora.data.text_video_dataset import TextVideoDataset
from transformers import CLIPTokenizer
import lightning as pl
import torch



class LightningVideoDiT(pl.LightningModule):
    def __init__(self, learning_rate=1e-5, compile=False):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained("configs/stable_diffusion/tokenizer")
        self.text_encoder = SDTextEncoder.from_diffusers("models/text_encoder_768/model.safetensors")
        self.video_encoder = SDVAEEncoder.from_diffusers("models/vae/model.safetensors")
        self.denoising_model = VideoDiT()
        self.noise_scheduler = DDIMScheduler()
        self.learning_rate = learning_rate
        self.text_encoder.requires_grad_(False)
        self.video_encoder.requires_grad_(False)
        if compile:
            self.compile()

    def compile(self):
        self.text_encoder = torch.compile(self.text_encoder)
        self.video_encoder = torch.compile(self.video_encoder)
        self.denoising_model = torch.compile(self.denoising_model)

    def tokenize(self, prompt):
        input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        return input_ids
    
    def process_text(self, text_batch):
        with torch.no_grad():
            input_ids = self.tokenize(text_batch).to(self.device)
            cross_emb, text_emb = self.text_encoder(
                input_ids, pooled_output_id=self.tokenizer.eos_token_id)
        return cross_emb, text_emb
    
    def process_video(self, frames_batch):
        with torch.no_grad():
            latents_batch = []
            for frames in frames_batch:
                latents = self.video_encoder.encode_video(frames, batch_size=16)
                latents_batch.append(latents)
        latents_batch = torch.stack(latents_batch)
        return latents_batch

    def training_step(self, batch, batch_idx):
        frames, text, start_sec, end_sec = batch["frames"], batch["text"], batch["start_sec"], batch["end_sec"]
        frames = frames.to(dtype=self.dtype, device=self.device)
        start_sec = start_sec.to(device=self.device)
        end_sec = end_sec.to(device=self.device)

        # Use pseudo start_sec and end_sec (TODO: We need to change the positional embeddings later)
        start_sec[:] = 0.0
        end_sec[:] = 8.0

        # Call text encoder
        cross_emb, text_emb = self.process_text(text)

        # Call video encoder
        latents = self.process_video(frames)

        # Call scheduler
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (1,), device=self.device)
        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        timesteps = torch.repeat_interleave(timesteps, latents.shape[0])

        # Calculate loss
        model_pred = self.denoising_model(noisy_latents, timesteps, start_sec, end_sec, cross_emb, text_emb)
        loss = torch.nn.functional.mse_loss(model_pred.to(torch.float32), noise.to(torch.float32), reduction="mean")

        # Record log
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.denoising_model.parameters(), lr=self.learning_rate)
        return optimizer


if __name__ == '__main__':
    # dataset and data loader
    dataset = TextVideoDataset("data/pixabay100", "data/pixabay100/metadata.json",
                               num_frames=128, height=512, width=512,
                               random_crop=False, random_interval=False, random_start=False,
                               steps_per_epoch=10000)
    train_loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=1, num_workers=8)

    # model
    model = LightningVideoDiT(learning_rate=1e-5)

    # train
    trainer = pl.Trainer(
        max_epochs=100000, accelerator="gpu", devices="auto", strategy="deepspeed_stage_1", precision="16-mixed",
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1)]
    )
    trainer.fit(model=model, train_dataloaders=train_loader)

