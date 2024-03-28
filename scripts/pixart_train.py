from litesora.models.pixart_video_dit import PixartVideoDiT
from litesora.models.svd_vae import SDVAEEncoder
from litesora.schedulers.ddim import DDIMScheduler
from litesora.data.text_video_dataset import TextVideoDataset
from litesora.models.utils import load_state_dict
from transformers import T5EncoderModel, T5Tokenizer
import lightning as pl
import torch



class LightningVideoDiT(pl.LightningModule):
    def __init__(self, learning_rate=1e-5):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained("models/PixArt-XL-2-512x512/tokenizer")
        self.text_encoder = T5EncoderModel.from_pretrained("models/PixArt-XL-2-512x512/text_encoder")
        self.video_encoder = SDVAEEncoder.from_diffusers("models/vae/model.safetensors")
        
        self.denoising_model = PixartVideoDiT()
        state_dict = load_state_dict("models/PixArt-XL-2-512x512/transformer/diffusion_pytorch_model.safetensors")
        self.denoising_model.load_from_diffusers(state_dict)

        self.noise_scheduler = DDIMScheduler()
        self.learning_rate = learning_rate
        self.text_encoder.requires_grad_(False)
        self.video_encoder.requires_grad_(False)

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
    
    def process_text(self, text_batch):
        with torch.no_grad():
            text_emb_batch, text_mask_batch = [], []
            for text in text_batch:
                text_emb, text_mask = self.encode_prompt(text)
                text_emb_batch.append(text_emb)
                text_mask_batch.append(text_mask)
        text_emb_batch = torch.concat(text_emb_batch, axis=0)
        text_mask_batch = torch.concat(text_mask_batch, axis=0)
        return text_emb_batch, text_mask_batch
    
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
        text_emb, text_flag = self.process_text(text)

        # Call video encoder
        latents = self.process_video(frames)

        # Call scheduler
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (1,), device=self.device)
        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        timesteps = torch.repeat_interleave(timesteps, latents.shape[0])

        # Calculate loss
        model_pred = self.denoising_model(noisy_latents, timesteps, text_emb, text_flag)
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

