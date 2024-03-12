import lightning as pl
import torch
from transformers import CLIPTokenizer
from ..models import SDXLTextEncoder2, VideoDiT
from ..schedulers import DDIMScheduler


class LightningVideoDiT(pl.LightningModule):
    def __init__(self, learning_rate=1e-5):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained("configs/stable_diffusion_xl/tokenizer_2")
        self.text_encoder = SDXLTextEncoder2()
        self.denoising_model = VideoDiT()
        self.noise_scheduler = DDIMScheduler()
        self.learning_rate = learning_rate
        self.text_encoder.requires_grad_(False)

    def tokenize(self, prompt):
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True
        ).input_ids
        return input_ids

    def training_step(self, batch, batch_idx):
        hidden_states, text = batch["frames"], batch["text"]

        with torch.no_grad():
            input_ids = self.tokenize(text[0]).to(self.device)
            text_emb = self.text_encoder(input_ids)

        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (1,), device="cuda")
        noise = torch.randn_like(hidden_states)
        noisy_latents = self.noise_scheduler.add_noise(hidden_states, noise, timesteps)

        model_pred = self.denoising_model(noisy_latents, timesteps, text_emb)
        loss = torch.nn.functional.mse_loss(model_pred, noise, reduction="mean")

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer