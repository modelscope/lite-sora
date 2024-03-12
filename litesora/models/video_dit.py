import torch, math
from .attention import Attention
from .utils import load_state_dict
from einops import rearrange, repeat
from functools import reduce


class Timesteps(torch.nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps):
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device) / half_dim
        timesteps = timesteps.unsqueeze(-1)
        emb = timesteps.float() * torch.exp(exponent)
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        return emb


class ConditioningFusion(torch.nn.Module):
    def __init__(self, dims_in, dim_out):
        super().__init__()
        self.projs = torch.nn.ModuleList([torch.nn.Linear(dim_in, dim_out) for dim_in in dims_in])

    def forward(self, conditionings):
        conditionings = [proj(conditioning) for conditioning, proj in zip(conditionings, self.projs)]
        conditionings = torch.stack(conditionings).sum(axis=0)
        return conditionings


class AdaLayerNormZero(torch.nn.Module):
    def __init__(self, dim_time, dim_text, dim_out):
        super().__init__()
        self.fusion = ConditioningFusion([dim_time, dim_text], dim_out)
        self.linear = torch.nn.Linear(dim_out, 6 * dim_out)

    def forward(self, time_emb, text_emb):
        conditionings = self.fusion([time_emb, text_emb])
        conditionings = self.linear(torch.nn.functional.silu(conditionings)).unsqueeze(1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = conditionings.chunk(6, dim=-1)
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp


class DiTFeedForward(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj_in = torch.nn.Linear(dim, dim * 4)
        self.proj_out = torch.nn.Linear(dim * 4, dim)

    def forward(self, hidden_states):
        dtype = hidden_states.dtype
        hidden_states = self.proj_in(hidden_states)
        hidden_states = torch.nn.functional.gelu(hidden_states.to(dtype=torch.float32), approximate="tanh").to(dtype=dtype)
        hidden_states = self.proj_out(hidden_states)
        return hidden_states


class DiTBlock(torch.nn.Module):
    def __init__(self, dim_out, dim_time, dim_text, dim_head):
        super().__init__()
        self.adaln = AdaLayerNormZero(dim_time, dim_text, dim_out)
        self.norm1 = torch.nn.LayerNorm(dim_out, eps=1e-5, elementwise_affine=False)
        self.attn1 = Attention(dim_out, dim_out // dim_head, dim_head, bias_q=True, bias_kv=True, bias_out=True)
        self.norm2 = torch.nn.LayerNorm(dim_out, 1e-5, elementwise_affine=False)
        self.ff = DiTFeedForward(dim_out)


    def forward(self, hidden_states, time_emb, text_emb):
        # 0. AdaLayerNormZero (Conditioning Fusion)
        beta_1, gamma_1, alpha_1, beta_2, gamma_2, alpha_2 = self.adaln(time_emb, text_emb)

        # 1. Layer Norm
        norm_hidden_states = self.norm1(hidden_states)

        # 2. Scale, Shift
        norm_hidden_states = norm_hidden_states * (1 + gamma_1) + beta_1

        # 3. Multi-Head Self-Attention
        attn_output = self.attn1(norm_hidden_states)

        # 4. Scale & Add
        hidden_states = alpha_1 * attn_output + hidden_states

        # 5. Layer Norm
        norm_hidden_states = self.norm2(hidden_states)

        # 6. Scale & Shift
        norm_hidden_states = norm_hidden_states * (1 + gamma_2) + beta_2

        # 7. Pointwise Feedforward
        ff_output = self.ff(norm_hidden_states)

        # 8. Scale & Add
        hidden_states = alpha_2 * ff_output + hidden_states

        return hidden_states
    

class VideoPatchEmbed(torch.nn.Module):
    def __init__(self, base_size=(16, 16, 16), patch_size=(16, 16, 16), in_channels=3, embed_dim=512):
        super().__init__()
        self.base_size = base_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj_pos = torch.nn.Linear(embed_dim*3, embed_dim)
        self.proj_latent = torch.nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

    def get_pos_embedding_1d(self, embed_dim, pos):
        omega = torch.arange(embed_dim // 2).to(torch.float64) * (2.0 / embed_dim)
        omega = 1.0 / 10000**omega

        pos = pos.reshape(-1)
        out = torch.einsum("m,d->md", pos, omega)
        emb_sin = torch.sin(out)
        emb_cos = torch.cos(out)

        emb = torch.concatenate([emb_sin, emb_cos], axis=1)
        return emb

    def get_pos_embedding_3d(self, embed_dim, grid_size, base_size):
        grid_t = torch.arange(grid_size[0]) / (grid_size[0] / base_size[0])
        grid_h = torch.arange(grid_size[1]) / (grid_size[1] / base_size[1])
        grid_w = torch.arange(grid_size[2]) / (grid_size[2] / base_size[2])
        grid = torch.stack([
            repeat(grid_t, "T -> T H W", T=grid_size[0], H=grid_size[1], W=grid_size[2]),
            repeat(grid_h, "H -> T H W", T=grid_size[0], H=grid_size[1], W=grid_size[2]),
            repeat(grid_w, "W -> T H W", T=grid_size[0], H=grid_size[1], W=grid_size[2]),
        ])
        pos_embed = self.get_pos_embedding_1d(embed_dim, grid)
        pos_embed = rearrange(pos_embed, "(C N) D -> N (C D)", C=3)
        return pos_embed

    def forward(self, latent):
        pos_embed = self.get_pos_embedding_3d(
            self.embed_dim,
            (latent.shape[-3] // self.patch_size[0], latent.shape[-2] // self.patch_size[1], latent.shape[-1] // self.patch_size[1]),
            self.base_size
        )
        pos_embed = pos_embed.unsqueeze(0).to(dtype=latent.dtype, device=latent.device)
        pos_embed = self.proj_pos(pos_embed)

        latent = self.proj_latent(latent)
        latent = rearrange(latent, "B C T H W -> B (T H W) C")

        return (latent + pos_embed).to(latent.dtype)
    

class TimeEmbed(torch.nn.Module):
    def __init__(self, dim_time):
        super().__init__()
        self.time_proj = Timesteps(dim_time)
        self.time_embedding = torch.nn.Sequential(
            torch.nn.Linear(dim_time, dim_time),
            torch.nn.SiLU(),
            torch.nn.Linear(dim_time, dim_time)
        )

    def forward(self, timesteps, dtype=torch.float32):
        time_emb = self.time_proj(timesteps).to(dtype=dtype)
        time_emb = self.time_embedding(time_emb)
        return time_emb


class VideoDiT(torch.nn.Module):
    def __init__(self, dim_hidden=1024, dim_time=1024, dim_text=1280, dim_head=64, num_blocks=16, patch_size=(4, 4, 4), in_channels=3):
        super().__init__()
        self.time_emb = TimeEmbed(dim_time)
        self.patchify = VideoPatchEmbed((16, 16, 16), patch_size, in_channels, dim_hidden)
        self.blocks = torch.nn.ModuleList([DiTBlock(dim_hidden, dim_time, dim_text, dim_head) for _ in range(num_blocks)])
        self.norm_out = torch.nn.LayerNorm(dim_hidden, eps=1e-5, elementwise_affine=False)
        self.proj_out = torch.nn.Linear(dim_hidden, reduce(lambda x,y: x*y, patch_size) * in_channels, bias=True)

    def forward(self, hidden_states, timesteps, text_emb):
        # Shape
        B, C, T, H, W = hidden_states.shape

        # Time Embedding
        time_emb = self.time_emb(timesteps, dtype=hidden_states.dtype)

        # Patchify
        hidden_states = self.patchify(hidden_states)

        # DiT Blocks
        for block in self.blocks:
            hidden_states = block(hidden_states, time_emb, text_emb)

        # The following computation is different from the original version of DiT
        # We make it simple.
        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = rearrange(
            hidden_states,
            "B (T H W) (PT PH PW C) -> B C (T PT) (H PH) (W PW)",
            T=T//self.patchify.patch_size[0], H=H//self.patchify.patch_size[1], W=W//self.patchify.patch_size[2],
            PT=self.patchify.patch_size[0], PH=self.patchify.patch_size[1], PW=self.patchify.patch_size[2]
        )

        return hidden_states
    
    @staticmethod
    def from_pretrained(file_path):
        state_dict = load_state_dict(file_path)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            state_dict = {i[len("denoising_model."):]: state_dict[i] for i in state_dict if i.startswith("denoising_model.")}

        model = VideoDiT()
        model.eval()
        model.load_state_dict(state_dict)
        return model
