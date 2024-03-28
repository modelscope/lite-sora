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
        conditionings = torch.stack(conditionings).sum(dim=0)
        return conditionings
    

class AdaLayerNormZero(torch.nn.Module):
    def __init__(self, dims_in, dim_out):
        super().__init__()
        self.fusion = ConditioningFusion(dims_in, dim_out)
        self.linear = torch.nn.Linear(dim_out, 3 * dim_out)

    def forward(self, conditionings):
        conditionings = self.fusion(conditionings)
        conditionings = self.linear(torch.nn.functional.silu(conditionings)).unsqueeze(1)
        shift, scale, gate = conditionings.chunk(3, dim=-1)
        return shift, scale, gate


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
    

class DiTBlockA(torch.nn.Module):
    def __init__(self, dim_out, dims_cond, dim_cross, dim_head):
        super().__init__()
        self.adaln = AdaLayerNormZero(dims_cond, dim_out)
        self.norm = torch.nn.LayerNorm(dim_out, eps=1e-5, elementwise_affine=False)
        self.attn = Attention(dim_out, dim_out // dim_head, dim_head, bias_q=True, bias_kv=True, bias_out=True)

    def forward(self, hidden_states, conditionings, cross_emb):
        # 0. AdaLayerNormZero (Conditioning Fusion)
        beta, gamma, alpha = self.adaln(conditionings)

        # 1. Layer Norm
        norm_hidden_states = self.norm(hidden_states)

        # 2. Scale, Shift
        norm_hidden_states = norm_hidden_states * (1 + gamma) + beta

        # 3. Self-Attention
        attn_output = self.attn(norm_hidden_states)

        # 4. Scale & Add
        hidden_states = alpha * attn_output + hidden_states

        return hidden_states
    

class DiTBlockB(torch.nn.Module):
    def __init__(self, dim_out, dims_cond, dim_cross, dim_head):
        super().__init__()
        self.adaln = AdaLayerNormZero(dims_cond, dim_out)
        self.norm = torch.nn.LayerNorm(dim_out, eps=1e-5, elementwise_affine=False)
        self.attn = Attention(dim_out, dim_out // dim_head, dim_head, kv_dim=dim_cross, bias_q=True, bias_kv=True, bias_out=True)

    def forward(self, hidden_states, conditionings, cross_emb):
        # 0. AdaLayerNormZero (Conditioning Fusion)
        beta, gamma, alpha = self.adaln(conditionings)

        # 1. Layer Norm
        norm_hidden_states = self.norm(hidden_states)

        # 2. Scale, Shift
        norm_hidden_states = norm_hidden_states * (1 + gamma) + beta

        # 3. Cross-Attention
        attn_output = self.attn(norm_hidden_states, encoder_hidden_states=cross_emb)

        # 4. Scale & Add
        hidden_states = alpha * attn_output + hidden_states

        return hidden_states
    

class DiTBlockC(torch.nn.Module):
    def __init__(self, dim_out, dims_cond, dim_cross, dim_head):
        super().__init__()
        self.adaln = AdaLayerNormZero(dims_cond, dim_out)
        self.norm = torch.nn.LayerNorm(dim_out, eps=1e-5, elementwise_affine=False)
        self.ff = DiTFeedForward(dim_out)

    def forward(self, hidden_states, conditionings, cross_emb):
        # 0. AdaLayerNormZero (Conditioning Fusion)
        beta, gamma, alpha = self.adaln(conditionings)

        # 1. Layer Norm
        norm_hidden_states = self.norm(hidden_states)

        # 2. Scale & Shift
        norm_hidden_states = norm_hidden_states * (1 + gamma) + beta

        # 3. Pointwise Feedforward
        ff_output = self.ff(norm_hidden_states)

        # 4. Scale & Add
        hidden_states = alpha * ff_output + hidden_states

        return hidden_states


class DiTBlock(torch.nn.Module):
    def __init__(self, dim_out, dims_cond, dim_cross, dim_head):
        super().__init__()
        self.blockA = DiTBlockA(dim_out, dims_cond, dim_cross, dim_head)
        self.blockB = DiTBlockB(dim_out, dims_cond, dim_cross, dim_head)
        self.blockC = DiTBlockC(dim_out, dims_cond, dim_cross, dim_head)

    def forward(self, hidden_states, conditionings, cross_emb):
        hidden_states = self.blockA(hidden_states, conditionings, cross_emb)
        hidden_states = self.blockB(hidden_states, conditionings, cross_emb)
        hidden_states = self.blockC(hidden_states, conditionings, cross_emb)
        return hidden_states
    

class Patchify(torch.nn.Module):
    def __init__(self, base_size=(None, 16, 16), patch_size=(16, 16, 16), in_channels=3, embed_dim=512):
        super().__init__()
        self.base_size = base_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj_pos = torch.nn.Linear(embed_dim*3, embed_dim)
        self.proj_latent = torch.nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

    def get_pos_embedding_1d(self, embed_dim, pos, device):
        omega = torch.arange(embed_dim // 2).to(dtype=torch.float64, device=device) * (2.0 / embed_dim)
        omega = 1.0 / 10000**omega

        pos = pos.reshape(-1)
        out = torch.einsum("m,d->md", pos, omega)
        emb_sin = torch.sin(out)
        emb_cos = torch.cos(out)

        emb = torch.concatenate([emb_sin, emb_cos], axis=1)
        return emb

    def get_pos_embedding_3d(self, embed_dim, grid_size, start_sec, end_sec, device):
        grid_t = torch.arange(grid_size[0], device=device) / grid_size[0] * (end_sec - start_sec) + start_sec
        grid_h = torch.arange(grid_size[1], device=device) / (grid_size[1] / self.base_size[1])
        grid_w = torch.arange(grid_size[2], device=device) / (grid_size[2] / self.base_size[2])
        grid = torch.stack([
            repeat(grid_t, "T -> T H W", T=grid_size[0], H=grid_size[1], W=grid_size[2]),
            repeat(grid_h, "H -> T H W", T=grid_size[0], H=grid_size[1], W=grid_size[2]),
            repeat(grid_w, "W -> T H W", T=grid_size[0], H=grid_size[1], W=grid_size[2]),
        ])
        pos_embed = self.get_pos_embedding_1d(embed_dim, grid, device)
        pos_embed = rearrange(pos_embed, "(C N) D -> N (C D)", C=3)
        return pos_embed

    def forward(self, latent, start_sec, end_sec):
        grid_size = (latent.shape[-3] // self.patch_size[0], latent.shape[-2] // self.patch_size[1], latent.shape[-1] // self.patch_size[2])
        pos_embed = torch.stack([
            self.get_pos_embedding_3d(self.embed_dim, grid_size, start_sec_, end_sec_, latent.device)
            for start_sec_, end_sec_ in zip(start_sec, end_sec)])
        pos_embed = pos_embed.to(dtype=latent.dtype, device=latent.device)
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
    

class UnPatchify(torch.nn.Module):
    def __init__(self, patch_size, in_channels, dim_hidden):
        super().__init__()
        num_values_per_patch = reduce(lambda x,y: x*y, patch_size)
        self.patch_size = patch_size

        self.norm_out = torch.nn.LayerNorm(dim_hidden, eps=1e-5, elementwise_affine=False)
        self.proj_act = torch.nn.SiLU()
        self.proj_out = torch.nn.Linear(dim_hidden, num_values_per_patch * in_channels, bias=True)
        self.conv_out = torch.nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, hidden_states, T, H, W):
        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.proj_act(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = rearrange(
            hidden_states,
            "B (T H W) (PT PH PW C) -> B C (T PT) (H PH) (W PW)",
            T=T//self.patch_size[0], H=H//self.patch_size[1], W=W//self.patch_size[2],
            PT=self.patch_size[0], PH=self.patch_size[1], PW=self.patch_size[2]
        )
        hidden_states = self.conv_out(hidden_states)
        return hidden_states


class VideoDiT(torch.nn.Module):
    def __init__(self, dim_hidden=1024, dim_time=1024, dim_text=768, dim_cross=768, dim_head=64, num_blocks=32, patch_size=(4, 4, 4), in_channels=4):
        super().__init__()
        self.time_emb = TimeEmbed(dim_time)
        self.patchify = Patchify((None, 16, 16), patch_size, in_channels, dim_hidden)
        self.blocks = torch.nn.ModuleList([DiTBlock(dim_hidden, [dim_time, dim_text], dim_cross, dim_head) for _ in range(num_blocks)])
        self.unpatchify = UnPatchify(patch_size, in_channels, dim_hidden)

    def forward(self, hidden_states, timesteps, start_sec, end_sec, cross_emb, text_emb):
        # Shape
        B, C, T, H, W = hidden_states.shape

        # Time Embedding
        time_emb = self.time_emb(timesteps, dtype=hidden_states.dtype)

        # Patchify
        hidden_states = self.patchify(hidden_states, start_sec, end_sec)

        # DiT Blocks
        for block in self.blocks:
            hidden_states = block(hidden_states, [time_emb, text_emb], cross_emb)

        # The following computation is different from the original version of DiT
        # We make it consistent with our VAE encoder.
        hidden_states = self.unpatchify(hidden_states, T, H, W)

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
