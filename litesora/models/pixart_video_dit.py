import torch, math
from einops import rearrange, repeat
from litesora.models.attention import Attention
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
    

class TimeEmbed(torch.nn.Module):
    def __init__(self, dim_time, dim_out):
        super().__init__()
        self.time_proj = Timesteps(dim_time)
        self.time_embedding = torch.nn.Sequential(
            torch.nn.Linear(dim_time, dim_out),
            torch.nn.SiLU(),
            torch.nn.Linear(dim_out, dim_out)
        )

    def forward(self, timesteps, dtype=torch.float32):
        time_emb = self.time_proj(timesteps).to(dtype=dtype)
        time_emb = self.time_embedding(time_emb)
        return time_emb


class GlobalAdaLayerNorm(torch.nn.Module):
    def __init__(self, dim_time, dim_out):
        super().__init__()
        self.time_emb = TimeEmbed(dim_time, dim_out)
        self.linear = torch.nn.Linear(dim_out, 6 * dim_out, bias=True)

    def forward(self, timestep, dtype=torch.float32):
        time_emb = self.time_emb(timestep, dtype=dtype)
        gates = self.linear(torch.nn.functional.silu(time_emb))
        gates = rearrange(gates, "B (N D) -> B N D", N=6)
        return gates, time_emb


class Patchify(torch.nn.Module):
    def __init__(self, base_size=(32, 32, 32), patch_size=(1, 2, 2), in_channels=4, pos_embed_dim=576, temporal=False):
        super().__init__()
        self.temporal = temporal
        self.base_size = base_size
        self.patch_size = patch_size
        self.embed_dim = pos_embed_dim
        pos_embed_dim = pos_embed_dim * (3 if temporal else 2)
        self.proj_latent = torch.nn.Conv3d(in_channels, pos_embed_dim, kernel_size=patch_size, stride=patch_size)

    def get_pos_embedding_1d(self, embed_dim, pos, device):
        omega = torch.arange(embed_dim // 2).to(dtype=torch.float64, device=device) * (2.0 / embed_dim)
        omega = 1.0 / 10000**omega

        pos = pos.reshape(-1)
        out = torch.einsum("m,d->md", pos, omega)
        emb_sin = torch.sin(out)
        emb_cos = torch.cos(out)

        emb = torch.concatenate([emb_sin, emb_cos], axis=1)
        return emb
    
    def get_pos_embedding_2d(self, embed_dim, grid_size, device):
        grid_h = torch.arange(grid_size[1], device=device) / (grid_size[1] / self.base_size[1])
        grid_w = torch.arange(grid_size[2], device=device) / (grid_size[2] / self.base_size[2])
        grid = torch.stack([
            repeat(grid_w, "W -> T H W", T=grid_size[0], H=grid_size[1], W=grid_size[2]),
            repeat(grid_h, "H -> T H W", T=grid_size[0], H=grid_size[1], W=grid_size[2]),
        ])
        pos_embed = self.get_pos_embedding_1d(embed_dim, grid, device)
        pos_embed = rearrange(pos_embed, "(C N) D -> 1 N (C D)", C=2)
        return pos_embed

    def get_pos_embedding_3d(self, embed_dim, grid_size, device):
        grid_t = torch.arange(grid_size[0], device=device) / (grid_size[0] / self.base_size[0])
        grid_h = torch.arange(grid_size[1], device=device) / (grid_size[1] / self.base_size[1])
        grid_w = torch.arange(grid_size[2], device=device) / (grid_size[2] / self.base_size[2])
        grid = torch.stack([
            repeat(grid_t, "T -> T H W", T=grid_size[0], H=grid_size[1], W=grid_size[2]),
            repeat(grid_h, "H -> T H W", T=grid_size[0], H=grid_size[1], W=grid_size[2]),
            repeat(grid_w, "W -> T H W", T=grid_size[0], H=grid_size[1], W=grid_size[2]),
        ])
        pos_embed = self.get_pos_embedding_1d(embed_dim, grid, device)
        pos_embed = rearrange(pos_embed, "(C N) D -> 1 N (C D)", C=3)
        return pos_embed

    def forward(self, latent):
        grid_size = (latent.shape[-3] // self.patch_size[0], latent.shape[-2] // self.patch_size[1], latent.shape[-1] // self.patch_size[2])
        if self.temporal:
            pos_embed = self.get_pos_embedding_3d(self.embed_dim, grid_size, latent.device)
        else:
            pos_embed = self.get_pos_embedding_2d(self.embed_dim, grid_size, latent.device)
        pos_embed = pos_embed.to(dtype=latent.dtype, device=latent.device)

        latent = self.proj_latent(latent)
        latent = rearrange(latent, "B C T H W -> B (T H W) C")

        return latent + pos_embed
    

class UnPatchify(torch.nn.Module):
    def __init__(self, dim_hidden=1152, dim_out=8, patch_size=(1, 2, 2)):
        super().__init__()
        self.patch_size = patch_size
        num_values_per_patch = reduce(lambda x,y: x*y, patch_size)
        self.scale_shift_table = torch.nn.Parameter(torch.zeros(1, 2, dim_hidden))
        self.norm_out = torch.nn.LayerNorm(dim_hidden, eps=1e-6, elementwise_affine=False)
        self.proj_out = torch.nn.Linear(in_features=dim_hidden, out_features=num_values_per_patch * dim_out, bias=True)

    def forward(self, hidden_states, time_emb, T, H, W):
        shift, scale = (self.scale_shift_table + time_emb.unsqueeze(1)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.proj_out(hidden_states)
        hidden_states = rearrange(
            hidden_states,
            "B (T H W) (PT PH PW C) -> B C (T PT) (H PH) (W PW)",
            T=T//self.patch_size[0], H=H//self.patch_size[1], W=W//self.patch_size[2],
            PT=self.patch_size[0], PH=self.patch_size[1], PW=self.patch_size[2]
        )
        return hidden_states


class DiTFeedForward(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj_in = torch.nn.Linear(dim, dim * 4)
        self.proj_out = torch.nn.Linear(dim * 4, dim)

    def forward(self, hidden_states):
        hidden_states = self.proj_in(hidden_states)
        hidden_states = torch.nn.functional.gelu(hidden_states, approximate="tanh")
        hidden_states = self.proj_out(hidden_states)
        return hidden_states


class PixartVideoDiTBlock(torch.nn.Module):
    def __init__(self, dim_hidden, num_heads, head_dim):
        super().__init__()
        self.scale_shift_table = torch.nn.Parameter(torch.zeros((1, 6, dim_hidden)))
        self.norm1 = torch.nn.LayerNorm(dim_hidden, eps=1e-6, elementwise_affine=False)
        self.attn1 = Attention(dim_hidden, num_heads, head_dim, bias_q=True, bias_kv=True, bias_out=True)
        self.norm2 = torch.nn.LayerNorm(dim_hidden, eps=1e-6, elementwise_affine=False)
        self.attn2 = Attention(dim_hidden, num_heads, head_dim, bias_q=True, bias_kv=True, bias_out=True)
        self.ff = DiTFeedForward(dim_hidden)

    def forward(self, hidden_states, time_emb, text_emb, text_mask, gates):
        # Gates
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table + gates).chunk(6, dim=1)

        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        attn_output = self.attn1(norm_hidden_states)
        hidden_states = gate_msa * attn_output + hidden_states

        norm_hidden_states = hidden_states
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states=text_emb, attn_mask=text_mask)
        hidden_states = attn_output + hidden_states

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
        ff_output = self.ff(norm_hidden_states)
        hidden_states = gate_mlp * ff_output + hidden_states

        return hidden_states


class PixartDiT(torch.nn.Module):
    def __init__(self, dim_hidden=1152, dim_time=256, dim_text=4096, dim_head=72, num_blocks=28, patch_size=(1, 2, 2), in_channels=4, out_channels=8):
        super().__init__()
        self.num_heads = dim_hidden // dim_head
        self.patchify = Patchify(base_size=(32, 32, 32), patch_size=patch_size, in_channels=in_channels, pos_embed_dim=dim_hidden // 2)
        self.global_adaln = GlobalAdaLayerNorm(dim_time=dim_time, dim_out=dim_hidden)
        self.caption_projection = torch.nn.Sequential(
            torch.nn.Linear(dim_text, dim_hidden),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Linear(dim_hidden, dim_hidden)
        )
        self.blocks = torch.nn.ModuleList([PixartVideoDiTBlock(dim_hidden, dim_hidden // dim_head, dim_head) for _ in range(num_blocks)])
        self.unpatchify = UnPatchify(dim_hidden=dim_hidden, dim_out=out_channels, patch_size=patch_size)

    def forward(self, hidden_states, timesteps, text_emb, text_flag):
        # Shape
        B, C, T, H, W = hidden_states.shape

        # Time Embedding
        gates, time_emb = self.global_adaln(timesteps, dtype=hidden_states.dtype)

        # Text Embedding
        text_emb = self.caption_projection(text_emb)

        # Text Mask
        text_mask = torch.ones((B, 120), dtype=hidden_states.dtype, device=hidden_states.device) * (-10000)
        text_mask[text_flag.to(torch.bool)] = 0
        text_mask = repeat(text_mask, "B L -> B N 1 L", N=self.num_heads)

        # Patchify
        hidden_states = self.patchify(hidden_states)

        # DiT Blocks
        for block in self.blocks:
            hidden_states = block(hidden_states, time_emb, text_emb, text_mask, gates)

        # Unpatchify
        hidden_states = self.unpatchify(hidden_states, time_emb, T, H, W)
        hidden_states, sigma = hidden_states.chunk(2, dim=1)

        return hidden_states
    
    def load_from_diffusers(self, state_dict):
        rename_dict = {
            "pos_embed.proj.weight": "patchify.proj_latent.weight",
            "pos_embed.proj.bias": "patchify.proj_latent.bias",
            "adaln_single.emb.timestep_embedder.linear_1.weight": "global_adaln.time_emb.time_embedding.0.weight",
            "adaln_single.emb.timestep_embedder.linear_1.bias": "global_adaln.time_emb.time_embedding.0.bias",
            "adaln_single.emb.timestep_embedder.linear_2.weight": "global_adaln.time_emb.time_embedding.2.weight",
            "adaln_single.emb.timestep_embedder.linear_2.bias": "global_adaln.time_emb.time_embedding.2.bias",
            "adaln_single.linear.weight": "global_adaln.linear.weight",
            "adaln_single.linear.bias": "global_adaln.linear.bias",
            "caption_projection.linear_1.weight": "caption_projection.0.weight",
            "caption_projection.linear_1.bias": "caption_projection.0.bias",
            "caption_projection.linear_2.weight": "caption_projection.2.weight",
            "caption_projection.linear_2.bias": "caption_projection.2.bias",
            "scale_shift_table": "unpatchify.scale_shift_table",
            "proj_out.weight": "unpatchify.proj_out.weight",
            "proj_out.bias": "unpatchify.proj_out.bias",
        }
        block_dict = {
            "attn1.to_q": "attn1.to_q",
            "attn1.to_k": "attn1.to_k",
            "attn1.to_v": "attn1.to_v",
            "attn1.to_out.0": "attn1.to_out",
            "attn2.to_q": "attn2.to_q",
            "attn2.to_k": "attn2.to_k",
            "attn2.to_v": "attn2.to_v",
            "attn2.to_out.0": "attn2.to_out",
            "ff.net.0.proj": "ff.proj_in",
            "ff.net.2": "ff.proj_out",
        }
        state_dict_ = {}
        for name in rename_dict:
            param = state_dict[name]
            if name == "pos_embed.proj.weight":
                param = param.unsqueeze(2)
            if name == "scale_shift_table":
                param = param.unsqueeze(0)
            state_dict_[rename_dict[name]] = param
        for i in range(28):
            for name in block_dict:
                for suffix in ["weight", "bias"]:
                    name_diffusers = f"transformer_blocks.{i}.{name}.{suffix}"
                    rename = f"blocks.{i}.{block_dict[name]}.{suffix}"
                    state_dict_[rename] = state_dict[name_diffusers]
            state_dict_[f"blocks.{i}.scale_shift_table"] = state_dict[f"transformer_blocks.{i}.scale_shift_table"].unsqueeze(0)
        self.load_state_dict(state_dict_)


class PixartVideoDiT(torch.nn.Module):
    def __init__(self, dim_hidden=1152, dim_time=256, dim_text=4096, dim_head=72, num_blocks=28, patch_size=(4, 4, 4), in_channels=4, out_channels=8):
        super().__init__()
        self.num_heads = dim_hidden // dim_head
        self.patchify = Patchify(base_size=(32, 32, 32), patch_size=patch_size, in_channels=in_channels, pos_embed_dim=dim_hidden // 3, temporal=True)
        self.global_adaln = GlobalAdaLayerNorm(dim_time=dim_time, dim_out=dim_hidden)
        self.caption_projection = torch.nn.Sequential(
            torch.nn.Linear(dim_text, dim_hidden),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Linear(dim_hidden, dim_hidden)
        )
        self.blocks = torch.nn.ModuleList([PixartVideoDiTBlock(dim_hidden, dim_hidden // dim_head, dim_head) for _ in range(num_blocks)])
        self.unpatchify = UnPatchify(dim_hidden=dim_hidden, dim_out=out_channels, patch_size=patch_size)

    def forward(self, hidden_states, timesteps, text_emb, text_flag):
        # Shape
        B, C, T, H, W = hidden_states.shape

        # Time Embedding
        gates, time_emb = self.global_adaln(timesteps, dtype=hidden_states.dtype)

        # Text Embedding
        text_emb = self.caption_projection(text_emb)

        # Text Mask
        text_mask = torch.ones((B, 120), dtype=hidden_states.dtype, device=hidden_states.device) * (-10000)
        text_mask[text_flag.to(torch.bool)] = 0
        text_mask = repeat(text_mask, "B L -> B N 1 L", N=self.num_heads)

        # Patchify
        hidden_states = self.patchify(hidden_states)

        # DiT Blocks
        for block in self.blocks:
            hidden_states = block(hidden_states, time_emb, text_emb, text_mask, gates)

        # Unpatchify
        hidden_states = self.unpatchify(hidden_states, time_emb, T, H, W)
        hidden_states, sigma = hidden_states.chunk(2, dim=1)

        return hidden_states
    
    def load_from_diffusers(self, state_dict):
        rename_dict = {
            "pos_embed.proj.weight": "patchify.proj_latent.weight",
            "pos_embed.proj.bias": "patchify.proj_latent.bias",
            "adaln_single.emb.timestep_embedder.linear_1.weight": "global_adaln.time_emb.time_embedding.0.weight",
            "adaln_single.emb.timestep_embedder.linear_1.bias": "global_adaln.time_emb.time_embedding.0.bias",
            "adaln_single.emb.timestep_embedder.linear_2.weight": "global_adaln.time_emb.time_embedding.2.weight",
            "adaln_single.emb.timestep_embedder.linear_2.bias": "global_adaln.time_emb.time_embedding.2.bias",
            "adaln_single.linear.weight": "global_adaln.linear.weight",
            "adaln_single.linear.bias": "global_adaln.linear.bias",
            "caption_projection.linear_1.weight": "caption_projection.0.weight",
            "caption_projection.linear_1.bias": "caption_projection.0.bias",
            "caption_projection.linear_2.weight": "caption_projection.2.weight",
            "caption_projection.linear_2.bias": "caption_projection.2.bias",
            "scale_shift_table": "unpatchify.scale_shift_table",
            "proj_out.weight": "unpatchify.proj_out.weight",
            "proj_out.bias": "unpatchify.proj_out.bias",
        }
        block_dict = {
            "attn1.to_q": "attn1.to_q",
            "attn1.to_k": "attn1.to_k",
            "attn1.to_v": "attn1.to_v",
            "attn1.to_out.0": "attn1.to_out",
            "attn2.to_q": "attn2.to_q",
            "attn2.to_k": "attn2.to_k",
            "attn2.to_v": "attn2.to_v",
            "attn2.to_out.0": "attn2.to_out",
            "ff.net.0.proj": "ff.proj_in",
            "ff.net.2": "ff.proj_out",
        }
        mismatch_list = [
            "patchify.proj_latent.weight",
            "patchify.proj_latent.bias",
            "unpatchify.proj_out.weight",
            "unpatchify.proj_out.bias",
        ]
        state_dict_ = {}
        for name in rename_dict:
            param = state_dict[name]
            if name == "pos_embed.proj.weight":
                param = param.unsqueeze(2)
            if name == "scale_shift_table":
                param = param.unsqueeze(0)
            state_dict_[rename_dict[name]] = param
        for i in range(28):
            for name in block_dict:
                for suffix in ["weight", "bias"]:
                    name_diffusers = f"transformer_blocks.{i}.{name}.{suffix}"
                    rename = f"blocks.{i}.{block_dict[name]}.{suffix}"
                    state_dict_[rename] = state_dict[name_diffusers]
            state_dict_[f"blocks.{i}.scale_shift_table"] = state_dict[f"transformer_blocks.{i}.scale_shift_table"].unsqueeze(0)
        for name in mismatch_list:
            state_dict_.pop(name)
        self.load_state_dict(state_dict_, strict=False)