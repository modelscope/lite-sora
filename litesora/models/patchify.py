import torch
from einops import repeat, rearrange


def get_pos_embedding_1d(embed_dim, pos):
    omega = torch.arange(embed_dim // 2).to(torch.float64) * (2.0 / embed_dim)
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    emb = torch.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def get_pos_embedding_2d(embed_dim, grid_size, base_size=16):
    grid_h = torch.arange(grid_size[0]) / (grid_size[0] / base_size)
    grid_w = torch.arange(grid_size[1]) / (grid_size[1] / base_size)
    # In the original implementation of DiT, the h and w seem to be reversed.
    grid = torch.stack([
        repeat(grid_w, "W -> H W", H=grid_size[0], W=grid_size[1]),
        repeat(grid_h, "H -> H W", H=grid_size[0], W=grid_size[1]),
    ])
    pos_embed = get_pos_embedding_1d(embed_dim // 2, grid)
    pos_embed = rearrange(pos_embed, "(C N) D -> N (C D)", C=2)
    return pos_embed


class PatchEmbed(torch.nn.Module):
    def __init__(
        self,
        base_size=16,
        patch_size=16,
        in_channels=3,
        embed_dim=768
    ):
        super().__init__()
        self.base_size = base_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = torch.nn.Conv2d(in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=True)

    def forward(self, latent):
        pos_embed = get_pos_embedding_2d(
            embed_dim=self.embed_dim,
            grid_size=(latent.shape[-2] // self.patch_size, latent.shape[-1] // self.patch_size),
            base_size=self.base_size
        )
        pos_embed = pos_embed.float().unsqueeze(0).to(latent.device)

        latent = self.proj(latent)
        latent = latent.flatten(2).transpose(1, 2)  # BCHW -> BNC

        return (latent + pos_embed).to(latent.dtype)
