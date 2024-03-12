import torch
from .attention import Attention
from .utils import load_state_dict


class CLIPEncoderLayer(torch.nn.Module):
    def __init__(self, embed_dim, intermediate_size, num_heads=12, head_dim=64, use_quick_gelu=True):
        super().__init__()
        self.attn = Attention(q_dim=embed_dim, num_heads=num_heads, head_dim=head_dim, bias_q=True, bias_kv=True, bias_out=True)
        self.layer_norm1 = torch.nn.LayerNorm(embed_dim)
        self.layer_norm2 = torch.nn.LayerNorm(embed_dim)
        self.fc1 = torch.nn.Linear(embed_dim, intermediate_size)
        self.fc2 = torch.nn.Linear(intermediate_size, embed_dim)

        self.use_quick_gelu = use_quick_gelu

    def quickGELU(self, x):
        return x * torch.sigmoid(1.702 * x)
    
    def forward(self, hidden_states, attn_mask=None):
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.attn(hidden_states, attn_mask=attn_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.fc1(hidden_states)
        if self.use_quick_gelu:
            hidden_states = self.quickGELU(hidden_states)
        else:
            hidden_states = torch.nn.functional.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
    

class SDXLTextEncoder2(torch.nn.Module):
    def __init__(self, embed_dim=1280, vocab_size=49408, max_position_embeddings=77, num_encoder_layers=32, encoder_intermediate_size=5120):
        super().__init__()

        # token_embedding
        self.token_embedding = torch.nn.Embedding(vocab_size, embed_dim)

        # position_embeds (This is a fixed tensor)
        self.position_embeds = torch.nn.Parameter(torch.zeros(1, max_position_embeddings, embed_dim))

        # encoders
        self.encoders = torch.nn.ModuleList([CLIPEncoderLayer(embed_dim, encoder_intermediate_size, num_heads=20, head_dim=64, use_quick_gelu=False) for _ in range(num_encoder_layers)])

        # attn_mask
        self.attn_mask = self.attention_mask(max_position_embeddings)

        # final_layer_norm
        self.final_layer_norm = torch.nn.LayerNorm(embed_dim)

        # text_projection
        self.text_projection = torch.nn.Linear(embed_dim, embed_dim, bias=False)

    def attention_mask(self, length):
        mask = torch.empty(length, length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def forward(self, input_ids):
        embeds = self.token_embedding(input_ids) + self.position_embeds
        attn_mask = self.attn_mask.to(device=embeds.device, dtype=embeds.dtype)
        for encoder_id, encoder in enumerate(self.encoders):
            embeds = encoder(embeds, attn_mask=attn_mask)
        embeds = self.final_layer_norm(embeds)
        pooled_embeds = embeds[torch.arange(embeds.shape[0]), input_ids.to(dtype=torch.int).argmax(dim=-1)]
        pooled_embeds = self.text_projection(pooled_embeds)
        return pooled_embeds
    
    def load_state_dict_from_diffusers(self, file_path=None, state_dict=None):
        if state_dict is None:
            state_dict = load_state_dict(file_path)
        rename_dict = {
            "text_model.embeddings.token_embedding.weight": "token_embedding.weight",
            "text_model.embeddings.position_embedding.weight": "position_embeds",
            "text_model.final_layer_norm.weight": "final_layer_norm.weight",
            "text_model.final_layer_norm.bias": "final_layer_norm.bias",
            "text_projection.weight": "text_projection.weight"
        }
        attn_rename_dict = {
            "self_attn.q_proj": "attn.to_q",
            "self_attn.k_proj": "attn.to_k",
            "self_attn.v_proj": "attn.to_v",
            "self_attn.out_proj": "attn.to_out",
            "layer_norm1": "layer_norm1",
            "layer_norm2": "layer_norm2",
            "mlp.fc1": "fc1",
            "mlp.fc2": "fc2",
        }
        state_dict_ = {}
        for name in state_dict:
            if name in rename_dict:
                param = state_dict[name]
                if name == "text_model.embeddings.position_embedding.weight":
                    param = param.reshape((1, param.shape[0], param.shape[1]))
                state_dict_[rename_dict[name]] = param
            elif name.startswith("text_model.encoder.layers."):
                param = state_dict[name]
                names = name.split(".")
                layer_id, layer_type, tail = names[3], ".".join(names[4:-1]), names[-1]
                name_ = ".".join(["encoders", layer_id, attn_rename_dict[layer_type], tail])
                state_dict_[name_] = param
        self.load_state_dict(state_dict_)
    
    @staticmethod
    def from_diffusers(file_path=None, state_dict=None):
        model = SDXLTextEncoder2()
        model.eval()
        model.load_state_dict_from_diffusers(file_path, state_dict)
        return model

