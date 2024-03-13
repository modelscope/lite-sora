import torch
from .attention import Attention
from .utils import load_state_dict
from einops import rearrange, repeat


class VAEAttentionBlock(torch.nn.Module):

    def __init__(self, num_attention_heads, attention_head_dim, in_channels, num_layers=1, norm_num_groups=32, eps=1e-5):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=eps, affine=True)

        self.transformer_blocks = torch.nn.ModuleList([
            Attention(
                inner_dim,
                num_attention_heads,
                attention_head_dim,
                bias_q=True,
                bias_kv=True,
                bias_out=True
            )
            for d in range(num_layers)
        ])

    def forward(self, hidden_states):
        batch, _, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)

        hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        hidden_states = hidden_states + residual

        return hidden_states



class ResnetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels=None, groups=32, eps=1e-5):
        super().__init__()
        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels is not None:
            self.time_emb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = torch.nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.nonlinearity = torch.nn.SiLU()
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, hidden_states):
        x = hidden_states
        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.conv2(x)
        if self.conv_shortcut is not None:
            hidden_states = self.conv_shortcut(hidden_states)
        hidden_states = hidden_states + x
        return hidden_states



class DownSampler(torch.nn.Module):
    def __init__(self, channels, padding=1, extra_padding=False):
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, channels, 3, stride=2, padding=padding)
        self.extra_padding = extra_padding

    def forward(self, hidden_states):
        if self.extra_padding:
            hidden_states = torch.nn.functional.pad(hidden_states, (0, 1, 0, 1), mode="constant", value=0)
        hidden_states = self.conv(hidden_states)
        return hidden_states



class UpSampler(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, hidden_states):
        hidden_states = torch.nn.functional.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        hidden_states = self.conv(hidden_states)
        return hidden_states
    


class TemporalResnetBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, groups=32, eps=1e-5):
        super().__init__()
        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = torch.nn.Conv3d(in_channels, out_channels, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
        self.norm2 = torch.nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.conv2 = torch.nn.Conv3d(out_channels, out_channels, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
        self.nonlinearity = torch.nn.SiLU()
        self.mix_factor = torch.nn.Parameter(torch.Tensor([0.5]))

    def forward(self, hidden_states):
        x_spatial = hidden_states
        x = rearrange(hidden_states, "T C H W -> 1 C T H W")
        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.conv2(x)
        x_temporal = hidden_states + x[0].permute(1, 0, 2, 3)
        alpha = torch.sigmoid(self.mix_factor)
        hidden_states = alpha * x_temporal + (1 - alpha) * x_spatial
        return hidden_states



class SDVAEEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scaling_factor = 0.18215
        self.quant_conv = torch.nn.Conv2d(8, 8, kernel_size=1)
        self.conv_in = torch.nn.Conv2d(3, 128, kernel_size=3, padding=1)

        self.blocks = torch.nn.ModuleList([
            # DownEncoderBlock2D
            ResnetBlock(128, 128, eps=1e-6),
            ResnetBlock(128, 128, eps=1e-6),
            DownSampler(128, padding=0, extra_padding=True),
            # DownEncoderBlock2D
            ResnetBlock(128, 256, eps=1e-6),
            ResnetBlock(256, 256, eps=1e-6),
            DownSampler(256, padding=0, extra_padding=True),
            # DownEncoderBlock2D
            ResnetBlock(256, 512, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            DownSampler(512, padding=0, extra_padding=True),
            # DownEncoderBlock2D
            ResnetBlock(512, 512, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            # UNetMidBlock2D
            ResnetBlock(512, 512, eps=1e-6),
            VAEAttentionBlock(1, 512, 512, 1, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
        ])

        self.conv_norm_out = torch.nn.GroupNorm(num_channels=512, num_groups=32, eps=1e-6)
        self.conv_act = torch.nn.SiLU()
        self.conv_out = torch.nn.Conv2d(512, 8, kernel_size=3, padding=1)


    def forward(self, sample):
        # 1. pre-process
        hidden_states = self.conv_in(sample)

        # 2. blocks
        for i, block in enumerate(self.blocks):
            hidden_states = block(hidden_states)
        
        # 3. output
        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        hidden_states = self.quant_conv(hidden_states)
        hidden_states = hidden_states[:, :4]
        hidden_states *= self.scaling_factor

        return hidden_states
    

    def encode_video(self, sample, batch_size=8, progress_bar=lambda x:x):
        data_device = sample.device
        computation_device = self.conv_in.weight.device
        hidden_states = []
        sample = rearrange(sample, "C T H W -> T C H W")

        for i in progress_bar(range(0, sample.shape[0], batch_size)):
            hidden_states_batch = self.forward(sample[i: i+batch_size].to(computation_device))
            hidden_states.append(hidden_states_batch.to(data_device))

        hidden_states = torch.concat(hidden_states, dim=0)
        hidden_states = rearrange(hidden_states, "T C H W -> C T H W")
        return hidden_states
    

    def load_state_dict_from_diffusers(self, file_path=None, state_dict=None):
        if state_dict is None:
            state_dict = load_state_dict(file_path)

        # architecture
        block_types = [
            'ResnetBlock', 'ResnetBlock', 'DownSampler',
            'ResnetBlock', 'ResnetBlock', 'DownSampler',
            'ResnetBlock', 'ResnetBlock', 'DownSampler',
            'ResnetBlock', 'ResnetBlock',
            'ResnetBlock', 'VAEAttentionBlock', 'ResnetBlock'
        ]

        # Rename each parameter
        local_rename_dict = {
            "quant_conv": "quant_conv",
            "encoder.conv_in": "conv_in",
            "encoder.mid_block.attentions.0.group_norm": "blocks.12.norm",
            "encoder.mid_block.attentions.0.to_q": "blocks.12.transformer_blocks.0.to_q",
            "encoder.mid_block.attentions.0.to_k": "blocks.12.transformer_blocks.0.to_k",
            "encoder.mid_block.attentions.0.to_v": "blocks.12.transformer_blocks.0.to_v",
            "encoder.mid_block.attentions.0.to_out.0": "blocks.12.transformer_blocks.0.to_out",
            "encoder.mid_block.resnets.0.norm1": "blocks.11.norm1",
            "encoder.mid_block.resnets.0.conv1": "blocks.11.conv1",
            "encoder.mid_block.resnets.0.norm2": "blocks.11.norm2",
            "encoder.mid_block.resnets.0.conv2": "blocks.11.conv2",
            "encoder.mid_block.resnets.1.norm1": "blocks.13.norm1",
            "encoder.mid_block.resnets.1.conv1": "blocks.13.conv1",
            "encoder.mid_block.resnets.1.norm2": "blocks.13.norm2",
            "encoder.mid_block.resnets.1.conv2": "blocks.13.conv2",
            "encoder.conv_norm_out": "conv_norm_out",
            "encoder.conv_out": "conv_out",
        }
        name_list = sorted([name for name in state_dict])
        rename_dict = {}
        block_id = {"ResnetBlock": -1, "DownSampler": -1, "UpSampler": -1}
        last_block_type_with_id = {"ResnetBlock": "", "DownSampler": "", "UpSampler": ""}
        for name in name_list:
            names = name.split(".")
            name_prefix = ".".join(names[:-1])
            if name_prefix in local_rename_dict:
                rename_dict[name] = local_rename_dict[name_prefix] + "." + names[-1]
            elif name.startswith("encoder.down_blocks"):
                block_type = {"resnets": "ResnetBlock", "downsamplers": "DownSampler", "upsamplers": "UpSampler"}[names[3]]
                block_type_with_id = ".".join(names[:5])
                if block_type_with_id != last_block_type_with_id[block_type]:
                    block_id[block_type] += 1
                last_block_type_with_id[block_type] = block_type_with_id
                while block_id[block_type] < len(block_types) and block_types[block_id[block_type]] != block_type:
                    block_id[block_type] += 1
                block_type_with_id = ".".join(names[:5])
                names = ["blocks", str(block_id[block_type])] + names[5:]
                rename_dict[name] = ".".join(names)

        # Convert state_dict
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                state_dict_[rename_dict[name]] = param
        self.load_state_dict(state_dict_)
        
    
    @staticmethod
    def from_diffusers(file_path=None, state_dict=None):
        model = SDVAEEncoder()
        model.eval()
        model.load_state_dict_from_diffusers(file_path, state_dict)
        return model



class SVDVAEDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scaling_factor = 0.18215
        self.conv_in = torch.nn.Conv2d(4, 512, kernel_size=3, padding=1)

        self.blocks = torch.nn.ModuleList([
            # UNetMidBlock
            ResnetBlock(512, 512, eps=1e-6),
            TemporalResnetBlock(512, 512, eps=1e-6),
            VAEAttentionBlock(1, 512, 512, 1, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            TemporalResnetBlock(512, 512, eps=1e-6),
            # UpDecoderBlock
            ResnetBlock(512, 512, eps=1e-6),
            TemporalResnetBlock(512, 512, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            TemporalResnetBlock(512, 512, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            TemporalResnetBlock(512, 512, eps=1e-6),
            UpSampler(512),
            # UpDecoderBlock
            ResnetBlock(512, 512, eps=1e-6),
            TemporalResnetBlock(512, 512, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            TemporalResnetBlock(512, 512, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            TemporalResnetBlock(512, 512, eps=1e-6),
            UpSampler(512),
            # UpDecoderBlock
            ResnetBlock(512, 256, eps=1e-6),
            TemporalResnetBlock(256, 256, eps=1e-6),
            ResnetBlock(256, 256, eps=1e-6),
            TemporalResnetBlock(256, 256, eps=1e-6),
            ResnetBlock(256, 256, eps=1e-6),
            TemporalResnetBlock(256, 256, eps=1e-6),
            UpSampler(256),
            # UpDecoderBlock
            ResnetBlock(256, 128, eps=1e-6),
            TemporalResnetBlock(128, 128, eps=1e-6),
            ResnetBlock(128, 128, eps=1e-6),
            TemporalResnetBlock(128, 128, eps=1e-6),
            ResnetBlock(128, 128, eps=1e-6),
            TemporalResnetBlock(128, 128, eps=1e-6),
        ])

        self.conv_norm_out = torch.nn.GroupNorm(num_channels=128, num_groups=32, eps=1e-5)
        self.conv_act = torch.nn.SiLU()
        self.conv_out = torch.nn.Conv2d(128, 3, kernel_size=3, padding=1)
        self.time_conv_out = torch.nn.Conv3d(3, 3, kernel_size=(3, 1, 1), padding=(1, 0, 0))


    def forward(self, sample):
        # 1. pre-process
        hidden_states = rearrange(sample, "C T H W -> T C H W")
        hidden_states = hidden_states / self.scaling_factor
        hidden_states = self.conv_in(hidden_states)

        # 2. blocks
        for i, block in enumerate(self.blocks):
            hidden_states = block(hidden_states)

        # 3. output
        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        hidden_states = rearrange(hidden_states, "T C H W -> C T H W")
        hidden_states = self.time_conv_out(hidden_states)

        return hidden_states
    

    def build_mask(self, data, is_bound):
        _, T, H, W = data.shape
        t = repeat(torch.arange(T), "T -> T H W", T=T, H=H, W=W)
        h = repeat(torch.arange(H), "H -> T H W", T=T, H=H, W=W)
        w = repeat(torch.arange(W), "W -> T H W", T=T, H=H, W=W)
        border_width = (T + H + W) // 6
        pad = torch.ones_like(t) * border_width
        mask = torch.stack([
            pad if is_bound[0] else t + 1,
            pad if is_bound[1] else T - t,
            pad if is_bound[2] else h + 1,
            pad if is_bound[3] else H - h,
            pad if is_bound[4] else w + 1,
            pad if is_bound[5] else W - w
        ]).min(dim=0).values
        mask = mask.clip(1, border_width)
        mask = (mask / border_width).to(dtype=data.dtype, device=data.device)
        mask = rearrange(mask, "T H W -> 1 T H W")
        return mask
    

    def decode_video(
        self, sample,
        batch_time=32, batch_height=64, batch_width=64,
        stride_time=16, stride_height=32, stride_width=32,
        progress_bar=lambda x:x
    ):
        data_device = sample.device
        computation_device = self.conv_in.weight.device
        torch_dtype = sample.dtype
        _, T, H, W = sample.shape

        weight = torch.zeros((1, T, H*8, W*8), dtype=torch_dtype, device=data_device)
        values = torch.zeros((3, T, H*8, W*8), dtype=torch_dtype, device=data_device)

        # Split tasks
        tasks = []
        for t in range(0, T, stride_time):
            for h in range(0, H, stride_height):
                for w in range(0, W, stride_width):
                    if (t-stride_time >= 0 and t-stride_time+batch_time >= T)\
                        or (h-stride_height >= 0 and h-stride_height+batch_height >= H)\
                        or (w-stride_width >= 0 and w-stride_width+batch_width >= W):
                        continue
                    tasks.append((t, t+batch_time, h, h+batch_height, w, w+batch_width))
        
        # Run
        for tl, tr, hl, hr, wl, wr in progress_bar(tasks):
            sample_batch = sample[:, tl:tr, hl:hr, wl:wr].to(computation_device)
            sample_batch = self.forward(sample_batch).to(data_device)
            mask = self.build_mask(sample_batch, is_bound=(tl==0, tr>=T, hl==0, hr>=H, wl==0, wr>=W))
            values[:, tl:tr, hl*8:hr*8, wl*8:wr*8] += sample_batch * mask
            weight[:, tl:tr, hl*8:hr*8, wl*8:wr*8] += mask
        values /= weight
        return values

    
    def load_state_dict_from_diffusers(self, file_path=None, state_dict=None):
        if state_dict is None:
            state_dict = load_state_dict(file_path)

        static_rename_dict = {
            "decoder.conv_in":  "conv_in",
            "decoder.mid_block.attentions.0.group_norm": "blocks.2.norm",
            "decoder.mid_block.attentions.0.to_q": "blocks.2.transformer_blocks.0.to_q",
            "decoder.mid_block.attentions.0.to_k": "blocks.2.transformer_blocks.0.to_k",
            "decoder.mid_block.attentions.0.to_v": "blocks.2.transformer_blocks.0.to_v",
            "decoder.mid_block.attentions.0.to_out.0": "blocks.2.transformer_blocks.0.to_out",
            "decoder.up_blocks.0.upsamplers.0.conv": "blocks.11.conv",
            "decoder.up_blocks.1.upsamplers.0.conv": "blocks.18.conv",
            "decoder.up_blocks.2.upsamplers.0.conv": "blocks.25.conv",
            "decoder.conv_norm_out": "conv_norm_out",
            "decoder.conv_out": "conv_out",
            "decoder.time_conv_out": "time_conv_out"
        }
        prefix_rename_dict = {
            "decoder.mid_block.resnets.0.spatial_res_block": "blocks.0",
            "decoder.mid_block.resnets.0.temporal_res_block": "blocks.1",
            "decoder.mid_block.resnets.0.time_mixer": "blocks.1",
            "decoder.mid_block.resnets.1.spatial_res_block": "blocks.3",
            "decoder.mid_block.resnets.1.temporal_res_block": "blocks.4",
            "decoder.mid_block.resnets.1.time_mixer": "blocks.4",

            "decoder.up_blocks.0.resnets.0.spatial_res_block": "blocks.5",
            "decoder.up_blocks.0.resnets.0.temporal_res_block": "blocks.6",
            "decoder.up_blocks.0.resnets.0.time_mixer": "blocks.6",
            "decoder.up_blocks.0.resnets.1.spatial_res_block": "blocks.7",
            "decoder.up_blocks.0.resnets.1.temporal_res_block": "blocks.8",
            "decoder.up_blocks.0.resnets.1.time_mixer": "blocks.8",
            "decoder.up_blocks.0.resnets.2.spatial_res_block": "blocks.9",
            "decoder.up_blocks.0.resnets.2.temporal_res_block": "blocks.10",
            "decoder.up_blocks.0.resnets.2.time_mixer": "blocks.10",

            "decoder.up_blocks.1.resnets.0.spatial_res_block": "blocks.12",
            "decoder.up_blocks.1.resnets.0.temporal_res_block": "blocks.13",
            "decoder.up_blocks.1.resnets.0.time_mixer": "blocks.13",
            "decoder.up_blocks.1.resnets.1.spatial_res_block": "blocks.14",
            "decoder.up_blocks.1.resnets.1.temporal_res_block": "blocks.15",
            "decoder.up_blocks.1.resnets.1.time_mixer": "blocks.15",
            "decoder.up_blocks.1.resnets.2.spatial_res_block": "blocks.16",
            "decoder.up_blocks.1.resnets.2.temporal_res_block": "blocks.17",
            "decoder.up_blocks.1.resnets.2.time_mixer": "blocks.17",

            "decoder.up_blocks.2.resnets.0.spatial_res_block": "blocks.19",
            "decoder.up_blocks.2.resnets.0.temporal_res_block": "blocks.20",
            "decoder.up_blocks.2.resnets.0.time_mixer": "blocks.20",
            "decoder.up_blocks.2.resnets.1.spatial_res_block": "blocks.21",
            "decoder.up_blocks.2.resnets.1.temporal_res_block": "blocks.22",
            "decoder.up_blocks.2.resnets.1.time_mixer": "blocks.22",
            "decoder.up_blocks.2.resnets.2.spatial_res_block": "blocks.23",
            "decoder.up_blocks.2.resnets.2.temporal_res_block": "blocks.24",
            "decoder.up_blocks.2.resnets.2.time_mixer": "blocks.24",

            "decoder.up_blocks.3.resnets.0.spatial_res_block": "blocks.26",
            "decoder.up_blocks.3.resnets.0.temporal_res_block": "blocks.27",
            "decoder.up_blocks.3.resnets.0.time_mixer": "blocks.27",
            "decoder.up_blocks.3.resnets.1.spatial_res_block": "blocks.28",
            "decoder.up_blocks.3.resnets.1.temporal_res_block": "blocks.29",
            "decoder.up_blocks.3.resnets.1.time_mixer": "blocks.29",
            "decoder.up_blocks.3.resnets.2.spatial_res_block": "blocks.30",
            "decoder.up_blocks.3.resnets.2.temporal_res_block": "blocks.31",
            "decoder.up_blocks.3.resnets.2.time_mixer": "blocks.31",
        }
        suffix_rename_dict = {
            "norm1.weight": "norm1.weight",
            "conv1.weight": "conv1.weight",
            "norm2.weight": "norm2.weight",
            "conv2.weight": "conv2.weight",
            "conv_shortcut.weight": "conv_shortcut.weight",
            "norm1.bias": "norm1.bias",
            "conv1.bias": "conv1.bias",
            "norm2.bias": "norm2.bias",
            "conv2.bias": "conv2.bias",
            "conv_shortcut.bias": "conv_shortcut.bias",
            "mix_factor": "mix_factor",
        }

        state_dict_ = {}
        for name in static_rename_dict:
            state_dict_[static_rename_dict[name] + ".weight"] = state_dict[name + ".weight"]
            state_dict_[static_rename_dict[name] + ".bias"] = state_dict[name + ".bias"]
        for prefix_name in prefix_rename_dict:
            for suffix_name in suffix_rename_dict:
                name = prefix_name + "." + suffix_name
                name_ = prefix_rename_dict[prefix_name] + "." + suffix_rename_dict[suffix_name]
                if name in state_dict:
                    state_dict_[name_] = state_dict[name]
        self.load_state_dict(state_dict_)
    
    @staticmethod
    def from_diffusers(file_path=None, state_dict=None):
        model = SVDVAEDecoder()
        model.eval()
        model.load_state_dict_from_diffusers(file_path, state_dict)
        return model
