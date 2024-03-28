from litesora.pipelines.pixart_video_dit import PixartVideoDiTPipeline, T5EncoderModel, SVDVAEDecoder
from litesora.models.pixart_video_dit import PixartVideoDiT
from litesora.data import save_video
import torch


def load_pl_state_dict(file_path):
    state_dict = torch.load(file_path, map_location="cpu")
    if "module" in state_dict:
        state_dict = state_dict["module"]
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    state_dict = {i[len("denoising_model."):]: state_dict[i] for i in state_dict if i.startswith("denoising_model.")}
    return state_dict


denoising_model = PixartVideoDiT()
state_dict = load_pl_state_dict("lightning_logs/version_xxx/checkpoints/xxx.ckpt/checkpoint/mp_rank_00_model_states.pt")
denoising_model.load_state_dict(state_dict)

pipe = PixartVideoDiTPipeline(device="cuda", torch_dtype=torch.bfloat16)
pipe.text_encoder = T5EncoderModel.from_pretrained("models/PixArt-XL-2-512x512/text_encoder", torch_dtype=torch.bfloat16).to("cuda")
pipe.denoising_model = denoising_model.to(dtype=torch.bfloat16, device="cuda")
pipe.video_decoder = SVDVAEDecoder.from_diffusers("models/vae/model.safetensors").to(dtype=torch.bfloat16, device="cuda")

prompt = "woman, flowers, plants, field, garden"
video = pipe(prompt=prompt, num_frames=128, num_inference_steps=100, height=512, width=512, use_cfg=True, cfg_scale=4.5)
save_video(video, f"video.mp4", fps=24)
