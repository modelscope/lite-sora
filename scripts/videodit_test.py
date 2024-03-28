import torch
from litesora.pipelines.latent_video_dit import LatentVideoDiTPipeline, SDTextEncoder, VideoDiT, SVDVAEDecoder
from litesora.data import save_video


def load_pl_state_dict(file_path):
    state_dict = torch.load(file_path, map_location="cpu")
    if "module" in state_dict:
        state_dict = state_dict["module"]
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    state_dict = {i[len("denoising_model."):]: state_dict[i] for i in state_dict if i.startswith("denoising_model.")}
    return state_dict


state_dict = load_pl_state_dict("lightning_logs/version_xxx/checkpoints/xxx.ckpt/checkpoint/mp_rank_00_model_states.pt")
denoising_model = VideoDiT().eval()
denoising_model.load_state_dict(state_dict)

pipe = LatentVideoDiTPipeline(device="cuda")
pipe.fetch_models(
    SDTextEncoder.from_diffusers("models/text_encoder_768/model.safetensors"),
    denoising_model,
    SVDVAEDecoder.from_diffusers("models/vae/model.safetensors")
)

prompt = "woman, flowers, plants, field, garden"
video = pipe(prompt=prompt, num_frames=128, num_inference_steps=100, start_sec=0.0, end_sec=8.0, use_cfg=False)
save_video(video, "video.mp4", fps=24)
