from litesora.models.utils import load_state_dict
from litesora.models.sdxl_3d_unet import SDXL3DUNet
from litesora.models.sd_vae import SDVAEDecoder
from litesora.pipelines.sdxl_3d import SDXLVideoPipeline, SDXLTextEncoder, SDXLTextEncoder2
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



# Load models
state_dict = load_state_dict("models/dreamshaperXL_v21TurboDPMSDE.safetensors")
pipe = SDXLVideoPipeline(device="cuda")

pipe.text_encoder = SDXLTextEncoder.from_civitai(state_dict=state_dict).to(dtype=torch.float16, device="cuda")
pipe.text_encoder_2 = SDXLTextEncoder2.from_civitai(state_dict=state_dict).to(dtype=torch.float16, device="cuda")
pipe.unet = SDXL3DUNet.from_civitai(state_dict=state_dict).to(dtype=torch.float16, device="cuda")
pipe.vae_decoder = SDVAEDecoder.from_civitai(state_dict=state_dict).to(dtype=torch.float16, device="cuda")
state_dict = load_pl_state_dict("lightning_logs/version_xxx/checkpoints/xxx.ckpt/checkpoint/mp_rank_00_model_states.pt")
pipe.unet.load_state_dict(state_dict)
pipe.unet.eval()

prompt = "woman, flowers, plants, field, garden"
negative_prompt = ""

torch.manual_seed(0)
video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    cfg_scale=5,
    num_frames=128, height=512, width=512, num_inference_steps=60,
)
save_video(video, "video.mp4")
