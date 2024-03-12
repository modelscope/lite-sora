import torch
from safetensors import safe_open


def load_state_dict(file_path, torch_dtype=None):
    if file_path.endswith(".safetensors"):
        state_dict = load_state_dict_from_safetensors(file_path, torch_dtype=torch_dtype)
    else:
        state_dict = load_state_dict_from_bin(file_path, torch_dtype=torch_dtype)
    return state_dict


def load_state_dict_from_safetensors(file_path, torch_dtype=None):
    state_dict = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
            if torch_dtype is not None:
                state_dict[k] = state_dict[k].to(torch_dtype)
    return state_dict


def load_state_dict_from_bin(file_path, torch_dtype=None):
    state_dict = torch.load(file_path, map_location="cpu")
    if torch_dtype is not None:
        state_dict = {i: state_dict[i].to(torch_dtype) for i in state_dict}
    return state_dict
