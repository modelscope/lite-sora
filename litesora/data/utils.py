import imageio
import numpy as np
from PIL import Image
import torch
from einops import rearrange


def save_video(frames, save_path, fps=30, quality=5, upscale=1):
    height, width, _ = frames[0].shape
    writer = imageio.get_writer(save_path, fps=fps, quality=quality)
    for frame in frames:
        frame = np.array(Image.fromarray(frame).resize((width*upscale, height*upscale), Image.NEAREST))
        writer.append_data(frame)
    writer.close()


def crop_and_resize(image, height, width, start_height, start_width):
    image = np.array(image)
    image_height, image_width, _ = image.shape
    if image_height / image_width < height / width:
        croped_width = int(image_height / height * width)
        left = start_width
        image = image[:, left: left+croped_width]
        image = Image.fromarray(image).convert("RGB").resize((width, height))
    else:
        croped_height = int(image_width / width * height)
        left = start_height
        image = image[left: left+croped_height, :]
        image = Image.fromarray(image).convert("RGB").resize((width, height))
    return image


def load_video(file_path, num_frames, height, width, random_crop=True):
    frames = []
    reader = imageio.get_reader(file_path)
    if reader.count_frames() < num_frames:
        return None
    if random_crop:
        start_frame = torch.randint(0, reader.count_frames() - num_frames + 1, (1,))[0]
    else:
        start_frame = 0
    w, h = reader.get_meta_data()["size"]
    if width / height < w / h:
        position = torch.rand(1)[0] if random_crop else 0.5
        start_width = int(position * (w - h / height * width))
        start_height = 0
    else:
        start_width = 0
        position = torch.rand(1)[0] if random_crop else 0.5
        start_height = int(position * (h - w / width * height))
    for frame_id in range(start_frame, start_frame + num_frames):
        frame = reader.get_data(frame_id)
        frame = crop_and_resize(frame, height, width, start_height, start_width)
        frames.append(frame)
    frames = torch.tensor(np.stack(frames))
    frames = frames / 127.5 - 1
    frames = rearrange(frames, "T H W C -> C T H W")
    reader.close()
    return frames


def tensor2video(frames):
    frames = rearrange(frames, "C T H W -> T H W C")
    frames = ((frames + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    return frames


def concat_video(videos):
    video = torch.concat(videos, dim=-1)
    return video
