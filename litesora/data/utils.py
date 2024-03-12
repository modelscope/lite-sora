import imageio
import numpy as np
from PIL import Image


def save_video(frames, save_path, fps=30, quality=5, upscale=1):
    height, width, _ = frames[0].shape
    writer = imageio.get_writer(save_path, fps=fps, quality=quality)
    for frame in frames:
        frame = np.array(Image.fromarray(frame).resize((width*upscale, height*upscale), Image.NEAREST))
        writer.append_data(frame)
    writer.close()
