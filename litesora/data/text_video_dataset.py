import torch, json, os, imageio
from einops import rearrange
from PIL import Image
import numpy as np


class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, num_frames=64, height=64, width=64):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        self.path = [os.path.join(base_path, i["path"]) for i in metadata]
        self.text = [i["text"] for i in metadata]
        self.num_frames = num_frames
        self.height = height
        self.width = width

    def crop_and_resize(self, image, height, width):
        image = np.array(image)
        image_height, image_width, _ = image.shape
        if image_height / image_width < height / width:
            croped_width = int(image_height / height * width)
            left = (image_width - croped_width) // 2
            image = image[:, left: left+croped_width]
            image = Image.fromarray(image).resize((width, height))
        else:
            croped_height = int(image_width / width * height)
            left = (image_height - croped_height) // 2
            image = image[left: left+croped_height, :]
            image = Image.fromarray(image).resize((width, height))
        return image

    def load_video(self, file_path, num_frames, height, width):
        frames = []
        reader = imageio.get_reader(file_path)
        for frame in reader:
            frame = self.crop_and_resize(frame, height, width)
            frames.append(frame)
            if len(frames)>=num_frames:
                break
        frames = torch.tensor(np.stack(frames))
        reader.close()
        return frames

    def process_video_frames(self, frames):
        frames = frames / 127.5 - 1
        frames = rearrange(frames, "T H W C -> C T H W")
        return frames

    def __getitem__(self, index):
        video_file = self.path[index % len(self.path)]
        text = self.text[index % len(self.path)]
        frames = self.load_video(video_file, self.num_frames, self.height, self.width)
        frames = self.process_video_frames(frames)
        return {"frames": frames, "text": text}

    def __len__(self):
        return len(self.path)
