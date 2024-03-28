import torch, json, os
from .utils import load_video


class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, steps_per_epoch=10000, num_frames=64, height=64, width=64,
                 random_crop=False, random_interval=True, random_start=True):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        self.path = [os.path.join(base_path, i["path"]) for i in metadata]
        self.text = [i["text"] for i in metadata]
        self.steps_per_epoch = steps_per_epoch
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.random_crop = random_crop
        self.random_interval = random_interval
        self.random_start = random_start

    def __getitem__(self, index):
        while True:
            index = torch.randint(0, len(self.path), (1,))[0]
            video_file = self.path[index]
            text = self.text[index % len(self.path)]
            try:
                frames, metadata = load_video(
                    video_file, self.num_frames, self.height, self.width,
                    random_crop=self.random_crop, random_interval=self.random_interval, random_start=self.random_start
                )
            except:
                frames, metadata = None, None
            if frames is not None:
                break
        return {"frames": frames, "text": text, "start_sec": metadata["start_sec"], "end_sec": metadata["end_sec"]}

    def __len__(self):
        return self.steps_per_epoch
