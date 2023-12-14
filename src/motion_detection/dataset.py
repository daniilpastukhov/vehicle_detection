import json
from pathlib import Path
from typing import Union, Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import VideoReader
from torchvision.transforms.functional import resize

__all__ = ['CustomDataset']


class Video:
    def __init__(self, path: Union[Path, str], annotation_dir: Union[Path, str]):
        self.path = Path(path)
        self.video_name = path.name
        self.frames = torch.tensor([])
        self.polygon = None
        self.bbox = None  # xyxy format
        self.labels = torch.tensor([])  # 0 - no vehicle, 1 - vehicle present
        self.current_frame = 0

        # read and parse polygon annotation
        polygons = json.load(Path(annotation_dir + '/polygons.json').open())
        self.polygon = np.array(polygons[self.video_name], dtype=np.int32)
        self.bbox = np.array([self.polygon[:, 0].min(),
                              self.polygon[:, 1].min(),
                              self.polygon[:, 0].max(),
                              self.polygon[:, 1].max()]).reshape(1, 4)

        self._load_frames()
        self._load_annotation(annotation_dir)

    def _load_frames(self):
        video = VideoReader(str(self.path), 'video')
        temp_frames = []

        for frame in video:
            cropped_frame = frame['data'][:, self.bbox[0, 1]:self.bbox[0, 3], self.bbox[0, 0]:self.bbox[0, 2]]
            # fill all regions outside polygon with gray color
            mask = np.zeros(cropped_frame.shape[1:], dtype=np.uint8)
            cv2.fillPoly(mask, [self.polygon - self.bbox[0, :2]], (255, 255, 255))
            cropped_frame[:, mask == 0] = 114
            cropped_frame = cropped_frame.to(torch.float32)
            cropped_frame = resize(cropped_frame, (384, 384))  # resize to 384x384 for EfficientNet
            cropped_frame /= 255  # normalize
            temp_frames.append(cropped_frame)

        self.frames = torch.stack(temp_frames)

    def _load_annotation(self, annotation_dir: Union[Path, str]):
        with open(annotation_dir + '/time_intervals.json', 'r') as f:
            time_interval = json.load(f)
        
        self.time_interval = time_interval[self.video_name]
        self.labels = torch.zeros(len(self.frames), dtype=torch.float32)

        # set labels to 1 for frames with vehicles
        for start, end in self.time_interval:
            self.labels[start:end + 1] = 1


class CustomDataset(Dataset):
    """
    Video dataset. Loads all videos from the given directory into memory.
    """
    def __init__(self,
                 videos_dir: Union[Path, str],
                 annotations_dir: Union[Path, str],
                 transform: Callable = None):
        super().__init__()

        # check if videos and annotations directories exist
        if not Path(videos_dir).exists():
            raise ValueError(f'Videos directory {videos_dir} does not exist.')
        if not Path(annotations_dir).exists():
            raise ValueError(f'Annotations directory {annotations_dir} does not exist.')

        # load all videos
        videos = [Video(video_path, annotations_dir)
                  for video_path in Path(videos_dir).glob('*.mp4')]
        
        # concatenate all frames and labels into one tensor
        self.video_frames = torch.cat([video.frames for video in videos])
        self.labels = torch.cat([video.labels for video in videos])

        self.transform = transform

    def __len__(self):
        return len(self.video_frames)

    def __getitem__(self, idx):
        frame = self.video_frames[idx]
        label = self.labels[idx]

        if self.transform:
            frame = self.transform(frame)

        return frame, label
