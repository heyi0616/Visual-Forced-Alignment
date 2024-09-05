import torch
import random
import numpy as np


class RandomCropping(object):
    def __init__(self, crop_size=(88, 88)):
        self.crop_size = crop_size

    def __call__(self, img_sequence):
        height, width = img_sequence.shape[2:]
        h_start = random.randint(0, height - self.crop_size[0])
        w_start = random.randint(0, width - self.crop_size[1])
        img_sequence = img_sequence[:, :, h_start:h_start + self.crop_size[0], w_start:w_start + self.crop_size[1]]
        return img_sequence


class CenterCropping(object):
    def __init__(self, crop_size=(88, 88)):
        self.crop_size = crop_size

    def __call__(self, img_sequence):
        height, width = img_sequence.shape[2:]
        h_start = (height - self.crop_size[0]) // 2
        w_start = (width - self.crop_size[1]) // 2
        img_sequence = img_sequence[:, :, h_start:h_start + self.crop_size[0], w_start:w_start + self.crop_size[1]]
        return img_sequence


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std


class TimeMasking(object):
    def __init__(self, video_fps):
        self.video_fps = video_fps

    def __call__(self, img_sequence):
        clip = int(np.ceil(img_sequence.shape[1] / self.video_fps))
        n_max = int(self.video_fps * 0.4)
        n_last = int(img_sequence.shape[1] % self.video_fps * 0.4)
        ori_sequence = img_sequence.clone()
        for i in range(clip):
            if i < clip - 1:
                mask_frame_per_clip = random.randint(0, n_max)
                start_mask_frame = random.randint(0, self.video_fps - mask_frame_per_clip)
            else:
                mask_frame_per_clip = random.randint(0, n_last)
                start_mask_frame = random.randint(0, img_sequence.shape[1] % self.video_fps - mask_frame_per_clip)
            mask_idx = np.arange(start_mask_frame + i * self.video_fps, start_mask_frame + mask_frame_per_clip + i * self.video_fps)
            alt_frame_id = start_mask_frame - 1 + i * self.video_fps if start_mask_frame > 0 else i * self.video_fps  # 用前一帧mask
            img_sequence[:, mask_idx, :, :] = ori_sequence[:, alt_frame_id, :, :].unsqueeze(1)
        return img_sequence


class RandomFlipping(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_sequence):
        if random.random() > self.p:
            img_sequence = torch.flip(img_sequence, dims=[3])
        return img_sequence


class Compose(object):
    """
    input sequence: (channels, frames, H, W), type: torch.Tensor
    """
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, img_sequence):
        for t in self.transforms:
            img_sequence = t(img_sequence)
        return img_sequence
