import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torchaudio

from utils import data_augment_util as transforms
from data import consts
CROP_SIZE = [88, 88]


def get_frame_label(m, phoneme_root, frame_len, index_start=0):
    label_path = phoneme_root + m + ".txt"
    with open(label_path, 'r') as f:
        splits = f.readlines()
    clip_splits = splits[1:-4]
    phoneme_label = []
    phoneme_label_with_sil = []
    # video_frames = int(splits[-1].split(":")[1])
    # frame_ids = np.arange(TIMIT_TRUNC_LEN)
    SIL_INDEX = consts.SIL
    frame_label = [SIL_INDEX] * frame_len
    pre_end = 0
    for index, split in enumerate(clip_splits):
        phone_id, start, end = [int(i) for i in split.split()]
        if start >= frame_len:
            break
        if end >= frame_len:
            end = frame_len
        phoneme_label.append(phone_id + index_start)
        if start > pre_end:
            phoneme_label_with_sil.append(SIL_INDEX)
        pre_end = end
        phoneme_label_with_sil.append(phone_id + index_start)

        for i in range(start, end):
            frame_label[i] = phone_id + index_start
    frame_label = torch.tensor(frame_label)
    phoneme_label.insert(0, consts.BOS)  # bos
    phoneme_label.append(consts.EOS)  # eos
    phoneme_label = torch.tensor(phoneme_label)

    if pre_end < frame_len:
        phoneme_label_with_sil.append(SIL_INDEX)
    phoneme_label_with_sil.insert(0, consts.BOS)  # bos
    phoneme_label_with_sil.append(consts.EOS)  # eos
    phoneme_label_with_sil = torch.tensor(phoneme_label_with_sil)
    return frame_label, phoneme_label, phoneme_label_with_sil


class LRS2Dataset(Dataset):
    def __init__(self, mode):
        super(LRS2Dataset, self).__init__()
        assert mode in ["train", "val", "test"]
        self.mouth_root = "./LRS2/mouth_grey_96/main/"
        split_root = "./LRS2/data_split/"
        self.align_root = "./LRS2/wav_align/main/"
        self.label_root = "./LRS2/audio/main/"
        self.mode = mode
        file_list = []
        with open(os.path.join(split_root, mode + ".txt"), "r") as f:
            for line in f.readlines():
                p = line.split()[0]
                mouth_path = os.path.join(self.mouth_root, p)
                if os.path.exists(mouth_path) and os.path.exists(self.align_root + p + ".txt"):
                    file_list.append(p)
        print("file length: ", len(file_list))
        self.file_list = file_list
        self.train_transform = transforms.Compose([
            transforms.RandomCropping(crop_size=CROP_SIZE),
            transforms.RandomFlipping(p=0.5),
            transforms.TimeMasking(video_fps=25),
        ])
        self.test_transform = transforms.Compose([
            transforms.CenterCropping(crop_size=CROP_SIZE)
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        m = self.file_list[index]
        # print(m)
        img_tensor_list = []

        mouth_path = os.path.join(self.mouth_root, m)
        _frame_length = len(os.listdir(mouth_path))
        for index, img_name in enumerate(sorted(os.listdir(mouth_path))):
            img = cv2.imread(os.path.join(mouth_path, img_name), cv2.IMREAD_GRAYSCALE)
            img = img / 255.0
            if len(img.shape) == 2:
                img = np.expand_dims(img, -1)
            img_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
            img_tensor_list.append(img_tensor)
        img_sequence = torch.stack(img_tensor_list, 1)  # (channels, frames, H, W)
        img_len = len(img_tensor_list)
        if self.mode in ["train"]:
            img_sequence = self.train_transform(img_sequence)
        elif self.mode in ["test", "val"]:
            img_sequence = self.test_transform(img_sequence)
        frame_label, phoneme_label, phoneme_label_with_sil = get_frame_label(m, self.align_root, img_len, index_start=2)
        return img_sequence, frame_label, phoneme_label, phoneme_label_with_sil


def collate_fn(batch):
    img_batch, frame_label_batch, phoneme_label_batch = [], [], []
    bound_batch = []
    for img_sequence, frame_label, phoneme_label, boundary_label in batch:
        img_batch.append(img_sequence.permute(1, 0, 2, 3))
        frame_label_batch.append(frame_label)
        phoneme_label_batch.append(phoneme_label)
        bound_batch.append(boundary_label)

    img_sequence = pad_sequence(img_batch, batch_first=True, padding_value=0)
    img_sequence = img_sequence.permute(0, 2, 1, 3, 4)
    frame_label = pad_sequence(frame_label_batch, batch_first=True, padding_value=0)
    phoneme_label = pad_sequence(phoneme_label_batch, batch_first=True, padding_value=0)
    boundary_label = pad_sequence(bound_batch, batch_first=True, padding_value=0)
    return img_sequence, frame_label, phoneme_label, boundary_label