import ipdb
import torch
import torch.nn as nn
import numpy as np
import os
import os.path as osp
import cv2
from PIL import Image
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for live, demo in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(live, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(live ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


class LiveDataset(Dataset):
    def __init__(self, live_loc, transform=None, train=True):
        self.live_img = sorted([osp.join(live_loc, f) for f in os.listdir(live_loc)])
        self.transform = transform

    def __len__(self):
        return len(self.live_img)

    def __getitem__(self, idx):
        live_img = Image.open(self.live_img[idx])

        if self.transform is not None:
            live_img = self.transform(live_img)

        return live_img, idx


class DemoDataset(Dataset):
    def __init__(self, demo_loc, transform=None, train=True):
        self.demo_img = sorted([osp.join(demo_loc, f) for f in os.listdir(demo_loc)])
        self.transform = transform

    def __len__(self):
        return len(self.demo_img)

    def __getitem__(self, idx):
        demo_img = Image.open(self.demo_img[idx])

        if self.transform is not None:
            demo_img = self.transform(demo_img)

        return demo_img, idx


class RewardDataset(Dataset):
    def __init__(self, target_file, live_loc, demo_loc, transform=None, train=True):
        self.target_file = np.load(target_file)['arr_0']
        self.live_img = sorted([osp.join(live_loc, f) for f in os.listdir(live_loc)])
        self.demo_img = sorted([osp.join(demo_loc, f) for f in os.listdir(demo_loc)])
        self.transform = transform

    def __len__(self):
        return len(self.live_img)

    def __getitem__(self, idx):
        live_img = Image.open(self.live_img[idx])
        demo_idx = np.random.randint(0, len(self.demo_img), 1)[0]
        demo_img = Image.open(self.demo_img[demo_idx])

        target = torch.Tensor([self.target_file[idx, demo_idx]]).to(torch.int64)
        target_value = F.one_hot(target, num_classes=2)[0]

        if self.transform is not None:
            live_img = self.transform(live_img)
            demo_img = self.transform(demo_img)

        return live_img, demo_img, target_value

