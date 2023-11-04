import numpy as np
import matplotlib.pyplot as plt

from  tqdm import tqdm

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from data.aircraft import Aircraft


IMG_SIZE = 380

train_transform = T.Compose(
    [
        T.Resize((IMG_SIZE,IMG_SIZE)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.3),
        T.RandomRotation(degrees=(-15, 15)),
        T.ToTensor()
    ]
)

test_transform = T.Compose(
    [
        T.Resize((IMG_SIZE,IMG_SIZE)),
        T.ToTensor()
    ]
)


train_ds = Aircraft('./data/aircraft', train=True, download=True, transform=train_transform)
test_ds = Aircraft('./data/aircraft', train=False, download=True, transform=test_transform)