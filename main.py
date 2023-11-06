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
from efficientnet_pytorch import EfficientNet


if __name__ == '__main__':
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



    image_ids, targets, classes, class_to_idx = train_ds.find_classes()

    CLASSES = [c[:-1] for c in classes]
    CLS2IDX = {c[:-1]:idx for c, idx in class_to_idx.items()}


    batch_size = 20
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,   
                                        shuffle=True, num_workers=2, pin_memory=True)
    train_dl_iter = iter(train_dl)




    mini_batch, labels = next(train_dl_iter)

    # 노멀라이즈 전으로 돌린 다음 64개만 잘라온다.
    # mini_batch = (mini_batch * stds_tns + means_tns)[:64]

   
    effnetb4 = EfficientNet.from_pretrained('efficientnet-b4')
    EfficientNet.get_image_size('efficientnet-b4')
    x = torch.rand((1,3,380,380))

    features = effnetb4.extract_features(x)
    features.shape