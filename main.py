import numpy as np

import gdown
from PIL import Image



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import matplotlib.pyplot as plt

from  tqdm import tqdm
from Model_class import Model
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from data.aircraft import Aircraft
from efficientnet_pytorch import EfficientNet



"""def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
"""

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
                                        shuffle=True, num_workers=2, pin_memory=True , drop_last=True)
    train_dl_iter = iter(train_dl)




    mini_batch, labels = next(train_dl_iter)

   
    effnetb4 = EfficientNet.from_pretrained('efficientnet-b4')
    EfficientNet.get_image_size('efficientnet-b4')
    x = torch.rand((1,3,380,380))

    features = effnetb4.extract_features(x)
    print(features.shape)

    print(effnetb4._fc.in_features)

 
    model = Model(effnetb4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model.load_state_dict(torch.load("fgvc-aircraft-30ep-cutmix.pth" , map_location=torch.device('cpu')))

    model.eval()
    torch.cuda.empty_cache()


    #TESTING

    test_img = Image.open("f-18.jpg")

    print (train_ds.classes)
    
    test_img_t = test_transform(test_img).unsqueeze(0).to(device)


    print( test_img_t.shape)


    pred = torch.exp(model(test_img_t))

    pred_idx = torch.argmax(pred, dim=-1).item()
    print("Classifier result = " , CLASSES[pred_idx])


    output_class_tensor = pred[0][pred_idx]
    output_class_tensor
    grads = torch.autograd.grad(output_class_tensor, model.fmap) 

    grad = grads[0]

    pooled_grads = torch.nn.AvgPool2d(grad.shape[2])(grad).squeeze()


    conv_layer_output = model.fmap[0]
    conv_layer_output *= pooled_grads.reshape(1792, 1, -1)

    heatmap = np.mean(conv_layer_output.cpu().detach().numpy(), axis=0)


    heatmap = np.maximum(heatmap, 0) # ReLU 
    heatmap /= np.max(heatmap)       
    

    width, height = test_img.size
    test_img = test_img.convert('RGBA')
    # print(img.mode)

    cm = plt.get_cmap('jet')
    heatmap2 = Image.fromarray(np.uint8(cm(heatmap)*255))
    heatmap2 = heatmap2.resize((width, height), Image.Resampling.LANCZOS)
    #print(heatmap2.mode)

    superimposed_img = Image.blend(test_img, heatmap2, alpha=0.3)
    plt.imshow(superimposed_img)


    plt.show()