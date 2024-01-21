import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from data.aircraft import Aircraft
from efficientnet_pytorch import EfficientNet
from Model_class import Model
IMG_SIZE = 380

file = "dddd"

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

effnetb4 = EfficientNet.from_pretrained('efficientnet-b4')
EfficientNet.get_image_size('efficientnet-b4')
x = torch.rand((1,3,380,380))

features = effnetb4.extract_features(x)


 
model = Model(effnetb4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.load_state_dict(torch.load("fgvc-aircraft-30ep-cutmix.pth" , map_location=torch.device('cpu')))

model.eval()
torch.cuda.empty_cache()


train_ds = Aircraft('./data/aircraft', train=True, download=True, transform=train_transform)
test_ds = Aircraft('./data/aircraft', train=False, download=True, transform=test_transform)



image_ids, targets, classes, class_to_idx = train_ds.find_classes()

CLASSES = [c[:-1] for c in classes]
CLS2IDX = {c[:-1]:idx for c, idx in class_to_idx.items()}


#batch_size = 20
#train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,   
                                        #shuffle=True, num_workers=2, pin_memory=True , drop_last=True)
#train_dl_iter = iter(train_dl)




def classify():


    file_path = selected_file_label.cget("text").replace("Selected File: ", "")
    fn = file_path.split("/")[-1]
    if not file_path:
        print("No file selected. Please open a file.")
        return
    print("FILE IS", file_path)
    test_img = Image.open(fn)

    
    test_img_t = test_transform(test_img).unsqueeze(0).to(device)


    print( test_img_t.shape)


    pred = torch.exp(model(test_img_t))

    pred_idx = torch.argmax(pred, dim=-1).item()

    result_label.config(text=f"Selected File: {CLASSES[pred_idx]}", fg="green")


def open_file_dialog():
    file_path = filedialog.askopenfilename(title="Select a File", filetypes=[("Image files", "*.jpg"), ("All files", "*.*")])
    if file_path:
        selected_file_label.config(text=f"Selected File: {file_path}")
        result_label.config(text="")
        process_file(file_path)

def process_file(file_path):
    try:
        image = Image.open(file_path)
        image.thumbnail((400, 400))  # Adjust the size as needed
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo
    except Exception as e:
        selected_file_label.config(text=f"Error: {str(e)}")

root = tk.Tk()
root.title("File Dialog Example")

open_button = tk.Button(root, text="Open File", command=open_file_dialog)
open_button.pack(padx=20, pady=20, anchor='w')




selected_file_label = tk.Label(root, text="Selected File:")
selected_file_label.pack()

result_label = tk.Label(root, text="Classifier result:")
result_label.pack()

classify_button = tk.Button(root, text="Classify", command=classify)
classify_button.pack(padx=20, pady=20, anchor='w')

image_label = tk.Label(root)
image_label.pack(padx=20, pady=20)

root.mainloop()
