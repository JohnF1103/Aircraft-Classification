import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from data.aircraft import Aircraft
from efficientnet_pytorch import EfficientNet
from Model_class import Model

import torch.nn.functional as nnf
import customtkinter


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
    test_img = Image.open("Test_Images/"+fn)

    
    test_img_t = test_transform(test_img).unsqueeze(0).to(device)


    print( test_img_t.shape)


    pred = torch.exp(model(test_img_t))



    # Get the predicted class and associated probability
    max_prob, predicted_class = torch.max(pred, 1)

    # Print the confidence (probability) of the model's prediction

    pred_idx = torch.argmax(pred, dim=-1).item()


    confidence_value = int(max_prob.item() * 100)

   

    result_label.config(text=f"Classifier Result: {CLASSES[pred_idx]}", fg="green",font=("Arial", 14))
    confidence_label.config(text=f"Confidence: {confidence_value}%", fg="green" if confidence_value > 70 else "yellow" if 45 <= confidence_value <= 70 else "red", font=("Arial", 14))

    makeHeatMap(pred,pred_idx, test_img)

def makeHeatMap(pred, pred_idx, img):



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
    

    width, height = img.size
    test_im = img.convert('RGBA')
    # print(img.mode)

    cm = plt.get_cmap('jet')
    heatmap2 = Image.fromarray(np.uint8(cm(heatmap)*255))
    heatmap2 = heatmap2.resize((width, height), Image.Resampling.LANCZOS)
    #print(heatmap2.mode)

    superimposed_img = Image.blend(test_im, heatmap2, alpha=0.3)
    superimposed_img.thumbnail((400, 400))  # Adjust the size as needed

    photo = ImageTk.PhotoImage(superimposed_img)
    image_label.config(image=photo)
    image_label.image = photo





def open_file_dialog():
    file_path = filedialog.askopenfilename(title="Select a File", filetypes=[("Image files", "*.jpg"), ("All files", "*.*")])
    if file_path:
        selected_file_label.config(text=f"Selected File: {file_path}")
        result_label.config(text="")
        confidence_label.config(text="")
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




root.title("Classifier")
root.geometry("500x500")


open_button = tk.Button(root, text="Open File", command=open_file_dialog)
open_button.pack(padx=20, pady=20, anchor='w')




selected_file_label = tk.Label(root, text="Selected File:")
selected_file_label.pack()

result_label = tk.Label(root, text="Classifier result:")
result_label.pack()

confidence_label = tk.Label(root, text="Confidence:")
confidence_label.pack()

classify_button = tk.Button(root, text="Classify", command=classify)
classify_button.pack(padx=20, pady=20, anchor='w')

image_label = tk.Label(root)
image_label.pack(padx=20, pady=20)

root.mainloop()
