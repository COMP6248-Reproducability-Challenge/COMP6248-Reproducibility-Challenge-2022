from torchvision.models import resnet50
import torch 
import torch.nn.functional as F
from torch import nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy

import load_data_script

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def idx_to_class(dictionary,lookup_val):
    for k,v in dictionary.items():
        if v == lookup_val:
            return k


labels_to_words = load_data_script.get_class_names()
enumerated_IN_classes = load_data_script.get_enumerated_IN_classes() # maps ImageNet folders to a number

model = resnet50(pretrained=True)
model.eval()

preprocess_input = transforms.Compose([
    transforms.Resize((224,224),interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
    ])

train_folder = '/home/kp6g18/mydocuments/tiny-imagenet-200/train'
val_folder = '/home/kp6g18/mydocuments/tiny-imagenet-200/val'
test_folder = '/home/kp6g18/mydocuments/tiny-imagenet-200/test'

train_dataset = ImageFolder(train_folder,preprocess_input)
val_dataset = ImageFolder(val_folder,preprocess_input)
test_dataset = ImageFolder(test_folder)

tin_mapping = val_dataset.class_to_idx # class to index mapping

# generate prediction
data = val_dataset[0][0]
preds = model(data.unsqueeze(0))
_,indexes = preds.topk(1)

predictions = []
accuracy = 0.0

for i in range(len(val_dataset)):
    data,true_label = val_dataset[i]
    preds = model(data.unsqueeze(0)).to(device)
    _,indexes = preds.topk(1) # returns a tensor
    
    # map index to folder name using ImageNet mapping
    folder_name = enumerated_IN_classes[indexes.item()]
   
    # map folder name to tiny image net label number
    if folder_name in tin_mapping:
        predicted_label = tin_mapping[folder_name]
        if predicted_label == true_label:
            accuracy += 1
        predictions.append(predicted_label)
    print(i)

accuracy /= len(val_dataset)
print(predictions)
print("Accuracy:",accuracy)

