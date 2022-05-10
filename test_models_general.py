from torchvision.models import resnet50
import torch 
import torch.nn.functional as F
from torch import nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy
import argparse

import load_data_script
import load_pretrained_models

def parser_models_help():
    Model_A  = "resnet_50_trained_on_SIN"
    Model_B = "resnet50_trained_on_SIN_and_IN"
    Model_C = "resnet50_trained_on_sin_and_IN_then_finetuned_on_IN"
    Model_D = "resnet50_trained_on_IN"
    Model_E = "resnet50_finetuned_on_TIN"
    text = 'Input a model name either: ' + "\n" + 'Model_A: ' + Model_A + "\n" + 'Model_B: ' + Model_B + "\n" + "Model_C: " + Model_C + "\n" + "Model_D: " + Model_D + "\n" + "Model_E: " + Model_E  
    return text    

model_names = ['Model_A','Model_B','Model_C','Model_D','Model_E']
model_name_dict = {
        "Model_A":"resnet50_trained_on_SIN",
        "Model_B":"resnet50_trained_on_SIN_and_IN",
        "Model_C":"resnet50_trained_on_sin_and_IN_then_finetuned_on_IN",
        "Model_D":"resnet50_trained_on_IN",
        "Model_E":"resnet50_finetuned_on_TIN"
        }

parser = argparse.ArgumentParser(description='Models Testing on tiny images')
parser.add_argument('--modelname',choices = model_names ,help=parser_models_help())
parser.add_argument('--testset',choices = ['tin','stin'],help='Enter name of dataset to test model on, either tin or stin')

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

def idx_to_class(dictionary,lookup_val):
    for k,v in dictionary.items():
        if v == lookup_val:
         return k

def main():
    args = parser.parse_args()

    labels_to_words = load_data_script.get_class_names()
    enumerated_IN_classes = load_data_script.get_enumerated_IN_classes() # maps ImageNet folders to a number

    model_name = str(args.modelname)

    if model_name == 'Model_D':
        model = resnet50(pretrained=True)
        model.eval()
    elif model_name == 'Model_E':
        model = torch.load('resnet_50_finetuned.pt')
    else:
        model_to_load = model_name_dict[model_name]
        print(model_to_load)

        model = load_pretrained_models.load_model(model_to_load)
        model.eval()
    

    preprocess_input = transforms.Compose([
        transforms.Resize((224,224),interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
        ])

    if args.testset == 'tin':
        print("testing on Tiny Image Net")
        train_folder = '/home/kp6g18/mydocuments/tiny-imagenet-200/train'
        val_folder = '/home/kp6g18/mydocuments/tiny-imagenet-200/val'
        test_folder = '/home/kp6g18/mydocuments/tiny-imagenet-200/test'

        #train_dataset = ImageFolder(train_folder,preprocess_input)
        val_dataset = ImageFolder(val_folder,preprocess_input)
        #test_dataset = ImageFolder(test_folder)
    
    elif args.testset == 'stin':
        print("Testing on Stylized Tiny Image Net")
        train_folder = '/home/kp6g18/Documents/stylized-images/train'
        val_folder = '/home/kp6g18/Documents/stylized-images/val'
        test_folder = '/home/kp6g18/Documents/stylized-images/test'

        #train_dataset = ImageFolder(train_folder,preprocess_input)
        val_dataset = ImageFolder(val_folder,preprocess_input)
        #test_dataset = ImageFolder(test_folder)


    tin_mapping = val_dataset.class_to_idx # class to index mapping

    #predictions = []
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
            #predictions.append(predicted_label)
        print(i)

    accuracy /= len(val_dataset)
    #print(predictions)
    print("Accuracy:",accuracy)

if __name__=='__main__':
    main()

