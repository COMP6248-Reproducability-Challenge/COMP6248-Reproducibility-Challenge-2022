from torchvision.models import resnet50
import torch 
import torch.nn.functional as F
from torch import nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy
import torchbearer
from torch import optim
from torch.optim import lr_scheduler
from torchbearer import Trial
from torchbearer.callbacks import Best

device = "cuda:0" if torch.cuda.is_available() else "cpu"


# custom
import load_data_script

val_image_label_dict = load_data_script.get_val_image_labels('/home/kp6g18/mydocuments/tiny-imagenet-200/val/val_annotations.txt')


train_folder = '/home/kp6g18/mydocuments/tiny-imagenet-200/train'
val_folder = '/home/kp6g18/mydocuments/tiny-imagenet-200/val'
test_folder = '/home/kp6g18/mydocuments/tiny-imagenet-200/test'

train_transform = transforms.Compose([
                        #transforms.RandomRotation(20), 
                       # transforms.RandomHorizontalFlip(0.5),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = [0.4802,0.4481,0.3975],std = [0.2302,0.2265,0.2262]),
                        ])

val_transform = transforms.Compose([          
                        transforms.ToTensor(),
                        transforms.Normalize(mean = [0.4802,0.4481,0.3975],std = [0.2302,0.2265,0.2262]),
                            ])

test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.4802,0.4481,0.3975],[0.2302,0.2265,0.2262]),
                    ])

train_dataset = ImageFolder(train_folder,train_transform)
train_loader = DataLoader(train_dataset,batch_size=100,shuffle=True)

#print(next(iter(train_loader)))

val_dataset = ImageFolder(val_folder,val_transform)
val_loader = DataLoader(val_dataset,batch_size=100,shuffle=False)

test_dataset = ImageFolder(test_folder)
test_loader = DataLoader(test_dataset,batch_size=100,shuffle=False)

# generate the first batch
(batch_images,batch_labels) = train_loader.__iter__().__next__()

#print(train_dataset.classes[batch_labels[0]])


model = resnet50(pretrained=True)
model.avgpool = nn.AdaptiveAvgPool2d(1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs,200)
model = model.to(device)
model.train()

# Freeze layers by not tracking gradients
for param in model.parameters():
    param.requires_grad = False
model.fc.weight.requires_grad = True # unfreeze last layer weights and biases
model.fc.bias.requires_grad = True
optimiser = optim.SGD(filter(lambda p: p.requires_grad,model.parameters()),lr=1e-3, momentum=0.9)
#optimiser = optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
loss_function = nn.CrossEntropyLoss()
#exp_lr_scheduler = lr_scheduler.StepLR(step_size=7,gamma=0.1)

path_to_save_cpnt = '/home/kp6g18/Documents/resnet_50_finetuned_checkpoint.pt'
checkpoint = Best(path_to_save_cpnt,save_model_params_only=True,period=5,monitor='val_acc',mode='max')

print("Using device: ",device)
trial = Trial(model,optimiser,loss_function,callbacks=[checkpoint],metrics=['loss','accuracy','top_5_acc']).to(device)
trial.with_generators(train_generator = train_loader, val_generator = val_loader)
trial.run(epochs=10)
results = trial.evaluate(data_key=torchbearer.VALIDATION_DATA)
print(results)

torch.save(model.state_dict(),"/home/kp6g18/Documents/resnet_50_finetuned.pt")


