# COMP6248-Reproducibility-Challenge-2022

Trying to reproduce the paper "ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness "
https://openreview.net/forum?id=Bygh9j09KX

## test_models_general.py
pretrained models from original paper are loaded using the code in this repository in models/load_pretrained_models.py:
https://github.com/rgeirhos/texture-vs-shape

## tin_data_folders_format.py and stin_data_folders_format.py
These scripts are used to format the validation folders for the downloaded tiny-imagenet and stylized tiny-imagenet into a format where each image is in the folder corresponding to its category. This means the data can be loaded easily using torchvision.datasets ImageFolder. 
