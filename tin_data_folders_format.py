import io
import glob
import os
from shutil import move
from os.path import join
from os import listdir, rmdir

# custom 
import load_data_script

target_folder = '/home/kp6g18/mydocuments/tiny-imagenet-200/val/'
test_folder = '/home/kp6g18/mydocuments/tiny-imagenet-200/test/'

val_dict = load_data_script.get_val_image_labels('/home/kp6g18/mydocuments/tiny-imagenet-200/val/val_annotations.txt')

paths = glob.glob('/home/kp6g18/mydocuments/tiny-imagenet-200/val/images/*')
for i,path in enumerate(paths):
    file = path.split('/')[-1]
    folder = val_dict[file]
    if not os.path.exists(target_folder + str(folder)):
        os.mkdir(target_folder + str(folder))
        os.mkdir(target_folder + str(folder) + '/images')
    if not os.path.exists(test_folder + str(folder)):
        os.mkdir(test_folder + str(folder))
        os.mkdir(test_folder + str(folder) + '/images')
    print(i)

print("finished making directories")
print("moving images....")
for path in paths:
    file = path.split('/')[-1]
    folder = val_dict[file]
    if len(glob.glob(target_folder + str(folder) + '/images/*')) < 25:
        dest = target_folder + str(folder) + '/images/' + str(file)
    else:
        dest = test_folder + str(folder) + '/images/' + str(file)
    move(path,dest)

