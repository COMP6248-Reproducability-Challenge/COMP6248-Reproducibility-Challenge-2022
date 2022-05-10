import glob
#import pandas as pd
import numpy as np
import os
import sys
#import matplotlib.iamge as mpllimg

def get_class_ids(path):
    id_label_dict = {}

    with open(path,"r") as f:
        for line in f:
            (key,val) = line.split('\t')
            id_label_dict[key] = val.replace('\n','')
    return id_label_dict

def get_class_names(path='/home/kp6g18/Documents/texture-vs-shape/code/helper/categories.txt'):
    """Function opens image net categories and returns a dictionary containing the folder name and its english category
    array contains all the folder names so that they can be enumerated
    returns: tuple (dict,array)
    """
    id_label_dict = {}
    keys = []

    with open(path,"r") as f:
        for line in f:
            split_line = line.split(' ')
            key = split_line[0]
            keys.append(key)
            split_line[-1] = split_line[-1][0:-1]
            val = ''
            for i in range(1,len(split_line)):
                    val = val + split_line[i]
         
            id_label_dict[key] = val
    return id_label_dict,keys

def get_enumerated_IN_classes():
    _,keys_arr = get_class_names()
    enumerated_classes = {}
    for i in range(len(keys_arr)):
        enumerated_classes[i] = keys_arr[i]
    return enumerated_classes



def get_val_image_labels(path):
    val_image_labels_dict = {}

    with open(path,"r") as f:
        for line in f:
            val_data = line.split('\t')
            #print(val_data[0],val_data[1])
            val_image_labels_dict[val_data[0]] = val_data[1]
    return val_image_labels_dict


def load_timg200_val(path,id_label_dict,df):
    val_data = pd.read_csv(path+'/val_annotations.txt',sep='\t',header=None,names=['File','Class_ID','X','Y','H','W'])
    data_path = os.path.joing(path+'\\images\\','*JPEG')
    images = glob.glob(data_path)

def get_tiny_imagenet_classes():
     
    in_categories = '/home/kp6g18/Documents/texture-vs-shape/code/helper/categories.txt'
    class_mappings = get_class_names(in_categories)
    #print(class_mappings) # full imagenet class mappings


    train_folder = '/home/kp6g18/mydocuments/tiny-imagenet-200/train/*'

    paths = glob.glob(train_folder)
    categories = []

    for path in paths:
        category = path.split('/')[-1]
        categories.append(category)
        #print(category, class_mappings[category])

    


#df_train = pd.DataFrame(columns=['Image','Class_OHE','Class_ID','Class_Labels'])
#df_val = pd.DataFrame(columns=['Image','Class_ID','Class_Labels'])i

#id_label_dict = get_val_image_labels('/home/kp6g18/mydocuments/tiny-imagenet-200/val/val_annotations.txt')
#print(id_label_dict)  
#get_tiny_imagenet_classes()
#print(get_enumerated_IN_classes())

