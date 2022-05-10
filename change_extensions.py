import os
import glob
import io
from os import rename,listdir,rmdir

def change_val_extensions():
    val_folder = '/home/kp6g18/mydocuments/tiny-imagenet-200/val/*'

    paths = glob.glob(val_folder)

    for path in paths[1:]:
        folder = path + '/*'
        paths_2 = glob.glob(folder)
        if paths_2:
            for fname in os.listdir(paths_2[0]): # returns 1D array of string = '/home/kp6g18/tiny-imagenet-200/val/n090321/images'
                prefix = fname.split('.')[0]
                old_name = paths_2[0] +'/' + fname
                new_name = paths_2[0] + '/' + prefix + '.jpeg'
                os.rename(old_name,new_name)

def change_train_extensions():
    train_folder = '/home/kp6g18/mydocuments/tiny-imagenet-200/train/*'

    paths = glob.glob(train_folder)

    for path in paths:
        folder = path + '/*'
        paths_2 = glob.glob(folder)
        if paths_2:
            for path2 in paths_2:
                if os.path.isdir(path2):
      
                    for fname in os.listdir(path2): # returns 1D array of string = '/home/kp6g18/tiny-imagenet-200/train/n090321/images'
                        prefix = fname.split('.')[0]
                        old_name = path2 +'/' + fname
                        new_name = path2 + '/' + prefix + '.jpeg'
                        #print(old_name,new_name)
                        os.rename(old_name,new_name)
 
def change_test_extensions():
    test_folder = '/home/kp6g18/mydocuments/tiny-imagenet-200/test/*'

    paths = glob.glob(test_folder)

    for path in paths:
        if not path.endswith('images'):
            folder = path + '/*'
            paths_2 = glob.glob(folder)
            if paths_2:
                for fname in os.listdir(paths_2[0]): # returns 1D array of string = '/home/kp6g18/tiny-imagenet-200/val/n090321/images'
                    prefix = fname.split('.')[0]
                    old_name = paths_2[0] +'/' + fname
                    new_name = paths_2[0] + '/' + prefix + '.jpeg'                    
                    #print(old_name,new_name)
                    os.rename(old_name,new_name)
   
 



change_test_extensions()




    
    
    

