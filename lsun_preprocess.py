# Author : Ishaan Bhat (i.r.bhat@student.tue.nl)

#!/usr/bin/python3

import numpy as np
import parser
import cv2
import os
import random


# TODO : Remove hard-coded path

def create_file_list():

    """
    Creates list of file names of all images in the dataset

    """
    dataset_path = '/home/ibhat/lsun/lsun/images'
    data_files = (os.listdir(dataset_path))
    files = [os.path.join(dataset_path,f) for f in data_files]
    return files

def generate_batch(files,batch_size):

    batch_files = random.sample(files,int(batch_size))
    norm_batch = []
    for bFile in batch_files:
        img = cv2.imread(bFile)
        # Resize image to 128x128
        img = cv2.resize(img,(128,128),interpolation = cv2.INTER_CUBIC)
        norm_img = normalize_image(img)
        norm_batch.append(norm_img)
    norm_batch = np.asarray(norm_batch)
    # Re-arrange the axes for TF compatibility
    return norm_batch

def normalize_image(img):
    img = np.array(img)/127.5 - 1.
    return img

if __name__ == '__main__':
    batch = generate_batch(64)
