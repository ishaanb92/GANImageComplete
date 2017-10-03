# Author : Ishaan Bhat [i.r.bhat@student.tue.nl]

#!/usr/bin/python3

from batchup.datasets import cifar10
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def generate_dataset():
    ds = cifar10.CIFAR10(n_val=0)
    dataset = np.transpose(ds.train_X,(0,2,3,1))
    print(dataset.shape)
    return dataset

def generate_batch(dataset,batch_size,shuffle = True):
    """
    Returns a randonly chosen normalized batch

    """
    if shuffle:
        np.random.shuffle(dataset)
    idx = np.random.randint(dataset.shape[0],size=batch_size)
    batch = dataset[idx,:,:,:]
    # Normalize
    for image in batch:
        image = (image/127.5) - 1

    return batch

if __name__ == '__main__':
    dataset = generate_dataset()
    generate_batch(dataset,64)
