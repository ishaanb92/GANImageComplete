# Author : Ishaan Bhat [i.r.bhat@student.tue.nl]

#!/usr/bin/python3

from batchup.datasets import cifar10
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def generate_dataset():
    ds = cifar10.CIFAR10(n_val=0)
    dataset = np.transpose(ds.train_X,(0,2,3,1))
    return dataset

def generate_batch(dataset,batch_size,shuffle = True):
    """
    Returns a randonly chosen normalized batch

    """
    if shuffle:
        np.random.shuffle(dataset)
    idx = np.random.randint(dataset.shape[0],size=batch_size)
    batch = dataset[idx,:,:,:]
    normalized_batch = []
    # Images must be moved from [0,1] -> [-1,1]
    for image in batch:
        norm_image = np.array(image)/0.5 - 1.
        normalized_batch.append(norm_image)
    return np.asarray(normalized_batch)

if __name__ == '__main__':
    dataset = generate_dataset()
    generate_batch(dataset,64)
