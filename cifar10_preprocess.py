from batchup.datasets import cifar10
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def generate_dataset():
    ds = cifar10.CIFAR10(n_val=0)
    dataset = np.transpose(ds.train_X,(0,2,3,1))
    print(dataset.shape)
    return dataset

def generate_batch(dataset,batch_size,shuffle = True):
    if shuffle:
        np.random.shuffle(dataset)
    idx = np.random.randint(dataset.shape[0],size=batch_size)
    batch = dataset[idx,:,:,:]
    #for image in batch:
    #    for channel in image:
    #        scaler = MinMaxScaler(feature_range = (-1,1), copy = False)
    #        scaler.fit(channel)
    #        scaler.transform(channel)
    return batch


if __name__ == '__main__':
    dataset = generate_dataset()
    generate_batch(dataset,64)
