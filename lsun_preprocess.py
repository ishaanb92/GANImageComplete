# Author : Ishaan Bhat (i.r.bhat@student.tue.nl)

#!/usr/bin/python3

import numpy as np
import parser
import cv2
import os
import random
import tensorflow as tf


# TODO : Remove hard-coded path

def create_file_list():

    """
    Creates list of file names of all images in the dataset

    """
    dataset_path = '/home/ibhat/lsun/lsun/images'
    data_files = (os.listdir(dataset_path))
    files = [os.path.join(dataset_path,f) for f in data_files]
    return files


def convert_file_format(files,size):
    """
    Takes filename queue and returns an example from it
    using the TF Reader structure

    """
    filename_queue = tf.train.string_input_producer(files,shuffle=True)
    image_reader = tf.WholeFileReader()
    _,image_file = image_reader.read(filename_queue)
    image = tf.image.decode_jpeg(image_file)
    image = tf.image.resize_images(image, [size,size])
    image.set_shape((size,size,3))
    return image

def generate_batch(files,batch_size,image_size):

    image = convert_file_format(files,image_size)
    # Generate batch
    num_preprocess_threads = 1
    min_queue_examples = 256
    images = tf.train.shuffle_batch(
            [image],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    normImages = tf.subtract(
                 tf.div(images,
                        127.5),
                 1.)
    return normImages


if __name__ == '__main__':
    """
    Test-case

    """
    sess = tf.InteractiveSession()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord,sess=sess)
    files = create_file_list()
    batch = generate_batch(files,batch_size = 64)
