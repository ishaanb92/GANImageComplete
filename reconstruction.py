#!/usr/bin/env python3

# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/main.py
#   + License: MIT
# [2016-08-05] Modifications for Inpainting: Brandon Amos (http://bamos.github.io)
#   + License: MIT
# [2017-10-21] Creating a script to reconstruct images : Ishaan Bhat

import os
import scipy.misc
import numpy as np

from model import DCGAN
from utils import pp, visualize, to_json

import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

flags = tf.app.flags
flags.DEFINE_float("learning_rate", 0.01, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam [0.5]")
flags.DEFINE_float("beta2",0.9999,"RMS term of adam")
flags.DEFINE_integer("batch_size", 1, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 64, "The size of image to use")
flags.DEFINE_integer("nIter",1500, "Iteration to update z")
flags.DEFINE_integer("samples_per_image",1,"Number of generated samples per true image")
flags.DEFINE_string("images", "lfw-aligned-64", "Images directory.")
flags.DEFINE_integer("num_images",50,"Number of images to be chosen from training data")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("outDir", "recon_celebA", "Directory name to save the image samples [samples]")
FLAGS = flags.FLAGS

if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    dcgan = DCGAN(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size,
                  is_crop=False, checkpoint_dir=FLAGS.checkpoint_dir,images_dir = FLAGS.images)

    dcgan.reconstruct(FLAGS)
