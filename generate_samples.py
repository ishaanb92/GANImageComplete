
#!/usr/bin/env python3
#
# Brandon Amos (http://bamos.github.io)
# License: MIT
# 2016-08-05

import argparse
import os
import tensorflow as tf

from model import DCGAN
from utils import visualize
import numpy as np

# Use GPU1
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size",64, "The size of batch images [64]")
flags.DEFINE_integer("image_size",64 , "The size of image to use")
flags.DEFINE_string("dataset", "lfw-aligned-64", "Dataset directory.")
flags.DEFINE_string("checkpoint_dir", "checkpoint_lsun_64", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples_lsun_64_test", "Directory name to save the image samples [samples]")
flags.DEFINE_string("num_batches",50, "Number of batches to generate")
FLAGS = flags.FLAGS


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    dcgan = DCGAN(sess, image_size=FLAGS.image_size,
                  batch_size=FLAGS.batch_size,
                  checkpoint_dir=FLAGS.checkpoint_dir,is_crop=False)
    dcgan.generate_samples(config,num_batches=FLAGS.num_batches,samples_dir=FLAGS.sample_dir)
