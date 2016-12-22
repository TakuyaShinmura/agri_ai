#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from data_sets import DataSets
import tensorflow as tf
import random


tf.app.flags.DEFINE_string("train_dir","data/","Work directory.")


FLAGS = tf.app.flags.FLAGS
DATA_SETS_FILE = 'data.csv'

def main(_):
    data_sets = DataSets(FLAGS.train_dir+DATA_SETS_FILE)
    data_sets.get_next_batch(3)
    print(data_sets.validation_data[0])

if __name__ == "__main__":
    tf.app.run()